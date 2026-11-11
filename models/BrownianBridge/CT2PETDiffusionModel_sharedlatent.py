import itertools
import torch
import os
import torch.nn as nn
from torch.optim import Adam  # Import Adam optimizer
from tqdm.autonotebook import tqdm
from models.BrownianBridge.BrownianBridgeModel import BrownianBridgeModel  # Import BrownianBridgeModel
from models.autoencoder_ctpet import Autoencoder  # Assuming you have the Autoencoder class defined
from models.BrownianBridge.base.modules.modules import SpatialRescaler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode does not change anymore."""
    return self

class CT2PETDiffusionModel(BrownianBridgeModel):  # Ensure BrownianBridgeModel is inherited
    def __init__(self, config, device):
        super().__init__(config)  # Initialize BrownianBridgeModel
        self.config = config

        # Initialize the autoencoder (already done in the previous code)
        self.autoencoder = Autoencoder(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            num_res_blocks=[2, 2, 2],
            attention_levels=[False, False, False],
            num_channels=[32, 64, 128],
            latent_channels=3,
            norm_num_groups=32,
            norm_eps=1e-6,
            with_encoder_nonlocal_attn=False,
            with_decoder_nonlocal_attn=False,
            include_fc=False,
            use_combined_linear=False,
            use_flash_attention=False,
            use_checkpointing=False,
            use_convtranspose=False,
            norm_float16=False,
            print_info=False,
            save_mem=True,
        ).to(device)


        # Load the district-specific checkpoint for the autoencoder
        self.autoencoder = self.load_autoencoder_checkpoint(self.autoencoder, config.district_path)
        print(f"Autoencoder loaded from {config.district_path}")
        self.autoencoder.eval()  # Set the autoencoder in eval mode for inference
        self.autoencoder.train = disabled_train
        for param in self.autoencoder.parameters():
            param.requires_grad = False  # Freeze the autoencoder to avoid gradients during training
        print("Autoencoder loadedcheckpointssharedlatent successfully.")

        # # Optimizer setup
        # self.optimizer = Adam(self.get_parameters(), lr=config.model.BB.optimizer.lr)

        # Conditioning mechanism (optional, used for conditioning during training)
        if self.condition_key == 'nocond':
            self.cond_stage_model = None
        elif self.condition_key == 'first_stage':
            self.cond_stage_model = self.autoencoder.encoder  # Encoder part of your autoencoder
        elif self.condition_key == 'SpatialRescaler':
            self.cond_stage_model = SpatialRescaler(**vars(config.CondStageParams))  # for rescale attention map
            self.cond_stage_model_1 = SpatialRescaler(**vars(config.CondStageParams)) # for rescale attenuation map
        else:
            raise NotImplementedError("Condition key not implemented: ", self.condition_key)

    def get_ema_net(self):
        return self

    def get_parameters(self):
        # The parameters for optimization (unet or denoise function)
        if self.condition_key == 'SpatialRescaler':
            print("get parameters to optimize: SpatialRescaler, UNet")
            params = itertools.chain(self.denoise_fn.parameters(),
                                     self.cond_stage_model.parameters(),
                                     self.cond_stage_model_1.parameters())
        else:
            print("get parameters to optimize: UNet")
            params = self.denoise_fn.parameters()
        return params

    def apply(self, weights_init):
        super().apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        if self.cond_stage_model is not None:
            self.cond_stage_model.apply(weights_init)
        return self
    
    def get_cond_stage_context(self, x_cond):
        if self.cond_stage_model is not None:
            context = self.cond_stage_model(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context
    
    def get_cond_stage_context_1(self, x_cond):
        if self.cond_stage_model_1 is not None:
            context = self.cond_stage_model_1(x_cond)
            if self.condition_key == 'first_stage':
                context = context.detach()
        else:
            context = None
        return context
    
    def load_autoencoder_checkpoint(self, autoencoder, checkpoint_path):
        # Load the checkpoint for the specific district
        checkpoint = torch.load(checkpoint_path)
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded autoencoder checkpoint from {checkpoint_path} successfully.")
        return autoencoder

    def forward(self, ct_image,  pet_image=None):
        """
        Forward pass for encoding CT and PET images into a shared latent space.
        During inference, only the CT image is used to generate the shared latent.

        Args:
            ct_image (Tensor): Input CT image.
            pet_image (Tensor, optional): Input PET image (used only during training).
        
        Returns:
            Tuple: Latent representations and distribution parameters (for training),
                or reconstructed PET image (for inference).
        """
        if pet_image is not None:
            # Training phase: Both CT and PET images are used
            with torch.no_grad():
                latent_ct = self.autoencoder.encoder(ct_image)
                latent_pet = self.autoencoder.encoder(pet_image)         
            # print(f"latent_ct shape: {latent_ct.shape}, latent_pet shape: {latent_pet.shape}")
            # Apply reparameterization BEFORE merging
            mu_ct, log_var_ct = torch.chunk(latent_ct, 2, dim=1)
            mu_pet, log_var_pet = torch.chunk(latent_pet, 2, dim=1)

            pet_residual = latent_pet - latent_ct
            latent_pet = latent_ct.clone() + pet_residual  # adjust PET retain metabolic information

            # Ensure correct latent channel shape
            assert mu_ct.shape[1] == mu_pet.shape[1] == self.autoencoder.latent_channels, \
                f"Mismatch in latent channels: mu_ct={mu_ct.shape}, mu_pet={mu_pet.shape}"

            # Create a shared latent representation by averaging the two latent representations
            shared_latent = 0.5 * (latent_ct + latent_pet)  # Enforcing a shared latent space
            # print(f"shared_latent shape: {shared_latent.shape}")
            mu, log_var = torch.chunk(shared_latent, 2, dim=1)  # Get mean & variance
        #     # # Diffusion loss here
            diffusion_loss, log_dict = super().forward(mu.detach(), mu_ct.detach()) # computes p_losses
        #     # print(f"log_dict: {log_dict}")  # Check the contents of log_dict
        #     # print(f"log_dict keys: {log_dict.keys()}")  # Check the keys in log_dict
        #     # Denoised latent
            z_pet_denoised_tensor = log_dict['x0_recon']
            with torch.no_grad():
                # Reconstruct the PET image using the denoised shared latent
                recon_pet = self.autoencoder.decoder(z_pet_denoised_tensor)      
            return mu, mu_ct, z_pet_denoised_tensor, recon_pet, diffusion_loss, log_dict


    def sample(self, x_cond, clip_denoised=False, sample_mid_step=False):
        """
        During inference, generate shared latent from CT image only and reconstruct PET image.
        """
        x_cond_latent = self.autoencoder.encoder(x_cond)
        mu_ct, log_var_ct = torch.chunk(x_cond_latent, 2, dim=1)
        # z_ct = self.autoencoder.reparameterize(mu_ct, log_var_ct)
        if sample_mid_step:
            temp, one_step_temp = self.p_sample_loop(y=mu_ct,
                                   clip_denoised=clip_denoised, 
                                   sample_mid_step=sample_mid_step)
            out_samples = []
            for i in tqdm(range(len(temp)), initial=0, desc="save output sample mid steps", dynamic_ncols=True,
                            smoothing=0.01):
                with torch.no_grad():
                    out = self.autoencoder.decoder(temp[i].detach())
                out_samples.append(out.to('cpu'))

            one_step_samples = []
            for i in tqdm(range(len(one_step_temp)), initial=0, desc="save one step sample mid steps",
                            dynamic_ncols=True,
                            smoothing=0.01):
                with torch.no_grad():
                    out = self.autoencoder.decoder(one_step_temp[i].detach())
                one_step_samples.append(out.to('cpu'))
            return out_samples, one_step_samples
        else:
            temp = self.p_sample_loop(y=mu_ct,
                                        clip_denoised=clip_denoised,
                                        sample_mid_step=sample_mid_step)
            x_latent = temp
            out = self.autoencoder.decoder(x_latent)
            return out