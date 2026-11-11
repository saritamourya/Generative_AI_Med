import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np
from models.utils import extract, default
from models.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
# from models.BrownianBridge.base.modules.diffusionmodules.unet import DiffusionModelUNet  # <- swap to 3D UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class BrownianBridgeModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config


        # model hyperparameters
        model_params = config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if hasattr(model_params, "max_var") else 1
        self.eta = model_params.eta if hasattr(model_params, "eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        # UNet3D
        self.image_size = model_params.UNetParams.image_size  # (D, H, W)
        self.channels = model_params.UNetParams.in_channels
        self.condition_key = model_params.UNetParams.condition_key
 

        self.denoise_fn = UNetModel(**vars(model_params.UNetParams))
        #Monai 3D UNet for denoising
        # self.denoise_fn = DiffusionModelUNet(spatial_dims=3,
        #                                   in_channels=3,
        #                                   # ho cambiato il numero di canali a 6 perchè faccio spatial conditioning
        #                                   out_channels=3,
        #                                   num_res_blocks=2,
        #                                   num_channels=(256, 512, 768, 1024),
        #                                   attention_levels=(False,True, True, True),
        #                                   norm_num_groups=32,
        #                                   norm_eps=1e-6,
        #                                   resblock_updown=True,
        #                                   num_head_channels=(0,256, 512, 768),
        #                                   transformer_num_layers=1,
        #                                   with_conditioning=False,
        #                                   # cross_attention_dim=8,
        #                                   num_class_embeds=None,
        #                                   upcast_attention=True,
        #                                   use_flash_attention=False)
        # self.denoise_fn.predict_codebook_ids = False 

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        
        m_tminus = np.append(0, m_t[:-1])
        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t
        # Debug prints
        # print(f"variance_t: {variance_t}")
        # print(f"variance_tminus: {variance_tminus}")
        # print(f"variance_t_tminus: {variance_t_tminus}")
        # print(f"posterior_variance_t: {posterior_variance_t}")

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x, y, context=None):
        """
        Performs a single training step by computing the diffusion loss.

        Args:
            x (Tensor): Latent representation of the source domain (CT), shape (B, C, D, H, W)
            y (Tensor): Latent representation of the target domain (PET), shape (B, C, D, H, W)
            context (Tensor, optional): Additional conditional context (e.g., external embeddings or patches)

        Returns:
            Tuple: (loss value, dictionary of logs including reconstructed x0)
        """

        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context
            
        b, c, d, h, w, device, img_size = *x.shape, x.device, self.image_size

        assert d == img_size and h == img_size and w == img_size, f' depth, height and width of the image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, y, context, t)

    def p_losses(self, x0, y, context, t, noise=None):
        """
        Computes the loss for a batch at a specific timestep.

        - Simulates noisy intermediate latent `x_t` between `x0` (CT) and `y` (PET).
        - Uses the denoising UNet to predict the noise or residual (`objective_recon`).
        - Reconstructs `x0` from this prediction and computes L1/L2 loss.

        Returns:
            - Scalar loss
            - Dictionary with reconstruction and diagnostic info
        """

        b, c, d, h, w = x0.shape
        noise = default(noise, lambda: torch.randn_like(x0))

        # Simulate noisy latent x_t by interpolating between x0 (CT) and y (PET), then add noise
        x_t, objective = self.q_sample(x0, y, t, noise)
        # Predict the noise (or transformation) from the noisy input x_t and optional context
        objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
        # print(f"objective_recon shape: {objective_recon.shape}, objective shape: {objective.shape}")

        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()
        
        # Reconstruct x0 (CT latent) from the denoised prediction
        x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon)

        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }

        return recloss, log_dict

    def q_sample(self, x0, y, t, noise=None):
        """
        Forward Brownian Bridge diffusion: interpolates between CT (x0) and PET (y),
        adds noise to simulate intermediate latent x_t.

        Returns:
            - x_t: Noisy intermediate latent
            - objective: Noise/residual the model is expected to learn (depending on mode)
        """


        noise = default(noise, lambda: torch.randn_like(x0))  # use standard Gaussian if not provided
        m_t = extract(self.m_t, t, x0.shape)                  # mixing coefficient at timestep t
        var_t = extract(self.variance_t, t, x0.shape)         # variance of noise at timestep t
        sigma_t = torch.sqrt(var_t)                           # std deviation of noise at timestep t

        if self.objective == 'grad':
            # Predicts the full Brownian bridge gradient: difference from CT to PET + noise
            objective = m_t * (y - x0) + sigma_t * noise
        elif self.objective == 'noise':
            # Predicts pure noise
            objective = noise
        elif self.objective == 'ysubx':
            # Predicts difference between PET and CT
            objective = y - x0
        else:
            raise NotImplementedError()
            # Print the noise and objective to inspect the added losses
        # print(f"Added Noise (objective) at timestep {t}: {objective.shape}")
        # print(f"Noise added: {noise.mean().item()}")

        # Return noisy latent and corresponding target
        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,  # interpolated latent + scaled noise
            objective
        )


    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        """
        Reconstructs the original CT latent (x0) from noisy x_t and model-predicted objective.
        Uses the inverse of the Brownian Bridge equation depending on objective type.
        """

        if self.objective == 'grad':
            """
            In 'grad' moe, we assume the model learned to predict the full perturbation
            so subtract the predicted transformation from x_t to recover CT
            """
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            """
            In 'noise' mode, model predicts only the noise; we must reverse th full Brownian update
            """
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            """
            In 'ysubx' mode, the model predicts the difference y - x0, so x0 = y -diff
            """
            x0_recon = y - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        """
        Visualizes the forward diffusion process from CT to PET latents.

        Used for inspection/debugging: returns a list of x_t latents from t=0 → T.

        Returns:
            List of tensors [x_0, x_1, ..., x_T]
        """

        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        """
        Single reverse denoising step in inference.

        Moves from x_t → x_{t-1} by estimating the clean latent x0,
        then sampling a step toward it based on Brownian Bridge posterior.

        Args:
            x_t (Tensor): Noisy latent at current timestep
            y (Tensor): Conditioning latent (CT), guides the denoising process
            context (Tensor): Optional conditioning
            i (int): Current step index
            clip_denoised (bool): Clamp x0 to [-1, 1] range if True

        Returns:
            - x_{t-1}: Next latent sample
            - x0_recon: Denoised latent estimate
        """

        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, timesteps=t, context=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)
            # print(f"x_tminus_mean shape: {x_tminus_mean.shape}, x0_recon shape: {x0_recon.shape}")
            # print(f"m_nt: {m_nt}, var_nt: {var_nt}, sigma2_t: {sigma2_t}, sigma_t: {sigma_t}")
            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        """
        Full reverse diffusion loop — reconstructs PET latent from CT latent (`y`).

        During inference, this is called with only `z_ct` (CT latent).
        The diffusion process gradually transforms `z_ct` to resemble `z_pet`.

        Args:
            y (Tensor): CT latent (z_ct) used as bridge source
            context (Tensor): Optional additional context
            clip_denoised (bool): Clamp results to [-1, 1] range
            sample_mid_step (bool): Return intermediate latents too

        Returns:
            Final predicted PET latent, or trajectory if sample_mid_step=True
        """

        if self.condition_key == "nocond":
            context = None
        else:
            context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img

    @torch.no_grad()
    def sample(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        """
        User-facing sampling API — runs reverse diffusion to get PET latent from CT latent.

        Calls p_sample_loop internally. Can return either only the final result or the full latent trajectory.
        """
        return self.p_sample_loop(y, context, clip_denoised, sample_mid_step)
