
import torch
import torch.nn.functional as F

from monai.losses import PatchAdversarialLoss, PerceptualLoss


class VAE_Losses:
    """Apply the same reconstruction objective to CT and PET separately."""

    def __init__(
        self,
        device,
        perceptual_weight=0.01,
        kl_weight=1e-6,
        adv_weight=0.01,
    ):
        # Starting weights; tune using separate CT/PET validation.
        self.perceptual_weight = perceptual_weight
        self.kl_weight = kl_weight
        self.adv_weight = adv_weight

        self.adv_loss = PatchAdversarialLoss(
            criterion="least_squares"
        ).to(device)

        self.perceptual_loss = PerceptualLoss(
            spatial_dims=3,
            network_type="squeeze",
            is_fake_3d=True,
            fake_3d_ratio=0.2,
            pretrained=True,
        ).to(device)

        self.perceptual_loss.requires_grad_(False)
        self.perceptual_loss.eval()

    @staticmethod
    def kl_loss(mu, sigma):
        variance = sigma.float().square()

        elementwise = 0.5 * (
            mu.float().square()
            + variance
            - torch.log(variance + 1e-10)
            - 1.0
        )

        return elementwise.flatten(start_dim=1).sum(dim=1).mean()

    def compute_losses(
        self,
        reconstruction,
        image,
        mu,
        sigma,
        discriminator=None,
        adversarial_active=False,
    ):
        """
        Call independently for CT and PET.

        Images are assumed normalized to [0, 1].
        Freeze discriminator parameters during generator updates.
        """
        reconstruction = reconstruction.float()
        image = image.float()

        losses = {
            "recon": F.l1_loss(reconstruction, image),
            "kl": self.kl_loss(mu, sigma),
            "perceptual": self.perceptual_loss(
                reconstruction,
                image,
            ),
            "generator_adv": image.new_zeros(()),
        }

        if adversarial_active:
            if discriminator is None:
                raise ValueError("An active GAN requires a discriminator")

            logits_fake = discriminator(reconstruction)[-1]
            losses["generator_adv"] = self.adv_loss(
                logits_fake,
                target_is_real=True,
                for_discriminator=False,
            )

        loss_g = (
            losses["recon"]
            + self.kl_weight * losses["kl"]
            + self.perceptual_weight * losses["perceptual"]
            + self.adv_weight * losses["generator_adv"]
        )

        return losses, loss_g

    def discriminator_loss(self, discriminator, image, reconstruction):
        """Detached images prevent discriminator updates entering the VAE."""
        logits_real = discriminator(image.detach().float())[-1]
        logits_fake = discriminator(
            reconstruction.detach().float()
        )[-1]

        loss_real = self.adv_loss(
            logits_real,
            target_is_real=True,
            for_discriminator=True,
        )
        loss_fake = self.adv_loss(
            logits_fake,
            target_is_real=False,
            for_discriminator=True,
        )

        return 0.5 * (loss_real + loss_fake)