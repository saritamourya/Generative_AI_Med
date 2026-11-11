import os

# Set before importing torch.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:256",
)

import argparse
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
)
from monai.networks.nets import PatchDiscriminator
from monai.utils import set_determinism

from models.data import config
from models.data.dataset_CACHE import CreateDataloader
from utils.losses import VAE_Losses


def build_autoencoder():
    """Architecture for the MAISI autoencoder checkpoint."""
    return AutoencoderKlMaisi(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        latent_channels=4,
        num_channels=(64, 128, 256),
        num_res_blocks=(2, 2, 2),
        attention_levels=(False, False, False),
        norm_num_groups=32,
        norm_eps=1e-6,
        with_encoder_nonlocal_attn=False,
        with_decoder_nonlocal_attn=False,
        use_checkpointing=False,
        use_convtranspose=False,
        norm_float16=False,
        num_splits=1,
        dim_split=0,
        save_mem=False,
    )


def build_discriminator():
    # Two downsampling layers suit your 32^3 training patches.
    return PatchDiscriminator(
        spatial_dims=3,
        num_layers_d=2,
        channels=32,
        in_channels=1,
        out_channels=1,
        norm="INSTANCE",
    )

def check_image(image, name):
    if image.ndim != 5 or image.shape[1] != 1:
        raise ValueError(
            f"{name}: expected [B, 1, D, H, W], "
            f"got {tuple(image.shape)}"
        )

    if any(size < 32 or size % 4 for size in image.shape[2:]):
        raise ValueError(
            f"{name}: spatial dimensions must be >=32 "
            "and divisible by 4"
        )

    if not torch.isfinite(image).all().item():
        raise ValueError(f"{name} contains NaN or infinite values")


def check_loss(loss, name):
    if not torch.isfinite(loss).item():
        raise FloatingPointError(f"Non-finite {name}")


def amp_context(enabled):
    if enabled:
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
        )
    return nullcontext()


def optimizer_step(scaler, optimizer, parameters):
    scaler.unscale_(optimizer)

    # GradScaler handles non-finite gradients when AMP is enabled.
    torch.nn.utils.clip_grad_norm_(
        parameters,
        max_norm=1.0,
        error_if_nonfinite=not scaler.is_enabled(),
    )

    scaler.step(optimizer)
    scaler.update()


def log_pet_reconstruction(writer, image, reconstruction, step):
    """Log the middle slice of the first PET sample."""
    middle = image.shape[-1] // 2

    writer.add_image(
        "PET/input",
        image[0, :, :, :, middle].detach().float().cpu(),
        step,
    )
    writer.add_image(
        "PET/reconstruction",
        reconstruction[0, :, :, :, middle]
        .detach().float().cpu().clamp(0, 1),
        step,
    )


def train_epoch(
    autoencoder,
    discriminator,
    loader,
    loss_handler,
    optimizer_g,
    optimizer_d,
    scaler_g,
    scaler_d,
    writer,
    device,
    epoch,
    global_step,
    args,
):
    autoencoder.train()

    adversarial_active = (
        epoch > args.warmup_epochs
        and args.adv_weight > 0
    )

    amp_enabled = args.amp and device.type == "cuda"
    accumulation = args.gradient_accumulation_steps

    totals = {
        "recon_pet": 0.0,
        "kl_pet": 0.0,
        "kl_ct": 0.0,
        "perceptual": 0.0,
        "generator_adv": 0.0,
        "generator": 0.0,
        "discriminator": 0.0,
    }

    num_batches = len(loader)
    if num_batches == 0:
        raise ValueError("Training loader has no batches")

    sample_count = 0

    for batch_index, batch in enumerate(loader):
        # Start a new gradient accumulation window.
        if batch_index % accumulation == 0:
            optimizer_g.zero_grad(set_to_none=True)
            optimizer_d.zero_grad(set_to_none=True)

        # Correctly normalize the final, possibly shorter window.
        window_start = (
            batch_index // accumulation
        ) * accumulation
        window_size = min(
            accumulation,
            num_batches - window_start,
        )

        ct_image = batch["A"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        pet_image = batch["B"].to(
            device, dtype=torch.float32, non_blocking=True
        )

        check_image(ct_image, "CT")
        check_image(pet_image, "PET")

        if ct_image.shape != pet_image.shape:
            raise ValueError(
                "CT and PET patches must have matching shapes"
            )

        # GENERATOR:
        # Freeze discriminator parameters, retaining the gradient
        # path from discriminator output to reconstructed PET.
        discriminator.requires_grad_(False)
        discriminator.eval()

        with amp_context(amp_enabled):
            # Independent calls through the same MAISI encoder.
            mu_ct, sigma_ct = autoencoder.encode(ct_image)
            mu_pet, sigma_pet = autoencoder.encode(pet_image)

            # Decode PET only.
            z_pet = autoencoder.sampling(mu_pet, sigma_pet)
            recon_pet = autoencoder.decode(z_pet)

        # Loss calculations and discriminator forward passes use FP32.
        # Gradients still flow through recon_pet to the VAE.
        losses, loss_g_pet = loss_handler.compute_losses(
            reconstruction=recon_pet.float(),
            image=pet_image,
            mu=mu_pet,
            sigma=sigma_pet,
            discriminator=discriminator,
            adversarial_active=adversarial_active,
        )

        loss_kl_ct = loss_handler.kl_loss(mu_ct, sigma_ct)

        loss_g = (
            loss_g_pet
            + args.ct_kl_scale
            * loss_handler.kl_weight
            * loss_kl_ct
        )

        check_loss(loss_g, "generator loss")
        scaler_g.scale(loss_g / window_size).backward()

        # DISCRIMINATOR:
        # The loss handler detaches recon_pet internally.
        loss_d = pet_image.new_zeros(())

        if adversarial_active:
            discriminator.requires_grad_(True)
            discriminator.train()

            loss_d = loss_handler.discriminator_loss(
                discriminator=discriminator,
                image=pet_image,
                reconstruction=recon_pet,
            )

            check_loss(loss_d, "discriminator loss")
            scaler_d.scale(loss_d / window_size).backward()

        end_of_window = (
            (batch_index + 1) % accumulation == 0
            or batch_index + 1 == num_batches
        )

        if end_of_window:
            optimizer_step(
                scaler_g,
                optimizer_g,
                autoencoder.parameters(),
            )

            if adversarial_active:
                optimizer_step(
                    scaler_d,
                    optimizer_d,
                    discriminator.parameters(),
                )

        values = {
            "recon_pet": losses["recon"],
            "kl_pet": losses["kl"],
            "kl_ct": loss_kl_ct,
            "perceptual": losses["perceptual"],
            "generator_adv": losses["generator_adv"],
            "generator": loss_g,
            "discriminator": loss_d,
        }

        batch_size = pet_image.shape[0]
        sample_count += batch_size
        global_step += 1

        for name, value in values.items():
            scalar = value.detach().item()
            totals[name] += scalar * batch_size

            if global_step % args.log_interval == 0:
                writer.add_scalar(
                    f"Step/{name}", scalar, global_step
                )

        if global_step % args.log_interval == 0:
            log_pet_reconstruction(
                writer, pet_image, recon_pet, global_step
            )

    averages = {
        name: value / sample_count
        for name, value in totals.items()
    }

    for name, value in averages.items():
        writer.add_scalar(f"Epoch/{name}", value, epoch)

    return averages, global_step


def train_autoencoder(args):
    set_determinism(seed=args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    amp_enabled = args.amp and device.type == "cuda"

    checkpoint_dir = Path(args.output)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Match your existing loader's option names.
    options = SimpleNamespace(
        dataroot=args.dataroot,
        phase="train",
        batchSize=args.batch_size,
    )

    print("[INFO] Loading dataset...")
    train_loader = CreateDataloader(
        options,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        cache=False,
    )

    if train_loader is None or len(train_loader) == 0:
        raise ValueError("No training data loaded")

    print("[INFO] Initializing MAISI autoencoder...")
    autoencoder = build_autoencoder().to(device)
    discriminator = build_discriminator().to(device)

    loss_handler = VAE_Losses(
        device=device,
        perceptual_weight=args.perceptual_weight,
        kl_weight=args.kl_weight,
        adv_weight=args.adv_weight,
    )

    optimizer_g = optim.Adam(
        autoencoder.parameters(),
        lr=args.lr_g,
        betas=(0.5, 0.999),
        eps=1e-6 if amp_enabled else 1e-8,
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.lr_d,
        betas=(0.5, 0.999),
        eps=1e-6 if amp_enabled else 1e-8,
    )

    # Gradually increase LR over the first few scheduler epochs.
    def lr_factor(index):
        return min(
            1.0,
            (index + 1) / max(1, args.lr_warmup_epochs),
        )

    scheduler_g = lr_scheduler.LambdaLR(
        optimizer_g, lr_lambda=lr_factor
    )
    scheduler_d = lr_scheduler.LambdaLR(
        optimizer_d, lr_lambda=lr_factor
    )

    scaler_g = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled
    )
    scaler_d = torch.amp.GradScaler(
        "cuda", enabled=amp_enabled
    )

    start_epoch = 1
    global_step = 0

    if args.resume is not None:
        checkpoint = torch.load(
            args.resume,
            map_location="cpu",
            weights_only=True,
        )

        autoencoder.load_state_dict(
            checkpoint["model_state_dict"]
        )
        discriminator.load_state_dict(
            checkpoint["discriminator_state_dict"]
        )
        optimizer_g.load_state_dict(
            checkpoint["optimizer_g_state_dict"]
        )
        optimizer_d.load_state_dict(
            checkpoint["optimizer_d_state_dict"]
        )
        scheduler_g.load_state_dict(
            checkpoint["scheduler_g_state_dict"]
        )
        scheduler_d.load_state_dict(
            checkpoint["scheduler_d_state_dict"]
        )

        if amp_enabled:
            if checkpoint["scaler_g_state_dict"]:
                scaler_g.load_state_dict(
                    checkpoint["scaler_g_state_dict"]
                )
            if checkpoint["scaler_d_state_dict"]:
                scaler_d.load_state_dict(
                    checkpoint["scaler_d_state_dict"]
                )

        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint.get("global_step", 0)

        print(f"[INFO] Resuming at epoch {start_epoch}")

    print(
        f"[INFO] Device={device}, AMP={amp_enabled}, "
        f"pairs={len(train_loader.dataset)}"
    )

    writer = SummaryWriter(
        log_dir=str(checkpoint_dir / "tensorboard"),
        purge_step=global_step + 1 if args.resume else None,
    )

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            # Log the learning rates actually used for this epoch.
            writer.add_scalar(
                "LR/Generator",
                optimizer_g.param_groups[0]["lr"],
                epoch,
            )
            writer.add_scalar(
                "LR/Discriminator",
                optimizer_d.param_groups[0]["lr"],
                epoch,
            )

            metrics, global_step = train_epoch(
                autoencoder=autoencoder,
                discriminator=discriminator,
                loader=train_loader,
                loss_handler=loss_handler,
                optimizer_g=optimizer_g,
                optimizer_d=optimizer_d,
                scaler_g=scaler_g,
                scaler_d=scaler_d,
                writer=writer,
                device=device,
                epoch=epoch,
                global_step=global_step,
                args=args,
            )

            scheduler_g.step()
            if (
                epoch > args.warmup_epochs
                and args.adv_weight > 0
            ):
                scheduler_d.step()

            print(
                f"Epoch [{epoch}/{args.epochs}] "
                f"PET L1={metrics['recon_pet']:.6f}, "
                f"PET KL={metrics['kl_pet']:.6f}, "
                f"CT KL={metrics['kl_ct']:.6f}, "
                f"Perceptual={metrics['perceptual']:.6f}, "
                f"G_adv={metrics['generator_adv']:.6f}, "
                f"D={metrics['discriminator']:.6f}",
                flush=True,
            )

            checkpoint = {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": autoencoder.state_dict(),
                "discriminator_state_dict":
                    discriminator.state_dict(),
                "optimizer_g_state_dict":
                    optimizer_g.state_dict(),
                "optimizer_d_state_dict":
                    optimizer_d.state_dict(),
                "scheduler_g_state_dict":
                    scheduler_g.state_dict(),
                "scheduler_d_state_dict":
                    scheduler_d.state_dict(),
                "scaler_g_state_dict": scaler_g.state_dict(),
                "scaler_d_state_dict": scaler_d.state_dict(),
                "metrics": metrics,
                "data_params": dict(config.params),
                "args": vars(args),
            }

            torch.save(
                checkpoint, checkpoint_dir / "latest.pt"
            )

            if (
                epoch % args.save_interval == 0
                or epoch == args.epochs
            ):
                torch.save(
                    checkpoint,
                    checkpoint_dir / f"epoch_{epoch:03d}.pt",
                )

            writer.flush()

    finally:
        writer.close()

    print(f"[INFO] Checkpoints saved to {checkpoint_dir}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--output", default="runs/maisi_pet_only")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=1
    )
    parser.add_argument("--amp", action="store_true")

    parser.add_argument("--lr-g", type=float, default=1e-5)
    parser.add_argument("--lr-d", type=float, default=1e-5)
    parser.add_argument("--lr-warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    parser.add_argument(
        "--perceptual-weight", type=float, default=0.01
    )
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--adv-weight", type=float, default=0.01)
    parser.add_argument(
        "--ct-kl-scale",
        type=float,
        default=1.0,
        help="Multiplier on CT KL; PET KL is already in losses.py",
    )

    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()


    for name in (
        "epochs", "batch_size", "gradient_accumulation_steps",
        "save_interval", "log_interval", "lr_warmup_epochs",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")

    if args.workers < 0 or args.warmup_epochs < 0:
        parser.error("workers and warmup-epochs must be nonnegative")

    if args.lr_g <= 0 or args.lr_d <= 0:
        parser.error("Learning rates must be positive")

    for name in (
        "perceptual_weight", "kl_weight",
        "adv_weight", "ct_kl_scale",
    ):
        if getattr(args, name) < 0:
            parser.error(f"{name} must be nonnegative")

    return args


if __name__ == "__main__":
    train_autoencoder(parse_args())