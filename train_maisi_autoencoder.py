import argparse
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from monai.apps.generation.maisi.networks.autoencoderkl_maisi import (
    AutoencoderKlMaisi,
)
from monai.utils import set_determinism

from models.data.dataset_CACHE import CreateDataloader


def build_autoencoder():
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


def load_weights(model, checkpoint_path):
    checkpoint = torch.load( checkpoint_path, map_location="cpu", weights_only=True, )

    if "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state = checkpoint["state_dict"]
    else:
        state = checkpoint

    # Fail explicitly if the architecture does not match the weights.
    model.load_state_dict(state, strict=True)


def compute_kl(mu, sigma):
    # MONAI returns standard deviation, not log variance.
    variance = sigma.float().square().clamp_min(1e-12)

    return 0.5 * ( mu.float().square() + variance - variance.log() - 1.0 ).mean()


def run_epoch(model, loader, optimizer, device, args):
    training = optimizer is not None
    model.train(training)

    totals = {"ct_l1": 0.0, "pet_l1": 0.0, "kl": 0.0}
    sample_count = 0
    weight_sum = args.ct_weight + args.pet_weight

    for batch in loader:
        if training:
            optimizer.zero_grad(set_to_none=True)

        batch_size = batch["A"].shape[0]

        # Separate forward/backward passes reduce peak GPU memory.
        for key, metric, weight in (
            ("A", "ct_l1", args.ct_weight),
            ("B", "pet_l1", args.pet_weight),
        ):
            image = batch[key].to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            if image.ndim != 5 or image.shape[1] != 1:
                raise ValueError(
                    f"{key}: expected [B, 1, D, H, W], "
                    f"received {tuple(image.shape)}"
                )

            if any(size % 4 != 0 for size in image.shape[2:]):
                raise ValueError(
                    "Spatial dimensions must be divisible by 4."
                )

            with torch.set_grad_enabled(training):
                mu, sigma = model.encode(image)

                latent = (
                    model.sampling(mu, sigma)
                    if training
                    else mu
                )
                reconstruction = model.decode(latent)

                reconstruction_loss = F.l1_loss(
                    reconstruction, image
                )
                kl = compute_kl(mu, sigma)

                loss = (weight / weight_sum) * (
                    reconstruction_loss + args.kl_weight * kl
                )

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"Non-finite loss for modality {key}"
                    )

                if training:
                    loss.backward()

            totals[metric] += (
                reconstruction_loss.detach().item() * batch_size
            )
            totals["kl"] += (
                weight / weight_sum
                * kl.detach().item()
                * batch_size
            )

        if training:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()

        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("DataLoader returned no samples")

    return {
        key: value / sample_count
        for key, value in totals.items()
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataroot", required=True)
    parser.add_argument("--pretrained", required=True)
    parser.add_argument("--output", default="runs/maisi_ctpet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-weight", type=float, default=1e-6)
    parser.add_argument("--ct-weight", type=float, default=1.0)
    parser.add_argument("--pet-weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--cache-rate", type=float, default=0.1)

    # Paths relative to each patient folder in the CSV.
    parser.add_argument(
        "--ct-relative-path", default="lung_CT.nii.gz"
    )
    parser.add_argument(
        "--pet-relative-path", default="lung_PET.nii.gz"
    )

    args = parser.parse_args()

    if args.ct_weight <= 0 or args.pet_weight <= 0:
        parser.error("Both modality weights must be positive")
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")

    set_determinism(seed=args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    options = SimpleNamespace(
        dataroot=args.dataroot,
        phase="train",
        batch_size=args.batch_size,
        cache_rate=args.cache_rate,
        ct_relative_path=args.ct_relative_path,
        pet_relative_path=args.pet_relative_path,
    )

    train_loader = CreateDataloader(
        options,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False,
        cache=args.cache,
    )

    # Optional patient-separated val.csv.
    val_loader = None
    if (Path(args.dataroot) / "val.csv").is_file():
        val_options = SimpleNamespace(
            **{
                **vars(options),
                "phase": "val",
                "batch_size": 1,
            }
        )
        val_loader = CreateDataloader(
            val_options,
            shuffle=False,
            num_workers=args.workers,
            drop_last=False,
            cache=args.cache,
        )

    model = build_autoencoder()
    load_weights(model, args.pretrained)
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        train = run_epoch(
            model, train_loader, optimizer, device, args
        )

        validation = None
        improved = False

        if val_loader is not None:
            with torch.no_grad():
                validation = run_epoch(
                    model, val_loader, None, device, args
                )

            score = (
                args.ct_weight * validation["ct_l1"]
                + args.pet_weight * validation["pet_l1"]
            ) / (args.ct_weight + args.pet_weight)

            improved = score < best_score
            if improved:
                best_score = score

        message = (
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train CT L1={train['ct_l1']:.6f} "
            f"PET L1={train['pet_l1']:.6f} "
            f"KL={train['kl']:.6f}"
        )
        if validation is not None:
            message += (
                f" | val CT L1={validation['ct_l1']:.6f} "
                f"PET L1={validation['pet_l1']:.6f}"
            )
        print(message, flush=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_score": (
                best_score if val_loader is not None else None
            ),
            "train_metrics": train,
            "validation_metrics": validation,
            "args": vars(args),
        }

        torch.save(checkpoint, output / "latest.pt")
        if improved:
            torch.save(checkpoint, output / "best.pt")

    print(f"Saved checkpoints to {output}")
    if val_loader is None:
        print(
            "No val.csv supplied: latest.pt saved; "
            "no validation-based best.pt selected."
        )


if __name__ == "__main__":
    main()