#!/bin/bash -l
#SBATCH --job-name=maisi-pet-scratch
#SBATCH --account=hpc2n2026-350-gpu
#SBATCH --partition=gpu_zen3
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --output=maisi-pet-scratch-%j.out
#SBATCH --error=maisi-pet-scratch-%j.err

set -euo pipefail

PROJECT_DIR="/pfs/proj/nobackup/fs/projnb11/genmed-imaging/saritam"

# CHANGE THIS: directory containing train.csv.
DATA_ROOT="/CHANGE/path/to/csv_directory"

OUTPUT_DIR="${PROJECT_DIR}/runs/maisi_pet_from_scratch"

# Empty = start with random weights.
# Resume only from a checkpoint produced by this training script.
RESUME=""

cd "${PROJECT_DIR}"

module load Python/3.11.3-GCCcore-12.3.0
source "${PROJECT_DIR}/genmed_venv/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:256"

if [[ ! -f "${DATA_ROOT}/train.csv" ]]; then
    echo "ERROR: Missing ${DATA_ROOT}/train.csv" >&2
    exit 1
fi

CHECKPOINT_ARGS=()

if [[ -n "${RESUME}" ]]; then
    if [[ ! -f "${RESUME}" ]]; then
        echo "ERROR: Missing checkpoint: ${RESUME}" >&2
        exit 1
    fi
    CHECKPOINT_ARGS=(--resume "${RESUME}")
elif [[ -f "${OUTPUT_DIR}/latest.pt" ]]; then
    echo "ERROR: Output already contains latest.pt." >&2
    echo "Set RESUME or choose a new OUTPUT_DIR." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "Job: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "Output: ${OUTPUT_DIR}"

nvidia-smi

python - <<'PY'
import torch
import monai

from models.data.dataset_CACHE import CreateDataloader
from utils.losses import VAE_Losses

print("PyTorch:", torch.__version__)
print("MONAI:", monai.__version__)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")

print("GPU:", torch.cuda.get_device_name(0))

# Verify the MAISI-style KL reduction:
# sum channels/spatial dimensions, then average the batch.
mu = torch.ones(2, 4, 8, 8, 8)
sigma = torch.ones_like(mu)
actual = VAE_Losses.kl_loss(mu, sigma)

if not torch.isclose(actual, actual.new_tensor(1024.0)):
    raise RuntimeError(
        f"Incorrect KL reduction: expected 1024, got {actual.item()}"
    )

print("KL reduction check passed.")
PY

srun python -u train_maisi_autoencoder.py \
    --dataroot "${DATA_ROOT}" \
    "${CHECKPOINT_ARGS[@]}" \
    --output "${OUTPUT_DIR}" \
    --epochs 300 \
    --batch-size 1 \
    --workers 4 \
    --gradient-accumulation-steps 4 \
    --lr-g 1e-4 \
    --lr-d 1e-4 \
    --lr-warmup-epochs 5 \
    --warmup-epochs 5 \
    --perceptual-weight 0.3 \
    --kl-weight 1e-7 \
    --adv-weight 0.1 \
    --ct-kl-scale 1.0 \
    --save-interval 10 \
    --log-interval 50 \
    --seed 42 \
    --amp