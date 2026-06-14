#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

LR="${LR:-5e-6}"
MIN_LR="${MIN_LR:-5e-8}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
TASK="${TASK:-retfound_dinov2_meh_rfmid_2xl40_lr5em6}"

python -m retfound.prepare_cache \
  --data-path dataset \
  --cache-dir dataset/.cache/rfmid_768 \
  --short-side 768 \
  --workers "${CACHE_WORKERS:-12}"

torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m retfound.main_finetune \
  --dataset rfmid \
  --data_path dataset \
  --image_cache_dir dataset/.cache/rfmid_768 \
  --model RETFound_dinov2 \
  --model_arch retfound_dinov2 \
  --finetune hf_models/RETFound_dinov2_meh/RETFound_dinov2_meh.pth \
  --adaptation finetune \
  --nb_classes 45 \
  --input_size 224 \
  --batch_size 16 \
  --accum_iter 1 \
  --epochs 50 \
  --lr "${LR}" \
  --min_lr "${MIN_LR}" \
  --warmup_epochs "${WARMUP_EPOCHS}" \
  --layer_decay 0.65 \
  --weight_decay 0.05 \
  --drop_path 0.2 \
  --threshold 0.5 \
  --use_pos_weight \
  --pos_weight_max 20 \
  --dist_eval \
  --num_workers "${NUM_WORKERS:-8}" \
  --prefetch_factor "${PREFETCH_FACTOR:-4}" \
  --persistent_workers \
  --fused_optimizer \
  --task "${TASK}"
