#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

TASK="${TASK:-retfound_dinov2_meh_rfmid_2xl40_dual_head}"
CHECKPOINT="${1:-output_dir/${TASK}/checkpoint-best.pth}"

torchrun \
  --standalone \
  --nproc_per_node=2 \
  -m retfound.main_finetune \
  --dataset rfmid \
  --data_path dataset \
  --model RETFound_dinov2 \
  --model_arch retfound_dinov2 \
  --adaptation finetune \
  --nb_classes 45 \
  --train_challenge_head \
  --input_size 224 \
  --batch_size 32 \
  --threshold 0.5 \
  --threshold_strategy per_class \
  --threshold_min 0.01 \
  --threshold_max 0.99 \
  --threshold_steps 99 \
  --threshold_min_positives 10 \
  --threshold_min_prevalence_multiplier 0.25 \
  --threshold_max_prevalence_multiplier 3.0 \
  --threshold_max_below_global 0.15 \
  --use_pos_weight \
  --pos_weight_max 20 \
  --dist_eval \
  --num_workers "${NUM_WORKERS:-4}" \
  --prefetch_factor "${PREFETCH_FACTOR:-2}" \
  --persistent_workers \
  --task "${TASK}" \
  --eval \
  --resume "${CHECKPOINT}"
