#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"

SEEDS_TEXT="${SEEDS:-0 1 2}"
read -r -a SEED_VALUES <<< "${SEEDS_TEXT}"
if [[ "${#SEED_VALUES[@]}" -ne 3 ]]; then
  echo "SEEDS must contain exactly three integers; got: ${SEEDS_TEXT}" >&2
  exit 2
fi

declare -A SEEN_SEEDS=()
for seed in "${SEED_VALUES[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed '${seed}': seeds must be non-negative integers" >&2
    exit 2
  fi
  if [[ -n "${SEEN_SEEDS[${seed}]:-}" ]]; then
    echo "Duplicate seed '${seed}' in SEEDS=${SEEDS_TEXT}" >&2
    exit 2
  fi
  SEEN_SEEDS["${seed}"]=1
done

LR="${LR:-5e-6}"
MIN_LR="${MIN_LR:-5e-8}"
EPOCHS="${EPOCHS:-100}"
TEST_EVAL_INTERVAL="${TEST_EVAL_INTERVAL:-10}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
CHALLENGE_LOSS_WEIGHT="${CHALLENGE_LOSS_WEIGHT:-1.0}"
POS_WEIGHT_MAX="${POS_WEIGHT_MAX:-15}"
DATASETS_SEED="${DATASETS_SEED:-2026}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-output_dir}"
TASK_PREFIX="${TASK_PREFIX:-retfound_dinov2_meh_rfmid_2xl40_dual_head_pw${POS_WEIGHT_MAX}}"
SUMMARY_DIR="${OUTPUT_DIR}/${TASK_PREFIX}_3seeds"

if [[ "${TEST_EVAL_INTERVAL}" -ne 10 ]]; then
  echo "TEST_EVAL_INTERVAL must be 10 for this three-seed experiment" >&2
  exit 2
fi

for seed in "${SEED_VALUES[@]}"; do
  task="${TASK_PREFIX}_seed${seed}"
  task_dir="${OUTPUT_DIR}/${task}"
  if [[ -e "${task_dir}" ]]; then
    echo "Refusing to append to existing run directory: ${task_dir}" >&2
    echo "Use a different TASK_PREFIX or move the existing directory." >&2
    exit 2
  fi
done

if [[ -e "${SUMMARY_DIR}" ]]; then
  echo "Refusing to overwrite existing summary directory: ${SUMMARY_DIR}" >&2
  echo "Use a different TASK_PREFIX or move the existing directory." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}" "${SUMMARY_DIR}"

python -m retfound.prepare_cache \
  --data-path dataset \
  --cache-dir dataset/.cache/rfmid_768 \
  --short-side 768 \
  --workers "${CACHE_WORKERS:-12}"

printf "seed\ttask\tstatus\n" > "${SUMMARY_DIR}/run_manifest.tsv"

for seed in "${SEED_VALUES[@]}"; do
  task="${TASK_PREFIX}_seed${seed}"
  task_dir="${OUTPUT_DIR}/${task}"
  mkdir -p "${task_dir}"

  printf '%s\t%s\ttraining\n' "${seed}" "${task}" \
    >> "${SUMMARY_DIR}/run_manifest.tsv"
  echo "Starting training: task=${task}, seed=${seed}"

  torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m retfound.main_finetune \
    --dataset rfmid \
    --data_path dataset \
    --image_cache_dir dataset/.cache/rfmid_768 \
    --model RETFound_dinov2 \
    --model_arch retfound_dinov2 \
    --finetune hf_models/RETFound_dinov2_meh/RETFound_dinov2_meh.pth \
    --adaptation finetune \
    --nb_classes 45 \
    --train_challenge_head \
    --challenge_loss_weight "${CHALLENGE_LOSS_WEIGHT}" \
    --model_selection mean \
    --input_size 224 \
    --batch_size 16 \
    --accum_iter 1 \
    --epochs "${EPOCHS}" \
    --test_eval_interval "${TEST_EVAL_INTERVAL}" \
    --test_eval_threshold_strategy per_class \
    --lr "${LR}" \
    --min_lr "${MIN_LR}" \
    --warmup_epochs "${WARMUP_EPOCHS}" \
    --layer_decay 0.65 \
    --weight_decay 0.05 \
    --drop_path 0.2 \
    --threshold 0.5 \
    --use_pos_weight \
    --pos_weight_max "${POS_WEIGHT_MAX}" \
    --seed "${seed}" \
    --datasets_seed "${DATASETS_SEED}" \
    --dist_eval \
    --num_workers "${NUM_WORKERS:-8}" \
    --prefetch_factor "${PREFETCH_FACTOR:-4}" \
    --persistent_workers \
    --fused_optimizer \
    --output_dir "${OUTPUT_DIR}" \
    --task "${task}" \
    2>&1 | tee "${task_dir}/train_console.log"

  printf '%s\t%s\tevaluating_best\n' "${seed}" "${task}" \
    >> "${SUMMARY_DIR}/run_manifest.tsv"
  echo "Evaluating best checkpoint: task=${task}, seed=${seed}"

  torchrun \
    --standalone \
    --nproc_per_node="${NPROC_PER_NODE}" \
    -m retfound.main_finetune \
    --dataset rfmid \
    --data_path dataset \
    --image_cache_dir dataset/.cache/rfmid_768 \
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
    --pos_weight_max "${POS_WEIGHT_MAX}" \
    --seed "${seed}" \
    --datasets_seed "${DATASETS_SEED}" \
    --dist_eval \
    --num_workers "${EVAL_NUM_WORKERS:-4}" \
    --prefetch_factor "${EVAL_PREFETCH_FACTOR:-2}" \
    --persistent_workers \
    --output_dir "${OUTPUT_DIR}" \
    --task "${task}" \
    --eval \
    --resume "${task_dir}/checkpoint-best.pth" \
    2>&1 | tee "${task_dir}/eval_console.log"

  printf '%s\t%s\tcomplete\n' "${seed}" "${task}" \
    >> "${SUMMARY_DIR}/run_manifest.tsv"
done

export OUTPUT_DIR TASK_PREFIX SUMMARY_DIR SEEDS_TEXT
python - <<'PY'
import csv
import os
import statistics
from pathlib import Path

output_dir = Path(os.environ["OUTPUT_DIR"])
task_prefix = os.environ["TASK_PREFIX"]
summary_dir = Path(os.environ["SUMMARY_DIR"])
seeds = os.environ["SEEDS_TEXT"].split()

periodic_rows = []
final_rows = []

for seed in seeds:
    task = f"{task_prefix}_seed{seed}"
    task_dir = output_dir / task

    with (task_dir / "epoch_curve.tsv").open(
        newline="", encoding="utf-8"
    ) as handle:
        data_lines = [
            line for line in handle if not line.startswith("#")
        ]
    for row in csv.DictReader(data_lines, delimiter="\t"):
        periodic_rows.append({"seed": seed, "task": task, **row})

    with (task_dir / "metrics_test_per_class.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        all_classes = list(csv.DictReader(handle))[-1]
    with (task_dir / "metrics_test_per_class_challenge28.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        challenge28 = list(csv.DictReader(handle))[-1]

    final_row = {"seed": seed, "task": task}
    final_row.update(
        {f"all_{key}": value for key, value in all_classes.items()}
    )
    final_row.update(
        {f"challenge28_{key}": value for key, value in challenge28.items()}
    )
    final_rows.append(final_row)

with (summary_dir / "test_every_10_epochs.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(periodic_rows[0]))
    writer.writeheader()
    writer.writerows(periodic_rows)

with (summary_dir / "best_checkpoint_test_metrics.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(final_rows[0]))
    writer.writeheader()
    writer.writerows(final_rows)

periodic_metric_names = [
    key
    for key in periodic_rows[0]
    if key.startswith("test_") or key.startswith("test28_")
]
periodic_stats = []
for epoch in sorted({int(row["epoch"]) for row in periodic_rows}):
    epoch_rows = [
        row for row in periodic_rows if int(row["epoch"]) == epoch
    ]
    stats_row = {"epoch": epoch, "seed_count": len(epoch_rows)}
    for metric_name in periodic_metric_names:
        values = [float(row[metric_name]) for row in epoch_rows]
        stats_row[f"{metric_name}_mean"] = statistics.mean(values)
        stats_row[f"{metric_name}_std"] = statistics.stdev(values)
    periodic_stats.append(stats_row)

with (summary_dir / "test_every_10_epochs_mean_std.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(periodic_stats[0]))
    writer.writeheader()
    writer.writerows(periodic_stats)

final_metric_names = [
    key
    for key in final_rows[0]
    if key not in {"seed", "task"}
]
final_stats = {"seed_count": len(final_rows)}
for metric_name in final_metric_names:
    values = [float(row[metric_name]) for row in final_rows]
    final_stats[f"{metric_name}_mean"] = statistics.mean(values)
    final_stats[f"{metric_name}_std"] = statistics.stdev(values)

with (summary_dir / "best_checkpoint_test_mean_std.csv").open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(handle, fieldnames=list(final_stats))
    writer.writeheader()
    writer.writerow(final_stats)
PY

echo "All three runs completed."
echo "Periodic test summary: ${SUMMARY_DIR}/test_every_10_epochs.csv"
echo "Periodic mean/std: ${SUMMARY_DIR}/test_every_10_epochs_mean_std.csv"
echo "Best-checkpoint summary: ${SUMMARY_DIR}/best_checkpoint_test_metrics.csv"
echo "Best-checkpoint mean/std: ${SUMMARY_DIR}/best_checkpoint_test_mean_std.csv"
