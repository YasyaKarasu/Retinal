# Retinal RETFound

Project code lives in `src/retfound`. RFMiD data, downloaded checkpoints, and
the upstream RETFound checkout remain ignored local artifacts.

## Environment

For an NVIDIA driver supporting CUDA 12.9:

```bash
conda activate retinal
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu129
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

Verify the two GPUs before training:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
print(torch.cuda.is_available(), torch.cuda.device_count()); \
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
```

## Local Checkpoint

Expected path:

```text
hf_models/RETFound_dinov2_meh/RETFound_dinov2_meh.pth
```

Verify architecture and checkpoint compatibility:

```bash
PYTHONPATH=src python -m retfound.verify_checkpoint \
  --checkpoint hf_models/RETFound_dinov2_meh/RETFound_dinov2_meh.pth \
  --num-classes 45
```

The checkpoint contains a DINOv2 teacher trained at a larger image resolution.
Its positional embedding is interpolated from 37x37 patches to the 16x16 patch
grid used by 224x224 inputs. The DINO/iBOT pre-training heads are ignored, and
a new 45-output classification head is initialized.

## RFMiD Data

The loader reads the existing official layout directly:

```text
dataset/
├── Training_Set/Training_Set/
│   ├── Training/*.png
│   └── RFMiD_Training_Labels.csv
├── Evaluation_Set/Evaluation_Set/
│   ├── Validation/*.png
│   └── RFMiD_Validation_Labels.csv
└── Test_Set/Test_Set/
    ├── Test/*.png
    └── RFMiD_Testing_Labels.csv
```

The task uses the 45 disease columns from `DR` through `CL`. `Disease_Risk` is
not included in the output. Training uses `BCEWithLogitsLoss`; evaluation uses
sigmoid probabilities and multi-label F1, AUROC, average precision, Hamming
loss, and related metrics.

## Fine-Tuning

For two NVIDIA L40 48GB GPUs, start full fine-tuning with:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_rfmid_2xl40.sh
```

The L40 script avoids gradient accumulation:

```text
effective batch size = 2 GPUs x 16 images = 32
1920 training images / 32 = 60 optimizer steps per epoch
```

Outputs are written to:

```text
output_dir/retfound_dinov2_meh_rfmid_2xl40/
output_logs/retfound_dinov2_meh_rfmid_2xl40/
```

Evaluate an existing best checkpoint without retraining:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/eval_rfmid_2xl40.sh
```

The evaluation script first calibrates thresholds on the validation split,
then applies the frozen thresholds to the test split. Classes with fewer than
10 positive or negative validation examples use the globally calibrated
threshold instead of an unstable class-specific threshold.

Detailed outputs are written beside the checkpoint:

```text
thresholds.json
per_class_val_per_class.csv
per_class_test_per_class.csv
predictions_val_per_class.csv
predictions_test_per_class.csv
metrics_val_per_class.csv
metrics_test_per_class.csv
```

An alternative checkpoint path can be passed as the first argument:

```bash
bash scripts/eval_rfmid_2xl40.sh /path/to/checkpoint-best.pth
```

To test the pipeline by training only the classification head, change:

```bash
--adaptation finetune
```

to:

```bash
--adaptation lp
```

If full fine-tuning runs out of memory, set `--batch_size 8` and
`--accum_iter 2`, preserving an effective batch size of 32.

## Six L40 GPUs

For six NVIDIA L40 48GB GPUs, use:

```bash
bash scripts/train_rfmid_6xl40.sh
```

The six-GPU configuration uses:

```text
6 GPUs x batch size 8 x accumulation 1 = global batch size 48
1920 training images / 48 = 40 optimizer steps per epoch
```

Gradient accumulation is intentionally disabled. With DDP, each backward pass
normally synchronizes gradients, so accumulation would add work without helping
this 48GB setup. The validation and test sets are sharded across the six ranks
without duplicate samples and gathered before computing metrics.

L40 uses PCIe Gen4 and does not support NVLink. Six-GPU scaling therefore will
not be linear for this approximately 300M-parameter model. Batch size 8 is the
recommended balance between GPU utilization, communication cost, and enough
optimizer updates for the small RFMiD dataset. After confirming stability,
batch sizes 12 or 16 can be benchmarked, but they reduce the number of optimizer
steps and may require retuning the learning rate or epoch count.
