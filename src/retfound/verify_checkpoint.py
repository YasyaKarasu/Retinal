"""Verify that a RETFound checkpoint can be loaded into the selected backbone."""

import argparse
from types import SimpleNamespace

from .checkpoints import load_pretrained_weights
from .models_vit import RETFound_dinov2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num-classes", type=int, default=45)
    args = parser.parse_args()

    model_args = SimpleNamespace(model_arch="retfound_dinov2")
    model = RETFound_dinov2(
        model_args,
        num_classes=args.num_classes,
        drop_path_rate=0.0,
    )
    path, result = load_pretrained_weights(
        model,
        model_name="RETFound_dinov2",
        finetune=args.checkpoint,
    )

    backbone_missing = [
        key for key in result.missing_keys if not key.startswith("head.")
    ]
    print(f"checkpoint: {path}")
    print(f"missing keys: {len(result.missing_keys)}")
    print(f"unexpected keys: {len(result.unexpected_keys)}")
    if backbone_missing:
        raise RuntimeError(
            f"Checkpoint is missing {len(backbone_missing)} backbone parameters"
        )
    print("RETFound-DINOv2 backbone weights loaded successfully")


if __name__ == "__main__":
    main()
