"""Checkpoint resolution and loading helpers."""

from pathlib import Path

import torch

from .util.pos_embed import interpolate_pos_embed


def resolve_pretrained_checkpoint(model_name: str, finetune: str) -> Path:
    """Resolve a local checkpoint or download an official RETFound checkpoint."""
    candidate = Path(finetune).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    if candidate.suffix in {".pth", ".pt", ".ckpt"} or candidate.parent != Path("."):
        raise FileNotFoundError(f"Pre-trained checkpoint not found: {candidate}")

    if model_name in {"RETFound_dinov2", "RETFound_mae"}:
        from huggingface_hub import hf_hub_download

        return Path(
            hf_hub_download(
                repo_id=f"YukunZhou/{finetune}",
                filename=f"{finetune}.pth",
            )
        )

    raise FileNotFoundError(
        f"Model '{model_name}' requires a local checkpoint path, got: {finetune}"
    )


def extract_model_state(checkpoint: dict, model_name: str) -> dict:
    """Extract and normalize the state dict used by a downstream model."""
    if model_name in {"Dinov3", "Dinov2"}:
        state_dict = checkpoint
    elif model_name == "RETFound_dinov2":
        if "teacher" not in checkpoint:
            raise KeyError("RETFound-DINOv2 checkpoint has no 'teacher' state dict")
        state_dict = checkpoint["teacher"]
    elif model_name == "RETFound_mae":
        if "model" not in checkpoint:
            raise KeyError("RETFound-MAE checkpoint has no 'model' state dict")
        state_dict = checkpoint["model"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    normalized = {}
    for key, value in state_dict.items():
        key = key.removeprefix("backbone.")
        key = key.replace("mlp.w12.", "mlp.fc1.")
        key = key.replace("mlp.w3.", "mlp.fc2.")
        normalized[key] = value
    return normalized


def load_pretrained_weights(model, model_name: str, finetune: str):
    """Load pre-trained weights and return the path and incompatibility report."""
    checkpoint_path = resolve_pretrained_checkpoint(model_name, finetune)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = extract_model_state(checkpoint, model_name)

    model_state = model.state_dict()
    for key in ("head.weight", "head.bias"):
        if (
            key in state_dict
            and key in model_state
            and state_dict[key].shape != model_state[key].shape
        ):
            del state_dict[key]

    interpolate_pos_embed(model, state_dict)
    load_result = model.load_state_dict(state_dict, strict=False)
    return checkpoint_path, load_result
