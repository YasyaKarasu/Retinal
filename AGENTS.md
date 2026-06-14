# Repository Guidelines

## Project Structure & Module Organization

Core project code lives in `src/retfound/`. `main_finetune.py` is the training and evaluation entry point, `engine_finetune.py` contains epoch and metric logic, and `models_vit.py` defines supported backbones. Shared checkpoint, dataset, scheduling, distributed-training, and positional-embedding helpers are under `src/retfound/` and `src/retfound/util/`.

`RETFound/` is the ignored upstream reference checkout. `dataset/` contains local retinal images, while `hf_models/` contains downloaded model weights and helper scripts. All three are ignored by the top-level Git repository; treat them as local artifacts and never commit patient data, checkpoints, or generated outputs.

## Build, Test, and Development Commands

Run commands from the repository root unless noted:

```bash
conda create -n retinal python=3.11 -y
conda activate retinal
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
python -m compileall src
PYTHONPATH=src python -m retfound.verify_checkpoint --checkpoint hf_models/RETFound_dinov2_meh/RETFound_dinov2_meh.pth
```

Use the `retinal` conda environment for all development, training, and evaluation commands. The install commands reproduce the documented CUDA environment. `compileall` is the fastest repository-wide syntax check. Run training as the module `retfound.main_finetune` through `torchrun`; pass the local checkpoint path through `--finetune`.

## Coding Style & Naming Conventions

Use Python 3.11, four-space indentation, PEP 8 naming, and concise docstrings for public functions. Use `snake_case` for functions and variables, `PascalCase` for classes, and uppercase names for shell configuration constants such as `MODEL_ARCH`. Keep CLI options lowercase with underscores, matching existing arguments like `--data_path`. No formatter or linter is configured, so preserve nearby style and keep imports grouped by standard library, third-party, and local modules.

## Testing Guidelines

There is currently no automated test suite or coverage threshold. At minimum, run `python -m compileall src` and exercise the changed path with a small dataset or evaluation checkpoint. For metric or dataset changes, report the command, model, dataset split, seed, and relevant output metrics in the pull request. Avoid using full training runs as the only validation.

## Commit & Pull Request Guidelines

The top-level repository has no commit history; the nested RETFound history favors short imperative subjects such as `update readme` and `remove unused file`. Use a focused subject under roughly 72 characters and explain behavioral or experiment-impacting changes in the body.

Pull requests should summarize the change, list validation commands, identify required weights/data, and link related issues. Include metric comparisons for training changes and screenshots only for notebook or visualization updates. Do not include secrets, Hugging Face tokens, datasets, checkpoints, `output_dir/`, or `output_logs/`.
