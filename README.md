# Inpainting Experiments

This repository trains and evaluates an image inpainting model using YAML configs for the dataset, loader, mask, model, and training settings. The dependency files are aligned to a Python 3.10 environment.

## Install

Create and activate a virtual environment from the repository root.

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux / macOS:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For a fully pinned environment, install from `requirements.lock.txt` instead:

```bash
pip install -r requirements.lock.txt
```

## Project Layout

Key directories:

- `configs/` - YAML configs used by training and evaluation
- `data/` - dataset and dataloader code
- `mask/` - mask generator code
- `models/` - model code
- `training/` - training loop, checkpointing, logging, optimizer setup
- `utils/` - metrics, visualization, notebook helpers
- `tools/` - helper scripts such as dataset extraction
- `runs/` - output directory for checkpoints, metrics, and visualizations

## Dataset Setup

Datasets are expected outside the repository and are located via the `DATA_PATH` environment variable.

Windows PowerShell:

```powershell
$env:DATA_PATH = "D:\datasets"
```

Windows `cmd`:

```bat
set DATA_PATH=D:\datasets
```

Linux / macOS:

```bash
export DATA_PATH=/data
```

Current dataset configs expect:

### Carpet

`configs/dataset/carpet.yaml` resolves to:

```text
%DATA_PATH%/carpet/
  images/
    train/
      Cam_L/
      Cam_R/
    val/
      Cam_L/
      Cam_R/
```

Supported file extensions are `jpg`, `jpeg`, and `png`.

### DTD

`configs/dataset/dtd.yaml` resolves to:

```text
%DATA_PATH%/dtd/
  images/
  labels/
    train1.txt
    val1.txt
    test1.txt
```

The split files list image paths relative to `images/`.

If `DATA_PATH` is missing or the expected folders/files are not present, training and evaluation will fail or produce empty datasets.

## Running Training

Run from the repository root:

```bash
python train.py --dataset carpet --mask mixed --model unet
```

Important defaults:

- loader config key: `default` → `configs/loader/default.yaml`
- train config key: `default` → `configs/train/default.yaml`
- dataset, mask, and model keys are required for fresh training

Outputs are written under an auto-generated run directory:
`runs/<model>__<dataset>__<mask>__...__s<seed>__<timestamp>/`.
Each run stores `run_meta.json` and `resolved_config.yaml` for reproducibility.

Example with explicit configs:

```bash
python train.py --dataset carpet --mask block --model unet
```

CPU sanity check (small run, fast feedback):

```bash
python train.py --dataset carpet --mask mixed --model unet --train sanity_cpu --batch_size 2 --limit 32
```

This runs only a tiny subset of data for 2 epochs, disables AMP/compile, and is intended only to verify the pipeline.

Resume training from an existing checkpoint:

```bash
python train.py --resume --resume_ckpt runs/<train_run>/checkpoints/last.pt
```

`--resume` now requires `--resume_ckpt` and continues in the same run folder.
Use `--strict_config_match` if you want resume to fail on dataset/mask/model key mismatches.

## Running Evaluation

Evaluation must be given a checkpoint path:

```bash
python eval.py --ckpt runs/<train_run>/checkpoints/last.pt
```

By default, evaluation infers model/dataset/mask/train/loader settings from checkpoint metadata and writes output to:
`runs/<train_run>/eval/default/<split>/epoch_<n>/eval_results.json`.

CPU sanity evaluation on the same tiny subset:

```bash
python eval.py --eval sanity_cpu --ckpt runs/<train_run>/checkpoints/last.pt --batch_size 2 --limit 16
```

`--eval` is optional. When not provided, eval runs a single default condition inferred from checkpoint metadata.
For advanced use, `--eval_yaml` can still be used with a custom path.
Use `--strict_config_match` to enforce that eval grid dataset defaults match checkpoint metadata.
Metric scope is consistent by default (`mask`) and can be overridden with:

```bash
python eval.py --ckpt runs/<train_run>/checkpoints/best.pt --metric_scope full
```

## What The Current Code Actually Does

- Training requires explicit dataset/mask/model keys and resolves them under `configs/<group>/<name>.yaml`.
- Evaluation is checkpoint-first and infers model/dataset/mask/train/loader configs from the checkpoint.
- Primary metric scope is consistent (`mask` by default), with optional reporting of both mask/full variants.
- The current implementation does not match older README claims about fixed multi-placement mask protocols such as "25 masks per image", so those claims have been removed here.

## Dataset Extraction Helper

If your dataset is packaged as an archive, `tools/extract.py` can unpack it into `DATA_PATH`:

```bash
python tools/extract.py --archive path/to/archive.zip --name carpet
```

## Notes

- Run commands from the repository root so the default relative config paths resolve correctly.
- `requirements.txt` is the editable dependency list.
- `requirements.lock.txt` is the pinned Python 3.10 environment file.
