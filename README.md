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
python train.py
```

Important defaults:

- dataset config: `configs/dataset/carpet.yaml`
- loader config: `configs/loader/default.yaml`
- mask config: `configs/mask/block.yaml`
- model config: `configs/model/unet.yaml`
- train config: `configs/train/default.yaml`

Outputs are written under `runs/<exp>/`, including checkpoints, metrics, and saved images.

Example with explicit configs:

```bash
python train.py --dataset_yaml configs/dataset/carpet.yaml --mask_yaml configs/mask/block.yaml --model_yaml configs/model/unet.yaml --exp train_carpet
```

## Running Evaluation

Evaluation must be given a checkpoint path:

```bash
python eval.py --ckpt runs/train/checkpoints/last.pt
```

Example with explicit configs:

```bash
python eval.py --ckpt runs/train/checkpoints/best.pt --dataset_yaml configs/dataset/carpet.yaml --mask_yaml configs/mask/block.yaml --model_yaml configs/model/unet.yaml --exp eval_carpet
```

Evaluation uses `configs/eval/default.yaml` by default and writes `eval_results.json` under `runs/<exp>/`.

## What The Current Code Actually Does

- The default training and evaluation path uses the UNet model in `configs/model/unet.yaml`.
- The active mask config is `configs/mask/block.yaml`.
- Evaluation reports masked-region L1 loss and full-image PSNR, SSIM, and LPIPS.
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
