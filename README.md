# Inpainting Experiments

Train and evaluate image inpainting models with key-based configs and checkpoint-first evaluation.

## Quick Start

1. Set `DATA_PATH`.
2. Run train.
3. Run eval from the generated checkpoint.

```bash
python train.py --dataset carpet --mask mixed --model unet
python eval.py --ckpt runs/<train_run>/checkpoints/best.pt
```

## Install

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

Pinned environment:

```bash
pip install -r requirements.lock.txt
```

## Dataset Setup

Datasets are resolved from `DATA_PATH`.

Windows PowerShell:

```powershell
$env:DATA_PATH = "D:\datasets"
```

Linux / macOS:

```bash
export DATA_PATH=/data
```

Expected layouts:

- `carpet`: `%DATA_PATH%/carpet/images/{train,val}/{Cam_L,Cam_R}`
- `dtd`: `%DATA_PATH%/dtd/images` and `%DATA_PATH%/dtd/labels/{train1,val1,test1}.txt`

If `DATA_PATH` is missing or folders are incorrect, train/eval will fail.

## Training

Fresh run:

```bash
python train.py --dataset carpet --mask mixed --model unet
```

Useful variants:

```bash
python train.py --dataset carpet --mask block --model unet
python train.py --dataset carpet --mask mixed --model unet --train sanity_cpu --batch_size 2 --limit 32
python train.py --dataset carpet --mask mixed --model unet --train benchmark_v1
```

Resume:

```bash
python train.py --resume --resume_ckpt runs/<train_run>/checkpoints/last.pt
```

`--strict_config_match` makes resume fail on dataset/mask/model key mismatches.

Defaults:

- `--loader default` -> `configs/loader/default.yaml`
- `--train default` -> `configs/train/default.yaml`
- `--dataset`, `--mask`, `--model` are required for fresh runs

Loss configuration (in train config, not CLI):

- `loss.name`: `l1`, `l1_perceptual`, `l1_perceptual_tv`
- `loss.weights`: weighted combination terms (`l1`, `perceptual`, `tv`)
- `loss.perceptual.net`: LPIPS backbone (`vgg` or `alex`)
- `ckpt.patience`: early-stop patience counted in validation checks (not raw wall-clock time)
- `ckpt.min_epochs`: warmup period before early stopping is allowed
- `ckpt.min_delta`: minimum val-loss improvement required to reset patience
- Early stopping monitors `val_loss` (masked L1) for stable cross-run comparison.

Outputs:

- `runs/<auto_run_name>/checkpoints/{best.pt,last.pt}`
- `runs/<auto_run_name>/metrics.csv`
- `runs/<auto_run_name>/run_meta.json`
- `runs/<auto_run_name>/resolved_config.yaml`

## Evaluation

Basic eval:

```bash
python eval.py --ckpt runs/<train_run>/checkpoints/best.pt
```

Eval profile:

```bash
python eval.py --eval sanity_cpu --ckpt runs/<train_run>/checkpoints/last.pt --batch_size 2 --limit 16
python eval.py --eval degradation_v1 --ckpt runs/<train_run>/checkpoints/best.pt
```

Advanced profile path:

```bash
python eval.py --eval_yaml configs/eval/default.yaml --ckpt runs/<train_run>/checkpoints/best.pt
```

Strict mode:

- `--strict_config_match` enforces checkpoint/profile consistency checks.

Metric scope:

```bash
python eval.py --ckpt runs/<train_run>/checkpoints/best.pt --metric_scope full
```

Default eval output:

`runs/<train_run>/eval/<eval_profile>/<split>/epoch_<n>/eval_results.json`

## Plot Degradation Curves

```bash
python tools/plot_degradation.py --results runs/<train_run>/eval/degradation_v1/val/epoch_<n>/eval_results.json
python tools/plot_degradation.py --results runs/<run_a>/eval/degradation_v1/val/epoch_<n>/eval_results.json runs/<run_b>/eval/degradation_v1/val/epoch_<n>/eval_results.json --labels "modelA-datasetX" "modelB-datasetY" --out_dir figures/degradation_compare
```

## Benchmark Protocol (v1)

Keep protocol fixed when comparing models/datasets.

```bash
python train.py --dataset <dataset_key> --mask <mask_key> --model <model_key> --train benchmark_v1
python eval.py --eval degradation_v1 --ckpt runs/<train_run>/checkpoints/best.pt
python tools/plot_degradation.py --results runs/<train_run>/eval/degradation_v1/val/epoch_<n>/eval_results.json
```

## Project Layout

- `configs/` configuration files
- `data/` dataset and dataloader code
- `mask/` mask generators
- `models/` model definitions
- `training/` train/eval loop helpers, checkpointing, optimizer, logger
- `evaluation/` eval profile/grid utilities
- `utils/` metrics, visualization, run/config helpers
- `tools/` helper scripts
- `runs/` outputs

## Notes

- Run commands from repository root.
- `requirements.txt` is editable dependency list.
- `requirements.lock.txt` is pinned environment.

## Troubleshooting

- `dataset_cfg.root is missing` or empty dataset:
  - Ensure `DATA_PATH` is set and dataset folders match expected layout.
- `Unsupported checkpoint format version`:
  - Checkpoint was created with older code; re-train or use a checkpoint from current code.
- `ImportError: lpips` during eval:
  - Install dependencies from `requirements.txt` (or run eval with `--no_lpips`).
- `ImportError: matplotlib` when plotting:
  - Install dependencies from `requirements.txt`.
