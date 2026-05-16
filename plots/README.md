# plots/

Standalone plotting scripts for the inpainting reconstructability paper.
No script modifies the training or evaluation codebase.

---

## Setup

All scripts require only `matplotlib`, `numpy`, and the Python standard library.
`qualitative.py` additionally requires `torch`, `torchvision`, and `omegaconf`
because it loads model checkpoints and runs inference.

---

## Collecting result paths (PowerShell)

Every script takes a `--results` argument — a list of `eval_results.json` paths.
Collect them once and reuse:

```powershell
$all = (Get-ChildItem runs -Recurse -Filter eval_results.json).FullName
```

---

## Scripts

### `domain_curves.py`
Model-averaged degradation curves with visual domains as curves.
Outputs: `figures/main_candidates/domain_psnr_ssim.png`, `domain_lpips_l1.png`

```powershell
python plots/domain_curves.py --results $all --out figures
```

Also accepts `--spread_only` to print the inter-model spread table without saving figures.

---

### `degradation_curves.py`
Model-comparison degradation curves (one curve per model).

**Default mode** — 2×2 grids per metric (one panel per domain) + individual single panels:

```powershell
python plots/degradation_curves.py --results $all --out figures
```

Outputs:
- `figures/model_comparison/model_{psnr,ssim,lpips,l1}_all_domains.png`
- `figures/model_comparison/single/{domain}_{metric}_models.png` (all 16 combinations)

**Selected cases** — compact figure for architecture-sensitive domain:metric pairs:

```powershell
python plots/degradation_curves.py --results $all --out figures `
  --select carpet:lpips imagenet-complex:psnr imagenet-complex:l1
```

Output: `figures/main_candidates/selected_architecture_sensitive_cases.png`

---

### `dispersion.py`
Cross-probe dispersion — one plot per metric, in two variants:

- **Raw**: population std of probe values in original metric units.
- **Normalized**: raw values min-max normalized within each (metric, domain) pair
  before computing std, making dispersion comparable across domains.

By default both variants are generated.

```powershell
python plots/dispersion.py --results $all --out figures
python plots/dispersion.py --results $all --out figures --normalized
python plots/dispersion.py --results $all --out figures --unnormalized
```

Outputs:
- `figures/dispersion/raw_dispersion_{psnr,ssim,lpips,l1}.png`
- `figures/dispersion/normalized_dispersion_{psnr,ssim,lpips,l1}.png`
- `figures/dispersion/normalized_dispersion_values.csv`

---

### `robustness_bar.py`
Normalized AUC robustness score R_m per (model, domain, metric).
Saves two CSV tables and prints a LaTeX table per metric to stdout.

```powershell
python plots/robustness_bar.py --results $all --out figures
```

Outputs:
- `figures/robustness/robustness_table.csv` — rows: domains, columns: metrics, values: model-averaged R_m
- `figures/robustness/robustness_by_model.csv` — rows: domain×metric, columns: U-Net / Partial Conv / Gated Conv

---

### `qualitative.py`
Qualitative reconstruction figure. Loads model checkpoints, applies deterministic
block masks at specified severities, runs inference, and composes a multi-panel
figure with: original | masked input | U-Net | Partial Conv | Gated Conv | error map.

Requires checkpoint paths; see `--help` for all arguments.

```powershell
python plots/qualitative.py --help
```

---

### `complexity_hist.py`
Histogram / KDE of the ImageNet complexity score c(I) for ImageNet-Simple
vs. ImageNet-Complex subsets. Validates the dataset split.

```powershell
python plots/complexity_hist.py --help
```

---

## Output folder structure

```
figures/
  main_candidates/          # paper-ready candidates
    domain_psnr_ssim.png
    domain_lpips_l1.png
    dispersion_psnr_lpips.png
    selected_architecture_sensitive_cases.png

  model_comparison/         # diagnostic model-comparison grids
    model_psnr_all_domains.png
    model_ssim_all_domains.png
    model_lpips_all_domains.png
    model_l1_all_domains.png
    single/                 # individual (domain, metric) panels
      carpet_psnr_models.png
      carpet_ssim_models.png
      ...

  dispersion/               # all dispersion variants
    raw_dispersion_psnr.png
    raw_dispersion_ssim.png
    raw_dispersion_lpips.png
    raw_dispersion_l1.png
    normalized_dispersion_psnr.png
    normalized_dispersion_ssim.png
    normalized_dispersion_lpips.png
    normalized_dispersion_l1.png
    normalized_dispersion_values.csv

  robustness/               # R_m tables
    robustness_table.csv
    robustness_by_model.csv

  qualitative/              # qualitative reconstruction figures
```

---

## Shared utilities (`_utils.py`)

Constants and helpers shared across all scripts:

| Symbol | Description |
|---|---|
| `METRICS` | `(key_candidates, short_title, y_axis_label, higher_better)` for PSNR, SSIM, LPIPS, Masked L1 |
| `DOMAIN_ORDER` / `DOMAIN_LABELS` | Canonical domain ordering and display names |
| `MODEL_ORDER` / `MODEL_LABELS` / `MODEL_COLORS` / `MODEL_MARKERS` | Canonical model ordering and style |
| `DOMAIN_COLORS` / `DOMAIN_LINESTYLES` | Domain colors and line styles for dispersion plots |
| `TRAIN_SEVERITY_MAX` | `30` — training maximum severity, shown as a dashed vertical line |
| `FIG_SINGLE` / `FIG_DOUBLE` / `FIG_2x2` | Standard figure sizes for IEEE double-column layout |
| `set_paper_style()` | Applies publication-quality `rcParams` |
| `extract_curve()` | Extracts `[(severity, metrics_dict)]` from `eval_results.json` |
| `normalized_auc()` | Computes R_m (normalized area under degradation curve) |
| `model_averaged_curve()` | Computes Q_hat (mean across probes at shared severities) |
| `cross_probe_dispersion()` | Computes V_hat (mean squared deviation from Q_hat) |
| `add_train_boundary()` | Draws the subtle dashed vertical line at severity 30% |
| `savefig()` | Saves PNG and closes the figure |
