"""
Plot degradation curves from eval_results.json produced by eval.py.

Severity is read directly from the structured JSON — no condition-name parsing:
  mask_ratios[0]            → x-axis for block / multi-block conditions
  mask_overrides.num_strokes → x-axis for freeform conditions

Mask type is derived from the mask_yaml filename stem
(e.g. block.yaml → "block", multi_block.yaml → "multi_block").

Produces one PNG per metric (PSNR, SSIM, LPIPS, masked L1), each with two
side-by-side subplots:
  Left  – block and multi-block  (x = mask area %)
  Right – freeform               (x = number of strokes)

Usage:
  python tools/plot_degradation.py --results runs/<train_run>/eval/default/val/epoch_<n>/eval_results.json
  python tools/plot_degradation.py --results runs/<train_run>/eval/default/val/epoch_<n>/eval_results.json --out_dir figures/
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


METRICS = [
    (["psnr", "psnr_mask", "psnr_full"], "PSNR (dB)", True),   # True  = higher is better
    (["ssim", "ssim_mask", "ssim_full"], "SSIM", True),
    (["lpips", "lpips_mask", "lpips_full"], "LPIPS", False),   # False = lower is better
    (["l1", "l1_mask", "l1_full"], "L1", False),
]

STYLE = {
    "block":       dict(color="#1f77b4", marker="o", linestyle="-",  label="Block"),
    "multi_block": dict(color="#ff7f0e", marker="s", linestyle="--", label="Multi-block"),
    "freeform":    dict(color="#2ca02c", marker="^", linestyle="-",  label="Freeform"),
}


def parse_conditions(conditions: list):
    """
    Returns:
      ratio_data:    {mask_type: [(ratio_pct, metrics_dict), ...]}  sorted ascending
      freeform_data: [(num_strokes, metrics_dict), ...]             sorted ascending
    """
    ratio_data = {}
    freeform_data = []

    for cond in conditions:
        mask_yaml = cond.get("mask_yaml", "")
        mask_type = Path(mask_yaml).stem          # "block", "multi_block", "freeform"
        metrics = cond["metrics"]
        mask_ratios = cond.get("mask_ratios") or []
        mask_overrides = cond.get("mask_overrides") or {}

        if mask_type in ("block", "multi_block") and mask_ratios:
            ratio = int(mask_ratios[0])
            ratio_data.setdefault(mask_type, []).append((ratio, metrics))

        elif mask_type == "freeform":
            num_strokes = mask_overrides.get("num_strokes")
            if num_strokes is not None:
                freeform_data.append((int(num_strokes), metrics))

    for v in ratio_data.values():
        v.sort(key=lambda x: x[0])
    freeform_data.sort(key=lambda x: x[0])

    return ratio_data, freeform_data


def _plot_ratio(ax, ratio_data, metric_key, metric_label, higher_better):
    for mask_type in ("block", "multi_block"):
        items = ratio_data.get(mask_type, [])
        if not items:
            continue
        xs = [r for r, _ in items]
        ys = [m.get(metric_key) for _, m in items]
        if any(y is not None for y in ys):
            ax.plot(xs, ys, **STYLE[mask_type])
    ax.set_xlabel("Mask area (%)")
    ax.set_ylabel(metric_label)
    direction = "↑ higher is better" if higher_better else "↓ lower is better"
    ax.set_title(f"Block / Multi-block  ({direction})", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))


def _plot_freeform(ax, freeform_data, metric_key, metric_label, higher_better):
    if not freeform_data:
        ax.set_visible(False)
        return
    xs = [n for n, _ in freeform_data]
    ys = [m.get(metric_key) for _, m in freeform_data]
    if any(y is not None for y in ys):
        ax.plot(xs, ys, **STYLE["freeform"])
    ax.set_xlabel("Number of strokes")
    ax.set_ylabel(metric_label)
    direction = "↑ higher is better" if higher_better else "↓ lower is better"
    ax.set_title(f"Freeform  ({direction})", fontsize=10)
    ax.set_xticks(xs)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot degradation curves from eval_results.json",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results", required=True,
                    help="Path to eval_results.json produced by eval.py")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory for figures (default: same directory as results)")
    ap.add_argument("--dpi", type=int, default=150)
    return ap.parse_args()


def main():
    args = parse_args()
    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(results_path) as f:
        data = json.load(f)

    epoch = data.get("epoch", "?")
    split = data.get("split", "val")
    conditions = data["conditions"]

    print(f"Epoch {epoch}  |  split: {split}  |  conditions: {len(conditions)}")

    ratio_data, freeform_data = parse_conditions(conditions)
    print(f"Ratio-based : {list(ratio_data.keys())}")
    print(f"Freeform    : {[n for n, _ in freeform_data]} strokes")

    if not ratio_data and not freeform_data:
        print("No plottable conditions found. Ensure eval_results.json contains "
              "ratio-based (mask_ratios) and/or freeform (mask_overrides.num_strokes) conditions.")
        return

    saved = []
    for metric_candidates, metric_label, higher_better in METRICS:
        all_items = [(r, m) for items in ratio_data.values() for r, m in items] + freeform_data
        metric_key = None
        for candidate in metric_candidates:
            if any(candidate in m for _, m in all_items):
                metric_key = candidate
                break
        if metric_key is None:
            print(f"Skipping {metric_candidates[0]} — not present in results")
            continue

        fig, (ax_ratio, ax_ff) = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"Degradation — {metric_label}  (epoch {epoch}, {split})", fontsize=12)

        _plot_ratio(ax_ratio, ratio_data, metric_key, metric_label, higher_better)
        _plot_freeform(ax_ff, freeform_data, metric_key, metric_label, higher_better)

        plt.tight_layout()
        out_path = out_dir / f"degradation_{metric_candidates[0]}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved {out_path}")

    print(f"\nDone. {len(saved)} figure(s) → {out_dir}")


if __name__ == "__main__":
    main()
