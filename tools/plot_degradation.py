"""
Plot paper-ready degradation curves from one or more eval_results.json files.

Supports:
  - Single-run curves (legacy behavior)
  - Multi-run overlays for cross-model/cross-dataset comparisons
  - Numeric summary export for reporting (CSV + JSON)

Usage:
  python tools/plot_degradation.py \
    --results runs/a/eval/degradation_v1/val/epoch_100/eval_results.json

  python tools/plot_degradation.py \
    --results runs/a/.../eval_results.json runs/b/.../eval_results.json \
    --labels "UNet-Carpet" "UNet-DTD" --out_dir figures/degradation_compare
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


METRICS = [
    (["psnr", "psnr_mask", "psnr_full"], "PSNR (dB)", True),   # higher is better
    (["ssim", "ssim_mask", "ssim_full"], "SSIM", True),
    (["lpips", "lpips_mask", "lpips_full"], "LPIPS", False),   # lower is better
    (["l1", "l1_mask", "l1_full"], "L1", False),
]

MASK_STYLE = {
    "block": {"linestyle": "-", "marker": "o", "display": "Block"},
    "multi_block": {"linestyle": "--", "marker": "s", "display": "Multi-block"},
    "freeform": {"linestyle": "-", "marker": "^", "display": "Freeform"},
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Plot degradation curves from one or more eval_results.json files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more eval_results.json paths",
    )
    ap.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional display labels (same count as --results)",
    )
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for figures (default: parent of first results file)",
    )
    ap.add_argument("--title_prefix", default="Degradation")
    ap.add_argument("--dpi", type=int, default=170)
    return ap.parse_args()


def load_result(path: Path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


def infer_run_label(data: dict, path: Path):
    dataset = data.get("dataset", "unknown")
    model = data.get("model", "unknown")
    ckpt_name = data.get("checkpoint_name", "ckpt")
    return f"{model}/{dataset}/{ckpt_name}"


def parse_conditions(conditions: list):
    ratio_data = {}
    freeform_data = []

    for cond in conditions:
        mask_yaml = cond.get("mask_yaml", "")
        mask_type = Path(mask_yaml).stem
        metrics = cond.get("metrics", {})
        mask_ratios = cond.get("mask_ratios") or []
        mask_overrides = cond.get("mask_overrides") or {}

        if mask_type in ("block", "multi_block") and mask_ratios:
            ratio_data.setdefault(mask_type, []).append((int(mask_ratios[0]), metrics))
        elif mask_type == "freeform":
            num_strokes = mask_overrides.get("num_strokes")
            if num_strokes is not None:
                freeform_data.append((int(num_strokes), metrics))

    for items in ratio_data.values():
        items.sort(key=lambda x: x[0])
    freeform_data.sort(key=lambda x: x[0])
    return ratio_data, freeform_data


def pick_metric_key(conditions: list, candidates: list[str]):
    for cand in candidates:
        if any(cand in (c.get("metrics") or {}) for c in conditions):
            return cand
    return None


def _valid_xy(items, metric_key):
    xs, ys = [], []
    for x, metrics in items:
        y = metrics.get(metric_key)
        if y is None:
            continue
        xs.append(float(x))
        ys.append(float(y))
    return xs, ys


def _trapz(xs, ys):
    if len(xs) < 2:
        return None
    area = 0.0
    for i in range(1, len(xs)):
        dx = xs[i] - xs[i - 1]
        area += dx * (ys[i] + ys[i - 1]) * 0.5
    return area


def _build_title(args, metric_label, runs):
    protocols = sorted({str(r["data"].get("eval_protocol", "default")) for r in runs})
    splits = sorted({str(r["data"].get("split", "val")) for r in runs})
    return f"{args.title_prefix} - {metric_label} | split={','.join(splits)} | protocol={','.join(protocols)}"


def _plot_ratio(ax, runs, metric_candidates, metric_label, higher_better):
    palette = plt.get_cmap("tab10")
    for idx, run in enumerate(runs):
        color = palette(idx % 10)
        ratio_data = run["ratio_data"]
        metric_key = pick_metric_key(run["data"].get("conditions", []), metric_candidates)
        if metric_key is None:
            continue
        for mask_type in ("block", "multi_block"):
            items = ratio_data.get(mask_type, [])
            if not items:
                continue
            xs, ys = _valid_xy(items, metric_key)
            if not xs:
                continue
            st = MASK_STYLE[mask_type]
            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=1.8,
                markersize=4,
                label=f"{run['label']} | {st['display']}",
            )
    direction = "higher is better" if higher_better else "lower is better"
    ax.set_title(f"Block / Multi-block ({direction})", fontsize=10)
    ax.set_xlabel("Mask area (%)")
    ax.set_ylabel(metric_label)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=1)


def _plot_freeform(ax, runs, metric_candidates, metric_label, higher_better):
    palette = plt.get_cmap("tab10")
    plotted = 0
    for idx, run in enumerate(runs):
        metric_key = pick_metric_key(run["data"].get("conditions", []), metric_candidates)
        if metric_key is None:
            continue
        items = run["freeform_data"]
        if not items:
            continue
        xs, ys = _valid_xy(items, metric_key)
        if not xs:
            continue
        ax.plot(
            xs,
            ys,
            color=palette(idx % 10),
            linestyle=MASK_STYLE["freeform"]["linestyle"],
            marker=MASK_STYLE["freeform"]["marker"],
            linewidth=1.8,
            markersize=4,
            label=run["label"],
        )
        plotted += 1
    if plotted == 0:
        ax.set_visible(False)
        return
    direction = "higher is better" if higher_better else "lower is better"
    ax.set_title(f"Freeform ({direction})", fontsize=10)
    ax.set_xlabel("Number of strokes")
    ax.set_ylabel(metric_label)
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=1)


def build_summary_rows(runs):
    rows = []
    families = ("block", "multi_block", "freeform")
    for run in runs:
        conditions = run["data"].get("conditions", [])
        for metric_candidates, metric_label, higher_better in METRICS:
            metric_key = pick_metric_key(conditions, metric_candidates)
            if metric_key is None:
                continue
            for family in families:
                if family == "freeform":
                    items = run["freeform_data"]
                else:
                    items = run["ratio_data"].get(family, [])
                xs, ys = _valid_xy(items, metric_key)
                if not xs:
                    continue
                first = ys[0]
                last = ys[-1]
                delta = last - first
                auc = _trapz(xs, ys)
                rows.append({
                    "label": run["label"],
                    "source": str(run["path"]),
                    "protocol": run["data"].get("eval_protocol", run["data"].get("eval_profile", "default")),
                    "split": run["data"].get("split", "val"),
                    "metric": metric_candidates[0],
                    "metric_key_used": metric_key,
                    "metric_label": metric_label,
                    "higher_is_better": bool(higher_better),
                    "family": family,
                    "n_points": len(xs),
                    "x_min": xs[0],
                    "x_max": xs[-1],
                    "y_first": first,
                    "y_last": last,
                    "delta": delta,
                    "auc": auc,
                })
    return rows


def save_summary(out_dir: Path, rows: list[dict]):
    if not rows:
        return
    json_path = out_dir / "degradation_summary.json"
    csv_path = out_dir / "degradation_summary.csv"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    keys = list(rows[0].keys())
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {json_path}")
    print(f"Saved {csv_path}")


def main():
    args = parse_args()
    result_paths = [Path(p) for p in args.results]
    if args.labels is not None and len(args.labels) != len(result_paths):
        raise ValueError("--labels must have the same count as --results.")
    out_dir = Path(args.out_dir) if args.out_dir else result_paths[0].parent
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for i, path in enumerate(result_paths):
        data = load_result(path)
        ratio_data, freeform_data = parse_conditions(data.get("conditions", []))
        label = args.labels[i] if args.labels is not None else infer_run_label(data, path)
        runs.append({
            "path": path,
            "label": label,
            "data": data,
            "ratio_data": ratio_data,
            "freeform_data": freeform_data,
        })
        print(
            f"[{i}] {label} | protocol={data.get('eval_protocol', data.get('eval_profile', 'default'))} "
            f"| split={data.get('split', 'val')} | conditions={len(data.get('conditions', []))}"
        )

    saved = []
    for metric_candidates, metric_label, higher_better in METRICS:
        if not any(pick_metric_key(r["data"].get("conditions", []), metric_candidates) for r in runs):
            print(f"Skipping {metric_candidates[0]} - metric not present in any run")
            continue
        fig, (ax_ratio, ax_ff) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(_build_title(args, metric_label, runs), fontsize=12)

        canonical_key = metric_candidates[0]
        _plot_ratio(ax_ratio, runs, metric_candidates, metric_label, higher_better)
        _plot_freeform(ax_ff, runs, metric_candidates, metric_label, higher_better)

        plt.tight_layout()
        out_path = out_dir / f"degradation_compare_{canonical_key}.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
        print(f"Saved {out_path}")

    summary_rows = build_summary_rows(runs)
    save_summary(out_dir, summary_rows)
    print(f"\nDone. {len(saved)} figure(s) -> {out_dir}")


if __name__ == "__main__":
    main()
