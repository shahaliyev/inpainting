"""
Domain-averaged degradation curves — Figures 2 and 3.

The primary reconstructability figure for the paper.

x-axis : mask severity (%)
y-axis : metric value
curves : visual domains (Carpet, DTD, ImageNet-Simple, ImageNet-Complex)
values : averaged over all available probe models (U-Net, Partial Conv, Gated Conv)

Produces (in figures/main_candidates/):
  domain_psnr_ssim.png   — 1×2: PSNR | SSIM
  domain_lpips_l1.png    — 1×2: LPIPS | Masked L1

Also prints a spread table showing max inter-model variation per (domain, metric),
useful for deciding which cases warrant a model-comparison plot.

Usage:
  python plots/domain_curves.py \\
    --results (Get-ChildItem runs -Recurse -Filter eval_results.json).FullName \\
    --out figures

Run --spread_only to just print the spread table without saving figures:
  python plots/domain_curves.py --results $all --spread_only
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import (
    DOMAIN_COLORS,
    DOMAIN_LABELS,
    DOMAIN_LINESTYLES,
    DOMAIN_ORDER,
    FIG_2x2,
    FIG_DOUBLE,
    METRICS,
    MODEL_ORDER,
    add_train_boundary,
    extract_curve,
    get_xy,
    load_result,
    result_identity,
    savefig,
    set_paper_style,
)

# Domain markers so curves are distinguishable in black-and-white print too
DOMAIN_MARKERS: dict[str, str] = {
    "carpet":           "o",
    "dtd":              "s",
    "imagenet-simple":  "^",
    "imagenet-complex": "D",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Model-averaged domain degradation curves (Figures 2 and 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results", nargs="+", required=True,
                    help="Paths to eval_results.json files (all 12).")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--family", default="block")
    ap.add_argument(
        "--spread_only", action="store_true", default=False,
        help="Only print the inter-model spread table; skip figure generation.",
    )
    return ap.parse_args()


# ── Data assembly ─────────────────────────────────────────────────────────────

def _build_domain_curves(
    by_domain_model: dict[tuple[str, str], list],
    candidates: list[str],
    domains: list[str],
) -> dict[str, tuple[list[float], list[float]]]:
    """
    For each domain, average the raw metric across all available probe models
    at each shared severity level.
    Returns {domain: (severities, mean_values)}.
    """
    result: dict[str, tuple[list[float], list[float]]] = {}
    for domain in domains:
        sev_vals: dict[float, list[float]] = defaultdict(list)
        for model in MODEL_ORDER:
            pts = by_domain_model.get((domain, model), [])
            xs, ys = get_xy(pts, candidates)
            for x, y in zip(xs, ys):
                sev_vals[x].append(y)
        if not sev_vals:
            continue
        sevs  = sorted(sev_vals)
        means = [float(np.mean(sev_vals[s])) for s in sevs]
        result[domain] = (sevs, means)
    return result


def _compute_spread(
    by_domain_model: dict[tuple[str, str], list],
    candidates: list[str],
    domains: list[str],
) -> dict[str, float]:
    """
    Max absolute range across models, averaged over severity levels.
    Returns {domain: spread_value}.  Higher = models disagree more.
    """
    spread: dict[str, float] = {}
    for domain in domains:
        sev_vals: dict[float, list[float]] = defaultdict(list)
        for model in MODEL_ORDER:
            pts = by_domain_model.get((domain, model), [])
            xs, ys = get_xy(pts, candidates)
            for x, y in zip(xs, ys):
                sev_vals[x].append(y)
        ranges = [max(v) - min(v) for v in sev_vals.values() if len(v) > 1]
        spread[domain] = float(np.mean(ranges)) if ranges else 0.0
    return spread


# ── Spread table ──────────────────────────────────────────────────────────────

def print_spread_table(
    by_domain_model: dict[tuple[str, str], list],
    domains: list[str],
) -> None:
    print("\n── Inter-model spread (mean range across severities) ────────────────")
    print(f"{'Domain':<22}", end="")
    for _, title, _, _ in METRICS:
        print(f"  {title:>10}", end="")
    print()
    print("-" * (22 + 12 * len(METRICS)))

    for domain in domains:
        print(f"{DOMAIN_LABELS.get(domain, domain):<22}", end="")
        for candidates, _, _, _ in METRICS:
            s = _compute_spread(by_domain_model, candidates, [domain])
            v = s.get(domain, 0.0)
            print(f"  {v:>10.4f}", end="")
        print()

    print("\nHigher spread → models disagree → candidate for Figure 4.")


# ── Plot helpers ──────────────────────────────────────────────────────────────

def _draw_panel(
    ax: plt.Axes,
    title: str,
    y_label: str,
    domain_curves: dict[str, tuple[list, list]],
    domains: list[str],
    family: str,
) -> tuple[list, list[str]]:
    """
    Draw domain curves into one axis.
    Returns (handles, labels) for building a shared legend.
    """
    handles, labels = [], []
    for domain in domains:
        if domain not in domain_curves:
            continue
        xs, ys = domain_curves[domain]
        if not xs:
            continue
        line, = ax.plot(
            xs, ys,
            color=DOMAIN_COLORS.get(domain, "#333333"),
            linestyle=DOMAIN_LINESTYLES.get(domain, "-"),
            marker=DOMAIN_MARKERS.get(domain, "o"),
            linewidth=1.4,
            markersize=3.5,
            zorder=3,
        )
        handles.append(line)
        labels.append(DOMAIN_LABELS.get(domain, domain))

    add_train_boundary(ax)

    ax.set_xlabel("Mask area (%)" if family != "freeform" else "Strokes", fontsize=8)
    ax.set_ylabel(y_label, fontsize=8)
    ax.set_title(title, pad=3)
    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    ax.grid(True)
    return handles, labels


def _save_pair(
    metric_a: tuple, metric_b: tuple,
    by_domain_model: dict,
    domains: list[str],
    family: str,
    stem: str,
    out_dir: Path,
) -> None:
    """Build and save a 1×2 figure for two metrics."""
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=FIG_DOUBLE)

    all_handles, all_labels = [], []

    for ax, (candidates, title, y_label, _) in [(ax_a, metric_a), (ax_b, metric_b)]:
        dc = _build_domain_curves(by_domain_model, candidates, domains)
        h, l = _draw_panel(ax, title, y_label, dc, domains, family)
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        ncol=len(all_handles),
        bbox_to_anchor=(0.5, -0.06),
        frameon=True,
        fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    savefig(fig, out_dir, stem)


def _save_2x2(
    by_domain_model: dict,
    domains: list[str],
    family: str,
    out_dir: Path,
) -> None:
    """Build and save a 2×2 figure with all four metrics."""
    fig, axes = plt.subplots(2, 2, figsize=FIG_2x2)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    all_handles, all_labels = [], []

    for ax, (candidates, title, y_label, _) in zip(axes_flat, METRICS):
        dc = _build_domain_curves(by_domain_model, candidates, domains)
        h, l = _draw_panel(ax, title, y_label, dc, domains, family)
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        ncol=len(all_handles),
        bbox_to_anchor=(0.5, -0.04),
        frameon=True,
        fontsize=7,
    )
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    savefig(fig, out_dir, "domain_all_metrics")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()
    out_dir = Path(args.out) / "main_candidates"

    # ── Load ────────────────────────────────────────────────────────────────
    by_domain_model: dict[tuple[str, str], list] = {}
    for p in args.results:
        data  = load_result(p)
        model, domain = result_identity(data)
        pts   = extract_curve(data, family=args.family)
        by_domain_model[(domain, model)] = pts
        print(f"  loaded  model={model:<14}  domain={domain}")

    present = {k[0] for k in by_domain_model}
    domains = [d for d in DOMAIN_ORDER if d in present] + \
              sorted(present - set(DOMAIN_ORDER))

    print(f"\nDomains  : {[DOMAIN_LABELS.get(d, d) for d in domains]}")
    print(f"Models   : {sorted({k[1] for k in by_domain_model})}")

    # ── Spread table (always) ────────────────────────────────────────────────
    print_spread_table(by_domain_model, domains)

    if args.spread_only:
        return

    # ── Figure 2: PSNR + SSIM ───────────────────────────────────────────────
    metric_lookup = {m[0][0]: m for m in METRICS}
    psnr_m  = metric_lookup["psnr"]
    ssim_m  = metric_lookup["ssim"]
    lpips_m = metric_lookup["lpips"]
    l1_m    = metric_lookup["l1"]

    print("\nPSNR + SSIM ...")
    _save_pair(psnr_m, ssim_m, by_domain_model, domains, args.family,
               "domain_psnr_ssim", out_dir)

    print("LPIPS + Masked L1 ...")
    _save_pair(lpips_m, l1_m, by_domain_model, domains, args.family,
               "domain_lpips_l1", out_dir)

    print("All metrics 2×2 ...")
    _save_2x2(by_domain_model, domains, args.family, out_dir)

    print(f"\nDone. Figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
