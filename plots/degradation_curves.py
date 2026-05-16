"""
Degradation curves — model comparison per domain/metric.

Two modes:

1. Default (diagnostic): 2×2 grid per metric, one panel per domain, model curves.
   Also saves every individual (domain, metric) panel as a standalone file.
   Produces (in figures/model_comparison/):
     model_psnr_all_domains.png
     model_ssim_all_domains.png
     model_lpips_all_domains.png
     model_l1_all_domains.png
   Produces (in figures/model_comparison/single/):
     {domain}_{metric}_models.png  for every combination
     e.g. carpet_lpips_models.png, imagenet_complex_psnr_models.png

2. --select (selected architecture-sensitive cases):
   Pass domain:metric pairs; each becomes one panel with model curves.
   Produces (in figures/main_candidates/):
     selected_architecture_sensitive_cases.png

Usage (diagnostic — all domains, all metrics):
  python plots/degradation_curves.py \\
    --results $all --out figures

Usage (selected cases):
  python plots/degradation_curves.py \\
    --results $all --out figures \\
    --select carpet:lpips imagenet-complex:psnr imagenet-complex:l1
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    FIG_2x2,
    FIG_SINGLE,
    METRICS,
    MODEL_COLORS,
    MODEL_LABELS,
    MODEL_MARKERS,
    MODEL_ORDER,
    add_train_boundary,
    extract_curve,
    get_xy,
    load_result,
    result_identity,
    savefig,
    set_paper_style,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Model-comparison degradation curves.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--family", default="block")
    ap.add_argument(
        "--select", nargs="+", default=None, metavar="DOMAIN:METRIC",
        help=(
            "Generate selected-cases figure with only these domain:metric pairs. "
            "Example: --select carpet:lpips imagenet-complex:psnr. "
            "Valid metric keys: psnr ssim lpips l1."
        ),
    )
    return ap.parse_args()


def _slug(name: str) -> str:
    """Convert domain/metric name to a LaTeX-friendly filename slug."""
    return name.replace("-", "_")


def _parse_select(tokens: list[str]) -> list[tuple[str, str]]:
    result = []
    for tok in tokens:
        if ":" not in tok:
            print(f"  WARNING: ignoring malformed --select token '{tok}' (expected domain:metric)")
            continue
        domain, metric = tok.split(":", 1)
        result.append((domain.strip().lower(), metric.strip().lower()))
    return result


def _metric_by_key(key: str):
    for m in METRICS:
        if m[0][0] == key:
            return m
    return None


# ── Shared panel drawing ──────────────────────────────────────────────────────

def _draw_model_panel(
    ax: plt.Axes,
    domain: str,
    candidates: list[str],
    y_label: str,
    runs: list[dict],
    family: str,
    y_lim: tuple[float, float] | None = None,
) -> tuple[list, list[str]]:
    """Draw model curves for one (domain, metric) panel. Returns (handles, labels)."""
    handles, labels = [], []
    for r in runs:
        xs, ys = get_xy(r["points"], candidates)
        if not xs:
            continue
        model = r["model"]
        line, = ax.plot(
            xs, ys,
            color=MODEL_COLORS.get(model, "#333333"),
            marker=MODEL_MARKERS.get(model, "o"),
            linewidth=1.4,
            markersize=3.5,
            zorder=3,
        )
        handles.append(line)
        labels.append(MODEL_LABELS.get(model, model))

    add_train_boundary(ax)
    if y_lim:
        ax.set_ylim(*y_lim)
    ax.set_title(DOMAIN_LABELS.get(domain, domain), pad=3)
    ax.set_xlabel("Mask area (%)" if family != "freeform" else "Strokes", fontsize=7)
    ax.set_ylabel(y_label, fontsize=7)
    ax.xaxis.set_major_locator(mticker.FixedLocator([2, 10, 20, 30, 40]))
    ax.grid(True)
    return handles, labels


# ── Mode 1: 2×2 diagnostic figures ───────────────────────────────────────────

def _make_2x2(
    candidates: list[str],
    title: str,
    y_label: str,
    by_domain: dict[str, list],
    domains: list[str],
    family: str,
) -> plt.Figure | None:
    fig, axes = plt.subplots(2, 2, figsize=FIG_2x2)
    axes_flat = [axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]]

    all_ys: list[float] = []
    for domain in domains:
        for r in by_domain.get(domain, []):
            _, ys = get_xy(r["points"], candidates)
            all_ys.extend(ys)

    if not all_ys:
        plt.close(fig)
        return None

    margin = (max(all_ys) - min(all_ys)) * 0.07
    y_lim  = (min(all_ys) - margin, max(all_ys) + margin)

    all_handles, all_labels = [], []
    for ax, domain in zip(axes_flat, domains):
        h, l = _draw_model_panel(ax, domain, candidates, y_label,
                                 by_domain.get(domain, []), family, y_lim)
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    for ax in axes_flat[len(domains):]:
        ax.set_visible(False)

    fig.legend(all_handles, all_labels,
               loc="lower center", ncol=len(all_handles),
               bbox_to_anchor=(0.5, -0.04), frameon=True, fontsize=7)
    fig.suptitle(title, fontsize=9, y=1.01)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def _make_single_panel(
    domain: str,
    candidates: list[str],
    y_label: str,
    runs: list[dict],
    family: str,
) -> plt.Figure | None:
    """Standalone single-panel figure for one (domain, metric) combination."""
    all_ys: list[float] = []
    for r in runs:
        _, ys = get_xy(r["points"], candidates)
        all_ys.extend(ys)
    if not all_ys:
        return None

    fig, ax = plt.subplots(1, 1, figsize=FIG_SINGLE)
    h, l = _draw_model_panel(ax, domain, candidates, y_label, runs, family)
    if h:
        ax.legend(h, l, fontsize=7, frameon=True)
    plt.tight_layout()
    return fig


# ── Mode 2: selected-cases figure ────────────────────────────────────────────

def _make_selected(
    selected: list[tuple[str, str]],
    by_domain: dict[str, list],
    family: str,
) -> plt.Figure | None:
    n = len(selected)
    if n == 0:
        return None

    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig_w = min(7.0, ncols * 2.5)
    fig_h = nrows * 2.6

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]

    all_handles, all_labels = [], []

    for ax, (domain, metric_key) in zip(axes_flat, selected):
        entry = _metric_by_key(metric_key)
        if entry is None:
            ax.set_visible(False)
            print(f"  WARNING: unknown metric key '{metric_key}', skipping")
            continue
        candidates, _, y_label, _ = entry
        runs = by_domain.get(domain, [])
        if not runs:
            ax.set_visible(False)
            print(f"  WARNING: no runs for domain '{domain}', skipping")
            continue

        panel_title = f"{DOMAIN_LABELS.get(domain, domain)} — {metric_key.upper()}"
        h, l = _draw_model_panel(ax, domain, candidates, y_label, runs, family)
        ax.set_title(panel_title, pad=3, fontsize=8)
        for handle, label in zip(h, l):
            if label not in all_labels:
                all_handles.append(handle)
                all_labels.append(label)

    for ax in axes_flat[len(selected):]:
        ax.set_visible(False)

    if all_handles:
        fig.legend(all_handles, all_labels,
                   loc="lower center", ncol=len(all_handles),
                   bbox_to_anchor=(0.5, -0.06), frameon=True, fontsize=7)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()
    base = Path(args.out)

    by_domain: dict[str, list] = defaultdict(list)
    for p in args.results:
        data  = load_result(p)
        model, domain = result_identity(data)
        pts   = extract_curve(data, family=args.family)
        by_domain[domain].append({"model": model, "points": pts})
        print(f"  loaded  model={model:<14}  domain={domain}")

    domains = [d for d in DOMAIN_ORDER if d in by_domain] + \
              [d for d in by_domain   if d not in DOMAIN_ORDER]

    # ── --select mode ────────────────────────────────────────────────────────
    if args.select:
        selected = _parse_select(args.select)
        print(f"\nSelected cases: {selected}")
        fig = _make_selected(selected, by_domain, args.family)
        if fig is not None:
            savefig(fig, base / "main_candidates",
                    "selected_architecture_sensitive_cases")
        print(f"\nDone.")
        return

    # ── Default mode: 2×2 figures + individual single panels ────────────────
    mc_dir     = base / "model_comparison"
    single_dir = base / "model_comparison" / "single"

    print(f"\nGenerating 2×2 model-comparison figures ...")
    for candidates, title, y_label, _ in METRICS:
        metric_key = candidates[0]
        print(f"  {title} ...")
        fig = _make_2x2(candidates, title, y_label, by_domain, domains, args.family)
        if fig is not None:
            savefig(fig, mc_dir, f"model_{metric_key}_all_domains")
        else:
            print(f"  skipping — no data")

    print(f"\nGenerating individual single-panel plots ...")
    for candidates, _, y_label, _ in METRICS:
        metric_key = candidates[0]
        for domain in domains:
            runs_d = by_domain.get(domain, [])
            if not runs_d:
                continue
            stem = f"{_slug(domain)}_{metric_key}_models"
            fig = _make_single_panel(domain, candidates, y_label, runs_d, args.family)
            if fig is not None:
                savefig(fig, single_dir, stem)

    print(f"\nDone. Figures saved under {base}/")


if __name__ == "__main__":
    main()
