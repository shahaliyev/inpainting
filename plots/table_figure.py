"""
Table figure — per-severity metric values as a publication-style PNG.

Reads eval_results.json files, extracts metric values at selected block-mask
severities, and renders a clean table figure for each metric.  Domain rows
are color-coded for readability.

Produces (in figures/tables/):
  table_psnr.png   / pdf/table_psnr.pdf
  table_ssim.png   / pdf/table_ssim.pdf
  table_lpips.png  / pdf/table_lpips.pdf
  table_l1.png     / pdf/table_l1.pdf

Optional flags:
  --metric psnr            generate only one metric table
  --severities 5 10 20 30 40   override default severity columns
  --family block           mask family (default: block)

Usage:
  python plots/table_figure.py --results $all --out figures
  python plots/table_figure.py --results $all --out figures --metric psnr
  python plots/table_figure.py --results $all --out figures --severities 10 20 30 40
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from plots._utils import (
    DOMAIN_LABELS,
    DOMAIN_ORDER,
    METRICS,
    MODEL_LABELS,
    MODEL_ORDER,
    extract_curve,
    get_xy,
    load_result,
    result_identity,
    savefig,
    set_paper_style,
)

# ── Default severity columns ──────────────────────────────────────────────────
DEFAULT_SEVERITIES = [5, 10, 20, 30, 40]

# ── Per-metric formatting ──────────────────────────────────────────────────────
METRIC_FMT: dict[str, str] = {
    "psnr":  "{:.2f}",
    "ssim":  "{:.3f}",
    "lpips": "{:.3f}",
    "l1":    "{:.3f}",
}

# ── Domain row colors (muted pastel per domain group) ────────────────────────
DOMAIN_ROW_COLORS: dict[str, str] = {
    "carpet":           "#e8f0f8",
    "dtd":              "#fff3ea",
    "imagenet-simple":  "#e8f5e8",
    "imagenet-complex": "#f5eaf5",
}
HEADER_BG   = "#2c3e50"
HEADER_FG   = "white"
SEP_COLOR   = "#999999"
EDGE_COLOR  = "#cccccc"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Render per-severity metric values as a table figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--family", default="block")
    ap.add_argument(
        "--severities", nargs="+", type=float, default=None,
        metavar="SEV",
        help="Severity levels to show as columns (default: 5 10 20 30 40).",
    )
    ap.add_argument(
        "--metric", default=None,
        choices=["psnr", "ssim", "lpips", "l1"],
        help="Generate only this metric table (default: all four).",
    )
    return ap.parse_args()


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(
    result_paths: list[str],
    family: str,
) -> dict[tuple[str, str], list]:
    """Return {(domain, model): [(severity, metrics_dict), ...]}."""
    data: dict[tuple[str, str], list] = {}
    for p in result_paths:
        result  = load_result(p)
        model, domain = result_identity(result)
        pts = extract_curve(result, family=family)
        data[(domain, model)] = pts
    return data


# ── Table rendering ───────────────────────────────────────────────────────────

def _make_table(
    by_domain_model: dict[tuple[str, str], list],
    domains: list[str],
    candidates: list[str],
    metric_key: str,
    metric_title: str,
    higher_better: bool,
    severities: list[float],
) -> tuple[list[list[str]], list[str], list[str], list[str]]:
    """
    Build (cell_text, row_labels, col_labels, row_domain_keys) for the table.
    cell_text[i][j] is the formatted value for row i, column j.
    row_domain_keys[i] is the domain key for row i (used for coloring).
    """
    fmt = METRIC_FMT.get(metric_key, "{:.3f}")
    direction = "↑" if higher_better else "↓"
    col_labels = [f"{int(s)}%" for s in severities]
    # Leading columns: Domain and Model
    col_labels = ["Domain", "Model"] + col_labels + [f"({direction})"]

    cell_text: list[list[str]] = []
    row_labels: list[str] = []
    row_domain_keys: list[str] = []

    for domain in domains:
        domain_label = DOMAIN_LABELS.get(domain, domain)
        first_in_domain = True
        for model in MODEL_ORDER:
            pts = by_domain_model.get((domain, model), [])
            xs, ys = get_xy(pts, candidates)
            sev_map = dict(zip(xs, ys))

            row: list[str] = []
            row.append(domain_label if first_in_domain else "")
            row.append(MODEL_LABELS.get(model, model))
            for s in severities:
                v = sev_map.get(float(s))
                row.append(fmt.format(v) if v is not None else "—")
            row.append("")   # direction column (header-only)
            cell_text.append(row)
            row_labels.append("")
            row_domain_keys.append(domain)
            first_in_domain = False

    return cell_text, row_labels, col_labels, row_domain_keys


def _render_table_figure(
    cell_text: list[list[str]],
    col_labels: list[str],
    row_domain_keys: list[str],
    title: str,
) -> mpl.figure.Figure:
    n_rows = len(cell_text)
    n_cols = len(col_labels)

    # Auto-size figure
    col_w   = 0.72      # inches per column
    row_h   = 0.28      # inches per row
    pad_top = 0.55      # space for title
    pad_bot = 0.15

    fig_w = max(5.5, n_cols * col_w)
    fig_h = (n_rows + 1) * row_h + pad_top + pad_bot

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)

    # Uniform row height
    for (row, col), cell in table.get_celld().items():
        cell.set_height(row_h / fig_h)
        cell.set_edgecolor(EDGE_COLOR)
        cell.set_linewidth(0.4)

        if row == 0:
            # Header row
            cell.set_facecolor(HEADER_BG)
            cell.get_text().set_color(HEADER_FG)
            cell.get_text().set_fontweight("bold")
            cell.get_text().set_fontsize(7.5)
        else:
            domain_key = row_domain_keys[row - 1]
            base_color = DOMAIN_ROW_COLORS.get(domain_key, "#f8f8f8")
            cell.set_facecolor(base_color)

            # Bold domain label (first column of first model in group)
            if col == 0 and cell.get_text().get_text():
                cell.get_text().set_fontweight("bold")

    # Draw separator lines between domain groups
    domain_changes = [
        i for i in range(1, len(row_domain_keys))
        if row_domain_keys[i] != row_domain_keys[i - 1]
    ]
    for change_row in domain_changes:
        for col in range(n_cols):
            cell = table[change_row + 1, col]
            cell.visible_edges = "TBL" if col == 0 else ("TBR" if col == n_cols - 1 else "TB")

    # Auto-fit column widths
    table.auto_set_column_width(list(range(n_cols)))

    ax.set_title(title, fontsize=9, fontweight="bold", pad=8, loc="left")
    plt.tight_layout(pad=0.5)
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    set_paper_style()
    out_dir    = Path(args.out) / "tables"
    severities = args.severities if args.severities else DEFAULT_SEVERITIES

    # ── Load ────────────────────────────────────────────────────────────────
    by_domain_model = load_data(args.results, args.family)
    print(f"  loaded {len(by_domain_model)} (domain, model) pairs")

    present = {k[0] for k in by_domain_model}
    domains = [d for d in DOMAIN_ORDER if d in present] + \
              sorted(present - set(DOMAIN_ORDER))

    # ── Select metrics ───────────────────────────────────────────────────────
    metrics_to_run = [
        m for m in METRICS
        if args.metric is None or m[0][0] == args.metric
    ]

    # ── Generate one table per metric ────────────────────────────────────────
    for candidates, title, _, higher_better in metrics_to_run:
        key = candidates[0]
        print(f"  {title} ...")

        cell_text, row_labels, col_labels, row_domain_keys = _make_table(
            by_domain_model=by_domain_model,
            domains=domains,
            candidates=candidates,
            metric_key=key,
            metric_title=title,
            higher_better=higher_better,
            severities=severities,
        )

        direction = "higher is better" if higher_better else "lower is better"
        fig_title = f"{title}  —  block mask, per severity  ({direction})"

        fig = _render_table_figure(cell_text, col_labels, row_domain_keys, fig_title)
        savefig(fig, out_dir, f"table_{key}")

    print(f"\nDone. Tables saved to {out_dir}/")


if __name__ == "__main__":
    main()
