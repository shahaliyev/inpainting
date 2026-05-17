"""
Paper sample demo: domains with block masks at each severity (black holes).

Usage:
  export DATA_PATH=/path/to/data
  python tools/sample_demo.py
  python tools/sample_demo.py --dataset carpet
  # -> figures/paper/sample_demo/carpet_s42_single.png
  python tools/sample_demo.py --n-samples 4 --seed 0
  # -> .../carpet_s0_multi4.png, etc.
  python tools/sample_demo.py --combined
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import ImageGrid
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.build import build_base_dataset
from mask.build import build_mask_generator
from plots._utils import set_paper_style
from utils.config_resolver import resolve_config_path
from utils.demo_utils import apply_demo_mask, denorm, get_norm_from_cfg

DOMAIN_ORDER = ("carpet", "dtd", "imagenet-simple", "imagenet-complex")
COMBINED_DOMAIN_ORDER = ("dtd", "carpet", "imagenet-simple", "imagenet-complex")

DATASET_DISPLAY = {
    "carpet": "Carpet",
    "dtd": "DTD",
    "imagenet-simple": "ImageNet-Simple",
    "imagenet-complex": "ImageNet-Complex",
}

DEFAULT_RATIOS = (5.0, 10.0, 20.0, 30.0)
DEFAULT_OUT_DIR = REPO_ROOT / "figures" / "paper" / "sample_demo"
DEFAULT_COMBINED_OUT = REPO_ROOT / "figures" / "paper" / "sample_demo.png"

RATIO_TITLE_COLOR = "#222222"
ORIGINAL_TITLE_COLOR = "#333333"
RATIO_TITLE_SIZE_DELTA = 0.5

# Fixed physical gap between images inside one dataset panel.
# This controls the red-arrow gaps. Because it is in inches, horizontal
# and vertical gaps are actually comparable.
INNER_IMAGE_PAD_INCH = 0.035
COMBINED_INNER_GAP_PX = 4
COMBINED_OUTER_PAD_PX = 8

# Outer 2x2 dataset spacing. These control the green/caption gaps.
COMBINED_OUTER_WSPACE = 0.01
COMBINED_OUTER_HSPACE = 0.01


def column_labels(ratios: list[float]) -> list[str]:
    return ["Original"] + [
        f"{int(r) if r == int(r) else r:g}%" for r in ratios
    ]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate dataset + block-mask sample demo figures for the paper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ds_group = ap.add_mutually_exclusive_group()
    ds_group.add_argument(
        "--dataset",
        choices=DOMAIN_ORDER,
        help="Single dataset only (default: all four domains)",
    )
    ds_group.add_argument(
        "--datasets",
        metavar="NAMES",
        help="Comma-separated dataset names (default: carpet,dtd,imagenet-simple,imagenet-complex)",
    )

    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--n-samples", type=int, default=1, help="Random images per domain")

    ap.add_argument(
        "--ratios",
        default=",".join(str(int(r)) if r == int(r) else str(r) for r in DEFAULT_RATIOS),
        help="Block mask area percentages",
    )

    ap.add_argument(
        "--mask-yaml",
        default=str(REPO_ROOT / "configs" / "mask" / "block.yaml"),
        help="Reference mask config (ratios overridden per column)",
    )

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help="Directory for per-dataset PNGs ({name}_s{N}_{single|multiK}.png)",
    )

    ap.add_argument(
        "--combined",
        action="store_true",
        help="Also save one 2x2 combined figure",
    )

    ap.add_argument(
        "--out",
        default=str(DEFAULT_COMBINED_OUT),
        help="Combined figure base path; s{N} and mode tag appended (only with --combined)",
    )

    ap.add_argument("--dataset-rows", type=int, default=2)
    ap.add_argument("--dataset-cols", type=int, default=2)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--combined-inner-gap",
        type=int,
        default=COMBINED_INNER_GAP_PX,
        help="Pixel gap between images inside one dataset panel for --combined.",
    )
    ap.add_argument(
        "--combined-dataset-gap",
        type=int,
        default=15,
        help="Pixel gap between dataset panels for --combined. Defaults to 2 * --combined-inner-gap.",
    )
    ap.add_argument(
        "--combined-outer-pad",
        type=int,
        default=COMBINED_OUTER_PAD_PX,
        help="White pixel padding around the whole --combined PNG.",
    )

    ap.add_argument(
        "--figsize",
        default=None,
        help="Figure size W,H in inches for each per-dataset figure (default: auto)",
    )

    ap.add_argument("--pdf", action="store_true", help="Also save PDF alongside PNG")

    return ap.parse_args()


def resolve_datasets(args: argparse.Namespace) -> list[str]:
    if args.dataset:
        return [args.dataset]

    if args.datasets:
        return [d.strip() for d in args.datasets.split(",") if d.strip()]

    if args.combined:
        return list(COMBINED_DOMAIN_ORDER)

    return list(DOMAIN_ORDER)


def sample_mode_tag(n_samples: int) -> str:
    n = max(1, int(n_samples))
    return "single" if n == 1 else f"multi{n}"


def tagged_stem(stem: str, seed: int, n_samples: int) -> str:
    return f"{stem}_s{seed}_{sample_mode_tag(n_samples)}"


def tagged_path(path: Path, seed: int, n_samples: int) -> Path:
    return path.parent / f"{tagged_stem(path.stem, seed, n_samples)}{path.suffix}"


def dataset_caption(panel_idx: int, name: str) -> str:
    display = DATASET_DISPLAY.get(name, name)
    panel_letter = chr(ord("a") + panel_idx)
    return f"({panel_letter}) {display}"


def _mask_at_ratio(
    image_shape: tuple[int, ...],
    ratio: float,
    mask_yaml: str,
) -> torch.Tensor:
    cfg = OmegaConf.load(mask_yaml)
    cfg = OmegaConf.merge(
        cfg,
        OmegaConf.create(
            {
                "ratios": [float(ratio)],
                "eval": {"deterministic": False},
            }
        ),
    )
    gen = build_mask_generator(cfg, split="eval")
    return gen(image_shape)


def _mask_area_fraction(mask: torch.Tensor) -> float:
    return float(mask.sum().item()) / float(mask.numel())


def _sample_indices(n_total: int, n_samples: int, rng: np.random.Generator) -> list[int]:
    n = min(n_samples, n_total)

    if n <= 0:
        return []

    return sorted(rng.choice(n_total, size=n, replace=False).tolist())


def load_dataset_cfg(name: str):
    path = resolve_config_path("dataset", name)
    cfg = OmegaConf.load(path)

    if "${oc.env:" in OmegaConf.to_yaml(cfg):
        OmegaConf.resolve(cfg)

    return cfg


def collect_domain_panels(
    dataset_name: str,
    split: str,
    indices: list[int],
    ratios: list[float],
    mask_yaml: str,
) -> list[list[np.ndarray]]:
    cfg = load_dataset_cfg(dataset_name)
    mean, std = get_norm_from_cfg(cfg)
    ds = build_base_dataset(cfg, split)

    rows: list[list[np.ndarray]] = []

    for image_idx in indices:
        item = ds[image_idx]
        image = item["image"]
        orig = denorm(image, mean, std)

        row = [orig]

        for ratio in ratios:
            mask = _mask_at_ratio(tuple(image.shape), ratio, mask_yaml)
            area = _mask_area_fraction(mask)

            if area <= 0:
                print(
                    f"WARNING: {dataset_name} idx={image_idx} ratio={ratio}% "
                    f"produced empty mask",
                    file=sys.stderr,
                )

            row.append(apply_demo_mask(orig, mask))

        rows.append(row)

    return rows


def _panel_figsize(n_sample_rows: int, n_cols: int) -> tuple[float, float]:
    return (1.35 * n_cols + 0.15, 1.35 * n_sample_rows + 0.55)


def _style_severity_axis(
    ax: plt.Axes,
    sc: int,
    ratios: list[float],
    *,
    show_title: bool,
    title_fontsize: float,
) -> None:
    labels = column_labels(ratios)

    if not show_title:
        return

    if sc == 0:
        ax.set_title(
            labels[0],
            fontsize=title_fontsize,
            fontweight="bold",
            pad=3,
            color=ORIGINAL_TITLE_COLOR,
        )
    else:
        ax.set_title(
            labels[sc],
            fontsize=title_fontsize + RATIO_TITLE_SIZE_DELTA,
            fontweight="bold",
            pad=3,
            color=RATIO_TITLE_COLOR,
        )


def build_domain_figure(
    rows: list[list[np.ndarray]],
    ratios: list[float],
    dataset_name: str,
    dpi: int,
    figsize: tuple[float, float] | None,
) -> plt.Figure:
    n_cols = 1 + len(ratios)
    n_rows = len(rows)

    if figsize is None:
        figsize = _panel_figsize(n_rows, n_cols)

    fig = plt.figure(figsize=figsize, dpi=dpi)

    top = 0.94 if n_rows > 1 else 0.88
    bottom = 0.08
    left = 0.01
    right = 0.995

    grid = ImageGrid(
        fig,
        rect=[left, bottom, right - left, top - bottom],
        nrows_ncols=(n_rows, n_cols),
        axes_pad=INNER_IMAGE_PAD_INCH,
        share_all=False,
        aspect=True,
    )

    bottom_sc = n_cols // 2

    for sr in range(n_rows):
        for sc in range(n_cols):
            ax = grid[sr * n_cols + sc]

            if sc < len(rows[sr]):
                ax.imshow(rows[sr][sc])

            ax.set_axis_off()

            _style_severity_axis(
                ax,
                sc,
                ratios,
                show_title=(sr == 0),
                title_fontsize=9,
            )

            if sr == n_rows - 1 and sc == bottom_sc:
                ax.text(
                    0.5,
                    -0.08,
                    dataset_name,
                    transform=ax.transAxes,
                    ha="center",
                    va="top",
                    fontsize=9,
                )

    return fig


def build_combined_figure(
    domain_rows: dict[str, list[list[np.ndarray]]],
    datasets: list[str],
    ratios: list[float],
    dataset_rows: int,
    dataset_cols: int,
    dpi: int,
    figsize: tuple[float, float] | None,
    inner_gap: int = COMBINED_INNER_GAP_PX,
    dataset_gap: int | None = None,
    outer_pad: int = COMBINED_OUTER_PAD_PX,
) -> plt.Figure:
    """Build the no-text combined demo figure using pixel-level composition."""
    n_cols = 1 + len(ratios)
    max_sample_rows = max(len(domain_rows[d]) for d in datasets)

    sample = next(
        img for name in datasets for row in domain_rows[name] for img in row
    )
    tile_h, tile_w = sample.shape[:2]
    panel_w = n_cols * tile_w + (n_cols - 1) * inner_gap
    panel_h = max_sample_rows * tile_h + (max_sample_rows - 1) * inner_gap
    dataset_gap = 2 * inner_gap if dataset_gap is None else dataset_gap

    canvas_w = dataset_cols * panel_w + (dataset_cols - 1) * dataset_gap + 2 * outer_pad
    canvas_h = dataset_rows * panel_h + (dataset_rows - 1) * dataset_gap + 2 * outer_pad
    canvas = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)

    for panel_idx, name in enumerate(datasets[: dataset_rows * dataset_cols]):
        pr, pc = divmod(panel_idx, dataset_cols)
        panel_y = outer_pad + pr * (panel_h + dataset_gap)
        panel_x = outer_pad + pc * (panel_w + dataset_gap)

        for sr, row in enumerate(domain_rows[name]):
            for sc, img in enumerate(row):
                tile = _as_uint8_rgb(img)
                y = panel_y + sr * (tile_h + inner_gap)
                x = panel_x + sc * (tile_w + inner_gap)
                canvas[y:y + tile_h, x:x + tile_w] = tile

    if figsize is None:
        figsize = (canvas_w / dpi, canvas_h / dpi)

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(canvas)
    ax.set_axis_off()
    return fig


def _as_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a [0,1] float RGB image to uint8 without changing its content."""
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def main() -> None:
    set_paper_style()

    args = parse_args()
    datasets = resolve_datasets(args)
    ratios = [float(x.strip()) for x in args.ratios.split(",") if x.strip()]

    if args.combined and len(datasets) > args.dataset_rows * args.dataset_cols:
        raise SystemExit(
            f"{len(datasets)} datasets exceed "
            f"--dataset-rows * --dataset-cols "
            f"({args.dataset_rows * args.dataset_cols})"
        )

    figsize = None

    if args.figsize:
        parts = [float(x.strip()) for x in args.figsize.split(",")]

        if len(parts) != 2:
            raise SystemExit("--figsize must be W,H")

        figsize = (parts[0], parts[1])

    rng = np.random.default_rng(args.seed)
    domain_rows: dict[str, list[list[np.ndarray]]] = {}
    errors: list[str] = []
    out_dir = Path(args.out_dir)

    for name in datasets:
        try:
            cfg = load_dataset_cfg(name)
        except FileNotFoundError as e:
            errors.append(str(e))
            continue

        root = Path(str(cfg.root))
        ds = build_base_dataset(cfg, args.split)

        if len(ds) == 0:
            errors.append(f"{name}: empty split={args.split!r} (root={root})")
            continue

        if not root.exists():
            errors.append(f"{name}: DATA_PATH root missing: {root}")
            continue

        indices = _sample_indices(len(ds), args.n_samples, rng)

        domain_rows[name] = collect_domain_panels(
            name,
            args.split,
            indices,
            ratios,
            args.mask_yaml,
        )

        display = DATASET_DISPLAY.get(name, name)

        fig = build_domain_figure(
            domain_rows[name],
            ratios,
            display,
            args.dpi,
            figsize,
        )

        out_path = out_dir / f"{tagged_stem(name, args.seed, args.n_samples)}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)
        print(f"Saved {out_path}")

        if args.pdf:
            pdf_path = out_path.with_suffix(".pdf")
            fig = build_domain_figure(
                domain_rows[name],
                ratios,
                display,
                args.dpi,
                figsize,
            )
            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
            plt.close(fig)
            print(f"Saved {pdf_path}")

    if errors:
        for msg in errors:
            print(f"ERROR: {msg}", file=sys.stderr)

    if not domain_rows:
        raise SystemExit("No datasets loaded. Set DATA_PATH and check configs.")

    if len(domain_rows) < len(datasets):
        print(
            f"WARNING: loaded {len(domain_rows)}/{len(datasets)} domains",
            file=sys.stderr,
        )

    if args.combined:
        loaded = [d for d in datasets if d in domain_rows]

        fig = build_combined_figure(
            domain_rows,
            loaded,
            ratios,
            args.dataset_rows,
            args.dataset_cols,
            args.dpi,
            figsize,
            inner_gap=args.combined_inner_gap,
            dataset_gap=args.combined_dataset_gap,
            outer_pad=args.combined_outer_pad,
        )

        out_path = tagged_path(Path(args.out), args.seed, args.n_samples)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
        plt.close(fig)
        print(f"Saved {out_path}")

        if args.pdf:
            pdf_path = out_path.with_suffix(".pdf")

            fig = build_combined_figure(
                domain_rows,
                loaded,
                ratios,
                args.dataset_rows,
                args.dataset_cols,
                args.dpi,
                figsize,
                inner_gap=args.combined_inner_gap,
                dataset_gap=args.combined_dataset_gap,
                outer_pad=args.combined_outer_pad,
            )

            fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.01)
            plt.close(fig)
            print(f"Saved {pdf_path}")


if __name__ == "__main__":
    main()
