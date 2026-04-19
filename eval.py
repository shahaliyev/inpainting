import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf, ListConfig

from data.build import build_dataloader
from models.build import build_model
from training.checkpoint import load_checkpoint
from training.engine import evaluate
from training.logger import MetricsLogger


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_yaml", default="configs/eval/default.yaml")
    ap.add_argument("--dataset_yaml", default="configs/dataset/carpet.yaml")
    ap.add_argument("--loader_yaml", default="configs/loader/default.yaml")
    ap.add_argument("--mask_yaml", default="configs/mask/block.yaml")
    ap.add_argument("--model_yaml", default="configs/model/unet.yaml")
    ap.add_argument("--train_yaml", default="configs/train/default.yaml")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--runs_dir", default="runs")
    ap.add_argument("--exp", default="eval")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_vis", action="store_true")
    ap.add_argument("--no_lpips", action="store_true",
                    help="Skip LPIPS computation (saves time on CPU)")
    return ap.parse_args()


def load_configs(args):
    dataset_cfg = OmegaConf.load(args.dataset_yaml)
    loader_cfg = OmegaConf.load(args.loader_yaml)
    mask_cfg = OmegaConf.load(args.mask_yaml)
    model_cfg = OmegaConf.load(args.model_yaml)
    train_cfg = OmegaConf.load(args.train_yaml)

    if getattr(dataset_cfg, "root", None) is None:
        raise ValueError("dataset_cfg.root is missing")
    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)

    return dataset_cfg, loader_cfg, mask_cfg, model_cfg, train_cfg


def apply_overrides(args, dataset_cfg, loader_cfg):
    if args.limit is not None:
        dataset_cfg.limit = int(args.limit)
        dataset_cfg.limit_shuffle = False
        dataset_cfg.limit_seed = int(args.seed)

    if args.batch_size is not None:
        loader_cfg.batch_size = int(args.batch_size)


def _ensure_list(x, default=None):
    """Normalize to list: str -> [str], list -> list, None -> default or []."""
    if x is None:
        return default if default is not None else []
    if isinstance(x, (list, tuple, ListConfig)):
        return list(x)
    return [x]


def _expand_grid_product(grid_cfg, defaults_cfg, args):
    """Expand to condition list.
    - dataset × mask_ratio × mask_yaml: one condition per (dataset, single ratio, mask type) for separate eval.
    - If add_mixed: true, append one condition per (dataset, mask_yaml) with all ratios (mixed eval).
    """
    datasets = _ensure_list(getattr(grid_cfg, "dataset_yaml", None), [args.dataset_yaml])
    ratios_list = _ensure_list(getattr(grid_cfg, "mask_ratios", None))
    default_mask = getattr(defaults_cfg, "mask_yaml", args.mask_yaml)
    mask_yamls = _ensure_list(getattr(grid_cfg, "mask_yaml", None), [default_mask])
    add_mixed = bool(getattr(grid_cfg, "add_mixed", False))

    conditions = []

    # Separate: one condition per (dataset, single ratio, mask type)
    ratios_for_product = ratios_list if ratios_list else [None]
    for dataset_yaml, ratio, mask_yaml in itertools.product(datasets, ratios_for_product, mask_yamls):
        name_parts = [Path(dataset_yaml).stem]
        if len(mask_yamls) > 1:
            name_parts.append(Path(mask_yaml).stem)
        if ratio is not None:
            name_parts.append(f"ratio_{ratio}")
        name = "_".join(name_parts) if name_parts else "default"
        cond = {"name": name, "dataset_yaml": dataset_yaml, "mask_yaml": mask_yaml,
                "mask_overrides": None}
        if ratio is not None:
            cond["mask_ratios"] = [int(ratio)]
        conditions.append(cond)

    # Mixed: one condition per (dataset, mask type) with all ratios (generator picks randomly)
    if add_mixed and ratios_list:
        for dataset_yaml, mask_yaml in itertools.product(datasets, mask_yamls):
            name_parts = [Path(dataset_yaml).stem]
            if len(mask_yamls) > 1:
                name_parts.append(Path(mask_yaml).stem)
            name_parts.append("mixed")
            name = "_".join(name_parts)
            conditions.append({
                "name": name,
                "dataset_yaml": dataset_yaml,
                "mask_yaml": mask_yaml,
                "mask_ratios": [int(r) for r in ratios_list],
                "mask_overrides": None,
            })

    if not conditions:
        conditions = [{"name": "default", "dataset_yaml": args.dataset_yaml, "mask_yaml": args.mask_yaml, "mask_ratios": None}]
    return conditions


def get_eval_grid(eval_cfg, args):
    """Return list of condition dicts: name, dataset_yaml, mask_yaml, mask_ratios, mask_overrides.

    Supports three formats (combinable — conditions and grid may coexist):
      conditions: explicit list, each entry may carry mask_overrides for arbitrary
                  mask-config field overrides (e.g. num_strokes for freeform masks).
      grid:       cartesian product of datasets × mask types × ratios.
      legacy:     top-level mask_ratios key (single dataset/mask from CLI args).
    """
    conditions = []

    for i, c in enumerate(list(getattr(eval_cfg, "conditions", None) or [])):
        raw_overrides = getattr(c, "mask_overrides", None)
        overrides = dict(OmegaConf.to_container(raw_overrides)) if raw_overrides else None
        conditions.append({
            "name": getattr(c, "name", f"cond_{i}"),
            "dataset_yaml": getattr(c, "dataset_yaml", args.dataset_yaml),
            "mask_yaml": getattr(c, "mask_yaml", args.mask_yaml),
            "mask_ratios": _ensure_list(getattr(c, "mask_ratios", None)) or None,
            "mask_overrides": overrides,
        })

    grid_cfg = getattr(eval_cfg, "grid", None)
    if grid_cfg is not None:
        defaults_cfg = getattr(eval_cfg, "defaults", OmegaConf.create({}))
        conditions.extend(_expand_grid_product(grid_cfg, defaults_cfg, args))

    if conditions:
        return conditions

    # Legacy: mask_ratios only (single dataset/mask from CLI)
    ratios = _ensure_list(getattr(eval_cfg, "mask_ratios", None))
    if not ratios:
        return [{"name": "default", "dataset_yaml": args.dataset_yaml, "mask_yaml": args.mask_yaml,
                 "mask_ratios": None, "mask_overrides": None}]
    return [
        {"name": f"ratio_{r}", "dataset_yaml": args.dataset_yaml, "mask_yaml": args.mask_yaml,
         "mask_ratios": [int(r)], "mask_overrides": None}
        for r in ratios
    ]


def load_dataset_and_mask(dataset_yaml, mask_yaml):
    """Load dataset and mask configs; resolve env refs."""
    dataset_cfg = OmegaConf.load(dataset_yaml)
    mask_cfg = OmegaConf.load(mask_yaml)
    if getattr(dataset_cfg, "root", None) is None:
        raise ValueError(f"dataset_cfg.root is missing in {dataset_yaml}")
    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)
    return dataset_cfg, mask_cfg


def merge_mask_cfg(base_mask_cfg, mask_ratios=None, mask_overrides=None):
    """New config = base + overrides. Does not mutate base."""
    if not mask_ratios and not mask_overrides:
        return base_mask_cfg
    cfg = OmegaConf.create(OmegaConf.to_container(base_mask_cfg, resolve=True))
    if mask_ratios:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"ratios": [int(r) for r in mask_ratios]}))
    if mask_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(mask_overrides)))
    return cfg


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    eval_cfg = OmegaConf.load(args.eval_yaml)
    _, loader_cfg, _, model_cfg, train_cfg = load_configs(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(getattr(train_cfg, "mixed_precision", False)) and device.type == "cuda"

    model = build_model(model_cfg).to(device)
    loss_fn = nn.L1Loss(reduction="none")
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    state = load_checkpoint(
        path=args.ckpt,
        model=model,
        optimizer=None,
        scheduler=None,
        scaler=scaler if (use_amp and device.type == "cuda") else None,
        device=device,
    )

    run_dir = Path(args.runs_dir) / args.exp
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(run_dir) if args.save_vis else None
    if args.no_lpips:
        lpips_net = None
    else:
        if device.type == "cpu":
            print("WARNING: LPIPS is enabled on CPU — this will be slow. Pass --no_lpips to skip it.")
        import lpips
        lpips_net = lpips.LPIPS(net="alex").to(device).eval()

    grid = get_eval_grid(eval_cfg, args)
    results = []
    print(f"checkpoint={args.ckpt}  epoch={state['epoch']} step={state['step']}  split={args.split}")
    print(f"grid: {[c['name'] for c in grid]}")

    for idx, cond in enumerate(grid):
        cond_name = cond["name"]
        dataset_cfg, mask_cfg = load_dataset_and_mask(cond["dataset_yaml"], cond["mask_yaml"])
        apply_overrides(args, dataset_cfg, loader_cfg)
        mask_cfg_cond = merge_mask_cfg(mask_cfg, cond.get("mask_ratios"), cond.get("mask_overrides"))

        dl = build_dataloader(
            dataset_cfg,
            loader_cfg,
            split=args.split,
            mask_cfg=mask_cfg_cond,
            global_seed=int(args.seed),
            eval_seed=int(args.seed) + 1,
        )
        mean_list = list(getattr(dataset_cfg.norm, "mean", [0.5, 0.5, 0.5]))
        std_list = list(getattr(dataset_cfg.norm, "std", [0.5, 0.5, 0.5]))
        mean = torch.tensor(mean_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std_list, device=device, dtype=torch.float32).view(1, 3, 1, 1)

        out = evaluate(
            model=model,
            dl=dl,
            device=device,
            loss_fn=loss_fn,
            use_amp=use_amp,
            epoch=state["epoch"],
            global_step=state["step"],
            logger=logger if idx == 0 else None,
            mean=mean,
            std=std,
            save_vis=args.save_vis and idx == 0,
            lpips_net=lpips_net,
        )
        metrics = {k: float(v) for k, v in out.items()}
        results.append({"condition": cond_name, "dataset_yaml": cond["dataset_yaml"], "mask_yaml": cond["mask_yaml"], "mask_ratios": cond.get("mask_ratios"), "mask_overrides": cond.get("mask_overrides"), "metrics": metrics})
        print(f"  {cond_name}: loss={metrics['val_loss']:.6f} psnr={metrics.get('psnr_full', 0):.4f} ssim={metrics.get('ssim_full', 0):.4f} lpips={metrics.get('lpips_full', 0):.4f}")

    summary = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "epoch": int(state["epoch"]),
        "step": int(state["step"]),
        "split": args.split,
        "conditions": results,
    }
    out_path = run_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved {out_path}")

    if logger is not None:
        logger.log(epoch=state["epoch"], step=state["step"], split=args.split, loss=results[0]["metrics"]["val_loss"], lr=0.0)


if __name__ == "__main__":
    main()
