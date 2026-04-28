import itertools
from pathlib import Path

from omegaconf import ListConfig, OmegaConf


def _ensure_list(x, default=None):
    if x is None:
        return default if default is not None else []
    if isinstance(x, (list, tuple, ListConfig)):
        return list(x)
    return [x]


def _expand_grid_product(grid_cfg, defaults_cfg, default_dataset_yaml, default_mask_yaml):
    datasets = _ensure_list(getattr(grid_cfg, "dataset_yaml", None), [default_dataset_yaml])
    ratios_list = _ensure_list(getattr(grid_cfg, "mask_ratios", None))
    default_mask = getattr(defaults_cfg, "mask_yaml", default_mask_yaml)
    mask_yamls = _ensure_list(getattr(grid_cfg, "mask_yaml", None), [default_mask])
    add_mixed = bool(getattr(grid_cfg, "add_mixed", False))
    conditions = []

    ratios_for_product = ratios_list if ratios_list else [None]
    for dataset_yaml, ratio, mask_yaml in itertools.product(datasets, ratios_for_product, mask_yamls):
        name_parts = [Path(dataset_yaml).stem]
        if len(mask_yamls) > 1:
            name_parts.append(Path(mask_yaml).stem)
        if ratio is not None:
            name_parts.append(f"ratio_{ratio}")
        cond = {
            "name": "_".join(name_parts) if name_parts else "default",
            "dataset_yaml": dataset_yaml,
            "mask_yaml": mask_yaml,
            "mask_overrides": None,
        }
        if ratio is not None:
            cond["mask_ratios"] = [int(ratio)]
        conditions.append(cond)

    if add_mixed and ratios_list:
        for dataset_yaml, mask_yaml in itertools.product(datasets, mask_yamls):
            name_parts = [Path(dataset_yaml).stem]
            if len(mask_yamls) > 1:
                name_parts.append(Path(mask_yaml).stem)
            name_parts.append("mixed")
            conditions.append({
                "name": "_".join(name_parts),
                "dataset_yaml": dataset_yaml,
                "mask_yaml": mask_yaml,
                "mask_ratios": [int(r) for r in ratios_list],
                "mask_overrides": None,
            })
    return conditions


def get_eval_grid(eval_cfg, default_dataset_yaml, default_mask_yaml):
    conditions = []

    for i, c in enumerate(list(getattr(eval_cfg, "conditions", None) or [])):
        raw_overrides = getattr(c, "mask_overrides", None)
        overrides = dict(OmegaConf.to_container(raw_overrides)) if raw_overrides else None
        conditions.append({
            "name": getattr(c, "name", f"cond_{i}"),
            "dataset_yaml": getattr(c, "dataset_yaml", default_dataset_yaml),
            "mask_yaml": getattr(c, "mask_yaml", default_mask_yaml),
            "mask_ratios": _ensure_list(getattr(c, "mask_ratios", None)) or None,
            "mask_overrides": overrides,
        })

    grid_cfg = getattr(eval_cfg, "grid", None)
    if grid_cfg is not None:
        defaults_cfg = getattr(eval_cfg, "defaults", OmegaConf.create({}))
        conditions.extend(_expand_grid_product(grid_cfg, defaults_cfg, default_dataset_yaml, default_mask_yaml))

    if conditions:
        return conditions

    ratios = _ensure_list(getattr(eval_cfg, "mask_ratios", None))
    if not ratios:
        return [{
            "name": "default",
            "dataset_yaml": default_dataset_yaml,
            "mask_yaml": default_mask_yaml,
            "mask_ratios": None,
            "mask_overrides": None,
        }]
    return [{
        "name": f"ratio_{r}",
        "dataset_yaml": default_dataset_yaml,
        "mask_yaml": default_mask_yaml,
        "mask_ratios": [int(r)],
        "mask_overrides": None,
    } for r in ratios]
