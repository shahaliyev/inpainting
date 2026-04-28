from omegaconf import OmegaConf

from utils.config_resolver import resolve_config_path


def load_dataset_and_mask(dataset_yaml, mask_yaml):
    dataset_cfg = OmegaConf.load(dataset_yaml)
    mask_cfg = OmegaConf.load(mask_yaml)
    if getattr(dataset_cfg, "root", None) is None:
        raise ValueError(f"dataset_cfg.root is missing in {dataset_yaml}")
    if "${oc.env:" in str(dataset_cfg.root):
        OmegaConf.resolve(dataset_cfg)
    return dataset_cfg, mask_cfg


def merge_mask_cfg(base_mask_cfg, mask_ratios=None, mask_overrides=None):
    if not mask_ratios and not mask_overrides:
        return base_mask_cfg
    cfg = OmegaConf.create(OmegaConf.to_container(base_mask_cfg, resolve=True))
    if mask_ratios:
        cfg = OmegaConf.merge(cfg, OmegaConf.create({"ratios": [int(r) for r in mask_ratios]}))
    if mask_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.create(dict(mask_overrides)))
    return cfg
