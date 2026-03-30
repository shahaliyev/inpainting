from typing import Any, Callable, Dict

from models.gated_conv import build_gated_conv
from models.partial_conv import build_partial_conv
from models.unet import build_unet


MODEL_BUILDERS: Dict[str, Callable[[Any], object]] = {
    "unet": build_unet,
    "partial_conv": build_partial_conv,
    "partialconv": build_partial_conv,
    "pconv": build_partial_conv,
    "gated_conv": build_gated_conv,
    "gatedconv": build_gated_conv,
}


def build_model(cfg: Any):
    name = str(getattr(cfg, "name", "unet")).lower()
    builder = MODEL_BUILDERS.get(name)
    if builder is None:
        available = ", ".join(sorted(set(MODEL_BUILDERS.keys())))
        raise ValueError(f"Unsupported model.name: {name}. Available: {available}")
    return builder(cfg)
