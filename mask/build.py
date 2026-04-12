from typing import Any, Callable, Dict, Optional

from mask.block import BlockMaskGenerator
from mask.freeform import FreeformMaskGenerator
from mask.mixed import MixedMaskGenerator
from mask.multi_block import MultiBlockMaskGenerator


def build_mask_generator(
    mask_cfg: Any,
    split: str,
    train_seed: Optional[int] = None,
    eval_seed: Optional[int] = None,
):
    name = str(getattr(mask_cfg, "name", "block")).lower()
    builders: Dict[str, Callable[..., object]] = {
        "block": BlockMaskGenerator,
        "freeform": FreeformMaskGenerator,
        "mixed": MixedMaskGenerator,
        "multi_block": MultiBlockMaskGenerator,
        "multiblock": MultiBlockMaskGenerator,
    }
    builder = builders.get(name)
    if builder is None:
        available = ", ".join(sorted(set(builders.keys())))
        raise ValueError(f"Unsupported mask.name: {name}. Available: {available}")
    return builder(mask_cfg, split=split, train_seed=train_seed, eval_seed=eval_seed)