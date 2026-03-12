from typing import Any, Optional

from mask.block import BlockMaskGenerator


def build_mask_generator(
    mask_cfg: Any,
    split: str,
    train_seed: Optional[int] = None,
    eval_seed: Optional[int] = None,
):
    return BlockMaskGenerator(mask_cfg, split=split, train_seed=train_seed, eval_seed=eval_seed)