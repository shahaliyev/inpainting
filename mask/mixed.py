from typing import Any, Optional

import torch

from mask.block import BlockMaskGenerator
from mask.freeform import FreeformMaskGenerator
from mask.multi_block import MultiBlockMaskGenerator


_BUILDERS = {
    "block": BlockMaskGenerator,
    "freeform": FreeformMaskGenerator,
    "multi_block": MultiBlockMaskGenerator,
    "multiblock": MultiBlockMaskGenerator,
}


class MixedMaskGenerator:
    """
    Uniformly samples one sub-generator per image from a list of mask types.
    Each sub-generator uses its own config (ratios, strokes, etc.), so both
    mask type and severity vary across the batch.

    cfg must have a `generators` list; each entry is a full sub-mask config
    (same fields as the standalone block/freeform/multi_block YAMLs).
    """

    def __init__(
        self,
        cfg: Any,
        split: str,
        train_seed: Optional[int] = None,
        eval_seed: Optional[int] = None,
    ):
        sub_cfgs = list(getattr(cfg, "generators", []))
        if not sub_cfgs:
            raise ValueError("MixedMaskGenerator requires cfg.generators to be a non-empty list")

        self.generators = []
        for sub_cfg in sub_cfgs:
            name = str(getattr(sub_cfg, "name", "block")).lower()
            builder = _BUILDERS.get(name)
            if builder is None:
                available = ", ".join(sorted(set(_BUILDERS.keys())))
                raise ValueError(f"Unknown mask name in generators: {name!r}. Available: {available}")
            self.generators.append(builder(sub_cfg, split=split, train_seed=train_seed, eval_seed=eval_seed))

        split_cfg = getattr(cfg, "train" if split == "train" else "eval", None)
        self.deterministic = bool(getattr(split_cfg, "deterministic", False)) if split_cfg is not None else False

        self._gen = None
        if self.deterministic:
            seed = train_seed if split == "train" else eval_seed
            if seed is None:
                seed = 0
            self._gen = torch.Generator()
            self._gen.manual_seed(int(seed) & 0x7FFFFFFF)

    def _randint(self, low: int, high: int) -> int:
        if self._gen is None:
            return int(torch.randint(low, high, (1,)).item())
        return int(torch.randint(low, high, (1,), generator=self._gen).item())

    def __call__(self, image_shape) -> torch.Tensor:
        j = self._randint(0, len(self.generators))
        return self.generators[j](image_shape)
