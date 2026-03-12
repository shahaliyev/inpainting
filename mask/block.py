from typing import Any, Tuple, Optional

import torch


class BlockMaskGenerator:
    def __init__(
        self,
        cfg: Any,
        split: str,
        train_seed: Optional[int] = None,
        eval_seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.split = split
        self.ratios = [float(x) for x in cfg.ratios]

        split_cfg = getattr(cfg, "train" if split == "train" else "eval", None)
        self.deterministic = bool(getattr(split_cfg, "deterministic", False)) if split_cfg is not None else False

        self._gen = None
        if self.deterministic:
            seed = train_seed if split == "train" else eval_seed
            if seed is None:
                seed = 0
            self._gen = torch.Generator()
            self._gen.manual_seed(int(seed) & 0x7FFFFFFF)

    def _block_side(self, H: int, W: int, ratio: float) -> int:
        area = max(1.0, (ratio / 100.0) * float(H * W))
        s = int(round(area ** 0.5))
        return max(1, min(s, H, W))

    def _randint(self, low: int, high: int) -> int:
        if self._gen is None:
            return int(torch.randint(low, high, (1,)).item())
        return int(torch.randint(low, high, (1,), generator=self._gen).item())

    def _top_left(self, H: int, W: int, s: int) -> Tuple[int, int]:
        max_y = max(H - s, 0)
        max_x = max(W - s, 0)
        y = self._randint(0, max_y + 1)
        x = self._randint(0, max_x + 1)
        return y, x

    def __call__(self, image_shape) -> torch.Tensor:
        if len(image_shape) != 3:
            raise ValueError(f"Expected image shape [C,H,W], got {image_shape}")

        _, H, W = image_shape

        j = self._randint(0, len(self.ratios))
        ratio = float(self.ratios[j])

        s = self._block_side(H, W, ratio)
        y, x = self._top_left(H, W, s)

        mask = torch.zeros((1, H, W), dtype=torch.float32)
        mask[:, y : y + s, x : x + s] = 1.0
        return mask