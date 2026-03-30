from typing import Any, Optional, Tuple

import torch


class MultiBlockMaskGenerator:
    """
    Generate a mask as the union of multiple rectangular blocks.
    Returns mask of shape [1, H, W], where 1.0 means hole (missing).
    """

    def __init__(
        self,
        cfg: Any,
        split: str,
        train_seed: Optional[int] = None,
        eval_seed: Optional[int] = None,
    ):
        self.cfg = cfg
        self.split = split

        split_cfg = getattr(cfg, "train" if split == "train" else "eval", None)
        self.deterministic = bool(getattr(split_cfg, "deterministic", False)) if split_cfg is not None else False

        # Per-block area ratios in percent of full image area
        ratios = getattr(cfg, "ratios", [3, 8, 15])
        self.ratios = [float(r) for r in ratios]

        # Number of blocks per image
        self.min_blocks = int(getattr(cfg, "min_blocks", 2))
        self.max_blocks = int(getattr(cfg, "max_blocks", 5))
        if self.max_blocks < self.min_blocks:
            self.max_blocks = self.min_blocks

        self._gen = None
        if self.deterministic:
            seed = train_seed if split == "train" else eval_seed
            if seed is None:
                seed = 0
            self._gen = torch.Generator()
            self._gen.manual_seed(int(seed) & 0x7FFFFFFF)

    def _randint(self, low: int, high: int) -> int:
        # [low, high)
        if self._gen is None:
            return int(torch.randint(low, high, (1,)).item())
        return int(torch.randint(low, high, (1,), generator=self._gen).item())

    def _randfloat(self, low: float, high: float) -> float:
        if self._gen is None:
            return float(torch.empty(1).uniform_(low, high).item())
        return float(torch.empty(1).uniform_(low, high, generator=self._gen).item())

    def _block_hw(self, H: int, W: int, ratio: float) -> Tuple[int, int]:
        # Area target
        target_area = max(1.0, (ratio / 100.0) * float(H * W))

        # Aspect ratio jitter for rectangles
        aspect = self._randfloat(0.5, 2.0)
        h = int(round((target_area / aspect) ** 0.5))
        w = int(round(target_area / max(1, h)))

        h = max(1, min(h, H))
        w = max(1, min(w, W))
        return h, w

    def __call__(self, image_shape) -> torch.Tensor:
        if len(image_shape) != 3:
            raise ValueError(f"Expected image shape [C,H,W], got {image_shape}")

        _, H, W = image_shape
        mask = torch.zeros((1, H, W), dtype=torch.float32)

        n_blocks = self._randint(self.min_blocks, self.max_blocks + 1)

        for _ in range(n_blocks):
            ratio = self.ratios[self._randint(0, len(self.ratios))]
            bh, bw = self._block_hw(H, W, float(ratio))

            max_y = max(H - bh, 0)
            max_x = max(W - bw, 0)
            y = self._randint(0, max_y + 1)
            x = self._randint(0, max_x + 1)

            mask[:, y : y + bh, x : x + bw] = 1.0

        return mask