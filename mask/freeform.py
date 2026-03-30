from typing import Any, Optional

import torch


class FreeformMaskGenerator:
    """
    Generate free-form brush-stroke masks.
    Returns mask [1, H, W], where 1.0 means hole (missing).
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

        self.num_strokes = int(getattr(cfg, "num_strokes", 6))
        self.min_vertices = int(getattr(cfg, "min_vertices", 4))
        self.max_vertices = int(getattr(cfg, "max_vertices", 12))
        self.min_brush_width = int(getattr(cfg, "min_brush_width", 8))
        self.max_brush_width = int(getattr(cfg, "max_brush_width", 30))
        self.max_angle = float(getattr(cfg, "max_angle", 3.14159 / 4.0))
        self.min_length = int(getattr(cfg, "min_length", 20))
        self.max_length = int(getattr(cfg, "max_length", 80))

        if self.max_vertices < self.min_vertices:
            self.max_vertices = self.min_vertices
        if self.max_brush_width < self.min_brush_width:
            self.max_brush_width = self.min_brush_width
        if self.max_length < self.min_length:
            self.max_length = self.min_length

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

    @staticmethod
    def _draw_disk(mask: torch.Tensor, cy: int, cx: int, r: int) -> None:
        _, H, W = mask.shape
        y0 = max(0, cy - r)
        y1 = min(H, cy + r + 1)
        x0 = max(0, cx - r)
        x1 = min(W, cx + r + 1)
        if y0 >= y1 or x0 >= x1:
            return

        ys = torch.arange(y0, y1, dtype=torch.float32).view(-1, 1)
        xs = torch.arange(x0, x1, dtype=torch.float32).view(1, -1)
        region = (ys - float(cy)) ** 2 + (xs - float(cx)) ** 2 <= float(r * r)
        mask[0, y0:y1, x0:x1] = torch.maximum(
            mask[0, y0:y1, x0:x1], region.to(mask.dtype)
        )

    def _draw_segment(self, mask: torch.Tensor, y0: int, x0: int, y1: int, x1: int, radius: int) -> None:
        dy = y1 - y0
        dx = x1 - x0
        steps = max(abs(dy), abs(dx), 1)
        for i in range(steps + 1):
            t = float(i) / float(steps)
            yy = int(round(y0 + t * dy))
            xx = int(round(x0 + t * dx))
            self._draw_disk(mask, yy, xx, radius)

    def __call__(self, image_shape) -> torch.Tensor:
        if len(image_shape) != 3:
            raise ValueError(f"Expected image shape [C,H,W], got {image_shape}")

        _, H, W = image_shape
        mask = torch.zeros((1, H, W), dtype=torch.float32)

        for _ in range(self.num_strokes):
            num_vertices = self._randint(self.min_vertices, self.max_vertices + 1)
            brush_w = self._randint(self.min_brush_width, self.max_brush_width + 1)
            radius = max(1, brush_w // 2)

            y = self._randint(0, H)
            x = self._randint(0, W)
            angle = self._randfloat(0.0, 2.0 * 3.141592653589793)

            self._draw_disk(mask, y, x, radius)

            for _ in range(num_vertices):
                delta = self._randfloat(-self.max_angle, self.max_angle)
                angle += delta

                length = self._randint(self.min_length, self.max_length + 1)
                ny = int(round(y + length * torch.sin(torch.tensor(angle)).item()))
                nx = int(round(x + length * torch.cos(torch.tensor(angle)).item()))

                ny = max(0, min(H - 1, ny))
                nx = max(0, min(W - 1, nx))

                self._draw_segment(mask, y, x, ny, nx, radius)

                y, x = ny, nx

        return mask.clamp_(0.0, 1.0)