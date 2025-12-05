from typing import Optional, Tuple

import torch

from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def __init__(self, *args, base_hw_tokens: Optional[Tuple[int, int]] = None, target_hw_tokens: Optional[Tuple[int, int]] = None, **kwargs):
        super().__init__(*args, base_hw_tokens=base_hw_tokens, **kwargs)
        # Fallback to square base if explicit anisotropic grid not provided.
        if self.base_hw_tokens is None:
            self.base_hw_tokens = (self.base_patches, self.base_patches)
        self.target_hw_tokens = target_hw_tokens

    def _scale_zimage_grid(self, pos: torch.Tensor) -> torch.Tensor:
        # Z-Image grids already encode the base-resolution spacing; DyPE extrapolation
        # must densify relative to that baseline without mutating the incoming layout.
        if pos.shape[-1] < 3 or self.target_hw_tokens is None or self.base_hw_tokens is None:
            return pos

        y_axis, x_axis = pos[..., 1], pos[..., 2]
        y_min, x_min = y_axis.min(), x_axis.min()

        base_h, base_w = self.base_hw_tokens
        target_h, target_w = self.target_hw_tokens

        h_scale = max(float(target_h) / max(float(base_h), 1.0), 1.0)
        w_scale = max(float(target_w) / max(float(base_w), 1.0), 1.0)

        pos_scaled = pos.clone()
        pos_scaled[..., 1] = (y_axis - y_min) * h_scale + y_min
        pos_scaled[..., 2] = (x_axis - x_min) * w_scale + x_min

        return pos_scaled

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = self._scale_zimage_grid(ids.float())
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(pos, freqs_dtype)

        emb_parts = []
        for cos, sin in components:
            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)
