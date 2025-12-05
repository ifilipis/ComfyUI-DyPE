from typing import Optional, Tuple

import torch

from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def __init__(self, *args, base_hw_tokens: Optional[Tuple[int, int]] = None, **kwargs):
        super().__init__(*args, base_hw_tokens=base_hw_tokens, **kwargs)
        # Fallback to square base if explicit anisotropic grid not provided.
        if self.base_hw_tokens is None:
            self.base_hw_tokens = (self.base_patches, self.base_patches)

    def _scale_zimage_grid(self, pos: torch.Tensor) -> torch.Tensor:
        if pos.shape[-1] < 3:
            return pos

        y_axis, x_axis = pos[..., 1], pos[..., 2]
        y_min, y_max = y_axis.min(), y_axis.max()
        x_min, x_max = x_axis.min(), x_axis.max()

        base_h = float(self.base_hw_tokens[0]) if self.base_hw_tokens else float(self.base_patches)
        base_w = float(self.base_hw_tokens[1]) if self.base_hw_tokens else float(self.base_patches)

        one_tensor = torch.tensor(1.0, device=pos.device, dtype=pos.dtype)
        y_span = torch.maximum(y_max - y_min + 1.0, one_tensor)
        x_span = torch.maximum(x_max - x_min + 1.0, one_tensor)

        h_scale = y_span / max(base_h, 1.0)
        w_scale = x_span / max(base_w, 1.0)

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
