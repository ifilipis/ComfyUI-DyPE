import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    @staticmethod
    def _infer_scale_and_offset(axis_values: torch.Tensor) -> tuple[float, torch.Tensor]:
        unique_vals = torch.unique(axis_values)
        offset = unique_vals.min()

        if unique_vals.numel() > 1:
            diffs = unique_vals[1:] - unique_vals[:-1]
            positive_diffs = diffs[diffs > 0]
            step = positive_diffs.min().item() if positive_diffs.numel() > 0 else 1.0
        else:
            step = 1.0

        return step, offset

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        grid_scale = torch.ones(pos.shape[-1], device=pos.device, dtype=pos.dtype)
        grid_offset = torch.zeros_like(grid_scale)

        for axis_idx in range(pos.shape[-1]):
            step, offset = self._infer_scale_and_offset(pos[..., axis_idx])
            grid_scale[axis_idx] = step
            grid_offset[axis_idx] = offset

        base_hw = None
        axes_lens = getattr(self, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 3:
            base_hw = (float(axes_lens[1]), float(axes_lens[2]))
        elif isinstance(self.base_patches, (list, tuple)):
            if len(self.base_patches) >= 2:
                base_hw = (float(self.base_patches[0]), float(self.base_patches[1]))
            elif len(self.base_patches) == 1:
                base_hw = (float(self.base_patches[0]), float(self.base_patches[0]))
        elif self.base_patches is not None:
            base_hw = (float(self.base_patches), float(self.base_patches))

        self._grid_scale = grid_scale
        self._grid_offset = grid_offset
        if base_hw is not None:
            self._grid_base_hw = base_hw

        normalized_pos = self._apply_grid_transform(pos, grid_scale, grid_offset)

        components = self.get_components(normalized_pos, freqs_dtype)

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
