import math
import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """

    def __init__(self, *args, base_hw: tuple[int, int] | None = None, **kwargs):
        super().__init__(*args, base_hw=base_hw, **kwargs)
        self.base_hw = base_hw

    def _scaled_spatial_positions(self, pos: torch.Tensor) -> torch.Tensor:
        if self.base_hw is None or pos.shape[-1] < 3:
            return pos

        spatial_mask = (pos[..., 1] != 0) | (pos[..., 2] != 0)
        if not spatial_mask.any():
            return pos

        base_h, base_w = self.base_hw
        pos_scaled = pos.clone()

        def _apply_scale(axis_values: torch.Tensor, base_axis: int) -> torch.Tensor:
            if base_axis <= 0:
                return axis_values

            unique_vals = torch.unique(axis_values)
            token_count = unique_vals.numel()
            if token_count == 0:
                return axis_values

            target_step = token_count / base_axis
            if token_count > 1:
                diffs = unique_vals[1:] - unique_vals[:-1]
                current_step = float(diffs.mean().item())
            else:
                current_step = target_step

            if not math.isfinite(current_step) or current_step == 0.0:
                return axis_values

            if math.isclose(current_step, target_step, rel_tol=1e-3, abs_tol=1e-3):
                return axis_values

            scale_factor = target_step / current_step
            return axis_values * scale_factor

        pos_scaled[..., 1] = _apply_scale(pos[..., 1], base_h)
        pos_scaled[..., 2] = _apply_scale(pos[..., 2], base_w)
        return pos_scaled

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        pos = self._scaled_spatial_positions(pos)
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
