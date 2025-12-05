import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def _compute_grid_metadata(self, ids: torch.Tensor):
        pos = ids.float()
        device = pos.device
        dtype = pos.dtype

        n_axes = pos.shape[-1]
        scales = torch.ones(n_axes, device=device, dtype=dtype)
        offsets = torch.zeros(n_axes, device=device, dtype=dtype)

        normalized = pos.clone()
        for axis in range(n_axes):
            axis_vals = pos[..., axis]
            offset_val = axis_vals.min()
            offsets[axis] = offset_val

            unique_vals = torch.unique(axis_vals)
            if unique_vals.numel() > 1:
                sorted_vals = torch.sort(unique_vals)[0]
                diffs = torch.diff(sorted_vals)
                positive_diffs = diffs[diffs > 0]
                if positive_diffs.numel() > 0:
                    step = positive_diffs.median()
                else:
                    step = torch.tensor(1.0, device=device, dtype=dtype)
            else:
                step = torch.tensor(1.0, device=device, dtype=dtype)

            step_val = float(step.item()) if torch.isfinite(step).item() else 1.0
            if step_val <= 0:
                step_val = 1.0

            scales[axis] = step_val
            normalized[..., axis] = (axis_vals - offset_val) / max(step_val, 1e-6)

        base_hw = (self.base_patches, self.base_patches) if self.base_patches is not None else None

        grid_meta = {
            "normalized_pos": normalized,
            "scales": scales,
            "offsets": offsets,
            "base_hw": base_hw,
        }

        return normalized, grid_meta

    def _normalize_for_scaling(self, pos: torch.Tensor):
        if self._grid_meta is not None and "normalized_pos" in self._grid_meta:
            return self._grid_meta["normalized_pos"], self._grid_meta

        normalized_pos, grid_meta = self._compute_grid_metadata(pos)
        self._grid_meta = grid_meta
        return normalized_pos, grid_meta

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos, grid_meta = self._compute_grid_metadata(ids)
        self._grid_meta = grid_meta
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(pos, freqs_dtype)

        self._grid_meta = None

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
