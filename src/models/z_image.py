import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos_full = ids.float()
        pos = pos_full.clone()

        grid_spans = []
        for axis in range(pos.shape[-1]):
            axis_vals = pos[..., axis]

            if axis == 0:
                grid_spans.append(self._axis_range(axis_vals))
                continue

            unique_vals = torch.unique(axis_vals)
            if unique_vals.numel() > 1:
                diffs = unique_vals[1:] - unique_vals[:-1]
                positive_diffs = diffs[diffs > 0]
                stride = positive_diffs.min() if positive_diffs.numel() > 0 else torch.tensor(1.0, device=axis_vals.device, dtype=axis_vals.dtype)
            else:
                stride = torch.tensor(1.0, device=axis_vals.device, dtype=axis_vals.dtype)

            axis_min = unique_vals.min()
            pos[..., axis] = (axis_vals - axis_min) / stride
            grid_spans.append(self._axis_range(pos[..., axis]))

        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32
        components = self.get_components(pos, freqs_dtype, grid_spans)

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
