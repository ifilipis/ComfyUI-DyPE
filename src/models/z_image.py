import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        spatial_step = self._infer_spatial_step(pos)

        components = self.get_components(pos, freqs_dtype, spatial_step=spatial_step)

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

    def _infer_spatial_step(self, pos: torch.Tensor):
        if pos.shape[-1] < 3:
            return None

        steps = []
        for axis in (1, 2):
            axis_values = pos[..., axis].reshape(-1)
            unique_vals = torch.unique(axis_values)

            if unique_vals.numel() <= 1:
                steps.append(1.0)
                continue

            sorted_vals, _ = torch.sort(unique_vals)
            diffs = torch.diff(sorted_vals)
            positive_diffs = diffs[diffs > 0]

            if positive_diffs.numel() == 0:
                steps.append(1.0)
            else:
                steps.append(max(1.0, float(positive_diffs.min().item())))

        return steps
