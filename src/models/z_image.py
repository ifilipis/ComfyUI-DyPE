import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos_full = ids.float()
        grid_spans = [self._axis_range(pos_full[..., i]) for i in range(pos_full.shape[-1])]

        pos = pos_full
        if self.base_hw is not None and pos.shape[-1] >= 3:
            pos = pos_full.clone()
            h_tokens = grid_spans[1] if len(grid_spans) > 1 else 0.0
            w_tokens = grid_spans[2] if len(grid_spans) > 2 else 0.0

            if self.base_hw[0] > 0:
                h_scale = h_tokens / float(self.base_hw[0])
                pos[..., 1] = pos[..., 1] * h_scale

            if self.base_hw[1] > 0:
                w_scale = w_tokens / float(self.base_hw[1])
                pos[..., 2] = pos[..., 2] * w_scale
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32
        components = self.get_components(pos, freqs_dtype, grid_spans, base_pos=pos_full)

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
