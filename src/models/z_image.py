import torch
from ..base import DyPEBasePosEmbed


class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Lumina/Z-Image Models.

    Mirrors the rope embedder used by Lumina/NextDiT: output shape (B, 1, L, D/2, 2, 2).
    """

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(pos, freqs_dtype)

        emb_parts = []
        for cos, sin in components:
            cos_half = cos[..., ::2]
            sin_half = sin[..., ::2]

            col0 = torch.stack([cos_half, sin_half], dim=-1)
            col1 = torch.stack([-sin_half, cos_half], dim=-1)

            matrix = torch.stack([col0, col1], dim=-1)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)

        out = emb.unsqueeze(1).to(ids.device)
        return out
