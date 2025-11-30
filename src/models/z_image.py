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

        components = self.get_components(pos, freqs_dtype)

        emb_parts = []
        for cos, sin in components:
            # `EmbedND` produces D/2 rotary pairs per axis.
            # Our raw DyPE components repeat each frequency twice, so decimate to the
            # expected (D/2) before forming the rotation matrix.
            cos_half = cos[..., ::2]
            sin_half = sin[..., ::2]

            row0 = torch.stack([cos_half, -sin_half], dim=-1)
            row1 = torch.stack([sin_half, cos_half], dim=-1)

            emb_parts.append(torch.stack([row0, row1], dim=-2))

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)
