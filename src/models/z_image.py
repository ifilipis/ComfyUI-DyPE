import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.float32

        components = self.get_components(pos, freqs_dtype)

        emb_parts = []
        for axis_idx, (cos, sin) in enumerate(components):
            axis_dim = self.axes_dim[axis_idx]

            cos = cos[..., :axis_dim]
            sin = sin[..., :axis_dim]

            cos_base = cos.view(*cos.shape[:-1], -1, 2)[..., 0]
            sin_base = sin.view(*sin.shape[:-1], -1, 2)[..., 0]

            axis_freq = torch.stack(
                [
                    torch.stack([cos_base, -sin_base], dim=-1),
                    torch.stack([sin_base, cos_base], dim=-1),
                ],
                dim=-2,
            )

            emb_parts.append(axis_freq)

        emb = torch.cat(emb_parts, dim=-3)
        return emb.unsqueeze(1).to(ids.device)
