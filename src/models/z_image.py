import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Mirror the native embedder contract for downstream consumers.
        self.dim = sum(self.axes_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(pos, freqs_dtype)

        emb_parts = []
        for cos, sin in components:
            cos_pairs = cos.view(*cos.shape[:-1], -1, 2)[..., 0]
            sin_pairs = sin.view(*sin.shape[:-1], -1, 2)[..., 0]

            matrix = torch.stack(
                (
                    torch.stack((cos_pairs, -sin_pairs), dim=-1),
                    torch.stack((sin_pairs, cos_pairs), dim=-1),
                ),
                dim=-2,
            )
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3).to(ids.device)
        return emb.unsqueeze(1)
