import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches the original RopeEmbedder: (L, D/2) complex tensor.
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(pos, freqs_dtype)

        freqs_parts = []
        for cos, sin in components:
            # Raw components are repeated for real-valued RoPE; decimate to per-frequency pairs
            cos_half = cos[..., ::2].float()
            sin_half = sin[..., ::2].float()

            freqs_parts.append(torch.complex(cos_half, sin_half))

        freqs_cis = torch.cat(freqs_parts, dim=-1)
        return freqs_cis.to(device=ids.device)
