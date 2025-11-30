import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `RopeEmbedder`: (L, sum(axes_dim) // 2) complex
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(pos, freqs_dtype)

        freqs_per_axis = []
        for cos, sin in components:
            # Reduce the interleaved real layout back to one rotation per axis.
            cos_half = cos.view(*cos.shape[:-1], -1, 2)[..., 0]
            sin_half = sin.view(*sin.shape[:-1], -1, 2)[..., 0]
            freqs_axis = torch.complex(cos_half, sin_half)
            freqs_per_axis.append(freqs_axis)

        freqs = torch.cat(freqs_per_axis, dim=-1)
        return freqs.to(ids.device)
