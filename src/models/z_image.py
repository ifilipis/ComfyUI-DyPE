import torch

from ..base import DyPEBasePosEmbed


class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE implementation for Z-Image models.

    The original Z-Image `RopeEmbedder` returns complex-valued frequency tensors.
    This implementation mirrors that behavior by converting the DyPE-adjusted
    cos/sin components into complex numbers, decimating interleaved values to
    match the expected `(D/2)` layout per axis.
    """

    def __init__(self, theta, axes_dim, axes_lens, *args, **kwargs):
        super().__init__(theta, axes_dim, *args, **kwargs)
        self.axes_lens = axes_lens

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()

        # Complex bfloat16 is not broadly supported; use float32 for stability.
        freqs_dtype = torch.float32

        components = self.get_components(pos, freqs_dtype)

        complex_parts = []
        for cos, sin in components:
            cos_half = cos[..., ::2].float()
            sin_half = sin[..., ::2].float()
            complex_axis = torch.complex(cos_half, sin_half)
            complex_parts.append(complex_axis)

        emb = torch.cat(complex_parts, dim=-1)

        return emb.to(pos.device)
