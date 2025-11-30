import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image (NextDiT) models.

    Matches the EmbedND output format used by Lumina/Z-Image models:
    Rotation matrices stacked per axis -> (B, 1, L, D/2, 2, 2)
    """

    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', yarn_alt_scaling: bool = False, dype: bool = True, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_resolution: int = 1024, dype_start_sigma: float = 1.0, axes_lens: list[int] | None = None):
        super().__init__(theta, axes_dim, method, yarn_alt_scaling, dype, dype_scale, dype_exponent, base_resolution, dype_start_sigma)

        if axes_lens:
            spatial_axes = axes_lens[1:] if len(axes_lens) > 1 else axes_lens
            if spatial_axes:
                self.base_patches = max(spatial_axes)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(pos, freqs_dtype)

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
