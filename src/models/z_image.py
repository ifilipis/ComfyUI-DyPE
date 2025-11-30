import torch
from ..base import DyPEBasePosEmbed


class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        method: str = 'yarn',
        yarn_alt_scaling: bool = False,
        dype: bool = True,
        dype_scale: float = 2.0,
        dype_exponent: float = 2.0,
        base_resolution: int = 1024,
        dype_start_sigma: float = 1.0,
        base_patches: int | None = None,
        axes_lens: list[int] | None = None,
    ):
        super().__init__(
            theta,
            axes_dim,
            method,
            yarn_alt_scaling,
            dype,
            dype_scale,
            dype_exponent,
            base_resolution,
            dype_start_sigma,
            base_patches,
        )

        if axes_lens is None:
            self.register_buffer("axes_lens", None, persistent=False)
        else:
            self.register_buffer(
                "axes_lens",
                torch.as_tensor(axes_lens, dtype=torch.float32),
                persistent=False,
            )

    def _scale_spatial_pos_ids(self, ids: torch.Tensor) -> torch.Tensor:
        if self.axes_lens is None or ids.shape[-1] < 3:
            return ids.float()

        pos = ids.float()
        axes_lens = self.axes_lens.to(device=pos.device, dtype=pos.dtype)

        for axis_idx, axis_limit in zip((1, 2), axes_lens[-2:]):
            axis_vals = pos[..., axis_idx]
            axis_min = axis_vals.min()
            axis_max = axis_vals.max()

            # `axes_lens` describes token counts (e.g., 512 for 1024px @ patch_size=2)
            # Only scale when incoming span exceeds the documented training window.
            allowed_span = axis_limit - 1.0
            current_span = axis_max - axis_min
            if current_span > allowed_span:
                scale = allowed_span / max(current_span, 1e-6)
                pos[..., axis_idx] = axis_min + (axis_vals - axis_min) * scale
                pos[..., axis_idx].clamp_(min=axis_min, max=axis_min + allowed_span)

        return pos

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = self._scale_spatial_pos_ids(ids)
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
