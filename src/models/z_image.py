from typing import Optional, Sequence

import torch

from ..base import DyPEBasePosEmbed


class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `RopeEmbedder`: (B, L, D/2) complex64
    """

    def __init__(
        self,
        theta: int,
        axes_dim: list[int],
        method: str = "yarn",
        yarn_alt_scaling: bool = False,
        dype: bool = True,
        dype_scale: float = 2.0,
        dype_exponent: float = 2.0,
        base_resolution: int = 1024,
        dype_start_sigma: float = 1.0,
        axes_lens: Optional[Sequence[int]] = None,
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
        )
        self.axes_lens = list(axes_lens) if axes_lens is not None else None

    def _validate_axes_lens(self, ids: torch.Tensor):
        if self.axes_lens is None or ids.numel() == 0:
            return

        if ids.shape[-1] != len(self.axes_lens):
            raise ValueError(
                f"Expected ids with {len(self.axes_lens)} axes, but got shape {ids.shape}"
            )

        max_ids = ids.reshape(-1, ids.shape[-1]).max(dim=0).values
        device_axes_lens = torch.as_tensor(self.axes_lens, device=ids.device, dtype=max_ids.dtype)
        if torch.any(max_ids >= device_axes_lens):
            exceeded = (max_ids >= device_axes_lens).nonzero(as_tuple=False).flatten().tolist()
            raise ValueError(
                f"Position ids exceed configured axes lengths at dimensions: {exceeded}."
            )

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        self._validate_axes_lens(ids)

        pos = ids.float()
        freqs_dtype = torch.float32

        components = self.get_components(pos, freqs_dtype)

        freqs_cis_parts: list[torch.Tensor] = []
        for cos, sin in components:
            # Reduce repeated pairs produced by `repeat_interleave_real` and construct complex freqs
            cos_half = cos[..., ::2]
            sin_half = sin[..., ::2]
            stacked = torch.stack([cos_half, sin_half], dim=-1).to(torch.float32)
            freqs_cis = torch.view_as_complex(stacked)
            freqs_cis_parts.append(freqs_cis)

        freqs_cis_full = torch.cat(freqs_cis_parts, dim=-1)
        return freqs_cis_full.to(ids.device)
