import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """

    def __init__(self, *args, axes_lens: list[int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.axes_lens = axes_lens

    def _scaled_pos(self, pos: torch.Tensor) -> torch.Tensor:
        if not self.axes_lens or len(self.axes_lens) < 3:
            return pos

        scaled_pos = pos
        for axis in range(1, min(pos.shape[-1], len(self.axes_lens))):
            max_tokens = self.axes_lens[axis]
            if max_tokens is None:
                continue

            axis_pos = pos[..., axis]
            axis_min = axis_pos.min()
            axis_span = axis_pos.max() - axis_min

            if axis_span + 1 > max_tokens:
                scale = (max_tokens - 1) / axis_span if axis_span > 0 else 1.0
                adjusted_axis_pos = (axis_pos - axis_min) * scale + axis_min
                adjusted_axis_pos = torch.clamp(adjusted_axis_pos, max=max_tokens - 1)

                if scaled_pos is pos:
                    scaled_pos = pos.clone()

                scaled_pos[..., axis] = adjusted_axis_pos

        return scaled_pos

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        pos = self._scaled_pos(pos)
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
