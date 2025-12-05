import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """

    def _infer_image_hw(self, pos: torch.Tensor) -> tuple[int | None, int | None]:
        image_mask = pos[..., 0] == pos[..., 0].max()
        if not image_mask.any():
            return None, None

        h_coords = pos[..., 1][image_mask]
        w_coords = pos[..., 2][image_mask]

        h_tokens = int(h_coords.max().item() - h_coords.min().item() + 1)
        w_tokens = int(w_coords.max().item() - w_coords.min().item() + 1)
        return h_tokens, w_tokens

    def _scale_spatial_grid(self, pos: torch.Tensor, current_hw: tuple[int | None, int | None]) -> torch.Tensor:
        if self.base_hw is None or current_hw[0] is None or current_hw[1] is None:
            return pos

        base_h, base_w = self.base_hw
        h_tokens, w_tokens = current_hw

        h_scale = h_tokens / base_h if base_h > 0 else 1.0
        w_scale = w_tokens / base_w if base_w > 0 else 1.0

        scaled = pos.clone()
        scaled[..., 1] = scaled[..., 1] * h_scale
        scaled[..., 2] = scaled[..., 2] * w_scale
        return scaled

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        current_hw = self._infer_image_hw(pos)
        if current_hw[0] is not None and current_hw[1] is not None:
            self.current_hw_override = (current_hw[0], current_hw[1])

        pos_scaled = self._scale_spatial_grid(pos, current_hw)
        components = self.get_components(pos_scaled, freqs_dtype)

        emb_parts = []
        for cos, sin in components:
            cos_reshaped = cos.view(*cos.shape[:-1], -1, 2)[..., :1]
            sin_reshaped = sin.view(*sin.shape[:-1], -1, 2)[..., :1]
            row1 = torch.cat([cos_reshaped, -sin_reshaped], dim=-1)
            row2 = torch.cat([sin_reshaped, cos_reshaped], dim=-1)
            matrix = torch.stack([row1, row2], dim=-2)
            emb_parts.append(matrix)

        emb = torch.cat(emb_parts, dim=-3)
        self.current_hw_override = None
        return emb.unsqueeze(1).to(ids.device)
