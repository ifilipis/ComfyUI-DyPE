import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """

    def __init__(self, *args, base_hw: tuple[int, int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if base_hw is not None and len(base_hw) == 2:
            self.base_hw = (max(1, int(base_hw[0])), max(1, int(base_hw[1])))
        else:
            self.base_hw = None

    @staticmethod
    def _build_scaled_grid(length: int, scale: float, device, dtype):
        coords = torch.arange(length, device=device, dtype=dtype)
        return coords * scale

    def _apply_scaled_grid(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.shape[-1] < 3 or self.base_hw is None:
            return ids.float()

        pos = ids.float().clone()
        batch, seq, _ = pos.shape
        spatial_token_id = pos[..., 0].amax(dim=1, keepdim=True)
        spatial_mask = pos[..., 0].unsqueeze(-1) == spatial_token_id.unsqueeze(-1)

        scaled_pos = pos
        for b in range(batch):
            mask = spatial_mask[b, :, 0]
            token_count = int(mask.sum().item())
            if token_count == 0:
                continue

            axis_y = pos[b, mask, 1]
            axis_x = pos[b, mask, 2]

            h_len, _, _ = self._infer_axis_info(axis_y)
            w_len, _, _ = self._infer_axis_info(axis_x)

            if h_len == 0 or w_len == 0 or h_len * w_len != token_count:
                continue

            h_scale = h_len / float(self.base_hw[0])
            w_scale = w_len / float(self.base_hw[1])

            scaled_y = self._build_scaled_grid(h_len, h_scale, pos.device, pos.dtype)
            scaled_x = self._build_scaled_grid(w_len, w_scale, pos.device, pos.dtype)

            grid_y = scaled_y.view(-1, 1).repeat(1, w_len).flatten()
            grid_x = scaled_x.view(1, -1).repeat(h_len, 1).flatten()

            scaled_pos[b, mask, 1] = grid_y[:token_count]
            scaled_pos[b, mask, 2] = grid_x[:token_count]

        return scaled_pos

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = self._apply_scaled_grid(ids)
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
