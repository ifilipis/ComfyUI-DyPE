import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """

    def __init__(self, *args, base_hw: tuple[int, int] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_hw = base_hw

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        # Scale Z-Image grids to the native base resolution while keeping DyPE aware of the
        # output token density for extrapolation.
        if pos.shape[-1] >= 3 and self.base_hw is not None:
            cap_token_id = pos[..., 0].max().item()
            image_mask = pos[..., 0] == cap_token_id

            if image_mask.any():
                base_h, base_w = self.base_hw
                image_rows = self._get_axis_patches(pos, 1)
                image_cols = self._get_axis_patches(pos, 2)

                # Preserve the existing Z-Image scaling strategy.
                scale_y = image_rows / base_h if base_h > 0 else 1.0
                scale_x = image_cols / base_w if base_w > 0 else 1.0

                pos = pos.clone()
                pos[..., 1] = torch.where(image_mask, pos[..., 1] * scale_y, pos[..., 1])
                pos[..., 2] = torch.where(image_mask, pos[..., 2] * scale_x, pos[..., 2])

                cap_count = int(cap_token_id - 1)
                self.grid_patch_counts = [cap_count if cap_count > 0 else None, image_rows, image_cols]

        components = self.get_components(pos, freqs_dtype)
        self.grid_patch_counts = None

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
