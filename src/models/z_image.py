import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def _blend_to_full_scale(self) -> float:
        t_effective = self.current_timestep

        if t_effective > self.dype_start_sigma:
            t_norm = 1.0
        else:
            t_norm = t_effective / self.dype_start_sigma

        t_factor = t_norm ** self.dype_exponent
        return 1.0 - t_factor

    def _resize_rope_grid(self, pos: torch.Tensor) -> torch.Tensor:
        image_mask = (pos[..., 1] != 0) | (pos[..., 2] != 0)
        if not image_mask.any():
            return pos

        blend = torch.tensor(self._blend_to_full_scale(), device=pos.device, dtype=pos.dtype)
        if blend.item() == 0:
            return pos

        pos_rescaled = pos.clone()

        for axis in (1, 2):
            coords = pos[..., axis]
            coords_image = coords[image_mask]

            if coords_image.numel() == 0:
                continue

            unique_coords = torch.unique(coords_image)
            if unique_coords.numel() <= 1:
                continue

            unique_sorted, _ = torch.sort(unique_coords)
            deltas = torch.diff(unique_sorted)
            if deltas.numel() == 0:
                continue

            step = torch.median(deltas)
            if torch.isclose(step, torch.tensor(0.0, device=pos.device, dtype=pos.dtype)):
                continue

            start = coords_image.min()
            full_scale_coords = (coords - start) / step + start

            pos_rescaled[..., axis] = coords + (full_scale_coords - coords) * blend

        return pos_rescaled

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = self._resize_rope_grid(ids.float())
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
