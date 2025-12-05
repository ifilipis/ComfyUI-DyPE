import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()

        h_coords = pos[..., 1]
        w_coords = pos[..., 2]

        def _normalize_axis(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            coords_flat = coords.reshape(-1)
            coord_min = coords_flat.min()
            unique_vals = torch.unique(coords_flat)

            if unique_vals.numel() > 1:
                deltas = unique_vals[1:] - unique_vals[:-1]
                non_zero = deltas[deltas != 0]
                step = non_zero.median() if non_zero.numel() > 0 else torch.tensor(0.0, device=coords.device, dtype=coords.dtype)
            else:
                step = torch.tensor(0.0, device=coords.device, dtype=coords.dtype)

            safe_step = step if step.abs().item() > 0 else torch.tensor(1.0, device=coords.device, dtype=coords.dtype)
            normalized = (coords - coord_min) / safe_step
            return normalized, coord_min

        norm_h, _ = _normalize_axis(h_coords)
        norm_w, _ = _normalize_axis(w_coords)

        rescaled_pos = torch.stack((pos[..., 0], norm_h, norm_w), dim=-1)
        freqs_dtype = torch.bfloat16 if pos.device.type == 'cuda' else torch.float32

        components = self.get_components(rescaled_pos, freqs_dtype)

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
