import math
import torch
from ..base import DyPEBasePosEmbed

class PosEmbedFlux(DyPEBasePosEmbed):
    """
    DyPE Implementation for Standard ComfyUI Flux Models.
    Output Format: Rotation Matrix (concatenated) -> (B, 1, L, D)
    """
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


class PosEmbedFlux2Klein(DyPEBasePosEmbed):
    """
    DyPE Implementation for Flux2 Klein models.
    Output Format: Rotation Matrix (concatenated) -> (B, 1, L, D)
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.external_scale_hint = 1.0

    def set_scale_hint(self, scale: float):
        self.external_scale_hint = scale

    def _blend_to_full_scale(self) -> float:
        t_effective = self.current_timestep
        if t_effective > self.dype_start_sigma:
            t_norm = 1.0
        else:
            t_norm = t_effective / self.dype_start_sigma

        t_factor = math.pow(t_norm, self.dype_exponent)
        return 1.0 - t_factor

    def _scale_rope_grid(self, pos: torch.Tensor) -> torch.Tensor:
        if self.external_scale_hint == 1.0:
            return pos

        pos_scaled = pos.clone()
        for axis in (1, 2):
            if axis < pos_scaled.shape[-1]:
                pos_scaled[..., axis] = pos_scaled[..., axis] * self.external_scale_hint
        return pos_scaled

    def _resize_rope_grid(self, pos: torch.Tensor) -> torch.Tensor:
        if not self.dype:
            return pos

        if pos.shape[-1] < 3:
            return pos

        image_mask = (pos[..., 1] != 0) | (pos[..., 2] != 0)
        if not image_mask.any():
            return pos

        blend_val = self._blend_to_full_scale()
        if blend_val <= 0.001:
            return pos

        blend = torch.tensor(blend_val, device=pos.device, dtype=pos.dtype)
        pos_rescaled = pos.clone()

        for axis in (1, 2):
            coords = pos[..., axis]
            coords_image = coords[image_mask]
            if coords_image.numel() <= 1:
                continue

            unique_coords = torch.unique(coords_image)
            if unique_coords.numel() <= 1:
                continue

            unique_sorted, _ = torch.sort(unique_coords)
            deltas = torch.diff(unique_sorted)
            if deltas.numel() == 0:
                continue
            step = torch.median(deltas)

            if torch.isclose(step, torch.tensor(1.0, device=pos.device, dtype=pos.dtype), atol=1e-3):
                continue
            if torch.isclose(step, torch.tensor(0.0, device=pos.device, dtype=pos.dtype)):
                continue

            start = coords_image.min()
            full_scale_coords = (coords - start) / step + start
            pos_rescaled[..., axis] = coords + (full_scale_coords - coords) * blend

        return pos_rescaled

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = ids.float()
        pos = self._scale_rope_grid(pos)
        pos = self._resize_rope_grid(pos)
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
