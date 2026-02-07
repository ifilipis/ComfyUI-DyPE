import math
import torch
from ..base import DyPEBasePosEmbed
from ..rope import get_1d_dype_yarn_pos_embed, get_1d_ntk_pos_embed, get_1d_yarn_pos_embed

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
        self.external_scale_hint = max(1.0, scale)

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

    def _calc_flux2_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        n_axes = pos.shape[-1]
        components = []

        scale_global = self.external_scale_hint
        current_mscale = self._get_mscale(scale_global) if (scale_global > 1.0 and self.dype) else 1.0

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {
                'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta,
                'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype
            }

            is_spatial = (i > 0)

            if is_spatial and scale_global > 1.0:
                grid_idx = i - 1
                base_axis_len = self.base_patch_grid[grid_idx] if grid_idx < len(self.base_patch_grid) else self.base_patches

                if self.method == 'vision_yarn':
                    dype_kwargs = {
                        'dype': self.dype, 'current_timestep': self.current_timestep,
                        'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent,
                        'ntk_scale': scale_global, 'override_mscale': current_mscale,
                        'linear_scale': scale_global
                    }
                    cos, sin = get_1d_dype_yarn_pos_embed(
                        **common_kwargs, ori_max_pe_len=base_axis_len, **dype_kwargs
                    )
                elif self.method == 'yarn':
                    fake_current_len = int(base_axis_len * scale_global)
                    max_pe_len = torch.tensor(fake_current_len, dtype=freqs_dtype, device=pos.device)
                    dype_kwargs = {
                        'dype': self.dype, 'current_timestep': self.current_timestep,
                        'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent
                    }
                    cos, sin = get_1d_yarn_pos_embed(
                        **common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=base_axis_len,
                        **dype_kwargs, use_aggressive_mscale=False
                    )
                    if self.dype:
                        mscale_tensor = torch.tensor(current_mscale, dtype=cos.dtype, device=cos.device)
                        cos = cos * mscale_tensor
                        sin = sin * mscale_tensor
                else:
                    base_ntk = scale_global ** (axis_dim / (axis_dim - 2))
                    if self.dype:
                        k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                        ntk_factor = base_ntk ** k_t
                    else:
                        ntk_factor = base_ntk
                    ntk_factor = max(1.0, ntk_factor)
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)
            else:
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

            components.append((cos, sin))

        return components

    def get_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        return self._calc_flux2_components(pos, freqs_dtype)

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
