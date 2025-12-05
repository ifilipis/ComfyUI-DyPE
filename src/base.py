import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .rope import get_1d_dype_yarn_pos_embed, get_1d_ntk_pos_embed, get_1d_yarn_pos_embed

class DyPEBasePosEmbed(nn.Module):
    """
    Base class for Dynamic Position Extrapolation.
    Handles the calculation of DyPE scaling factors and raw (cos, sin) components.
    Subclasses must implement `forward` to format the output for specific model architectures.
    """
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', yarn_alt_scaling: bool = False, dype: bool = True, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_resolution: int = 1024, dype_start_sigma: float = 1.0, base_patches: int | None = None, base_hw_tokens: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim
        self.method = method
        self.yarn_alt_scaling = yarn_alt_scaling
        self.dype = True if method == 'vision_yarn' else (dype if method != 'base' else False)
        self.dype_scale = dype_scale
        self.dype_exponent = dype_exponent
        self.base_resolution = base_resolution
        self.dype_start_sigma = max(0.001, min(1.0, dype_start_sigma)) # Clamp 0.001-1.0
        
        self.current_timestep = 1.0

        # Dynamic Base Patches: configurable per-model to align with native patch grids.
        # Flux/Qwen default: (Resolution // 8) // 2
        self.base_patches = base_patches if base_patches is not None else (self.base_resolution // 8) // 2
        # Optional per-axis base token counts for anisotropic grids (e.g., Z-Image).
        self.base_hw_tokens: Optional[Tuple[int, int]] = base_hw_tokens

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def _get_base_for_axis(self, axis_index: int) -> float:
        if self.base_hw_tokens is not None and axis_index > 0:
            idx = min(axis_index - 1, len(self.base_hw_tokens) - 1)
            return float(self.base_hw_tokens[idx])
        return float(self.base_patches)

    def _calc_vision_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Calculates raw (cos, sin) pairs using DyPE Vision YaRN (Decoupled + Quadratic Aggressive).
        Returns a list of (cos, sin) tuples per axis.
        """
        n_axes = pos.shape[-1]
        components = []
        
        global_scales = []
        if pos.shape[-1] >= 3:
            h_span = float(pos[..., 1].max().item() - pos[..., 1].min().item() + 1.0)
            w_span = float(pos[..., 2].max().item() - pos[..., 2].min().item() + 1.0)
            global_scales.append(h_span / max(1.0, self._get_base_for_axis(1)))
            global_scales.append(w_span / max(1.0, self._get_base_for_axis(2)))
        else:
            span = float(pos.max().item() - pos.min().item() + 1.0)
            global_scales.append(span / max(1.0, self._get_base_for_axis(1)))

        scale_global = max(1.0, max(global_scales) if len(global_scales) > 0 else 1.0)
            
        mscale_start = 0.1 * math.log(scale_global) + 1.0
        mscale_end = 1.0
        
        t_effective = self.current_timestep
        
        if t_effective > self.dype_start_sigma:
            t_norm = 1.0
        else:
            t_norm = t_effective / self.dype_start_sigma

        t_factor = math.pow(t_norm, self.dype_exponent)
        current_mscale = mscale_end + (mscale_start - mscale_end) * t_factor

        # Low Theta Heuristic (Z-Image / Lumina)
        force_isotropic = self.theta < 1000.0

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            current_patches = float(axis_pos.max().item() - axis_pos.min().item() + 1.0)
            base_for_axis = self._get_base_for_axis(i)
            
            common_kwargs = {
                'dim': axis_dim, 
                'pos': axis_pos, 
                'theta': self.theta, 
                'use_real': True, 
                'repeat_interleave_real': True, 
                'freqs_dtype': freqs_dtype
            }
            
            dype_kwargs = {
                'dype': self.dype, 
                'current_timestep': self.current_timestep, 
                'dype_scale': self.dype_scale, 
                'dype_exponent': self.dype_exponent,
                'ntk_scale': scale_global,      
                'override_mscale': current_mscale 
            }

            if i > 0:
                scale_local = max(1.0, current_patches / max(1.0, base_for_axis))
                
                # Apply Low Theta protection
                if force_isotropic:
                    dype_kwargs['linear_scale'] = 1.0
                else:
                    dype_kwargs['linear_scale'] = scale_local 
                
                if scale_global > 1.0:
                    cos, sin = get_1d_dype_yarn_pos_embed(
                        **common_kwargs,
                        ori_max_pe_len=base_for_axis,
                        **dype_kwargs
                    )
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            else:
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

            components.append((cos, sin))
            
        return components

    def _calc_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Legacy Method: Standard YaRN
        Returns a list of (cos, sin) tuples per axis.
        """
        n_axes = pos.shape[-1]
        components = []
        
        axis_spans = []
        if pos.shape[-1] >= 3:
            h_span = float(pos[..., 1].max().item() - pos[..., 1].min().item() + 1.0)
            w_span = float(pos[..., 2].max().item() - pos[..., 2].min().item() + 1.0)
            axis_spans.append((1, h_span))
            axis_spans.append((2, w_span))
        else:
            span = float(pos.max().item() - pos.min().item() + 1.0)
            axis_spans.append((1, span))

        needs_extrapolation = any(span > self._get_base_for_axis(idx) for idx, span in axis_spans)

        force_isotropic = self.theta < 1000.0
        use_anisotropic = self.yarn_alt_scaling and not force_isotropic

        if needs_extrapolation and use_anisotropic:
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                current_patches_on_axis = float(axis_pos.max().item() - axis_pos.min().item() + 1.0)
                base_for_axis = self._get_base_for_axis(i)
                if i > 0 and current_patches_on_axis > base_for_axis:
                    max_pe_len = torch.tensor(current_patches_on_axis, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_yarn_pos_embed(**common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=base_for_axis, **dype_kwargs, use_aggressive_mscale=True)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
                
                components.append((cos, sin))
        else:
            cos_full_spatial, sin_full_spatial = None, None
            if needs_extrapolation:
                spatial_axis_dim = self.axes_dim[1]
                max_span = math.ceil(max(span for _, span in axis_spans))
                square_pos = torch.arange(0, max_span, device=pos.device).float()
                max_pe_len = torch.tensor(max_span, dtype=freqs_dtype, device=pos.device)
                
                common_kwargs_spatial = {'dim': spatial_axis_dim, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                cos_full_spatial, sin_full_spatial = get_1d_yarn_pos_embed(
                    **common_kwargs_spatial, pos=square_pos, max_pe_len=max_pe_len, ori_max_pe_len=self._get_base_for_axis(1), **dype_kwargs, use_aggressive_mscale=False
                )

            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]

                if i > 0 and needs_extrapolation:
                    offset_indices = axis_pos.long() - axis_pos.long().min()
                    pos_indices = offset_indices.view(-1)

                    cos = cos_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                    sin = sin_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                else:
                    common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

                components.append((cos, sin))
            
        return components

    def _calc_ntk_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Returns a list of (cos, sin) tuples per axis using NTK.
        """
        n_axes = pos.shape[-1]
        components = []

        axis_spans = []
        if pos.shape[-1] >= 3:
            h_span = float(pos[..., 1].max().item() - pos[..., 1].min().item() + 1.0)
            w_span = float(pos[..., 2].max().item() - pos[..., 2].min().item() + 1.0)
            axis_spans.append((1, h_span))
            axis_spans.append((2, w_span))
        else:
            span = float(pos.max().item() - pos.min().item() + 1.0)
            axis_spans.append((1, span))

        axis_scale_lookup = {
            idx: span / max(1.0, self._get_base_for_axis(idx))
            for idx, span in axis_spans
        }

        unified_scale = max(1.0, max(axis_scale_lookup.values()) if len(axis_scale_lookup) > 0 else 1.0)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            
            ntk_factor = 1.0
            axis_scale = max(1.0, axis_scale_lookup.get(i, unified_scale))
            if i > 0 and axis_scale > 1.0:
                base_ntk = axis_scale ** (axis_dim / (axis_dim - 2))
                if self.dype:
                    k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                    ntk_factor = base_ntk ** k_t
                else:
                    ntk_factor = base_ntk
                ntk_factor = max(1.0, ntk_factor)
            
            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)
            components.append((cos, sin))
            
        return components

    def get_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        if self.method == 'vision_yarn':
            return self._calc_vision_yarn_components(pos, freqs_dtype)
        elif self.method == 'yarn':
            return self._calc_yarn_components(pos, freqs_dtype)
        else: # 'ntk' or 'base'
            return self._calc_ntk_components(pos, freqs_dtype)
            
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class does not implement forward. Use a specific model subclass.")