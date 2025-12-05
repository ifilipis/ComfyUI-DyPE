import torch
import torch.nn as nn
import math
from .rope import get_1d_dype_yarn_pos_embed, get_1d_yarn_pos_embed, get_1d_ntk_pos_embed

class DyPEBasePosEmbed(nn.Module):
    """
    Base class for Dynamic Position Extrapolation.
    Handles the calculation of DyPE scaling factors and raw (cos, sin) components.
    Subclasses must implement `forward` to format the output for specific model architectures.
    """
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', yarn_alt_scaling: bool = False, dype: bool = True, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_resolution: int = 1024, dype_start_sigma: float = 1.0, base_patches: int | None = None):
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

        # Z-Image grid inference uses token counts rather than raw coordinate spans.
        self.use_scaled_grid_tokens = False

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def _infer_axis_stats(self, pos: torch.Tensor):
        axis_stats = []
        for i in range(pos.shape[-1]):
            axis_pos = pos[..., i].reshape(-1)
            unique_vals = torch.unique(axis_pos)

            token_count = max(1, int(unique_vals.numel()))

            if token_count > 1:
                diffs = unique_vals[1:] - unique_vals[:-1]
                step = float(torch.median(diffs).item())
            else:
                step = 1.0

            axis_stats.append({
                'tokens': token_count,
                'step': max(step, 1e-6)
            })

        return axis_stats

    def _calc_vision_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Calculates raw (cos, sin) pairs using DyPE Vision YaRN (Decoupled + Quadratic Aggressive).
        Returns a list of (cos, sin) tuples per axis.
        """
        n_axes = pos.shape[-1]
        components = []
        
        axis_stats = self._infer_axis_stats(pos) if self.use_scaled_grid_tokens else None

        if axis_stats and pos.shape[-1] >= 3:
            h_tokens = axis_stats[1]['tokens']
            w_tokens = axis_stats[2]['tokens']
            h_scaled = h_tokens * axis_stats[1]['step']
            w_scaled = w_tokens * axis_stats[2]['step']
            max_current_patches = max(h_tokens, w_tokens)
            max_scaled_patches = max(h_scaled, w_scaled)
        elif axis_stats:
            max_current_patches = axis_stats[0]['tokens']
            max_scaled_patches = axis_stats[0]['tokens'] * axis_stats[0]['step']
        else:
            if pos.shape[-1] >= 3:
                h_span = int(pos[..., 1].max().item() - pos[..., 1].min().item() + 1)
                w_span = int(pos[..., 2].max().item() - pos[..., 2].min().item() + 1)
                max_current_patches = max(h_span, w_span)
            else:
                max_current_patches = int(pos.max().item() - pos.min().item() + 1)
            max_scaled_patches = max_current_patches

        scale_global_tokens = max(1.0, max_current_patches / self.base_patches)
        scale_global_span = max(1.0, max_scaled_patches / self.base_patches)
        scale_global = max(scale_global_tokens, scale_global_span)
            
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
            if axis_stats:
                current_patches = axis_stats[i]['tokens']
                scaled_patches = axis_stats[i]['tokens'] * axis_stats[i]['step']
            else:
                current_patches = int(axis_pos.max().item() - axis_pos.min().item() + 1)
                scaled_patches = current_patches
            
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
                scale_local_tokens = max(1.0, current_patches / self.base_patches)
                scale_local_span = max(1.0, scaled_patches / self.base_patches)
                scale_local = max(scale_local_tokens, scale_local_span)
                
                # Apply Low Theta protection
                if force_isotropic:
                    dype_kwargs['linear_scale'] = 1.0
                else:
                    dype_kwargs['linear_scale'] = scale_local 
                
                if scale_global > 1.0:
                    cos, sin = get_1d_dype_yarn_pos_embed(
                        **common_kwargs,
                        ori_max_pe_len=self.base_patches,
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
        
        axis_stats = self._infer_axis_stats(pos) if self.use_scaled_grid_tokens else None

        if axis_stats and pos.shape[-1] >= 3:
            h_tokens = axis_stats[1]['tokens']
            w_tokens = axis_stats[2]['tokens']
            h_scaled = h_tokens * axis_stats[1]['step']
            w_scaled = w_tokens * axis_stats[2]['step']
            max_current_patches = max(h_tokens, w_tokens)
            max_scaled_patches = max(h_scaled, w_scaled)
        elif axis_stats:
            max_current_patches = axis_stats[0]['tokens']
            max_scaled_patches = axis_stats[0]['tokens'] * axis_stats[0]['step']
        else:
            if pos.shape[-1] >= 3:
                h_span = int(pos[..., 1].max().item() - pos[..., 1].min().item() + 1)
                w_span = int(pos[..., 2].max().item() - pos[..., 2].min().item() + 1)
                max_current_patches = max(h_span, w_span)
            else:
                max_current_patches = int(pos.max().item() - pos.min().item() + 1)
            max_scaled_patches = max_current_patches

        needs_extrapolation = (max_scaled_patches > self.base_patches)

        force_isotropic = self.theta < 1000.0
        use_anisotropic = self.yarn_alt_scaling and not force_isotropic

        if needs_extrapolation and use_anisotropic:
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                if axis_stats:
                    current_patches_on_axis = axis_stats[i]['tokens'] * axis_stats[i]['step']
                else:
                    current_patches_on_axis = int(axis_pos.max().item() - axis_pos.min().item() + 1)

                if i > 0 and current_patches_on_axis > self.base_patches:
                    max_pe_len = torch.tensor(current_patches_on_axis, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_yarn_pos_embed(**common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs, use_aggressive_mscale=True)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
                
                components.append((cos, sin))
        else:
            cos_full_spatial, sin_full_spatial = None, None
            if needs_extrapolation:
                spatial_axis_dim = self.axes_dim[1]
                square_length = math.ceil(max_scaled_patches)
                square_pos = torch.arange(0, square_length, device=pos.device).float()
                max_pe_len = torch.tensor(square_length, dtype=freqs_dtype, device=pos.device)
                
                common_kwargs_spatial = {'dim': spatial_axis_dim, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                cos_full_spatial, sin_full_spatial = get_1d_yarn_pos_embed(
                    **common_kwargs_spatial, pos=square_pos, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs, use_aggressive_mscale=False
                )

            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                
                if i > 0 and needs_extrapolation:
                    if axis_stats:
                        step = axis_stats[i]['step']
                        offset_indices = torch.round((axis_pos - axis_pos.min()) / step).long()
                    else:
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

        axis_stats = self._infer_axis_stats(pos) if self.use_scaled_grid_tokens else None

        if axis_stats and pos.shape[-1] >= 3:
            h_tokens = axis_stats[1]['tokens']
            w_tokens = axis_stats[2]['tokens']
            h_scaled = h_tokens * axis_stats[1]['step']
            w_scaled = w_tokens * axis_stats[2]['step']
            max_patches = max(h_tokens, w_tokens)
            max_scaled = max(h_scaled, w_scaled)
        elif axis_stats:
            max_patches = axis_stats[0]['tokens']
            max_scaled = axis_stats[0]['tokens'] * axis_stats[0]['step']
        else:
            if pos.shape[-1] >= 3:
                h_span = int(pos[..., 1].max().item() - pos[..., 1].min().item() + 1)
                w_span = int(pos[..., 2].max().item() - pos[..., 2].min().item() + 1)
                max_patches = max(h_span, w_span)
            else:
                max_patches = int(pos.max().item() - pos.min().item() + 1)
            max_scaled = max_patches

        unified_scale = max(max_patches, max_scaled) / self.base_patches if max(max_patches, max_scaled) > self.base_patches else 1.0

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}

            ntk_factor = 1.0
            if axis_stats:
                axis_extent = max(axis_stats[i]['tokens'], axis_stats[i]['tokens'] * axis_stats[i]['step'])
            else:
                axis_extent = int(axis_pos.max().item() - axis_pos.min().item() + 1)

            if i > 0 and unified_scale > 1.0:
                base_ntk = unified_scale ** (axis_dim / (axis_dim - 2))
                if self.dype:
                    k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                    ntk_factor = base_ntk ** k_t
                else:
                    ntk_factor = base_ntk
                ntk_factor = max(1.0, ntk_factor)
            elif i > 0 and axis_extent > self.base_patches:
                base_ntk = (axis_extent / self.base_patches) ** (axis_dim / (axis_dim - 2))
                ntk_factor = max(1.0, base_ntk)

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
