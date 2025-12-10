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

        is_z_image = self.__class__.__name__.lower().endswith("zimage")
        if is_z_image:
            self.dype_start_sigma = 1.0
        else:
            self.dype_start_sigma = max(0.001, min(1.0, dype_start_sigma)) # Clamp 0.001-1.0

        self.current_timestep = 1.0
        self.use_basic_scaling = False

        # Dynamic Base Patches: configurable per-model to align with native patch grids.
        # Flux/Qwen default: (Resolution // 8) // 2
        self.base_patches = base_patches if base_patches is not None else (self.base_resolution // 8) // 2

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def set_basic_scaling(self, use_basic_scaling: bool):
        self.use_basic_scaling = use_basic_scaling

    def _axis_grid_stats(self, axis_pos: torch.Tensor):
        axis_flat = axis_pos.reshape(-1)
        unique_vals = torch.unique(axis_flat)
        token_count = unique_vals.numel()

        if token_count <= 1:
            step = 1.0
        else:
            sorted_vals, _ = torch.sort(unique_vals)
            diffs = torch.diff(sorted_vals)
            positive_diffs = diffs[diffs > 0]
            step = positive_diffs.min().item() if positive_diffs.numel() > 0 else 1.0

        min_val = axis_flat.min().item() if axis_flat.numel() > 0 else 0.0
        max_val = axis_flat.max().item() if axis_flat.numel() > 0 else 0.0

        scaled_len = token_count * step if token_count > 0 else 0.0

        return {
            "token_count": token_count,
            "step": step,
            "scaled_len": scaled_len,
            "min": min_val,
            "max": max_val,
        }

    def _calc_vision_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Calculates raw (cos, sin) pairs using DyPE Vision YaRN (Decoupled + Quadratic Aggressive).
        Returns a list of (cos, sin) tuples per axis.
        """
        n_axes = pos.shape[-1]
        components = []
        
        axis_stats = [self._axis_grid_stats(pos[..., i]) for i in range(n_axes)]

        if pos.shape[-1] >= 3:
            max_current_patches = max(axis_stats[1]["scaled_len"], axis_stats[2]["scaled_len"])
        else:
            max_current_patches = axis_stats[0]["scaled_len"]
        
        scale_global = max(1.0, max_current_patches / self.base_patches)
            
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
            axis_scaled_len = axis_stats[i]["scaled_len"]

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
                scale_local = max(1.0, axis_scaled_len / self.base_patches)
                
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

        axis_stats = [self._axis_grid_stats(pos[..., i]) for i in range(n_axes)]

        if pos.shape[-1] >= 3:
            max_current_patches = max(axis_stats[1]["scaled_len"], axis_stats[2]["scaled_len"])
        else:
            max_current_patches = axis_stats[0]["scaled_len"]

        needs_extrapolation = (max_current_patches > self.base_patches)

        force_isotropic = self.theta < 1000.0
        use_anisotropic = self.yarn_alt_scaling and not force_isotropic

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            axis_scaled_len = axis_stats[i]["scaled_len"]

            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

            axis_needs_extrapolation = (i > 0 and needs_extrapolation)
            if use_anisotropic:
                axis_needs_extrapolation = axis_needs_extrapolation and (axis_scaled_len > self.base_patches)

            if axis_needs_extrapolation:
                target_len = axis_scaled_len if use_anisotropic else max_current_patches
                max_pe_len = torch.tensor(target_len, dtype=freqs_dtype, device=pos.device)
                cos, sin = get_1d_yarn_pos_embed(**common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=self.base_patches, **dype_kwargs, use_aggressive_mscale=use_anisotropic)
            else:
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

            components.append((cos, sin))
            
        return components

    def _calc_ntk_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Returns a list of (cos, sin) tuples per axis using NTK.
        """
        n_axes = pos.shape[-1]
        components = []

        axis_stats = [self._axis_grid_stats(pos[..., i]) for i in range(n_axes)]

        if pos.shape[-1] >= 3:
            max_patches = max(axis_stats[1]["scaled_len"], axis_stats[2]["scaled_len"])
        else:
            max_patches = axis_stats[0]["scaled_len"]

        unified_scale = max_patches / self.base_patches if max_patches > self.base_patches else 1.0

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            
            ntk_factor = 1.0
            if i > 0 and unified_scale > 1.0:
                base_ntk = unified_scale ** (axis_dim / (axis_dim - 2))
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

        if self.method in ('yarn', 'ntk') and self.use_basic_scaling:
            return self._calc_ntk_components(pos, freqs_dtype)

        if self.method == 'yarn':
            return self._calc_yarn_components(pos, freqs_dtype)
        else: # 'ntk' or 'base'
            return self._calc_ntk_components(pos, freqs_dtype)
            
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class does not implement forward. Use a specific model subclass.")