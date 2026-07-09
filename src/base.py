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
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', yarn_alt_scaling: bool = False, dype: bool = True, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_resolution: int = 1024, dype_start_sigma: float = 1.0, base_patch_grid: tuple[int, int] = None, spatial_axes: tuple[int, ...] = None, yarn_base_patch_grid: tuple[int, int] = None, direct_yarn_positions: bool = False):
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
        
        # Determine Base Patch Grid and Max Patches
        if base_patch_grid is None:
            # Default heuristic: 1024px -> 128 latent -> 64 patches (assuming patch_size=2)
            val = (self.base_resolution // 8) // 2
            self.base_patch_grid = (val, val)
        elif isinstance(base_patch_grid, int):
             self.base_patch_grid = (base_patch_grid, base_patch_grid)
        else:
            self.base_patch_grid = base_patch_grid
            
        self.base_patches = max(self.base_patch_grid)

        # Most existing adapters use every non-zero axis for image RoPE.  Some
        # architectures (Flux2) reserve an extra axis for text positions, so
        # they can opt into an explicit spatial-axis contract.
        self.spatial_axes = None if spatial_axes is None else tuple(
            axis for axis in spatial_axes if 0 <= axis < len(self.axes_dim)
        )

        # Adapters can retain an established NTK base grid while giving YaRN
        # its architecture-specific image-token base grid.  By default both
        # methods use the same grid.
        if yarn_base_patch_grid is None:
            self.yarn_base_patch_grid = self.base_patch_grid
        elif isinstance(yarn_base_patch_grid, int):
            self.yarn_base_patch_grid = (yarn_base_patch_grid, yarn_base_patch_grid)
        else:
            self.yarn_base_patch_grid = yarn_base_patch_grid
        self.yarn_base_patches = max(self.yarn_base_patch_grid)
        self.direct_yarn_positions = direct_yarn_positions

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    @staticmethod
    def _axis_token_span(axis_pos: torch.Tensor) -> float:
        flat = axis_pos.float().reshape(-1)

        if flat.numel() <= 1: return 1.0

        min_val, max_val = flat.min(), flat.max()
        span = max_val - min_val

        if span <= 0: return 1.0

        unique_vals = torch.unique(flat)

        if unique_vals.numel() <= 1: return 1.0

        step = torch.diff(unique_vals).min().item()

        if step <= 1e-6: return float(flat.numel())
        return float((span / step) + 1.0)

    def _get_mscale(self, scale_global):
        mscale_start = 0.1 * math.log(scale_global) + 1.0
        mscale_end = 1.0
        t_effective = self.current_timestep
        t_norm = 1.0 if t_effective > self.dype_start_sigma else (t_effective / self.dype_start_sigma)
        return mscale_end + (mscale_start - mscale_end) * math.pow(t_norm, self.dype_exponent)

    def _is_spatial_axis(self, axis_index: int) -> bool:
        if self.spatial_axes is None:
            return axis_index > 0
        return axis_index in self.spatial_axes

    def _base_axis_len(self, axis_index: int, n_axes: int, use_yarn_grid: bool = False) -> float:
        grid = self.yarn_base_patch_grid if use_yarn_grid else self.base_patch_grid
        fallback = self.yarn_base_patches if use_yarn_grid else self.base_patches

        if self.spatial_axes is None:
            grid_index = axis_index - 1
            if n_axes >= 3 and axis_index > 0 and grid_index < len(grid):
                return grid[grid_index]
            return fallback

        try:
            grid_index = self.spatial_axes.index(axis_index)
        except ValueError:
            return fallback

        return grid[grid_index] if grid_index < len(grid) else fallback

    @staticmethod
    def _rounded_token_span(span: float) -> float:
        """Remove insignificant FP32 error from a regular, scaled token grid."""
        rounded = round(span)
        return float(rounded) if abs(span - rounded) <= 1e-3 else span

    def _spatial_max_patches(self, pos: torch.Tensor, discrete: bool = False) -> float:
        n_axes = pos.shape[-1]

        if self.spatial_axes is None:
            if n_axes >= 3:
                spans = (self._axis_token_span(pos[..., 1]), self._axis_token_span(pos[..., 2]))
                result = max(spans)
            else:
                result = self._axis_token_span(pos)
        elif len(self.spatial_axes) > 0:
            result = max(self._axis_token_span(pos[..., axis]) for axis in self.spatial_axes)
        else:
            result = self._axis_token_span(pos)

        return self._rounded_token_span(result) if discrete else result

    def _spatial_scale_global(self, pos: torch.Tensor, use_yarn_grid: bool = False, discrete: bool = False) -> float:
        n_axes = pos.shape[-1]

        if self.spatial_axes is None:
            # Preserve the legacy calculation for all existing adapters.
            if n_axes >= 3:
                h_span = self._axis_token_span(pos[..., 1])
                w_span = self._axis_token_span(pos[..., 2])
                if discrete:
                    h_span = self._rounded_token_span(h_span)
                    w_span = self._rounded_token_span(w_span)
                h_base = self._base_axis_len(1, n_axes, use_yarn_grid)
                w_base = self._base_axis_len(2, n_axes, use_yarn_grid)
                return max(1.0, max(h_span / h_base, w_span / w_base))

            max_current_patches = self._axis_token_span(pos)
            if discrete:
                max_current_patches = self._rounded_token_span(max_current_patches)
            base_patches = self.yarn_base_patches if use_yarn_grid else self.base_patches
            return max(1.0, max_current_patches / base_patches)

        if len(self.spatial_axes) == 0:
            max_current_patches = self._axis_token_span(pos)
            if discrete:
                max_current_patches = self._rounded_token_span(max_current_patches)
            base_patches = self.yarn_base_patches if use_yarn_grid else self.base_patches
            return max(1.0, max_current_patches / base_patches)

        scales = []
        for axis_index in self.spatial_axes:
            span = self._axis_token_span(pos[..., axis_index])
            if discrete:
                span = self._rounded_token_span(span)
            scales.append(span / self._base_axis_len(axis_index, n_axes, use_yarn_grid))
        return max(1.0, max(scales))

    def _calc_vision_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        n_axes = pos.shape[-1]
        components = []

        use_discrete_span = self.spatial_axes is not None
        scale_global = self._spatial_scale_global(
            pos, use_yarn_grid=True, discrete=use_discrete_span
        )
            
        current_mscale = self._get_mscale(scale_global)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            current_patches = self._axis_token_span(axis_pos)
            if use_discrete_span:
                current_patches = self._rounded_token_span(current_patches)
            
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent, 'ntk_scale': scale_global, 'override_mscale': current_mscale}

            if self._is_spatial_axis(i):
                base_axis_len = self._base_axis_len(i, n_axes, use_yarn_grid=True)
                
                scale_local = max(1.0, current_patches / base_axis_len)
                dype_kwargs['linear_scale'] = scale_local 
                
                if scale_global > 1.0:
                    cos, sin = get_1d_dype_yarn_pos_embed(**common_kwargs, ori_max_pe_len=base_axis_len, **dype_kwargs)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            else:
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

            components.append((cos, sin))
            
        return components

    def _calc_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        n_axes = pos.shape[-1]
        components = []

        use_discrete_span = self.spatial_axes is not None
        max_current_patches = self._spatial_max_patches(pos, discrete=use_discrete_span)
        yarn_base_patches = self.yarn_base_patches
        needs_extrapolation = (max_current_patches > yarn_base_patches)

        if needs_extrapolation and self.yarn_alt_scaling:
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                current_patches = self._axis_token_span(axis_pos)
                if use_discrete_span:
                    current_patches = self._rounded_token_span(current_patches)
                base_axis_len = self._base_axis_len(i, n_axes, use_yarn_grid=True)

                if self._is_spatial_axis(i) and current_patches > base_axis_len:
                    max_pe_len = torch.tensor(current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_yarn_pos_embed(**common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=base_axis_len, **dype_kwargs, use_aggressive_mscale=True)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
                
                components.append((cos, sin))
        else:
            cos_full_spatial, sin_full_spatial = None, None
            if needs_extrapolation and not self.direct_yarn_positions:
                spatial_axis = 1 if self.spatial_axes is None else self.spatial_axes[0]
                spatial_axis_dim = self.axes_dim[spatial_axis]
                square_pos = torch.arange(0, max_current_patches, device=pos.device).float()
                max_pe_len = torch.tensor(max_current_patches, dtype=freqs_dtype, device=pos.device)
                
                common_kwargs_spatial = {'dim': spatial_axis_dim, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                cos_full_spatial, sin_full_spatial = get_1d_yarn_pos_embed(
                    **common_kwargs_spatial, pos=square_pos, max_pe_len=max_pe_len, ori_max_pe_len=yarn_base_patches, **dype_kwargs, use_aggressive_mscale=False
                )

            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                
                if self._is_spatial_axis(i) and needs_extrapolation and self.direct_yarn_positions:
                    common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                    dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}
                    max_pe_len = torch.tensor(max_current_patches, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_yarn_pos_embed(
                        **common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=yarn_base_patches,
                        **dype_kwargs, use_aggressive_mscale=False
                    )
                elif self._is_spatial_axis(i) and needs_extrapolation:
                    offset_indices = axis_pos.long() - axis_pos.long().min()
                    pos_indices = offset_indices.view(-1)
                    pos_indices = torch.clamp(pos_indices, max=cos_full_spatial.shape[0]-1)
                    
                    cos = cos_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                    sin = sin_full_spatial[pos_indices].view(*axis_pos.shape, -1)
                else:
                    common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)

                components.append((cos, sin))
            
        return components

    def _calc_ntk_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        n_axes = pos.shape[-1]
        components = []

        scale_global = self._spatial_scale_global(pos)

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            
            ntk_factor = 1.0
            if self._is_spatial_axis(i) and scale_global > 1.0:
                base_ntk = scale_global ** (axis_dim / (axis_dim - 2))
                if self.dype:
                    k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                    ntk_factor = base_ntk ** k_t
                else:
                    ntk_factor = base_ntk
                ntk_factor = max(1.0, ntk_factor)
            
            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)
            components.append((cos, sin))
        return components

    # Public Interface
    def get_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        if self.method == 'vision_yarn':
            return self._calc_vision_yarn_components(pos, freqs_dtype)
        elif self.method == 'yarn':
            return self._calc_yarn_components(pos, freqs_dtype)
        else:
            return self._calc_ntk_components(pos, freqs_dtype)
            
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class does not implement forward. Use a specific model subclass.")
