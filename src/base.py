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
    def __init__(self, theta: int, axes_dim: list[int], method: str = 'yarn', yarn_alt_scaling: bool = False, dype: bool = True, dype_scale: float = 2.0, dype_exponent: float = 2.0, base_resolution: int = 1024, dype_start_sigma: float = 1.0, base_patches: int | tuple[int, int] | None = None):
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
        
        # Dynamic Base Patches: configurable per-model to align with native patch grids.
        # Flux/Qwen default: (Resolution // 8) // 2
        self.base_hw = None
        if isinstance(base_patches, (tuple, list)) and len(base_patches) == 2:
            self.base_hw = (int(base_patches[0]), int(base_patches[1]))
            self.base_patches = max(self.base_hw)
        else:
            self.base_patches = base_patches if base_patches is not None else (self.base_resolution // 8) // 2

        self.current_grid_hw = None
        self.current_base_hw_override = None
        self.start_grid_scale = None

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def _time_norm(self) -> float:
        if self.current_timestep > self.dype_start_sigma:
            return 1.0

        return self.current_timestep / self.dype_start_sigma if self.dype_start_sigma > 0 else 1.0

    def set_grid_hw(self, grid_hw: tuple[int, int] | None, base_hw: tuple[int, int] | None, grid_scale: tuple[float, float] | None = None):
        if grid_hw is None:
            self.current_grid_hw = None
            self.current_base_hw_override = None
            return

        self.current_grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        if base_hw is not None:
            self.current_base_hw_override = (int(base_hw[0]), int(base_hw[1]))
        if grid_scale is not None:
            if self.start_grid_scale is None:
                self.start_grid_scale = (float(grid_scale[0]), float(grid_scale[1]))

    def _base_len_for_axis(self, axis_index: int) -> int:
        base_hw = self.current_base_hw_override if self.current_base_hw_override is not None else self.base_hw
        if base_hw is None:
            return self.base_patches
        if axis_index == 1:
            return max(1, base_hw[0])
        if axis_index == 2:
            return max(1, base_hw[1])
        return self.base_patches

    def _scaled_base_len_for_axis(self, axis_index: int) -> float:
        return float(self._base_len_for_axis(axis_index))

    def _axis_scale_ratio(self, axis_index: int) -> float:
        if self.start_grid_scale is None or axis_index == 0:
            return 1.0

        axis_offset = min(axis_index - 1, len(self.start_grid_scale) - 1)
        start_scale = max(self.start_grid_scale[axis_offset], 1e-8)

        t_norm = self._time_norm()

        return 1.0 + (start_scale - 1.0) * (1.0 - t_norm)

    def _current_scale_value(self) -> float:
        if self.start_grid_scale is None:
            return 1.0

        ratios = []
        for idx, start in enumerate(self.start_grid_scale):
            axis_idx = idx + 1
            ratios.append(self._axis_scale_ratio(axis_idx))

        return max(ratios) if ratios else 1.0

    def _calc_vision_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Calculates raw (cos, sin) pairs using DyPE Vision YaRN (Decoupled + Quadratic Aggressive).
        Returns a list of (cos, sin) tuples per axis.
        """
        n_axes = pos.shape[-1]
        components = []
        
        scale_global = self._current_scale_value()
            
        mscale_start = 0.1 * math.log(scale_global) + 1.0
        mscale_end = 1.0

        t_norm = self._time_norm()

        t_factor = math.pow(t_norm, self.dype_exponent)
        current_mscale = mscale_end + (mscale_start - mscale_end) * t_factor

        # Low Theta Heuristic (Z-Image / Lumina)
        force_isotropic = self.theta < 1000.0

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            scale_local = self._axis_scale_ratio(i)
            
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
                axis_base_len = self._scaled_base_len_for_axis(i)

                # Apply Low Theta protection
                if force_isotropic:
                    dype_kwargs['linear_scale'] = 1.0
                else:
                    dype_kwargs['linear_scale'] = scale_local

                cos, sin = get_1d_dype_yarn_pos_embed(
                    **common_kwargs,
                    ori_max_pe_len=axis_base_len,
                    **dype_kwargs
                )
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
        
        scale_value = self._current_scale_value()

        needs_extrapolation = scale_value > 1.0

        force_isotropic = self.theta < 1000.0
        use_anisotropic = self.yarn_alt_scaling and not force_isotropic

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

            axis_base_len = self._scaled_base_len_for_axis(i)
            axis_scale = self._axis_scale_ratio(i)
            current_len = axis_base_len * axis_scale

            if i > 0 and needs_extrapolation:
                max_pe_len = torch.tensor(current_len, dtype=freqs_dtype, device=pos.device)
                cos, sin = get_1d_yarn_pos_embed(
                    **common_kwargs,
                    max_pe_len=max_pe_len,
                    ori_max_pe_len=axis_base_len,
                    **dype_kwargs,
                    use_aggressive_mscale=use_anisotropic
                )
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

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}

            ntk_factor = 1.0
            if i > 0:
                scale_local = self._axis_scale_ratio(i)
                blend = (1.0 - self._time_norm()) ** self.dype_exponent
                if self.dype:
                    blend = min(1.0, self.dype_scale * blend)

                ntk_factor = 1.0 + (scale_local - 1.0) * blend

            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)
            components.append((cos, sin))
            
        return components

    def get_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        if self.method == 'vision_yarn':
            return self._calc_vision_yarn_components(pos, freqs_dtype)
        elif self.method == 'yarn':
            return self._calc_yarn_components(pos, freqs_dtype)
        elif self.method == 'ntk':
            return self._calc_ntk_components(pos, freqs_dtype)
        else: # 'base'
            return self._calc_base_components(pos, freqs_dtype)

    def _calc_base_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Returns unscaled (cos, sin) tuples per axis (no NTK/YaRN scaling).
        """
        n_axes = pos.shape[-1]
        components = []

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
            cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            components.append((cos, sin))

        return components
            
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Base class does not implement forward. Use a specific model subclass.")
