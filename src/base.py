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
        self.current_grid_scale = None

    def set_timestep(self, timestep: float):
        self.current_timestep = timestep

    def set_grid_hw(self, grid_hw: tuple[int, int] | None, base_hw: tuple[int, int] | None, grid_scale: tuple[float, float] | None = None):
        if grid_hw is None:
            self.current_grid_hw = None
            self.current_base_hw_override = None
            self.current_grid_scale = None
            return

        self.current_grid_hw = (int(grid_hw[0]), int(grid_hw[1]))
        if base_hw is not None:
            self.current_base_hw_override = (int(base_hw[0]), int(base_hw[1]))
        if grid_scale is not None:
            self.current_grid_scale = (float(grid_scale[0]), float(grid_scale[1]))
        else:
            self.current_grid_scale = None

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
        base_len = float(self._base_len_for_axis(axis_index))

        if axis_index > 0 and self.current_grid_scale is not None:
            axis_offset = axis_index - 1
            if axis_offset < len(self.current_grid_scale):
                scale = self.current_grid_scale[axis_offset]
                base_len = base_len / max(scale, 1e-8)

        return base_len

    def _axis_grid_effective_len(self, axis_index: int) -> float:
        if self.current_grid_hw is None or axis_index == 0:
            return -1.0

        if axis_index == 1:
            tokens = self.current_grid_hw[0]
            scale = self.current_grid_scale[0] if self.current_grid_scale is not None else 1.0
        elif axis_index == 2:
            tokens = self.current_grid_hw[1]
            scale = self.current_grid_scale[1] if self.current_grid_scale is not None else 1.0
        else:
            tokens = self.current_grid_hw[axis_index - 1]
            scale = self.current_grid_scale[axis_index - 1] if self.current_grid_scale is not None else 1.0

        return max(1.0, tokens * scale)

    def _axis_current_len(self, pos: torch.Tensor, axis_index: int) -> float:
        grid_len = self._axis_grid_effective_len(axis_index)
        if grid_len > 0:
            return grid_len
        return float(pos.max().item() - pos.min().item() + 1.0)

    def _calc_vision_yarn_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        """
        Calculates raw (cos, sin) pairs using DyPE Vision YaRN (Decoupled + Quadratic Aggressive).
        Returns a list of (cos, sin) tuples per axis.
        """
        n_axes = pos.shape[-1]
        components = []
        
        if pos.shape[-1] >= 3 and self.current_grid_hw is not None:
            effective_h = self._axis_grid_effective_len(1)
            effective_w = self._axis_grid_effective_len(2)
            base_h = self._scaled_base_len_for_axis(1)
            base_w = self._scaled_base_len_for_axis(2)
            scale_h = effective_h / base_h
            scale_w = effective_w / base_w
            scale_global = max(scale_h, scale_w)
        else:
            if pos.shape[-1] >= 3:
                h_span = int(pos[..., 1].max().item() - pos[..., 1].min().item() + 1)
                w_span = int(pos[..., 2].max().item() - pos[..., 2].min().item() + 1)
                base_h = self._scaled_base_len_for_axis(1)
                base_w = self._scaled_base_len_for_axis(2)
                scale_global = max(h_span / base_h, w_span / base_w)
            else:
                max_current_patches = int(pos.max().item() - pos.min().item() + 1)
                scale_global = max_current_patches / self.base_patches
            
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
            current_patches = self._axis_current_len(axis_pos, i)
            
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
                scale_local = current_patches / axis_base_len

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
        
        if pos.shape[-1] >= 3 and self.current_grid_hw is not None:
            effective_h = self._axis_grid_effective_len(1)
            effective_w = self._axis_grid_effective_len(2)
            base_h = self._scaled_base_len_for_axis(1)
            base_w = self._scaled_base_len_for_axis(2)
            max_base = max(base_h, base_w)
            max_current_patches = max(effective_h, effective_w)
        else:
            if pos.shape[-1] >= 3:
                h_span = int(pos[..., 1].max().item() - pos[..., 1].min().item() + 1)
                w_span = int(pos[..., 2].max().item() - pos[..., 2].min().item() + 1)
                base_h = self._scaled_base_len_for_axis(1)
                base_w = self._scaled_base_len_for_axis(2)
                max_base = max(base_h, base_w)
                max_current_patches = max(h_span, w_span)
            else:
                max_base = self.base_patches
                max_current_patches = int(pos.max().item() - pos.min().item() + 1)

        needs_extrapolation = (max_current_patches > max_base)

        force_isotropic = self.theta < 1000.0
        use_anisotropic = self.yarn_alt_scaling and not force_isotropic

        if needs_extrapolation and use_anisotropic:
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]
                common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                current_patches_on_axis = self._axis_current_len(axis_pos, i)
                axis_base_len = self._scaled_base_len_for_axis(i)
                if i > 0 and current_patches_on_axis > axis_base_len:
                    max_pe_len = torch.tensor(current_patches_on_axis, dtype=freqs_dtype, device=pos.device)
                    cos, sin = get_1d_yarn_pos_embed(**common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=axis_base_len, **dype_kwargs, use_aggressive_mscale=True)
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
                
                components.append((cos, sin))
        else:
            for i in range(n_axes):
                axis_pos = pos[..., i]
                axis_dim = self.axes_dim[i]

                if i > 0 and needs_extrapolation:
                    current_patches_on_axis = self._axis_current_len(axis_pos, i)
                    axis_base_len = self._scaled_base_len_for_axis(i)

                    max_pe_len = torch.tensor(current_patches_on_axis, dtype=freqs_dtype, device=pos.device)
                    common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}
                    dype_kwargs = {'dype': self.dype, 'current_timestep': self.current_timestep, 'dype_scale': self.dype_scale, 'dype_exponent': self.dype_exponent}

                    cos, sin = get_1d_yarn_pos_embed(
                        **common_kwargs, max_pe_len=max_pe_len, ori_max_pe_len=axis_base_len, **dype_kwargs, use_aggressive_mscale=False
                    )
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

        for i in range(n_axes):
            axis_pos = pos[..., i]
            axis_dim = self.axes_dim[i]
            common_kwargs = {'dim': axis_dim, 'pos': axis_pos, 'theta': self.theta, 'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype}

            ntk_factor = 1.0
            if i > 0:
                base_len = self._scaled_base_len_for_axis(i)
                scale_local = (self._axis_current_len(axis_pos, i)) / base_len
                base_ntk = scale_local ** (axis_dim / (axis_dim - 2))
                if self.dype:
                    k_t = self.dype_scale * (self.current_timestep ** self.dype_exponent)
                    ntk_factor = base_ntk ** k_t
                else:
                    ntk_factor = base_ntk

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
