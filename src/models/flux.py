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
        self.yarn_base_patches = math.ceil(self.base_resolution / 16)

    def set_scale_hint(self, scale: float):
        self.external_scale_hint = scale

    def _blend_to_full_scale(self) -> float:
        t_norm = min(self.current_timestep / self.dype_start_sigma, 1.0)
        return 1.0 - math.pow(t_norm, self.dype_exponent)

    def _scale_rope_grid(self, pos: torch.Tensor) -> torch.Tensor:
        pos = pos.clone()
        pos[..., 1:3] *= self.external_scale_hint
        return pos

    def _resize_rope_grid(self, pos: torch.Tensor) -> torch.Tensor:
        if not self.dype:
            return pos

        blend = self._blend_to_full_scale()
        if blend <= 0.001:
            return pos

        scaled = self.external_scale_hint
        pos = pos.clone()
        pos[..., 1:3] += (pos[..., 1:3] / scaled - pos[..., 1:3]) * blend
        return pos

    def get_components(self, pos: torch.Tensor, freqs_dtype: torch.dtype):
        spatial_spans = [round(self._axis_token_span(pos[..., i])) for i in (1, 2)]
        yarn_scale = max(1.0, max(spatial_spans) / self.yarn_base_patches)
        ntk_scale = max(1.0, max(
            spatial_spans[i] / self.base_patch_grid[i] for i in range(2)
        ))
        components = []

        for i, axis_dim in enumerate(self.axes_dim):
            common_kwargs = {
                'dim': axis_dim, 'pos': pos[..., i], 'theta': self.theta,
                'use_real': True, 'repeat_interleave_real': True, 'freqs_dtype': freqs_dtype,
            }

            if i not in (1, 2):
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            elif self.method == 'vision_yarn':
                if yarn_scale > 1.0:
                    cos, sin = get_1d_dype_yarn_pos_embed(
                        **common_kwargs,
                        linear_scale=max(1.0, spatial_spans[i - 1] / self.yarn_base_patches),
                        ntk_scale=yarn_scale,
                        ori_max_pe_len=self.yarn_base_patches,
                        dype=self.dype,
                        current_timestep=self.current_timestep,
                        dype_scale=self.dype_scale,
                        dype_exponent=self.dype_exponent,
                        override_mscale=self._get_mscale(yarn_scale),
                    )
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            elif self.method == 'yarn':
                scale = spatial_spans[i - 1] if self.yarn_alt_scaling else max(spatial_spans)
                if scale > self.yarn_base_patches:
                    cos, sin = get_1d_yarn_pos_embed(
                        **common_kwargs,
                        max_pe_len=torch.tensor(scale, dtype=freqs_dtype, device=pos.device),
                        ori_max_pe_len=self.yarn_base_patches,
                        dype=self.dype,
                        current_timestep=self.current_timestep,
                        dype_scale=self.dype_scale,
                        dype_exponent=self.dype_exponent,
                        use_aggressive_mscale=self.yarn_alt_scaling,
                    )
                else:
                    cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=1.0)
            else:
                ntk_factor = 1.0
                if ntk_scale > 1.0:
                    ntk_factor = ntk_scale ** (axis_dim / (axis_dim - 2))
                    if self.dype:
                        ntk_factor **= self.dype_scale * self.current_timestep ** self.dype_exponent
                cos, sin = get_1d_ntk_pos_embed(**common_kwargs, ntk_factor=ntk_factor)

            components.append((cos, sin))

        return components

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos = self._resize_rope_grid(self._scale_rope_grid(ids.float()))
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
