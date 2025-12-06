import torch
from ..base import DyPEBasePosEmbed

class PosEmbedZImage(DyPEBasePosEmbed):
    """
    DyPE Implementation for Z-Image / NextDiT models.

    Output Format matches `EmbedND`: (B, 1, L, D/2, 2, 2)
    """
    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pos_full = ids.float()
        token_spans = [self._axis_range(pos_full[..., i]) for i in range(pos_full.shape[-1])]

        # Remap RoPE features to the native base grid while preserving full-resolution
        # token coordinates for DyPE range calculations.
        if self.base_hw is not None and len(token_spans) >= 3:
            pos_features = pos_full.clone()
            h_scale = token_spans[1] / max(self.base_hw[0], 1e-6)
            w_scale = token_spans[2] / max(self.base_hw[1], 1e-6)

            pos_features[..., 1] = pos_full[..., 1] / max(h_scale, 1e-6)
            pos_features[..., 2] = pos_full[..., 2] / max(w_scale, 1e-6)

            feature_spans = [token_spans[0], float(self.base_hw[0]), float(self.base_hw[1])]
        else:
            pos_features = pos_full
            feature_spans = token_spans

        freqs_dtype = torch.bfloat16 if pos_features.device.type == 'cuda' else torch.float32
        components = self.get_components(pos_features, freqs_dtype, feature_spans, base_pos=pos_full)

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
