import torch
from ..base import DyPEBasePosEmbed

class PosEmbedFlux(DyPEBasePosEmbed):
    """
    DyPE Implementation for Standard ComfyUI Flux Models.
    Output Format: Rotation Matrix (concatenated) -> (B, 1, L, D)
    """
    klein_cross_rope = False
    cross_scale_h = 1.0
    cross_scale_w = 1.0

    def set_klein_cross_rope(self, scale_h: float, scale_w: float):
        self.klein_cross_rope = True
        self.cross_scale_h = max(float(scale_h), 1e-6)
        self.cross_scale_w = max(float(scale_w), 1e-6)

    def _embed(self, ids: torch.Tensor) -> torch.Tensor:
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

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pe_self = self._embed(ids)
        if not self.klein_cross_rope:
            return pe_self

        ids_cross = ids.float().clone()
        if ids_cross.shape[-1] > 1:
            ids_cross[..., 1] /= self.cross_scale_h
        if ids_cross.shape[-1] > 2:
            ids_cross[..., 2] /= self.cross_scale_w
        return pe_self, self._embed(ids_cross)
