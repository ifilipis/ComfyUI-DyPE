import math
from types import MethodType
from typing import Tuple, List

import torch
import torch.nn.functional as F


def _compute_token_grid(resolution: Tuple[int, int], patch_size: int) -> Tuple[int, int, int, int]:
    h, w = resolution
    h_tokens = math.ceil(h / patch_size)
    w_tokens = math.ceil(w / patch_size)
    return h_tokens, w_tokens, h_tokens * patch_size, w_tokens * patch_size


def _z_image_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, transformer_options={}):
    bsz = len(x)
    pH = pW = self.patch_size
    device = x.device if isinstance(x, torch.Tensor) else x[0].device

    if self.pad_tokens_multiple is not None:
        pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
        cap_feats = torch.cat(
            (cap_feats, self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0).repeat(cap_feats.shape[0], pad_extra, 1)),
            dim=1,
        )

    cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
    cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

    B, C, H, W = x.shape
    h_tokens, w_tokens, padded_h, padded_w = _compute_token_grid((H, W), pH)

    if padded_h != H or padded_w != W:
        x = F.pad(x, (0, padded_w - W, 0, padded_h - H))

    x = self.x_embedder(
        x.view(B, C, padded_h // pH, pH, padded_w // pW, pW)
        .permute(0, 2, 4, 3, 5, 1)
        .flatten(3)
        .flatten(1, 2)
    )

    rope_options = transformer_options.get("rope_options", None)
    h_scale = 1.0
    w_scale = 1.0
    h_start = 0.0
    w_start = 0.0
    if rope_options is not None:
        h_scale = rope_options.get("scale_y", 1.0)
        w_scale = rope_options.get("scale_x", 1.0)
        h_start = rope_options.get("shift_y", 0.0)
        w_start = rope_options.get("shift_x", 0.0)

    h_positions = torch.linspace(0, max(H / pH - 1, 0.0), steps=h_tokens, device=device, dtype=torch.float32)
    w_positions = torch.linspace(0, max(W / pW - 1, 0.0), steps=w_tokens, device=device, dtype=torch.float32)

    x_pos_ids = torch.zeros((bsz, x.shape[1], 3), dtype=torch.float32, device=device)
    x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
    x_pos_ids[:, :, 1] = (h_positions * h_scale + h_start).view(-1, 1).repeat(1, w_tokens).flatten()
    x_pos_ids[:, :, 2] = (w_positions * w_scale + w_start).view(1, -1).repeat(h_tokens, 1).flatten()

    if self.pad_tokens_multiple is not None:
        pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
        x = torch.cat(
            (x, self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)),
            dim=1,
        )
        x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))

    freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

    for layer in self.context_refiner:
        cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=transformer_options)

    padded_img_mask = None
    for layer in self.noise_refiner:
        x = layer(x, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1] :], t, transformer_options=transformer_options)

    padded_full_embed = torch.cat((cap_feats, x), dim=1)
    mask = None
    img_sizes = [(padded_h, padded_w)] * bsz
    l_effective_cap_len = [cap_feats.shape[1]] * bsz
    return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis


def _z_image_unpatchify(self, x: torch.Tensor, img_size: List[Tuple[int, int]], cap_size: List[int], return_tensor=False):
    pH = pW = self.patch_size
    imgs = []
    for i in range(x.size(0)):
        H, W = img_size[i]
        begin = cap_size[i]
        end = begin + (H // pH) * (W // pW)
        imgs.append(
            x[i][begin:end]
            .view(H // pH, W // pW, pH, pW, self.out_channels)
            .permute(4, 0, 2, 1, 3)
            .flatten(3, 4)
            .flatten(1, 2)
        )

    if return_tensor:
        imgs = torch.stack(imgs, dim=0)
    return imgs


def _z_image_forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
    t = 1.0 - timesteps
    cap_feats = context
    cap_mask = attention_mask
    _, _, h, w = x.shape

    t = self.t_embedder(t * self.time_scale, dtype=x.dtype)
    adaln_input = t

    cap_feats = self.cap_embedder(cap_feats)

    transformer_options = kwargs.get("transformer_options", {})
    x_is_tensor = isinstance(x, torch.Tensor)
    x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(x, cap_feats, cap_mask, t, num_tokens, transformer_options=transformer_options)
    freqs_cis = freqs_cis.to(x.device)

    for layer in self.layers:
        x = layer(x, mask, freqs_cis, adaln_input, transformer_options=transformer_options)

    x = self.final_layer(x, adaln_input)
    x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]

    return -x


def patch_z_image_methods(patcher):
    diffusion_model = patcher.model.diffusion_model

    patcher.add_object_patch("diffusion_model.patchify_and_embed", MethodType(_z_image_patchify_and_embed, diffusion_model))
    patcher.add_object_patch("diffusion_model.unpatchify", MethodType(_z_image_unpatchify, diffusion_model))
    patcher.add_object_patch("diffusion_model._forward", MethodType(_z_image_forward, diffusion_model))
