import types
import types
from typing import List, Tuple

import torch


def _scaled_linspace(steps: int, target_steps: int, device: torch.device, scale: float, start: float) -> torch.Tensor:
    if target_steps < 1:
        target_steps = 1
    if steps == 1:
        base = torch.zeros(1, device=device, dtype=torch.float32)
    else:
        base = torch.linspace(0, target_steps - 1, steps=steps, device=device, dtype=torch.float32)
    return base * scale + start


def patch_nextdit_resolution_handling(model, target_width: int, target_height: int) -> None:
    """Monkey-patch NextDiT to avoid circular padding and align RoPE with the real grid."""
    if getattr(model, "_dype_resolution_patch", False):
        return

    target_latent_size = (target_height // 8, target_width // 8)
    model._dype_target_latent_size = target_latent_size

    def unpatchify(self, x: torch.Tensor, img_size: List[Tuple[int, int]], cap_size: List[int], return_tensor: bool = False):
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

    def patchify_and_embed(
        self,
        x: torch.Tensor,
        cap_feats: torch.Tensor,
        cap_mask: torch.Tensor,
        t: torch.Tensor,
        num_tokens,
        transformer_options=None,
    ):
        if transformer_options is None:
            transformer_options = {}

        bsz = x.shape[0]
        pH = pW = self.patch_size
        device = x.device

        if self.pad_tokens_multiple is not None:
            pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
            cap_feats = torch.cat(
                (
                    cap_feats,
                    self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True)
                    .unsqueeze(0)
                    .repeat(cap_feats.shape[0], pad_extra, 1),
                ),
                dim=1,
            )

        cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

        B, C, H, W = x.shape
        x = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

        rope_options = transformer_options.get("rope_options", None)
        h_scale = rope_options.get("scale_y", 1.0) if rope_options is not None else 1.0
        w_scale = rope_options.get("scale_x", 1.0) if rope_options is not None else 1.0
        h_start = rope_options.get("shift_y", 0.0) if rope_options is not None else 0.0
        w_start = rope_options.get("shift_x", 0.0) if rope_options is not None else 0.0

        H_tokens = H // pH
        W_tokens = W // pW
        target_H_tokens = max(1, target_latent_size[0] // pH)
        target_W_tokens = max(1, target_latent_size[1] // pW)

        h_positions = _scaled_linspace(H_tokens, target_H_tokens, device, h_scale, h_start)
        w_positions = _scaled_linspace(W_tokens, target_W_tokens, device, w_scale, w_start)

        x_pos_ids = torch.zeros((bsz, x.shape[1], 3), dtype=torch.float32, device=device)
        x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
        x_pos_ids[:, :, 1] = h_positions.view(-1, 1).repeat(1, W_tokens).flatten()
        x_pos_ids[:, :, 2] = w_positions.view(1, -1).repeat(H_tokens, 1).flatten()

        if self.pad_tokens_multiple is not None:
            pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
            x = torch.cat(
                (
                    x,
                    self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True)
                    .unsqueeze(0)
                    .repeat(x.shape[0], pad_extra, 1),
                ),
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
        img_sizes = [target_latent_size] * bsz
        l_effective_cap_len = [cap_feats.shape[1]] * bsz
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    def _forward(self, x, timesteps, context, num_tokens, attention_mask=None, **kwargs):
        t = 1.0 - timesteps
        cap_feats = context
        cap_mask = attention_mask
        bs, c, h, w = x.shape

        t = self.t_embedder(t * self.time_scale, dtype=x.dtype)
        adaln_input = t

        cap_feats = self.cap_embedder(cap_feats)

        transformer_options = kwargs.get("transformer_options", {})
        x_is_tensor = isinstance(x, torch.Tensor)
        x, mask, img_size, cap_size, freqs_cis = self.patchify_and_embed(
            x, cap_feats, cap_mask, t, num_tokens, transformer_options=transformer_options
        )
        freqs_cis = freqs_cis.to(x.device)

        for layer in self.layers:
            x = layer(x, mask, freqs_cis, adaln_input, transformer_options=transformer_options)

        x = self.final_layer(x, adaln_input)
        x = self.unpatchify(x, img_size, cap_size, return_tensor=x_is_tensor)[:, :, :h, :w]

        return -x

    model.unpatchify = types.MethodType(unpatchify, model)
    model.patchify_and_embed = types.MethodType(patchify_and_embed, model)
    model._forward = types.MethodType(_forward, model)
    model._dype_resolution_patch = True
