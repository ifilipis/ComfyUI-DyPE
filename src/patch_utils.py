import math
import types
from typing import Dict, Tuple

import torch
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling

from .models.flux import PosEmbedFlux
from .models.nunchaku import PosEmbedNunchaku
from .models.qwen import PosEmbedQwen
from .models.z_image import PosEmbedZImage


def _get_transformer_options(model) -> Dict:
    try:
        if hasattr(model, "transformer_options"):
            opts = model.transformer_options or {}
        else:
            opts = getattr(model.model_config, "transformer_options", {}) or {}
    except Exception:
        opts = {}
    return opts if isinstance(opts, dict) else {}


def _compute_token_spans(
    transformer_options: Dict,
    height_tokens: int,
    width_tokens: int,
    patch_size: int,
) -> Tuple[int, int]:
    dype_original_hw = transformer_options.get("dype_original_hw")
    if isinstance(dype_original_hw, (list, tuple)) and len(dype_original_hw) >= 2:
        orig_h, orig_w = float(dype_original_hw[0]), float(dype_original_hw[1])
    else:
        orig_h, orig_w = float(height_tokens), float(width_tokens)

    stride = transformer_options.get("token_stride") or transformer_options.get("token_strides")
    if isinstance(stride, (list, tuple)) and len(stride) >= 2:
        stride_h, stride_w = int(stride[0]), int(stride[1])
    elif isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h = stride_w = patch_size

    pad_multiple = transformer_options.get("pad_tokens_multiple")
    if not isinstance(pad_multiple, int) or pad_multiple <= 0:
        pad_multiple = patch_size

    stride_h = max(1, stride_h)
    stride_w = max(1, stride_w)

    h_tokens = math.ceil(orig_h / stride_h)
    w_tokens = math.ceil(orig_w / stride_w)

    h_tokens = math.ceil(h_tokens / pad_multiple) * pad_multiple
    w_tokens = math.ceil(w_tokens / pad_multiple) * pad_multiple

    return int(h_tokens), int(w_tokens)


def _patch_z_image_patchify(dm):
    if getattr(dm, "_dype_patchify_patched", False):
        return

    def patched_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, transformer_options={}):
        transformer_options = transformer_options or {}
        pH = pW = self.patch_size

        h_scale = 1.0
        w_scale = 1.0
        h_start = 0.0
        w_start = 0.0

        rope_options = dict(transformer_options.get("rope_options", {}) or {})

        B, C, H, W = x.shape
        base_transformer_opts = _get_transformer_options(self)
        token_spans = _compute_token_spans(transformer_options or base_transformer_opts, H // pH, W // pW, pH)

        base_grid_h = None
        base_grid_w = None

        axes_lens = getattr(self, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 3:
            base_grid_h, base_grid_w = axes_lens[1], axes_lens[2]
        elif hasattr(self, "rope_embedder") and hasattr(self.rope_embedder, "axes_lens"):
            rope_axes = getattr(self.rope_embedder, "axes_lens", None)
            if isinstance(rope_axes, (list, tuple)) and len(rope_axes) >= 3:
                base_grid_h, base_grid_w = rope_axes[1], rope_axes[2]

        if (base_grid_h is None or base_grid_w is None):
            base_res = transformer_options.get("base_resolution") or base_transformer_opts.get("base_resolution")
            if isinstance(base_res, int) and base_res > 0:
                base_grid_h = base_grid_w = (base_res // 8) // pH

        if base_grid_h:
            h_scale = base_grid_h / max(1, token_spans[0])
        if base_grid_w:
            w_scale = base_grid_w / max(1, token_spans[1])

        tile_offsets = transformer_options.get("tile_offsets") or transformer_options.get("tile_offset")
        if isinstance(tile_offsets, (list, tuple)) and len(tile_offsets) >= 2:
            h_start += float(tile_offsets[0])
            w_start += float(tile_offsets[1])
        else:
            h_start += float(transformer_options.get("tile_offset_y", 0.0))
            w_start += float(transformer_options.get("tile_offset_x", 0.0))

        rope_options.update({
            "scale_y": h_scale,
            "scale_x": w_scale,
            "shift_y": h_start,
            "shift_x": w_start,
        })

        updated_transformer_options = dict(transformer_options)
        updated_transformer_options["rope_options"] = rope_options

        if self.pad_tokens_multiple is not None:
            pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
            cap_feats = torch.cat((cap_feats, self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0).repeat(cap_feats.shape[0], pad_extra, 1)), dim=1)

        cap_pos_ids = torch.zeros(B, cap_feats.shape[1], 3, dtype=torch.float32, device=x.device)
        cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=x.device) + 1.0

        x = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

        H_tokens, W_tokens = token_spans
        x_pos_ids = torch.zeros((B, x.shape[1], 3), dtype=torch.float32, device=x.device)
        x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
        x_pos_ids[:, :, 1] = (torch.arange(H_tokens, dtype=torch.float32, device=x.device) * h_scale + h_start).view(-1, 1).repeat(1, W_tokens).flatten()[:x.shape[1]]
        x_pos_ids[:, :, 2] = (torch.arange(W_tokens, dtype=torch.float32, device=x.device) * w_scale + w_start).view(1, -1).repeat(H_tokens, 1).flatten()[:x.shape[1]]

        if self.pad_tokens_multiple is not None:
            pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
            if pad_extra:
                x = torch.cat((x, self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)), dim=1)
                total_tokens = x.shape[1]
                w_tokens_adj = max(W_tokens, math.ceil(total_tokens / max(1, H_tokens)))
                H_tokens = math.ceil(total_tokens / max(1, w_tokens_adj))
                W_tokens = w_tokens_adj
                x_pos_ids = torch.zeros((B, total_tokens, 3), dtype=torch.float32, device=x.device)
                x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
                x_pos_ids[:, :, 1] = (torch.arange(H_tokens, dtype=torch.float32, device=x.device) * h_scale + h_start).view(-1, 1).repeat(1, W_tokens).flatten()[:total_tokens]
                x_pos_ids[:, :, 2] = (torch.arange(W_tokens, dtype=torch.float32, device=x.device) * w_scale + w_start).view(1, -1).repeat(H_tokens, 1).flatten()[:total_tokens]
            else:
                x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))

        freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

        for layer in self.context_refiner:
            cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=updated_transformer_options)

        padded_img_mask = None
        for layer in self.noise_refiner:
            x = layer(x, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1]:], t, transformer_options=updated_transformer_options)

        padded_full_embed = torch.cat((cap_feats, x), dim=1)
        mask = None
        img_sizes = [(H, W)] * B
        l_effective_cap_len = [cap_feats.shape[1]] * B
        return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

    dm.patchify_and_embed = types.MethodType(patched_patchify_and_embed, dm)
    dm._dype_patchify_patched = True


def apply_dype_to_model(model: ModelPatcher, model_type: str, width: int, height: int, method: str, yarn_alt_scaling: bool, enable_dype: bool, dype_scale: float, dype_exponent: float, base_shift: float, max_shift: float, base_resolution: int = 1024, dype_start_sigma: float = 1.0) -> ModelPatcher:
    m = model.clone()

    is_nunchaku = False
    is_qwen = False
    is_z_image = False

    if model_type == "nunchaku":
        is_nunchaku = True
    elif model_type == "qwen":
        is_qwen = True
    elif model_type == "z_image":
        is_z_image = True
    elif model_type == "flux":
        pass
    else: # auto
        if hasattr(m.model, "diffusion_model"):
            dm = m.model.diffusion_model
            model_class_name = dm.__class__.__name__

            # ToDo: add normal logging
            if "QwenImage" in model_class_name:
                is_qwen = True
                # print("[DyPE] Auto-detected Qwen Image model.")
            elif hasattr(dm, "rope_embedder"):
                is_z_image = True
                # print("[DyPE] Auto-detected Z-Image / NextDiT model.")
            elif hasattr(dm, "model") and hasattr(dm.model, "pos_embed"):
                is_nunchaku = True
                # print("[DyPE] Auto-detected Nunchaku Flux model.")
            elif hasattr(dm, "pe_embedder"):
                # print("[DyPE] Auto-detected Standard Flux model.")
                pass
            else:
                # print("[DyPE] Warning: Could not auto-detect model type. Assuming Standard Flux.")
                pass
        else:
            raise ValueError("The provided model is not a compatible model.")

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_z_image)
    
    should_patch_schedule = True
    if hasattr(m.model, "_dype_params"):
        if m.model._dype_params == new_dype_params:
            should_patch_schedule = False

    if is_z_image:
        try:
            _patch_z_image_patchify(m.model.diffusion_model)
        except Exception:
            pass

    if enable_dype and should_patch_schedule:
        patch_size = 2 # Default Flux/Qwen
        try:
            if is_nunchaku:
                patch_size = m.model.diffusion_model.model.config.patch_size
            else:
                patch_size = m.model.diffusion_model.patch_size
        except:
            pass

        try:
            if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux) or is_qwen or is_z_image:
                transformer_options = _get_transformer_options(m.model)
                latent_h, latent_w = height // 8, width // 8
                h_tokens, w_tokens = _compute_token_spans(transformer_options, latent_h, latent_w, patch_size)
                image_seq_len = h_tokens * w_tokens

                base_patches = transformer_options.get("base_patches_override") or (base_resolution // 8) // 2
                if isinstance(base_patches, (list, tuple)):
                    base_patches = base_patches[0]
                base_patches = int(base_patches)
                base_seq_len = base_patches * base_patches
                max_seq_len = image_seq_len

                if max_seq_len <= base_seq_len:
                    dype_shift = base_shift
                else:
                    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                    intercept = base_shift - slope * base_seq_len
                    dype_shift = image_seq_len * slope + intercept
                
                dype_shift = max(0.0, dype_shift)
                # print(f"[DyPE DEBUG] Calculated dype_shift (mu): {dype_shift:.4f} for resolution {width}x{height} (Base: {base_resolution})")

                class DypeModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST):
                    pass

                new_model_sampler = DypeModelSamplingFlux(m.model.model_config)
                new_model_sampler.set_parameters(shift=dype_shift)
                
                m.add_object_patch("model_sampling", new_model_sampler)
                m.model._dype_params = new_dype_params
        except:
            pass

    elif not enable_dype:
        if hasattr(m.model, "_dype_params"):
            class DefaultModelSamplingFlux(model_sampling.ModelSamplingFlux, model_sampling.CONST): pass
            default_sampler = DefaultModelSamplingFlux(m.model.model_config)
            m.add_object_patch("model_sampling", default_sampler)
            del m.model._dype_params

    try:
        if is_nunchaku:
            orig_embedder = m.model.diffusion_model.model.pos_embed
            target_patch_path = "diffusion_model.model.pos_embed"
        elif is_z_image:
            orig_embedder = m.model.diffusion_model.rope_embedder
            target_patch_path = "diffusion_model.rope_embedder"
        else:
            orig_embedder = m.model.diffusion_model.pe_embedder
            target_patch_path = "diffusion_model.pe_embedder"

        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX/Qwen model structure.")

    embedder_cls = PosEmbedFlux
    if is_nunchaku:
        embedder_cls = PosEmbedNunchaku
    elif is_qwen:
        embedder_cls = PosEmbedQwen
    elif is_z_image:
        embedder_cls = PosEmbedZImage

    new_pe_embedder = embedder_cls(
        theta, axes_dim, method, yarn_alt_scaling, enable_dype, 
        dype_scale, dype_exponent, base_resolution, dype_start_sigma
    )
        
    m.add_object_patch(target_patch_path, new_pe_embedder)
    
    sigma_max = m.model.model_sampling.sigma_max.item()
    
    def dype_wrapper_function(model_function, args_dict):
        timestep_tensor = args_dict.get("timestep")
        if timestep_tensor is not None and timestep_tensor.numel() > 0:
            current_sigma = timestep_tensor.flatten()[0].item()
            
            if sigma_max > 0:
                normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                new_pe_embedder.set_timestep(normalized_timestep)
        
        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)

    return m
