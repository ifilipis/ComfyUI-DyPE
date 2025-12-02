import math
import types
import torch
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling

from .models.flux import PosEmbedFlux
from .models.nunchaku import PosEmbedNunchaku
from .models.qwen import PosEmbedQwen
from .models.z_image import PosEmbedZImage

def apply_dype_to_model(model: ModelPatcher, model_type: str, width: int, height: int, method: str, yarn_alt_scaling: bool, enable_dype: bool, dype_scale: float, dype_exponent: float, base_shift: float, max_shift: float, base_resolution: int = 1024, dype_start_sigma: float = 1.0) -> ModelPatcher:
    m = model.clone()

    is_nunchaku = False
    is_qwen = False
    is_z_image = False

    base_patches_override = None

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

    if is_z_image:
        axes_lens = getattr(m.model.diffusion_model, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 2:
            base_patches_override = axes_lens[1]
        elif isinstance(axes_lens, torch.Tensor) and axes_lens.numel() >= 2:
            base_patches_override = axes_lens.flatten()[1].item()

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_z_image)
    
    should_patch_schedule = True
    if hasattr(m.model, "_dype_params"):
        if m.model._dype_params == new_dype_params:
            should_patch_schedule = False

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
                pad_multiple = getattr(m.model.diffusion_model, "pad_tokens_multiple", None)

                latent_h, latent_w = height // 8, width // 8
                h_tokens = math.ceil(latent_h / patch_size)
                w_tokens = math.ceil(latent_w / patch_size)
                total_tokens = h_tokens * w_tokens

                if pad_multiple is not None and pad_multiple > 0:
                    pad_extra = (-total_tokens) % pad_multiple
                    if pad_extra:
                        total_tokens += pad_extra
                        w_tokens = math.ceil(total_tokens / h_tokens)

                h_span = h_tokens
                w_span = w_tokens
                image_seq_len = h_span * w_span

                base_patches = base_patches_override if base_patches_override is not None else (base_resolution // 8) // 2
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
        dype_scale, dype_exponent, base_resolution, dype_start_sigma, base_patches_override
    )

    m.add_object_patch(target_patch_path, new_pe_embedder)

    if is_z_image:
        def _build_spatial_pos_ids(batch: int, total_len: int, width_tokens: int, cap_len: int, stride_y: float, stride_x: float, h_start: float, w_start: float, device: torch.device):
            base_pos = torch.arange(total_len, device=device, dtype=torch.float32)
            y = torch.div(base_pos, width_tokens, rounding_mode='floor') * stride_y + h_start
            x = torch.remainder(base_pos, width_tokens) * stride_x + w_start

            pos = torch.stack([
                torch.full_like(base_pos, cap_len + 1),
                y,
                x
            ], dim=-1)
            return pos.unsqueeze(0).repeat(batch, 1, 1)

        def patched_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, transformer_options={}):
            bsz = len(x)
            pH = pW = self.patch_size
            device = x[0].device

            if self.pad_tokens_multiple is not None:
                pad_extra = (-cap_feats.shape[1]) % self.pad_tokens_multiple
                if pad_extra:
                    cap_pad = self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0)
                    cap_feats = torch.cat((cap_feats, cap_pad.repeat(cap_feats.shape[0], pad_extra, 1)), dim=1)

            cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
            cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

            B, C, H, W = x.shape
            x = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

            requested_hw = transformer_options.get("dype_requested_hw", (height, width))
            rope_base_resolution = transformer_options.get("dype_base_resolution", base_resolution)

            base_grid = None
            axes_meta = transformer_options.get("dype_axes_lens", getattr(self, "axes_lens", None))
            if axes_meta is not None:
                if isinstance(axes_meta, torch.Tensor):
                    axes_values = axes_meta.flatten().tolist()
                else:
                    axes_values = list(axes_meta)

                if len(axes_values) >= 3:
                    base_grid = (float(axes_values[-2]), float(axes_values[-1]))
                elif len(axes_values) >= 2:
                    base_grid = (float(axes_values[0]), float(axes_values[1]))

            if base_grid is None:
                rope_base_shape = getattr(self.rope_embedder, "base_shape", None)
                rope_base_hw = getattr(self.rope_embedder, "base_hw", None)
                rope_grid = rope_base_shape or rope_base_hw
                if rope_grid is not None and isinstance(rope_grid, (list, tuple)) and len(rope_grid) >= 2:
                    base_grid = (float(rope_grid[0]), float(rope_grid[1]))

            if base_grid is None and hasattr(self.rope_embedder, "base_resolution"):
                base_res = getattr(self.rope_embedder, "base_resolution")
                if isinstance(base_res, (list, tuple)) and len(base_res) >= 2:
                    base_grid = (float(base_res[0]), float(base_res[1]))

            if base_grid is None:
                base_grid = (float(rope_base_resolution), float(rope_base_resolution))

            rope_scale_y = float(base_grid[0]) / max(1.0, float(requested_hw[0]))
            rope_scale_x = float(base_grid[1]) / max(1.0, float(requested_hw[1]))

            h_start = 0.0
            w_start = 0.0

            tile_offset = transformer_options.get("tile_offset")
            if isinstance(tile_offset, (list, tuple)) and len(tile_offset) >= 2:
                h_start = float(tile_offset[0])
                w_start = float(tile_offset[1])

            h_start = float(transformer_options.get("tile_offset_y", h_start))
            w_start = float(transformer_options.get("tile_offset_x", w_start))
            h_start = float(transformer_options.get("h_start", h_start))
            w_start = float(transformer_options.get("w_start", w_start))

            sampler_meta = transformer_options.get("sampler_meta", {})
            if isinstance(sampler_meta, dict):
                h_start = float(sampler_meta.get("h_start", sampler_meta.get("tile_offset_y", h_start)))
                w_start = float(sampler_meta.get("w_start", sampler_meta.get("tile_offset_x", w_start)))

            original_hw = transformer_options.get("dype_original_hw")
            if original_hw is None:
                original_hw = (H, W)

            H_tokens = math.ceil(original_hw[0] / pH)
            W_tokens = math.ceil(original_hw[1] / pW)
            token_stride_y = (original_hw[0] / max(1, H_tokens)) * rope_scale_y
            token_stride_x = (original_hw[1] / max(1, W_tokens)) * rope_scale_x
            shift_y = h_start * (original_hw[0] / max(1, H_tokens))
            shift_x = w_start * (original_hw[1] / max(1, W_tokens))
            base_img_tokens = H_tokens * W_tokens

            x_pos_ids = _build_spatial_pos_ids(bsz, base_img_tokens, W_tokens, cap_feats.shape[1], token_stride_y, token_stride_x, shift_y, shift_x, device)

            if self.pad_tokens_multiple is not None:
                pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
                if pad_extra:
                    x = torch.cat((x, self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)), dim=1)

            total_img_tokens = x.shape[1]
            if total_img_tokens != x_pos_ids.shape[1]:
                padded_W_tokens = max(W_tokens, math.ceil(total_img_tokens / max(1, H_tokens)))
                x_pos_ids = _build_spatial_pos_ids(bsz, total_img_tokens, padded_W_tokens, cap_feats.shape[1], token_stride_y, token_stride_x, shift_y, shift_x, device)

            if hasattr(self.rope_embedder, "set_axis_strides"):
                try:
                    self.rope_embedder.set_axis_strides((None, token_stride_y, token_stride_x))
                except Exception:
                    pass

            freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=transformer_options)

            padded_img_mask = None
            for layer in self.noise_refiner:
                x = layer(x, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1]:], t, transformer_options=transformer_options)

            padded_full_embed = torch.cat((cap_feats, x), dim=1)
            mask = None
            img_sizes = [(H, W)] * bsz
            l_effective_cap_len = [cap_feats.shape[1]] * bsz
            return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

        m.add_object_patch(
            "diffusion_model.patchify_and_embed",
            types.MethodType(patched_patchify_and_embed, m.model.diffusion_model)
        )

    sigma_max = m.model.model_sampling.sigma_max.item()
    
    def dype_wrapper_function(model_function, args_dict):
        timestep_tensor = args_dict.get("timestep")
        if timestep_tensor is not None and timestep_tensor.numel() > 0:
            current_sigma = timestep_tensor.flatten()[0].item()
            
            if sigma_max > 0:
                normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                new_pe_embedder.set_timestep(normalized_timestep)
        
        input_x, c = args_dict.get("input"), args_dict.get("c", {})

        if is_z_image and isinstance(input_x, torch.Tensor) and input_x.dim() >= 4:
            c = dict(c)
            transformer_options = dict(c.get("transformer_options", {}))
            transformer_options["dype_original_hw"] = (input_x.shape[-2], input_x.shape[-1])
            transformer_options["dype_requested_hw"] = (height, width)
            transformer_options["dype_base_resolution"] = base_resolution
            if hasattr(m.model.diffusion_model, "axes_lens"):
                transformer_options["dype_axes_lens"] = m.model.diffusion_model.axes_lens
            c["transformer_options"] = transformer_options

        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)

    return m
