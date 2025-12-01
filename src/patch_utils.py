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

    patch_size = 2 # Default Flux/Qwen
    base_patch_grid = None
    try:
        if is_nunchaku:
            patch_size = m.model.diffusion_model.model.config.patch_size
        else:
            patch_size = m.model.diffusion_model.patch_size
    except Exception:
        pass

    if is_z_image:
        try:
            axes_lens = getattr(m.model.diffusion_model, "axes_lens", None)
            if axes_lens and len(axes_lens) >= 3:
                base_patch_grid = (int(axes_lens[-2]), int(axes_lens[-1]))
        except Exception:
            base_patch_grid = None

    if base_patch_grid is None:
        base_latent = max(1, base_resolution // 8)
        base_patch_tokens = max(1, base_latent // patch_size)
        base_patch_grid = (base_patch_tokens, base_patch_tokens)

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_z_image, base_patch_grid)
    
    should_patch_schedule = True
    if hasattr(m.model, "_dype_params"):
        if m.model._dype_params == new_dype_params:
            should_patch_schedule = False

    if enable_dype and should_patch_schedule:
        try:
            if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux) or is_qwen or is_z_image:
                latent_h, latent_w = height // 8, width // 8
                padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
                image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)
                
                base_seq_len = base_patch_grid[0] * base_patch_grid[1]
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
        dype_scale, dype_exponent, base_resolution, dype_start_sigma, base_patch_grid
    )
        
    m.add_object_patch(target_patch_path, new_pe_embedder)

    if is_z_image and enable_dype:
        diffusion_model = m.model.diffusion_model

        def dype_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, transformer_options={}):
            bsz = len(x)
            pH = pW = self.patch_size
            device = x.device if isinstance(x, torch.Tensor) else x[0].device

            pad_tokens_multiple = self.pad_tokens_multiple
            if pad_tokens_multiple is not None:
                cap_pad_extra = (-cap_feats.shape[1]) % pad_tokens_multiple
                if cap_pad_extra:
                    cap_pad = self.cap_pad_token.to(device=cap_feats.device, dtype=cap_feats.dtype, copy=True).unsqueeze(0).repeat(cap_feats.shape[0], cap_pad_extra, 1)
                    cap_feats = torch.cat((cap_feats, cap_pad), dim=1)

            cap_pos_ids = torch.zeros(bsz, cap_feats.shape[1], 3, dtype=torch.float32, device=device)
            cap_pos_ids[:, :, 0] = torch.arange(cap_feats.shape[1], dtype=torch.float32, device=device) + 1.0

            B, C, H, W = x.shape
            x = self.x_embedder(x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 3, 5, 1).flatten(3).flatten(1, 2))

            rope_options = transformer_options.get("rope_options", None)
            h_scale = rope_options.get("scale_y", 1.0) if rope_options is not None else 1.0
            w_scale = rope_options.get("scale_x", 1.0) if rope_options is not None else 1.0
            h_start = rope_options.get("shift_y", 0.0) if rope_options is not None else 0.0
            w_start = rope_options.get("shift_x", 0.0) if rope_options is not None else 0.0

            H_tokens, W_tokens = H // pH, W // pW
            x_pos_ids = torch.zeros((bsz, x.shape[1], 3), dtype=torch.float32, device=device)
            x_pos_ids[:, :, 0] = cap_feats.shape[1] + 1
            x_pos_ids[:, :, 1] = (torch.arange(H_tokens, dtype=torch.float32, device=device) * h_scale + h_start).view(-1, 1).repeat(1, W_tokens).flatten()
            x_pos_ids[:, :, 2] = (torch.arange(W_tokens, dtype=torch.float32, device=device) * w_scale + w_start).view(1, -1).repeat(H_tokens, 1).flatten()

            padded_img_mask = None
            if pad_tokens_multiple is not None:
                pad_extra = (-x.shape[1]) % pad_tokens_multiple
                if pad_extra:
                    pad_token = self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)
                    x = torch.cat((x, pad_token), dim=1)
                    x_pos_ids = torch.nn.functional.pad(x_pos_ids, (0, 0, 0, pad_extra))
                    padded_img_mask = torch.zeros((bsz, x.shape[1]), device=device, dtype=torch.bool)
                    padded_img_mask[:, -pad_extra:] = True

            freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=transformer_options)

            for layer in self.noise_refiner:
                x = layer(x, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1]:], t, transformer_options=transformer_options)

            padded_full_embed = torch.cat((cap_feats, x), dim=1)
            mask = None
            img_sizes = [(H, W)] * bsz
            l_effective_cap_len = [cap_feats.shape[1]] * bsz
            return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis

        m.add_object_patch("diffusion_model.patchify_and_embed", types.MethodType(dype_patchify_and_embed, diffusion_model))
    
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
