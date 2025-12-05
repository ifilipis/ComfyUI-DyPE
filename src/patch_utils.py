import math
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
    is_zimage = False

    normalized_model_type = model_type.replace("_", "").lower()

    if normalized_model_type == "nunchaku":
        is_nunchaku = True
    elif normalized_model_type == "qwen":
        is_qwen = True
    elif normalized_model_type == "zimage":
        is_zimage = True
    elif model_type == "flux":
        pass # defaults false
    else: # auto
        if hasattr(m.model, "diffusion_model"):
            dm = m.model.diffusion_model
            model_class_name = dm.__class__.__name__

            if "QwenImage" in model_class_name:
                is_qwen = True
            elif "NextDiT" in model_class_name or hasattr(dm, "rope_embedder"):
                is_zimage = True
            elif hasattr(dm, "model") and hasattr(dm.model, "pos_embed"):
                is_nunchaku = True
            elif hasattr(dm, "pe_embedder"):
                pass
            else:
                pass
        else:
            raise ValueError("The provided model is not a compatible model.")

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_zimage)

    should_patch_schedule = True
    if hasattr(m.model, "_dype_params"):
        if m.model._dype_params == new_dype_params:
            should_patch_schedule = False
        else:
            pass

    base_patch_h_tokens = None
    base_patch_w_tokens = None
    default_base_patches = (base_resolution // 8) // 2
    default_base_seq_len = default_base_patches * default_base_patches

    if is_zimage:
        axes_lens = getattr(m.model.diffusion_model, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 3:
            base_patch_h_tokens = int(axes_lens[1])
            base_patch_w_tokens = int(axes_lens[2])

    patch_size = 2 # Default Flux/Qwen
    try:
        if is_nunchaku:
            patch_size = m.model.diffusion_model.model.config.patch_size
        else:
            patch_size = m.model.diffusion_model.patch_size
    except:
        pass

    if base_patch_h_tokens is not None and base_patch_w_tokens is not None:
        derived_base_patches = max(base_patch_h_tokens, base_patch_w_tokens)
        derived_base_seq_len = base_patch_h_tokens * base_patch_w_tokens
    else:
        derived_base_patches = default_base_patches
        derived_base_seq_len = default_base_seq_len

    if enable_dype and should_patch_schedule:
        try:
            if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux) or is_qwen or is_zimage:
                latent_h, latent_w = height // 8, width // 8
                padded_h, padded_w = math.ceil(latent_h / patch_size) * patch_size, math.ceil(latent_w / patch_size) * patch_size
                image_seq_len = (padded_h // patch_size) * (padded_w // patch_size)

                base_seq_len = derived_base_seq_len
                max_seq_len = image_seq_len

                if max_seq_len <= base_seq_len:
                    dype_shift = base_shift
                else:
                    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
                    intercept = base_shift - slope * base_seq_len
                    dype_shift = image_seq_len * slope + intercept

                dype_shift = max(0.0, dype_shift)

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
        elif is_zimage:
            orig_embedder = m.model.diffusion_model.rope_embedder
            target_patch_path = "diffusion_model.rope_embedder"
        else:
            orig_embedder = m.model.diffusion_model.pe_embedder
            target_patch_path = "diffusion_model.pe_embedder"

        theta, axes_dim = orig_embedder.theta, orig_embedder.axes_dim
    except AttributeError:
        raise ValueError("The provided model is not a compatible FLUX/Qwen/Z-Image model structure.")

    embedder_cls = PosEmbedFlux
    if is_nunchaku:
        embedder_cls = PosEmbedNunchaku
    elif is_qwen:
        embedder_cls = PosEmbedQwen
    elif is_zimage:
        embedder_cls = PosEmbedZImage

    embedder_base_patches = derived_base_patches if is_zimage else None

    new_pe_embedder = embedder_cls(
        theta, axes_dim, method, yarn_alt_scaling, enable_dype,
        dype_scale, dype_exponent, base_resolution, dype_start_sigma, embedder_base_patches
    )

    m.add_object_patch(target_patch_path, new_pe_embedder)

    sigma_max = m.model.model_sampling.sigma_max.item()

    def dype_wrapper_function(model_function, args_dict):
        current_sigma = None
        timestep_tensor = args_dict.get("timestep")
        if timestep_tensor is not None and timestep_tensor.numel() > 0:
            current_sigma = timestep_tensor.flatten()[0].item()

            if sigma_max > 0:
                normalized_timestep = min(max(current_sigma / sigma_max, 0.0), 1.0)
                new_pe_embedder.set_timestep(normalized_timestep)

        input_x, c = args_dict.get("input"), args_dict.get("c", {})
        output = model_function(input_x, args_dict.get("timestep"), **c)
        return output

    m.set_model_unet_function_wrapper(dype_wrapper_function)

    return m
