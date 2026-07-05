import math
import types
import torch
import torch.nn.functional as F
import comfy
from comfy.model_patcher import ModelPatcher
from comfy import model_sampling

from .models.flux import PosEmbedFlux
from .models.nunchaku import PosEmbedNunchaku
from .models.qwen import PosEmbedQwen
from .models.zimage import PosEmbedZImage


def _flux_klein_attention(q, k, v, pe, num_txt_tokens):
    from comfy.ldm.flux.math import apply_rope

    if not isinstance(pe, tuple):
        from comfy.ldm.flux.math import attention
        return attention(q, k, v, pe=pe)

    pe_self, pe_cross = pe
    pe_txt = pe_self[:, :, :num_txt_tokens]
    pe_img_self = pe_self[:, :, num_txt_tokens:]
    pe_img_cross = pe_cross[:, :, num_txt_tokens:]

    q_txt, q_img = q[:, :, :num_txt_tokens], q[:, :, num_txt_tokens:]
    k_txt, k_img = k[:, :, :num_txt_tokens], k[:, :, num_txt_tokens:]
    v_txt, v_img = v[:, :, :num_txt_tokens], v[:, :, num_txt_tokens:]

    q_txt_r = apply_rope(q_txt, q_txt, pe_txt)[0]
    k_txt_r = apply_rope(k_txt, k_txt, pe_txt)[0]
    q_img_self = apply_rope(q_img, q_img, pe_img_self)[0]
    k_img_self = apply_rope(k_img, k_img, pe_img_self)[0]
    q_img_cross = apply_rope(q_img, q_img, pe_img_cross)[0]
    k_img_cross = apply_rope(k_img, k_img, pe_img_cross)[0]

    scale = q.shape[-1] ** -0.5
    logits_tt = torch.matmul(q_txt_r, k_txt_r.transpose(-2, -1)) * scale
    logits_ti = torch.matmul(q_txt_r, k_img_cross.transpose(-2, -1)) * scale
    logits_it = torch.matmul(q_img_cross, k_txt_r.transpose(-2, -1)) * scale
    logits_ii = torch.matmul(q_img_self, k_img_self.transpose(-2, -1)) * scale

    values = torch.cat([v_txt, v_img], dim=2)
    probs_txt = torch.softmax(torch.cat([logits_tt, logits_ti], dim=-1).float(), dim=-1).to(v.dtype)
    probs_img = torch.softmax(torch.cat([logits_it, logits_ii], dim=-1).float(), dim=-1).to(v.dtype)
    out_txt = torch.matmul(probs_txt, values)
    out_img = torch.matmul(probs_img, values)
    out = torch.cat([out_txt, out_img], dim=2)
    return out.transpose(1, 2).reshape(out.shape[0], out.shape[2], -1)


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
            if "QwenImage" in model_class_name:
                is_qwen = True
            elif hasattr(dm, "rope_embedder"):
                is_z_image = True
            elif hasattr(dm, "model") and hasattr(dm.model, "pos_embed"):
                is_nunchaku = True
        else:
            raise ValueError("The provided model is not a compatible model.")

    new_dype_params = (width, height, base_shift, max_shift, method, yarn_alt_scaling, base_resolution, dype_start_sigma, is_nunchaku, is_qwen, is_z_image)

    should_patch_schedule = True
    if hasattr(m.model, "_dype_params"):
        if m.model._dype_params == new_dype_params:
            should_patch_schedule = False

    base_patch_h_tokens = None
    base_patch_w_tokens = None
    if is_z_image:
        axes_lens = getattr(m.model.diffusion_model, "axes_lens", None)
        if isinstance(axes_lens, (list, tuple)) and len(axes_lens) >= 3:
            base_patch_h_tokens = int(axes_lens[1])
            base_patch_w_tokens = int(axes_lens[2])

    patch_size = 2
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
        derived_base_patches = (base_resolution // 8) // 2
        derived_base_seq_len = derived_base_patches * derived_base_patches

    if enable_dype and should_patch_schedule:
        try:
            if isinstance(m.model.model_sampling, model_sampling.ModelSamplingFlux) or is_qwen or is_z_image:
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

    embedder_base_patches = derived_base_patches if is_z_image else None

    new_pe_embedder = embedder_cls(
        theta, axes_dim, method, yarn_alt_scaling, enable_dype,
        dype_scale, dype_exponent, base_resolution, dype_start_sigma, embedder_base_patches
    )

    if not (is_nunchaku or is_qwen or is_z_image):
        new_pe_embedder.set_klein_cross_rope(
            max(1.0, float(height) / float(base_resolution)),
            max(1.0, float(width) / float(base_resolution)),
        )
        _patch_flux_klein_blocks(m.model.diffusion_model)
        
    m.add_object_patch(target_patch_path, new_pe_embedder)

    if is_z_image:
        base_hw_override = None
        if base_patch_h_tokens is not None and base_patch_w_tokens is not None:
            base_hw_override = (base_patch_h_tokens, base_patch_w_tokens)
        elif derived_base_patches is not None:
            base_hw_override = (derived_base_patches, derived_base_patches)

        if base_hw_override is not None:
            m.model.diffusion_model._dype_base_hw = base_hw_override

        def dype_patchify_and_embed(self, x, cap_feats, cap_mask, t, num_tokens, ref_latents=[], ref_contexts=[], siglip_feats=[], transformer_options={}):
            bsz = len(x)
            pH = pW = self.patch_size
            device = x[0].device
            cap_feats = self.cap_embedder(cap_feats)

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

            raw_scale_y = float(rope_base_resolution) / max(1.0, float(requested_hw[0]))
            raw_scale_x = float(rope_base_resolution) / max(1.0, float(requested_hw[1]))
            
            iso_scale = min(raw_scale_y, raw_scale_x)
            rope_scale_y = iso_scale
            rope_scale_x = iso_scale
            
            freq_scale_factor = 1.0 / iso_scale
            new_pe_embedder.set_scale_hint(freq_scale_factor)

            h_start = 0.0
            w_start = 0.0

            original_hw = transformer_options.get("dype_original_hw")
            if original_hw is None:
                original_hw = (H, W)

            H_tokens = math.ceil(original_hw[0] / pH)
            W_tokens = math.ceil(original_hw[1] / pW)
            
            token_stride_y = rope_scale_y
            token_stride_x = rope_scale_x

            shift_y = h_start
            shift_x = w_start
            
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

            base_img_tokens = H_tokens * W_tokens
            x_pos_ids = _build_spatial_pos_ids(bsz, base_img_tokens, W_tokens, cap_feats.shape[1], token_stride_y, token_stride_x, shift_y, shift_x, device)

            if self.pad_tokens_multiple is not None:
                pad_extra = (-x.shape[1]) % self.pad_tokens_multiple
                if pad_extra:
                    x = torch.cat((x, self.x_pad_token.to(device=x.device, dtype=x.dtype, copy=True).unsqueeze(0).repeat(x.shape[0], pad_extra, 1)), dim=1)

            if x.shape[1] != x_pos_ids.shape[1]:
                x_pos_ids = _build_spatial_pos_ids(bsz, x.shape[1], W_tokens, cap_feats.shape[1], token_stride_y, token_stride_x, shift_y, shift_x, device)

            freqs_cis = self.rope_embedder(torch.cat((cap_pos_ids, x_pos_ids), dim=1)).movedim(1, 2)

            for layer in self.context_refiner:
                cap_feats = layer(cap_feats, cap_mask, freqs_cis[:, :cap_pos_ids.shape[1]], transformer_options=transformer_options)

            padded_img_mask = None
            timestep_zero_index = None
            for layer in self.noise_refiner:
                x = layer(x, padded_img_mask, freqs_cis[:, cap_pos_ids.shape[1]:], t, timestep_zero_index=timestep_zero_index, transformer_options=transformer_options)

            padded_full_embed = torch.cat((cap_feats, x), dim=1)
            mask = None
            img_sizes = [(H, W)] * bsz
            l_effective_cap_len = [cap_feats.shape[1]] * bsz
            return padded_full_embed, mask, img_sizes, l_effective_cap_len, freqs_cis, timestep_zero_index

        m.add_object_patch(
            "diffusion_model.patchify_and_embed",
            types.MethodType(dype_patchify_and_embed, m.model.diffusion_model)
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
            c["transformer_options"] = transformer_options

        return model_function(input_x, args_dict.get("timestep"), **c)

    m.set_model_unet_function_wrapper(dype_wrapper_function)

    return m

def _patch_flux_klein_blocks(dm):
    from comfy.ldm.flux.layers import apply_mod

    def double_forward(self, img, txt, vec, pe, attn_mask=None, modulation_dims_img=None, modulation_dims_txt=None, transformer_options={}):
        if attn_mask is not None or not isinstance(pe, tuple):
            return self._dype_orig_forward(img, txt, vec, pe, attn_mask, modulation_dims_img, modulation_dims_txt, transformer_options)
        if self.modulation:
            img_mod1, img_mod2 = self.img_mod(vec)
            txt_mod1, txt_mod2 = self.txt_mod(vec)
        else:
            (img_mod1, img_mod2), (txt_mod1, txt_mod2) = vec

        transformer_patches = transformer_options.get("patches", {})
        extra_options = transformer_options.copy()

        img_modulated = self.img_norm1(img)
        img_modulated = apply_mod(img_modulated, (1 + img_mod1.scale), img_mod1.shift, modulation_dims_img)
        img_qkv = self.img_attn.qkv(img_modulated)
        del img_modulated
        img_q, img_k, img_v = img_qkv.view(img_qkv.shape[0], img_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del img_qkv
        img_q, img_k = self.img_attn.norm(img_q, img_k, img_v)

        txt_modulated = self.txt_norm1(txt)
        txt_modulated = apply_mod(txt_modulated, (1 + txt_mod1.scale), txt_mod1.shift, modulation_dims_txt)
        txt_qkv = self.txt_attn.qkv(txt_modulated)
        del txt_modulated
        txt_q, txt_k, txt_v = txt_qkv.view(txt_qkv.shape[0], txt_qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del txt_qkv
        txt_q, txt_k = self.txt_attn.norm(txt_q, txt_k, txt_v)

        q = torch.cat((txt_q, img_q), dim=2)
        del txt_q, img_q
        k = torch.cat((txt_k, img_k), dim=2)
        del txt_k, img_k
        v = torch.cat((txt_v, img_v), dim=2)
        del txt_v, img_v

        num_txt_tokens = txt.shape[1]
        extra_options["img_slice"] = [num_txt_tokens, q.shape[2]]
        if "attn1_patch" in transformer_patches:
            for p in transformer_patches["attn1_patch"]:
                out = p(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
                q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)

        attn = _flux_klein_attention(q, k, v, pe, num_txt_tokens)
        del q, k, v

        if "attn1_output_patch" in transformer_patches:
            for p in transformer_patches["attn1_output_patch"]:
                attn = p(attn, extra_options)

        txt_attn, img_attn = attn[:, :num_txt_tokens], attn[:, num_txt_tokens:]
        img += apply_mod(self.img_attn.proj(img_attn), img_mod1.gate, None, modulation_dims_img)
        del img_attn
        img += apply_mod(self.img_mlp(apply_mod(self.img_norm2(img), (1 + img_mod2.scale), img_mod2.shift, modulation_dims_img)), img_mod2.gate, None, modulation_dims_img)
        txt += apply_mod(self.txt_attn.proj(txt_attn), txt_mod1.gate, None, modulation_dims_txt)
        del txt_attn
        txt += apply_mod(self.txt_mlp(apply_mod(self.txt_norm2(txt), (1 + txt_mod2.scale), txt_mod2.shift, modulation_dims_txt)), txt_mod2.gate, None, modulation_dims_txt)
        if txt.dtype == torch.float16:
            txt = torch.nan_to_num(txt, nan=0.0, posinf=65504, neginf=-65504)
        return img, txt

    def single_forward(self, x, vec, pe, attn_mask=None, modulation_dims=None, transformer_options={}):
        if attn_mask is not None or not isinstance(pe, tuple):
            return self._dype_orig_forward(x, vec, pe, attn_mask, modulation_dims, transformer_options)
        if self.modulation:
            mod, _ = self.modulation(vec)
        else:
            mod = vec

        transformer_patches = transformer_options.get("patches", {})
        extra_options = transformer_options.copy()
        qkv, mlp = torch.split(self.linear1(apply_mod(self.pre_norm(x), (1 + mod.scale), mod.shift, modulation_dims)), [3 * self.hidden_size, self.mlp_hidden_dim_first], dim=-1)
        q, k, v = qkv.view(qkv.shape[0], qkv.shape[1], 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        del qkv
        q, k = self.norm(q, k, v)

        if "attn1_patch" in transformer_patches:
            for p in transformer_patches["attn1_patch"]:
                out = p(q, k, v, pe=pe, attn_mask=attn_mask, extra_options=extra_options)
                q, k, v, pe, attn_mask = out.get("q", q), out.get("k", k), out.get("v", v), out.get("pe", pe), out.get("attn_mask", attn_mask)

        num_txt_tokens = transformer_options["img_slice"][0]
        attn = _flux_klein_attention(q, k, v, pe, num_txt_tokens)
        del q, k, v

        if "attn1_output_patch" in transformer_patches:
            for p in transformer_patches["attn1_output_patch"]:
                attn = p(attn, extra_options)

        if self.yak_mlp:
            mlp = self.mlp_act(mlp[..., self.mlp_hidden_dim_first // 2:]) * mlp[..., :self.mlp_hidden_dim_first // 2]
        else:
            mlp = self.mlp_act(mlp)
        output = self.linear2(torch.cat((attn, mlp), 2))
        x += apply_mod(output, mod.gate, None, modulation_dims)
        if x.dtype == torch.float16:
            x = torch.nan_to_num(x, nan=0.0, posinf=65504, neginf=-65504)
        return x

    for block in list(dm.double_blocks) + list(dm.single_blocks):
        if hasattr(block, "_dype_orig_forward"):
            continue
        block._dype_orig_forward = block.forward
        if block in dm.double_blocks:
            block.forward = types.MethodType(double_forward, block)
        else:
            block.forward = types.MethodType(single_forward, block)
