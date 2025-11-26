<a id="readme-top"></a>

<div align="center">
  <h1 align="center">ComfyUI-DyPE</h1>

<img src="https://github.com/user-attachments/assets/4f11966b-86f7-4bdb-acd4-ada6135db2f8" alt="ComfyUI-DyPE Banner" width="70%">

  
  <p align="center">
    A ComfyUI custom node that implements <strong>DyPE (Dynamic Position Extrapolation)</strong>, enabling FLUX-based models to generate ultra-high-resolution images (4K and beyond) with exceptional coherence and detail.
    <br />
    <br />
    <a href="https://github.com/wildminder/ComfyUI-DyPE/issues/new?labels=bug&template=bug-report---.md">Report Bug</a>
    ·
    <a href="https://github.com/wildminder/ComfyUI-DyPE/issues/new?labels=enhancement&template=feature-request---.md">Request Feature</a>
  </p>
</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![Forks][forks-shield]][forks-url]

</div>

<br>

## About The Project

DyPE is a training-free method that allows pre-trained DiT like FLUX to generate images at resolutions far beyond their training data, with no additional sampling cost.

It works by taking advantage of the spectral progression inherent to the diffusion process. By dynamically adjusting the model's positional encodings at each step, DyPE matches their frequency spectrum with the current stage of the generative process—focusing on low-frequency structures early on and resolving high-frequency details in later steps. This prevents the repeating artifacts and structural degradation typically seen when pushing models beyond their native resolution.

<div align="center">

  <img alt="ComfyUI-DyPE example workflow" width="70%" src="https://github.com/user-attachments/assets/31f5d254-68a7-435b-8e1f-c4e636d4f3c2" />
      <p><sub><i>A simple, single-node integration to patch your FLUX model for high-resolution generation.</i></sub></p>
  </div>


  
This node provides a seamless, "plug-and-play" integration of DyPE into any FLUX-based workflow.

**✨ Key Features:**
*   **High-Resolution Generation:** Push FLUX models to 4096x4096 and beyond.
*   **Single-Node Integration:** Simply place the `DyPE for FLUX` node after your model loader to patch the model. No complex workflow changes required.
*   **Full Compatibility:** Works seamlessly with your existing ComfyUI workflows, samplers, schedulers, and other optimization nodes like Self-Attention or quantization.
*   **Fine-Grained Control:** Exposes key DyPE hyperparameters, allowing you to tune the algorithm's strength and behavior for optimal results at different target resolutions.
*   **Zero Inference Overhead:** DyPE's adjustments happen on-the-fly with negligible performance impact.

<div align="center">
<img alt="Node" width="70%" src="https://github.com/user-attachments/assets/ef900ba2-e019-496a-89f6-abd92c857029" />
</div>


<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Getting Started

The easiest way to install is via **ComfyUI Manager**. Search for `ComfyUI-DyPE` and click "Install".

Alternatively, to install manually:

1.  **Clone the Repository:**

    Navigate to your `ComfyUI/custom_nodes/` directory and clone this repository:
    ```sh
    git clone https://github.com/wildminder/ComfyUI-DyPE.git
    ```
2. **Start/Restart ComfyUI:**
   Launch ComfyUI. No further dependency installation is required.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## 🛠️ Usage

Using the node is straightforward and designed for minimal workflow disruption.

1.  **Load Your FLUX Model:** Use a standard `Load Checkpoint` node to load your FLUX model (e.g., `FLUX.1-Krea-dev`).
2.  **Add the DyPE Node:** Add the `DyPE for FLUX` node to your graph (found under `model_patches/unet`).
3.  **Connect the Model:** Connect the `MODEL` output from your loader to the `model` input of the DyPE node.
4.  **Set Resolution:** Set the `width` and `height` on the DyPE node to match the resolution of your `Empty Latent Image`. (See note below).
5.  **Connect to KSampler:** Use the `MODEL` output from the DyPE node as the input for your `KSampler`.
6.  **Generate!** That's it. Your workflow is now DyPE-enabled.

> [!NOTE]
> This node specifically patches the **diffusion model (UNet)**. It does not modify the CLIP or VAE models. It is designed exclusively for **FLUX-based** architectures.

### Node Inputs

#### 1. Method Selection
*   **`method`**:
    *   **`vision_yarn`:** A novel variant designed specifically for aspect-ratio robustness. It decouples structure from texture: low frequencies (shapes) are scaled to fit your canvas aspect ratio, while high frequencies (details) are scaled uniformly to prevent stretching or deformation of features like faces. It also uses a dynamic attention schedule to ensure sharpness without "speckle" artifacts.
    *   **`yarn`:** The standard YaRN method. Good general performance but can struggle with extreme aspect ratios.
    *   **`ntk`:** Neural Tangent Kernel scaling. Very stable but tends to be softer/blurrier at high resolutions.
    *   **`base`:** No positional interpolation (standard FLUX behavior).

#### 2. Scaling Options
*   **`yarn_alt_scaling`** (Only affects `yarn` method):
    *   **Anisotropic (High-Res):** Scales Height and Width independently. Can cause geometric stretching if the aspect ratio differs significantly from the training data.
    *   **Isotropic (Stable Default):** Scales both dimensions based on the largest axis. .
    *   *Note: `vision_yarn` automatically handles this balance internally, so this switch is ignored when `vision_yarn` is selected.*

#### 3. Dynamic Control
*   **`enable_dype`**: Enables or disables the **dynamic, time-aware** component of DyPE.
    *   **Enabled (True):** Both the noise schedule and RoPE will be dynamically adjusted throughout sampling. This is the full DyPE algorithm.
    *   **Disabled (False):** The node will only apply the dynamic noise schedule shift. The RoPE will use static extrapolation.
*   **`dype_scale`**: (λs) Controls the "magnitude" of the DyPE modulation.
    *   Determines how much the frequency bands are shifted.
    *   Default is `2.0`, which works well for most high-res upscaling.
*   **`dype_exponent`**: (λt) Controls the "strength" of the dynamic effect over time.
    *   `2.0`: Recommended for **4K+** resolutions. Aggressive schedule that transitions quickly to clean up artifacts.
    *   `1.0`: Good starting point for **~2K-3K** resolutions.
    *   `0.5`: Gentler schedule for resolutions just above native 1K.

#### 4. Advanced Noise Scheduling
*   **`base_shift` / `max_shift`**: These parameters control the Noise Schedule Shift (`mu`).
    *   In this implementation, `max_shift` (Default 1.15) acts as the target shift for any resolution larger than the base (256 tokens).
    *   While modifying this changes the signal-to-noise ratio, it does not drastically alter the output composition.

*   **`width` / `height`**: These inputs are used to calculate the Noise Schedule interpolation.
    *   Since the shift caps at `max_shift`, exact matching with your generation resolution is **not strictly required** for the algorithm to function, provided the values are above the minimum threshold (1024x1024).
    *   The **RoPE scaling** (which prevents artifacts) is calculated dynamically from the actual tensors during generation, so it ignores these inputs and will always adapt perfectly to your actual resolution.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## ❗ Important Notes & Best Practices

> [!IMPORTANT]
> **Limitations at Extreme Resolutions (4K)**
> While DyPE significantly extends the capabilities of FLUX, enabling generation far beyond its training distribution, **generating perfectly clean 4096x4096 images is still a limitation of the base model itself.**
>
> Even with DyPE, you are pushing a model trained on ~1 megapixel to generate 16 megapixels. You may still encounter composition errors, repeating patterns, or minor artifacts at these extreme scales. DyPE mitigates these issues, but it cannot add semantic understanding that the model does not possess.

> [!TIP]
> **Experimentation is Required**
> There is no single "magic setting" that works for every prompt and every resolution. To achieve the best results:
> *   **Test different Methods:** Start with `vision_yarn`, but try `yarn` (Isotropic) if you encounter issues.
> *   **Adjust `dype_exponent`:** This is your main knob for balancing sharpness vs. artifacts.
> *   **Tune Scaling:** If using legacy `yarn`, toggle between Anisotropic and Isotropic scaling to see which suits your subject better.

<p align="center">══════════════════════════════════</p>

Beyond the code, I believe in the power of community and continuous learning. I invite you to join the 'TokenDiff AI News' and 'TokenDiff Community Hub'

<p align="center">══════════════════════════════════</p>

Beyond the code, I believe in the power of community and continuous learning. I invite you to join the 'TokenDiff AI News' and 'TokenDiff Community Hub'

<table border="0" align="center" cellspacing="10" cellpadding="0">
  <tr>
    <td align="center" valign="top">
      <h4>TokenDiff AI News</h4>
      <a href="https://t.me/TokenDiff">
        <img width="40%" alt="tokendiff-tg-qw" src="https://github.com/user-attachments/assets/e29f6b3c-52e5-4150-8088-12163a2e1e78" />
      </a>
      <p><sub>🗞️ AI for every home, creativity for every mind!</sub></p>
    </td>
    <td align="center" valign="top">
      <h4>TokenDiff Community Hub</h4>
      <a href="https://t.me/TokenDiff_hub">
        <img width="40%" alt="token_hub-tg-qr" src="https://github.com/user-attachments/assets/da544121-5f5b-4e3d-a3ef-02272535929e" />
      </a>
      <p><sub>💬 questions, help, and thoughtful discussion.</sub> </p>
    </td>
  </tr>
</table>

<p align="center">══════════════════════════════════</p>

## ⚠️ Known Issues
*   **FLUX Only:** This implementation is highly specific to the architecture of the FLUX model and will not work on standard U-Net models (like SD 1.5/SDXL) or other Diffusion Transformers.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License
The original DyPE project is patent pending. For commercial use or licensing inquiries regarding the underlying method, please contact the [original authors](mailto:noam.issachar@mail.huji.ac.il).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

*   **Noam Issachar, Guy Yariv, and the co-authors** for their groundbreaking research and for open-sourcing the [DyPE](https://github.com/guyyariv/DyPE) project.
*   **The ComfyUI team** for creating such a powerful and extensible platform for diffusion model research and creativity.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[stars-shield]: https://img.shields.io/github/stars/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[stars-url]: https://github.com/wildminder/ComfyUI-DyPE/stargazers
[issues-shield]: https://img.shields.io/github/issues/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[issues-url]: https://github.com/wildminder/ComfyUI-DyPE/issues
[forks-shield]: https://img.shields.io/github/forks/wildminder/ComfyUI-DyPE.svg?style=for-the-badge
[forks-url]: https://github.com/wildminder/ComfyUI-DyPE/network/members
