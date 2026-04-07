# State-of-the-Art Image Editing & Enhancement — Implementation Report

## For: Local AI Langchain Platform (RTX 4060 8GB VRAM, 32GB RAM, Windows 11)

---

## What You Already Have vs What's Missing

### Already Implemented (39 operations)

| Category | Operations | Implementation |
|---|---|---|
| **Classical adjust** | Brightness, contrast, saturation, sharpness, color temp, hue, gamma, shadows/highlights, clarity, vibrance, auto levels, auto white balance | Pillow + OpenCV + numpy |
| **Filters** | Blur, sharpen (unsharp mask), denoise (fastNlMeans), vignette, grain, grayscale, sepia, invert, emboss, skin smooth (bilateral) | Pillow + OpenCV |
| **Transform** | Crop, resize, rotate, flip, auto crop, straighten | Pillow |
| **AI enhance** | Background removal (rembg/BiRefNet), face restoration (GFPGAN), upscale (RealESRGAN), auto enhance, portrait bokeh (MiDaS depth) | rembg + gfpgan + realesrgan + torch |
| **AI edit** | InstructPix2Pix, FLUX Kontext (text-guided editing) | diffusers |
| **Presets** | Vivid, cinematic, vintage, B&W dramatic, portrait, landscape | Chained classical ops |

### Missing — High Impact (Should Implement)

| Algorithm | Why It Matters | Library | VRAM/CPU | Difficulty |
|---|---|---|---|---|
| **CLAHE** (Contrast Limited Adaptive Histogram Equalization) | Far better than global auto_levels for local contrast recovery. Standard in medical/satellite imaging. | OpenCV `cv2.createCLAHE()` | CPU only, instant | Trivial |
| **Wavelet denoising** | Multi-scale noise removal with fine-grained threshold control. Preserves detail better than bilateral filter for certain noise types. | scikit-image `denoise_wavelet()` | CPU, ~100ms | Easy |
| **Total variation denoising** | Preserves edges while removing noise. Better for piecewise-smooth images (graphics, illustrations). | scikit-image `denoise_tv_chambolle()` | CPU, ~200ms | Easy |
| **Non-local means (color)** | Already have OpenCV fastNlMeans — but should expose strength/search window controls. | OpenCV (already using) | CPU, ~500ms | Already done, needs UI |
| **Wiener/Richardson-Lucy deconvolution** | Recovers blur from known/estimated PSF. Useful for motion blur, defocus. | scikit-image `wiener()`, `richardson_lucy()` | CPU, ~300ms | Medium |
| **LaMa inpainting** | SOTA non-diffusion inpainting. Excellent for object removal — much faster than diffusion inpaint. ~200MB model. | torch, ONNX | ~1GB VRAM or CPU | Medium |
| **Retinexformer / LYT-Net** | SOTA low-light enhancement. Dramatically better than gamma adjustment for dark photos. | torch | ~2GB VRAM | Medium |
| **NAFNet** | SOTA general restoration (denoise + deblur + dejpeg). Single model handles mixed degradations. Simple architecture, fast. | torch, BasicSR | ~1GB VRAM | Medium |
| **SwinIR** | Best quality super-resolution (better than RealESRGAN for clean images). Also does JPEG artifact removal. | torch, BasicSR | ~2GB VRAM | Medium |
| **Color grading with 3D LUTs** | Professional color grading. Ship 5-10 `.cube` LUT files for cinematic looks. | numpy + trilinear interpolation | CPU only, instant | Easy |
| **HDR tone mapping** | Reinhard, Drago, Mantiuk tone mapping for HDR images or to create HDR-like effect from SDR. | OpenCV `cv2.createTonemapReinhard()` | CPU only, instant | Easy |
| **Perspective correction** | 4-point transform for straightening buildings, documents. | OpenCV `cv2.getPerspectiveTransform()` | CPU only, instant | Easy (already in code as function, needs UI) |
| **Chromatic aberration correction** | Removes color fringing on edges. Common in phone photos. | OpenCV channel shift + alignment | CPU only, ~50ms | Easy |
| **Lens distortion correction** | Barrel/pincushion distortion fix. | OpenCV `cv2.undistort()` | CPU only, instant | Easy |
| **Object-aware segmentation** | SAM 2 or Grounding DINO for click-to-select objects, then edit only that region. | torch, transformers | ~2GB VRAM | Hard |

### Missing — Medium Impact (Nice to Have)

| Algorithm | Why | Library | Notes |
|---|---|---|---|
| **Frequency separation** (advanced) | Pro retouching technique: edit texture and color independently. | numpy FFT | CPU, manual workflow |
| **Focus stacking** | Combine multiple images with different focus planes. | OpenCV Laplacian pyramid | CPU, multi-image input needed |
| **Panorama stitching** | Stitch multiple images into panorama. | OpenCV `cv2.Stitcher` | CPU, multi-image |
| **Image registration/alignment** | Align two images of same scene (before/after, HDR bracketing). | OpenCV ORB + homography | CPU |
| **Morphological operations** | Erosion, dilation, opening, closing — useful for mask refinement. | OpenCV | CPU, instant |
| **Color transfer** | Transfer color palette from one image to another. | numpy LAB color space matching | CPU, instant |
| **Dodge & burn** | Localized brightness painting (like Photoshop). | Pillow blend with masks | CPU, needs brush UI |

### Missing — Low Priority (Research Grade)

| Algorithm | Why Not Yet | Notes |
|---|---|---|
| **All-in-one restoration (PromptIR, AirNet)** | Swiss army knife but specialist models beat it per-task | Watch for maturity |
| **Diffusion-based restoration** | Slow, unstable, needs heavy VRAM | Keep for semantic edits only |
| **Neural style transfer** | Mostly a novelty at this point, IP-Adapter is better | Already have InstructPix2Pix |
| **CodeFormer** | Duplicate of GFPGAN niche, adds complexity | GFPGAN is sufficient |

---

## Recommended Implementation Priority

### Phase 1: Quick Wins (CPU only, 1-2 hours each)

These add professional-grade tools with zero new dependencies:

1. **CLAHE** — `cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))` → apply to L channel of LAB
2. **HDR tone mapping** — `cv2.createTonemapReinhard(gamma=1.0, intensity=0, light_adapt=0.8)`
3. **3D LUT color grading** — parse `.cube` files, trilinear interpolation, ship 5-10 presets
4. **Wavelet denoising** — `skimage.restoration.denoise_wavelet(image, method='BayesShrink')`
5. **Total variation denoising** — `skimage.restoration.denoise_tv_chambolle(image, weight=0.1)`
6. **Lens distortion correction** — `cv2.undistort()` with barrel/pincushion slider
7. **Chromatic aberration fix** — shift R/B channels by 1-2px, align to green
8. **Perspective correction** — already have the function, add UI with 4-point selector

### Phase 2: AI Enhancement Models (~4 hours each, need model downloads)

These are the biggest quality jumps:

9. **LaMa inpainting** — ONNX model (~200MB), paint mask → fill with coherent texture
10. **NAFNet** — general restoration, ONNX export from BasicSR
11. **SwinIR** — quality super-resolution + JPEG artifact removal
12. **Retinexformer or LYT-Net** — low-light enhancement (dramatic improvement over gamma)

### Phase 3: Semantic Editing (complex, high impact)

13. **SAM 2 click-to-segment** — click on object → get mask → edit only that region
14. **Grounding DINO + SAM 2** — text-to-mask ("select the car") → targeted editing

---

## Library Stack Recommendation

### Already Installed (keep using)
```
Pillow          — image I/O, basic transforms
OpenCV          — filtering, denoising, morphology, color spaces
numpy           — array ops, color math
torch           — neural model inference
diffusers       — text-guided editing, inpainting
rembg           — background removal
gfpgan          — face restoration
realesrgan      — super-resolution
```

### Should Add
```
scikit-image    — wavelet/TV denoising, deconvolution, feature extraction (already installed)
kornia          — differentiable CV ops inside PyTorch (useful for batch processing)
onnxruntime     — fast cross-platform inference for exported models (already installed)
basicsr         — restoration model zoo (SwinIR, NAFNet, ESRGAN) (already installed)
```

### For Future
```
segment-anything — SAM 2 for object segmentation
groundingdino    — text-to-bbox detection
lama-cleaner     — LaMa inpainting wrapper
openvino         — Intel GPU/NPU acceleration (already in optional deps)
```

---

## Architecture Recommendation (from research)

```
Stage 1: Fast Analysis (CPU, <100ms)
├── Estimate brightness, noise level, blur, saturation
├── Detect faces (for face-specific tools)
├── Classify image type (photo/illustration/document)
└── Suggest appropriate tools

Stage 2: Route by Task
├── Low-light → Retinexformer / LYT-Net
├── Generic denoise/deblur → NAFNet / Restormer
├── Super-resolution → Real-ESRGAN (practical) / SwinIR (quality)
├── Face cleanup → GFPGAN / CodeFormer
├── Background removal → BiRefNet / RMBG 2.0
├── Object removal → LaMa (fast) / Diffusers inpaint (semantic)
├── Color grading → 3D LUT + CLAHE + tone mapping
└── Semantic editing → InstructPix2Pix / FLUX Kontext

Stage 3: Deploy Intelligently
├── ONNX Runtime for cross-platform inference
├── OpenVINO for Intel laptops
├── DirectML for AMD GPUs
└── ncnn Vulkan for packaged desktop tools
```

---

## Key Insight from Research

> "State of the art now has two heads: one chases maximum perceptual quality, the other chases deployable quality. Laptop work lives in the second head, with occasional raids into the first."

For your app: **classical algorithms for speed, neural models for quality, diffusion only for semantic edits**. This three-tier approach matches your 8GB VRAM constraint perfectly.

---

## What Makes Your App Better Than Luminar Neo

| Feature | Luminar Neo | Your App (with additions) |
|---|---|---|
| Classical adjustments | Basic sliders | 15+ adjustment tools with professional algorithms (CLAHE, wavelet denoise, etc.) |
| AI Enhancement | Cloud-dependent for some | Fully local, 7 AI operations |
| Background removal | Built-in | BiRefNet (SOTA, 6 models to choose from) |
| Face restoration | Basic | GFPGAN (research-grade) |
| Super-resolution | Basic | RealESRGAN + SwinIR option |
| Text-guided editing | None | InstructPix2Pix + FLUX Kontext |
| Low-light enhancement | Built-in Relight | Retinexformer (NTIRE award-winning) |
| LUT color grading | Built-in Mood | 3D LUT with custom .cube files |
| Object removal | GenErase (cloud) | LaMa (local, fast) + diffusion inpaint |
| Privacy | Cloud processing | 100% local, zero cloud |
| Undo/redo | Limited | Full non-destructive history |
| Cost | $99/year subscription | Free, open source |
