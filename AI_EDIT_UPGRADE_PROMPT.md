# AI Image Edit Upgrade — Implementation Prompt

## System Specs
- GPU: NVIDIA RTX 4060 Laptop, 8.6GB VRAM
- RAM: 32GB
- OS: Windows 11
- Python: 3.11, PyTorch with CUDA
- diffusers: 0.37.1 (FluxKontextPipeline available)
- HF Token: in .env as HF_TOKEN

## Current State
- InstructPix2Pix (SD 1.5, `timbrooks/instruct-pix2pix`) produces poor quality results
- Pipeline code in: `src/local_ai_platform/images/ai_enhance.py`
- Editor dispatches via: `src/local_ai_platform/images/editor.py`
- Flutter UI panel: `flutter_client/lib/pages/editor_page.dart` (AI Edit section)
- Models cached in: `data/models/` and HuggingFace cache

## Task: Implement 3 AI editing approaches, user-selectable

### 1. FLUX.1 Kontext [dev] GGUF (Primary — Best Quality)

**Model**: `black-forest-labs/FLUX.1-Kontext-dev` (gated, HF token required)
**GGUF**: `QuantStack/FLUX.1-Kontext-dev-GGUF` — download Q4_K_S (~6.8GB)
**Pipeline**: `FluxKontextPipeline` from diffusers 0.37+

**Implementation requirements**:
- Download the GGUF quantized model on first use (auto-download with progress)
- Load with `GGUFQuantizationConfig` from diffusers
- Use `enable_model_cpu_offload()` for 8GB VRAM
- Input: original image + text instruction
- Parameters: `guidance_scale` (1.0-10.0, default 2.5), `num_inference_steps` (default 24), `max_sequence_length` (default 256)
- Output: edited image at same resolution as input

**Key code pattern** (from diffusers docs):
```python
from diffusers import FluxKontextPipeline, GGUFQuantizationConfig
import torch

gguf_path = "data/models/flux-kontext-dev-Q4_K_S.gguf"
quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    transformer_path=gguf_path,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)
pipe.enable_model_cpu_offload()

result = pipe(
    image=input_image,
    prompt="Change the hair color to blonde",
    guidance_scale=2.5,
    num_inference_steps=24,
    max_sequence_length=256,
).images[0]
```

**GGUF download**: Use `huggingface_hub.hf_hub_download(repo_id="QuantStack/FLUX.1-Kontext-dev-GGUF", filename="flux.1-kontext-dev-Q4_K_S.gguf")` to download to local cache.

**Optimal parameters** (from research):
- guidance_scale: 2.5 (default, good balance)
- num_inference_steps: 24 (sweet spot for quality/speed)
- For subtle edits: guidance 1.5-2.5
- For dramatic changes: guidance 3.0-5.0

### 2. CosXL Edit (Secondary — Fast SDXL Quality)

**Model**: `stabilityai/cosxl` — file `cosxl_edit.safetensors`
**Pipeline**: `StableDiffusionXLInstructPix2PixPipeline` with `is_cosxl_edit=True`

**Implementation requirements**:
- Download `cosxl_edit.safetensors` from stabilityai/cosxl on first use
- Load as SDXL InstructPix2Pix pipeline
- Use `enable_model_cpu_offload()` + `vae.enable_slicing()` + `vae.enable_tiling()` for 8GB
- Same input/output as current IP2P (instruction + image → edited image)

**Key code pattern**:
```python
from diffusers import StableDiffusionXLInstructPix2PixPipeline, AutoencoderKL, EulerDiscreteScheduler
import torch

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    "data/models/cosxl_edit.safetensors",
    vae=vae,
    torch_dtype=torch.float16,
    is_cosxl_edit=True,
)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

result = pipe(
    prompt="Turn the sky to sunset",
    image=input_image,
    guidance_scale=7.0,
    image_guidance_scale=1.5,
    num_inference_steps=20,
).images[0]
```

**Optimal parameters**:
- guidance_scale: 7.0 (same as IP2P)
- image_guidance_scale: 1.5 (preserves original)
- num_inference_steps: 20
- Uses Cosine-Continuous EDM VPred schedule

### 3. Standard img2img + ControlNet (Fallback — Structure Preservation)

**Models**: 
- SD 1.5 (`runwayml/stable-diffusion-v1-5`) — already cached from IP2P
- ControlNet Depth (`lllyasviel/control_v11f1p_sd15_depth`)
- ControlNet Canny (`lllyasviel/control_v11p_sd15_canny`)

**Implementation requirements**:
- Use `StableDiffusionControlNetImg2ImgPipeline`
- Extract depth or canny edges from input image
- Feed as ControlNet conditioning + img2img with denoising strength
- User provides descriptive prompt (describe desired result, not instruction)

**Key code pattern**:
```python
from diffusers import StableDiffusionControlNetImg2ImgPipeline, ControlNetModel
import torch, cv2, numpy as np

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)
pipe.enable_model_cpu_offload()

# Extract depth from MiDaS or Depth Anything
depth_image = get_depth_map(input_image)

result = pipe(
    prompt="A woman with blonde hair in a church",
    image=input_image,
    control_image=depth_image,
    strength=0.5,  # denoising strength: 0.3=subtle, 0.5=moderate, 0.7=dramatic
    guidance_scale=7.5,
    num_inference_steps=25,
).images[0]
```

**Optimal parameters**:
- strength (denoising): 0.3-0.5 for subtle, 0.5-0.7 for moderate, 0.7+ for dramatic
- guidance_scale: 7-9
- num_inference_steps: 25
- ControlNet conditioning_scale: 0.8-1.0

## Integration Plan

### Backend (`ai_enhance.py`)

1. Add `INSTRUCT_MODELS` entries for all 3:
```python
INSTRUCT_MODELS = {
    "kontext": {
        "name": "FLUX Kontext (best quality)",
        "description": "State-of-the-art editing. Q4 quantized, ~7GB. First use downloads model.",
        "default_guidance": 2.5,
        "default_steps": 24,
    },
    "cosxl": {
        "name": "CosXL Edit (SDXL quality)",
        "description": "SDXL-based editing. Good quality, faster than Kontext.",
        "default_guidance": 7.0,
        "default_image_guidance": 1.5,
        "default_steps": 20,
    },
    "pix2pix": {
        "name": "InstructPix2Pix (legacy)",
        "description": "SD 1.5 editing. Fastest but lower quality.",
        "default_guidance": 7.5,
        "default_image_guidance": 1.5,
        "default_steps": 20,
    },
    "controlnet": {
        "name": "ControlNet img2img (structure)",
        "description": "SD 1.5 + depth/canny. Best for structure-preserving edits.",
        "default_guidance": 7.5,
        "default_steps": 25,
        "default_strength": 0.5,
    },
}
```

2. Add separate loader functions:
- `_load_kontext_pipeline()` — downloads GGUF, loads FluxKontextPipeline
- `_load_cosxl_pipeline()` — downloads cosxl_edit.safetensors, loads SDXL IP2P
- `_load_controlnet_pipeline()` — loads SD 1.5 + ControlNet

3. Update `instruct_edit()` to dispatch based on `model` parameter

4. Each pipeline should:
- Lazy-load on first use
- Cache in `_instruct_pipes` dict
- Use `enable_model_cpu_offload()` for VRAM management
- Disable safety checker
- Show download progress for first-time model downloads

### Frontend (`editor_page.dart`)

Update the AI Edit panel to show model selector:
```
[Kontext (Best)] [CosXL] [IP2P] [ControlNet]
```
With description text changing per selection.

For ControlNet mode, show additional controls:
- Denoising strength slider (0.3-0.9)
- ControlNet type: Depth / Canny
- Prompt changes from "instruction" to "description" mode

### Model Downloads (auto on first use)

| Model | Source | Size | Destination |
|-------|--------|------|-------------|
| FLUX Kontext GGUF Q4 | QuantStack/FLUX.1-Kontext-dev-GGUF | ~6.8GB | HF cache |
| FLUX Kontext config | black-forest-labs/FLUX.1-Kontext-dev | ~100MB | HF cache |
| CosXL Edit | stabilityai/cosxl | ~6.5GB | data/models/ |
| SDXL VAE fix | madebyollin/sdxl-vae-fp16-fix | ~300MB | HF cache |
| ControlNet Depth | lllyasviel/control_v11f1p_sd15_depth | ~1.3GB | HF cache |

Total new downloads: ~15GB (one-time, cached)

### Verification Checklist

After implementation, test each model with the same image + instruction:

1. **Kontext**: "Change the dress color to blue" → should produce high-quality blue dress
2. **CosXL**: "Make it look like sunset" → should add warm golden tones
3. **IP2P (legacy)**: "Add snow" → verify still works
4. **ControlNet**: "A painting of two women" (with depth conditioning) → should preserve pose/composition

For each, verify:
- Image returned (not blank/error)
- VRAM usage < 8GB (check `nvidia-smi`)
- Time < 30 seconds
- Quality is visibly different between models (Kontext should be clearly best)

### IMPORTANT Notes
- FLUX Kontext needs `torch.bfloat16` (not float16) for GGUF
- CosXL needs `is_cosxl_edit=True` flag or results will be garbage
- ControlNet needs depth/canny preprocessing — use existing Depth Anything v2 ONNX
- All models need `safety_checker=None` (local tool, false positives on clothing/skin)
- HF Token is in .env as `HF_TOKEN`
- Free VRAM before loading new model: `torch.cuda.empty_cache()`
- Only one editing model should be loaded at a time (unload others)
