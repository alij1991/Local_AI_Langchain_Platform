# AI Image Edit & Prompt Enhancer — Bug Fixes and Enhancements

## Overview

The AI edit section in the editor (`editor_page.dart`) supports 4 models: Kontext (FLUX), CosXL, InstructPix2Pix, and ControlNet. Several have broken model loading, incorrect API parameters, and the prompt enhancer has logic bugs. This document catalogs every bug found and the exact fix required.

---

## Critical Bugs

### BUG 1: FLUX Kontext — Wrong Pipeline Loading (CRASHES)

**File:** `src/local_ai_platform/images/ai_enhance.py` lines 510-530
**Severity:** CRITICAL — model will not load

**Current (broken):**
```python
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    transformer_path=gguf_path,           # <-- DOES NOT EXIST
    quantization_config=quantization_config, # <-- WRONG PLACE
    torch_dtype=torch.bfloat16,
    token=token,
)
```

**Problem:** `FluxKontextPipeline.from_pretrained()` does NOT accept `transformer_path` or `quantization_config`. These are not valid parameters. The GGUF quantization must be applied to the transformer model separately via `FluxTransformer2DModel.from_single_file()`, then passed to the pipeline as `transformer=`.

**Fix:**
```python
from diffusers import FluxKontextPipeline, FluxTransformer2DModel, GGUFQuantizationConfig

gguf_path = _download_kontext_gguf()
token = _get_hf_token()

logger.info("Loading FLUX Kontext transformer with GGUF quantization...")
quantization_config = GGUFQuantizationConfig(compute_dtype=torch.bfloat16)

# Step 1: Load the quantized transformer separately
transformer = FluxTransformer2DModel.from_single_file(
    gguf_path,
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
)

# Step 2: Load the pipeline with the pre-loaded transformer
logger.info("Loading FLUX Kontext pipeline...")
pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev",
    transformer=transformer,
    torch_dtype=torch.bfloat16,
    token=token,
)
# IMPORTANT: Use enable_model_cpu_offload(), NOT enable_sequential_cpu_offload()
# (sequential is incompatible with GGUF quantized models — causes KeyError: None)
pipe.enable_model_cpu_offload()
```

**Version requirement:** `diffusers >= 0.35.0` (FluxKontextPipeline was added in v0.35.0). Also needs `pip install gguf`.

**Lines to change:** 510-530 in `_load_kontext_pipeline()`

---

### BUG 2: CosXL — Missing Required Parameters (MAY CRASH or PRODUCE BAD RESULTS)

**File:** `src/local_ai_platform/images/ai_enhance.py` lines 574-578
**Severity:** HIGH — shape mismatch or wrong image latent scaling

**Current:**
```python
pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    str(model_path),
    vae=vae,
    torch_dtype=torch.float16,
    is_cosxl_edit=True,
)
```

**Problems:**
1. `is_cosxl_edit=True` is passed to `from_single_file()` which forwards it to the pipeline constructor — this part IS correct.
2. **Missing `num_in_channels=8`**: The CosXL Edit checkpoint has a UNet with 8 input channels (4 for image latents + 4 for conditioning), but the default init creates a UNet with 4 input channels. This causes a shape mismatch: `RuntimeError: expected (320, 8, 3, 3) but got (320, 4, 3, 3)`.
3. **Missing `config` parameter**: Without a config reference, `from_single_file` may not correctly detect the SDXL InstructPix2Pix configuration from the safetensors file.

**Fix:**
```python
pipe = StableDiffusionXLInstructPix2PixPipeline.from_single_file(
    str(model_path),
    config="diffusers/sdxl-instructpix2pix-768",
    vae=vae,
    torch_dtype=torch.float16,
    is_cosxl_edit=True,
    num_in_channels=8,
)
```

**Lines to change:** 574-579

---

### BUG 3: CosXL — Missing HF Token Validation (SILENT DOWNLOAD FAILURE)

**File:** `src/local_ai_platform/images/ai_enhance.py` lines 535-555
**Severity:** HIGH — download fails with cryptic 401 error

**Current:** `_download_cosxl_model()` does NOT validate the HF token, unlike `_download_kontext_gguf()` which properly checks at line 485-489.

**Problem:** The `stabilityai/cosxl` repo IS gated (requires accepted license + token). If `_get_hf_token()` returns `None`, the download fails with a confusing 401/403 HTTP error instead of a clear message.

**Fix:** Add token validation at the start of `_download_cosxl_model()`:
```python
def _download_cosxl_model() -> Path:
    """Download cosxl_edit.safetensors if not present."""
    model_path = Path("data/models/cosxl_edit.safetensors")
    if model_path.exists():
        return model_path

    from huggingface_hub import hf_hub_download

    token = _get_hf_token()
    if not token:
        raise RuntimeError(
            "CosXL Edit requires a HuggingFace token (gated model). "
            "Set HF_TOKEN in .env or run `huggingface-cli login`."
        )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    # ... rest unchanged
```

**Lines to change:** 535-555

---

### BUG 4: Prompt Enhancer — `or True` Logic Bug (WRONG BEHAVIOR)

**File:** `src/local_ai_platform/images/ai_enhance.py` line 1038
**Severity:** MEDIUM — always executes Ollama path, even when router LLM succeeded

**Current:**
```python
if not router or True:  # Always try Ollama as fallback
```

**Problem:** `or True` makes the condition ALWAYS true, so the Ollama direct fallback ALWAYS runs, even after the router LLM path at line 1015-1035 has already been tried. This means:
1. If router succeeds and returns an enhanced prompt at line 1031, the function returns correctly.
2. If router fails (line 1034-1035), it falls through to line 1038 — but the `or True` means this block runs regardless of whether `router` exists.
3. The actual intent was: "if router was not available, try Ollama directly as fallback." The `or True` defeats this logic and causes Ollama to be tried even after the router path already ran and failed.

**The real issue:** When the router LLM path at line 1015-1035 fails (LLM produces bad output that gets rejected at line 1029), it falls through. Then line 1038 always tries Ollama direct, which makes a SECOND LLM call with a different model. This double-call adds latency and may return conflicting results.

**Fix:** Change to:
```python
if not router:  # Ollama direct fallback (only when router unavailable)
```

Or if the intent is "always try Ollama as a second fallback after router failure":
```python
# Ollama direct fallback (try if LLM enhancement above didn't return)
try:
    import json
    import urllib.request
    # ... rest unchanged
```

**Lines to change:** 1038

---

### BUG 5: `instruct_edit()` — No Empty Instruction Validation (CRYPTIC MODEL ERRORS)

**File:** `src/local_ai_platform/images/ai_enhance.py` lines 743-782
**Severity:** MEDIUM — produces cryptic "prompt length cannot be 0" errors

**Problem:** If an empty string passes through to any of the 4 model paths, the diffusers pipeline will throw an unclear error. There is no early validation.

**Fix:** Add at the top of `instruct_edit()` (after line 756):
```python
if not instruction or not instruction.strip():
    raise ValueError("Edit instruction cannot be empty. Describe what you want to change.")
```

---

### BUG 6: ControlNet — No Try/Except Around Model Loading (APP CRASH)

**File:** `src/local_ai_platform/images/ai_enhance.py` lines 610-630
**Severity:** MEDIUM — unhandled exception crashes the entire operation

**Problem:** `ControlNetModel.from_pretrained()` and `StableDiffusionControlNetImg2ImgPipeline.from_pretrained()` have no error handling. If the download fails (network issue, disk space, etc.), it crashes without a useful message.

**Fix:** Wrap in try/except:
```python
try:
    controlnet = ControlNetModel.from_pretrained(
        controlnet_repo, torch_dtype=torch.float16,
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
except Exception as e:
    raise RuntimeError(
        f"Failed to load ControlNet ({control_type}) pipeline: {e}. "
        "Check your internet connection and disk space."
    )
```

---

### BUG 7: ControlNet Depth — Silent Failure When No Depth Model Available

**File:** `src/local_ai_platform/images/ai_enhance.py` lines 643-657
**Severity:** MEDIUM — `_get_depth_map()` returns None silently

**Problem:** `_get_depth_image()` calls `_get_depth_map(arr, h, w)` which relies on Depth Anything v2 ONNX or MiDaS. If neither is installed/downloaded, `_get_depth_map` returns `None` and the function raises `RuntimeError("Could not generate depth map — no depth model available")`. But this error only appears at runtime during an edit, not at model loading time.

**Enhancement:** Add a check during `_load_controlnet_pipeline("depth")` that verifies depth model availability before caching the pipeline:
```python
if control_type == "depth":
    # Verify depth model is available before loading ControlNet
    try:
        test_img = Image.new("RGB", (64, 64), "white")
        _ = _get_depth_image(test_img)
    except RuntimeError:
        raise RuntimeError(
            "ControlNet depth requires a depth estimation model. "
            "The Depth Anything v2 ONNX model will be downloaded on first use, "
            "or you can use 'canny' control type which has no additional dependencies."
        )
```

---

## Prompt Enhancer Bugs

### BUG 8: `/images/enhance-prompt` — Prompt Template Embeds User Input in JSON Template (BREAKS JSON)

**File:** `api_server.py` lines 4718-4721
**Severity:** MEDIUM — JSON parsing fails when user input contains quotes or braces

**Current (non-weighted path):**
```python
generate_prompt = f"""/no_think
IMPORTANT: Keep the prompt UNDER 60 words. CLIP only accepts 77 tokens — longer prompts get silently truncated.
Output ONLY this JSON object, nothing else:
{{"prompt": "[concise Stable Diffusion {model_family} prompt for: {user_prompt}. Include key quality tags ...]", "negative_prompt": "[things to avoid: ...]"}}"""
```

**Problem:** This embeds `{user_prompt}` INSIDE the JSON template example. If the user's prompt contains quotes (`"`) or curly braces (`{}`), it breaks the JSON structure that the LLM is trying to complete. The LLM sees a malformed JSON example and produces garbage output.

**Fix:** Restructure the prompt to separate the user input from the JSON template:
```python
generate_prompt = f"""/no_think
You are a Stable Diffusion {model_family} prompt expert. Enhance this description into a detailed image prompt.

User's description: {user_prompt}

RULES:
1. MAX 60 words (CLIP truncates at 77 tokens)
2. Include quality tags: masterpiece, best quality, highly detailed
3. Add style, lighting, composition details
4. Comma-separated tag format

Output ONLY this JSON format:
{{"prompt": "your enhanced prompt here", "negative_prompt": "worst quality, low quality, blurry, deformed, ugly, bad anatomy, watermark, text, plus scene-specific negatives"}}"""
```

---

### BUG 9: Prompt Enhancer — `/no_think` Not Supported by All Models

**File:** `api_server.py` lines 4703, 4718 and `ai_enhance.py` line 1055
**Severity:** LOW-MEDIUM — some models output `/no_think` as literal text

**Problem:** The `/no_think` directive is Qwen3-specific. Other models (Gemma, LLaMA, Phi) don't understand it and will either:
- Include `/no_think` as literal text in their response
- Ignore it but still use chain-of-thought reasoning (polluting output with `<think>` tags)

**Fix:** Only prepend `/no_think` for Qwen models:
```python
no_think_prefix = ""
if any(kw in ollama_model.lower() for kw in ("qwen", )):
    no_think_prefix = "/no_think\n"

generate_prompt = f"""{no_think_prefix}..."""
```

Also strip `<think>...</think>` tags from ALL model responses:
```python
import re
content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
```

---

### BUG 10: `_extract_json_from_llm` — Fails on Nested JSON with Inner Braces

**File:** `api_server.py` lines 4561-4575
**Severity:** LOW — misses valid JSON with escaped characters

**Current step 3:**
```python
brace_match = _re.search(r"\{[^{}]*\}", text, _re.DOTALL)
```

**Problem:** The regex `[^{}]*` matches JSON with NO nested braces. But if the LLM wraps its JSON in explanation text with other braces, or if the JSON contains nested objects, this fails. Step 4 (greedy `\{.*\}`) then matches too much (includes explanation text).

**Enhancement:** Add a balanced-brace extraction step between 3 and 4:
```python
# 3.5: Try to find balanced braces using a counter
def _find_balanced_json(text):
    start = text.find('{')
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i+1])
                except json.JSONDecodeError:
                    return None
    return None
```

---

## Enhancement Opportunities

### ENHANCEMENT 1: Add Diffusers Version Check

Before loading any pipeline, verify diffusers version:
```python
def _check_diffusers_version(min_version: str):
    import diffusers
    from packaging import version
    if version.parse(diffusers.__version__) < version.parse(min_version):
        raise RuntimeError(
            f"diffusers >= {min_version} required (installed: {diffusers.__version__}). "
            f"Upgrade with: pip install -U diffusers"
        )
```

Call in `_load_kontext_pipeline()`:
```python
_check_diffusers_version("0.35.0")  # FluxKontextPipeline added in 0.35.0
```

### ENHANCEMENT 2: Add Progress Logging During Model Downloads

The download functions (`_download_kontext_gguf`, `_download_cosxl_model`) download 6-7GB files with no progress indication. Add callback:
```python
from huggingface_hub import hf_hub_download
from tqdm import tqdm

# Already using logger, but add periodic progress:
logger.info("Downloading %s (~%.1fGB, this may take a while)...", filename, size_gb)
```

### ENHANCEMENT 3: Better Model Error Messages

When `instruct_edit()` fails, the error message should include which model was being used and a hint about requirements:
```python
except Exception as e:
    model_hints = {
        "kontext": "Requires: diffusers>=0.35.0, pip install gguf, HF_TOKEN set, ~7GB VRAM",
        "cosxl": "Requires: HF_TOKEN set, ~8GB VRAM, first use downloads ~6.5GB",
        "pix2pix": "Requires: ~4GB VRAM, first use downloads ~4GB",
        "controlnet": "Requires: ~4GB VRAM, depth model for depth mode",
    }
    hint = model_hints.get(model, "")
    raise RuntimeError(f"instruct_edit with model '{model}' failed: {e}. {hint}")
```

### ENHANCEMENT 4: Prompt Enhancer — Strip Thinking Tags from All Responses

Add thinking tag cleanup to ALL response processing paths in `/images/enhance-prompt`:
```python
# After getting content from Ollama response:
content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
# Also strip /no_think echo
content = content.replace('/no_think', '').strip()
```

### ENHANCEMENT 5: Editor Prompt Enhancer Should Support All Models, Not Just InstructPix2Pix

The `enhance_edit_prompt()` function's LLM system prompt says "for InstructPix2Pix" but is used for all models including Kontext and CosXL. Kontext works best with different prompt styles (more descriptive, less command-like).

**Fix:** Accept `model` parameter and adjust the system prompt:
```python
def enhance_edit_prompt(instruction: str, model: str = "pix2pix", router=None, config=None) -> str:
    if model == "kontext":
        # Kontext works best with detailed natural language descriptions
        _LLM_SYSTEM = (
            "You enhance image editing instructions for FLUX Kontext. "
            "Kontext understands natural language well. Be descriptive and specific. "
            "Use adjectives and details about the desired result. "
            "Keep it 1-3 sentences. Output ONLY the improved instruction."
        )
    elif model == "controlnet":
        # ControlNet needs a description of the desired OUTPUT, not an edit command
        _LLM_SYSTEM = (
            "You write image descriptions for ControlNet img2img. "
            "Describe the DESIRED RESULT, not what to change. "
            "Include style, lighting, quality details. "
            "Keep it 1-2 sentences. Output ONLY the description."
        )
    else:
        # IP2P/CosXL: edit commands with verbs
        _LLM_SYSTEM = (
            "You enhance image EDITING instructions for InstructPix2Pix. "
            # ... existing prompt
        )
```

---

## Files to Modify

### Python
- `src/local_ai_platform/images/ai_enhance.py`
  - Lines 510-530: Fix Kontext GGUF loading (BUG 1)
  - Lines 535-555: Add CosXL token validation (BUG 3)
  - Lines 574-579: Add CosXL missing params (BUG 2)
  - Lines 610-630: Add ControlNet error handling (BUG 6)
  - Lines 643-657: Add depth model check (BUG 7)
  - Lines 743-756: Add instruction validation (BUG 5)
  - Line 1038: Fix `or True` logic bug (BUG 4)
  - Lines 950-1081: Enhance edit prompt model awareness (ENHANCEMENT 5)

- `api_server.py`
  - Lines 4718-4721: Fix prompt template (BUG 8)
  - Lines 4703, 4718: Fix `/no_think` for non-Qwen models (BUG 9)
  - Lines 4561-4575: Improve JSON extraction (BUG 10)
  - All response paths: Strip thinking tags (ENHANCEMENT 4)

### Flutter
- `flutter_client/lib/pages/editor_page.dart`
  - Pass `model` parameter to `/editor/enhance-prompt` endpoint (ENHANCEMENT 5)
  - Show model-specific error hints in UI (ENHANCEMENT 3)

---

## Priority Order

1. **BUG 1** (Kontext GGUF loading) — completely broken, model won't load
2. **BUG 2** (CosXL missing params) — may crash or produce garbage
3. **BUG 3** (CosXL token validation) — confusing download failure
4. **BUG 5** (empty instruction validation) — easy fix, prevents cryptic errors
5. **BUG 4** (or True logic) — prompt enhancer double-calls LLM
6. **BUG 8** (prompt template JSON) — prompt enhancer fails with special chars
7. **BUG 9** (no_think compatibility) — prompt enhancer polluted output
8. **BUG 6** (ControlNet error handling) — crash prevention
9. **BUG 7** (depth model check) — better error messages
10. **BUG 10** (JSON extraction) — edge case improvement
11. **ENHANCEMENTS 1-5** — polish and robustness
