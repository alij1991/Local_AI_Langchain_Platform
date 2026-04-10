# FLUX.1 Kontext Pipeline — Reference & Tuning Guide

This document is the canonical reference for how the FLUX.1-Kontext-dev image
editing pipeline is wired into this project, why the code is structured the
way it is, and how to tune it for different hardware. It exists so future
changes don't re-litigate decisions that were already hard-won through
debugging.

**Target hardware baseline:** NVIDIA RTX 4060 Laptop, 8 GB VRAM, 32 GB system
RAM, Windows 11, torch 2.11.0 + CUDA 13.0, diffusers 0.37.1.

**File that owns all of this:** `src/local_ai_platform/images/ai_enhance.py`
(search for `[KONTEXT]` log lines and the `_load_kontext_pipeline` function).

---

## Table of Contents

1. [What FLUX.1-Kontext-dev is and how it differs from txt2img](#1-what-flux1-kontext-dev-is)
2. [Memory architecture and device layout](#2-memory-architecture-and-device-layout)
3. [The GGUF quantization path](#3-the-gguf-quantization-path)
4. [Environment variables — the tuning knobs](#4-environment-variables)
5. [Recommended settings by hardware class](#5-recommended-settings-by-hardware-class)
6. [Tuning guide — steps, guidance, resolution](#6-tuning-guide)
7. [Troubleshooting common errors](#7-troubleshooting-common-errors)
8. [Performance expectations and benchmarks](#8-performance-expectations)
9. [Design decisions and rationale (history)](#9-design-decisions-and-rationale)
10. [Log line reference](#10-log-line-reference)

---

## 1. What FLUX.1-Kontext-dev is

FLUX.1-Kontext-dev is an instruction-based **image editing** model from Black
Forest Labs. You give it an input image + a natural language instruction
("make all the girls clothed", "turn this into a sunset scene") and it
produces an edited version of the image.

Architecturally it shares the backbone of FLUX.1-dev — a 12B-parameter
multimodal DiT transformer. The key difference for us:

- **Standard FLUX.1-dev (txt2img):** conditions only on text embeddings.
- **FLUX.1-Kontext-dev:** concatenates **image tokens** (from VAE-encoding the
  input image) **with text tokens** before feeding to the transformer.
  This roughly **doubles the attention sequence length**, making attention
  computation **~4× more expensive** at the same resolution than plain FLUX.

That sequence-length doubling is why a 1024×1024 Kontext edit feels much
heavier than a 1024×1024 FLUX.1-dev generation on the same hardware.

### The guidance scale is NOT classic classifier-free guidance

FLUX.1 is **guidance-distilled**. The `guidance_scale` parameter controls an
internal guidance embedding, not a separate unconditional forward pass.
Consequences:

- The sweet spot is roughly **2.5 – 3.5** (BFL's published recommendation).
- Values above ~4.0 often produce **worse** output, not stronger — the
  guidance embedding saturates and the model starts producing flat/blurry
  results or ignoring edit instructions.
- There is no negative prompt mechanism (`negative_prompt` is ignored at this
  guidance scale). Phrase your edit as a positive instruction.

**If your edits look weak or washed out, first thing to check is guidance ≤ 3.5.**

---

## 2. Memory architecture and device layout

On an 8 GB VRAM card with ~1 GB held by Windows + other apps, we have roughly
**6.7 GB of usable VRAM**. The FLUX Kontext pipeline has four components and
they do **not** all fit on the GPU simultaneously:

| Component | Dtype / quant | Size | Where it lives |
|---|---|---|---|
| `transformer` | GGUF Q3_K_S / Q4_K_S | ~4.9 – 6.3 GB | CPU at rest → cycled to GPU during denoise |
| `text_encoder_2` (T5-XXL) | bf16 (no quant) | ~9.5 GB | **CPU permanently** via `device_map="cpu"` |
| `text_encoder` (CLIP-L) | bf16 | ~0.25 GB | CPU at rest → cycled to GPU for encode |
| `vae` | bf16 | ~0.35 GB | CPU at rest → cycled to GPU for encode/decode |

### Why `enable_model_cpu_offload()` and not manual placement

We tried both. Manual placement (pinning transformer + CLIP + VAE to GPU) was
measured as **worse** on our 8 GB card because CLIP + VAE added ~0.6 GB of
permanent residents, pushing us over the physical VRAM limit and triggering
**Windows WDDM shared memory paging** (the driver silently pages GPU memory
into system RAM over PCIe). Symptom: step times spike from ~7s to 108-200s
with high variance because some weight accesses hit paged pages.

`enable_model_cpu_offload()` keeps only the currently-executing component on
GPU. During the 20+ denoising steps, **only the transformer is resident on
GPU** — CLIP and VAE have already been evicted by the accelerate hooks.
Peak VRAM ≈ `transformer_size + ~0.5 GB activations`.

### Why T5 stays on CPU via `device_map="cpu"`

T5-XXL is 9.5 GB in bf16 — it cannot fit on an 8 GB card alongside the
transformer. We tried NF4 quantization (1.2 GB in theory) but bitsandbytes
NF4 models **cannot be moved to CPU** (they require CUDA kernels), which
breaks `enable_model_cpu_offload()`'s whole premise — the offload hook tries
to move NF4 T5 to CPU as the rest state, fails silently, and the GGUF
transformer ends up running on CPU as a side effect. **Result was 0% GPU
utilization despite nvidia-smi showing memory allocated.**

Standard bf16 T5 loaded with `device_map="cpu"` is given an `hf_device_map`
attribute, which accelerate checks and **respects** — it does not try to
cycle T5 through GPU at all. T5 runs on CPU once per inference (not once per
step), producing text embeddings which are tiny (few MB) and cheap to move
to GPU before the denoising loop.

### VRAM budget checklist

Before inference, the pipeline logs:

```
[KONTEXT] Before load — VRAM: driver_free=X.XXGB driver_used=X.XXGB (pytorch=X.XXGB other_procs=X.XXGB) total=8.6GB
```

- `driver_free` / `driver_used` — true memory state from `torch.cuda.mem_get_info()`
- `other_procs` — VRAM held by processes **other than** our Python server
- `pytorch` — VRAM allocated by our PyTorch process

If `other_procs > 0.9 GB` and you're on Q4_K_S or bigger, the pipeline emits
a **VRAM pressure warning** telling you to either close GPU apps or switch
to Q3_K_S.

---

## 3. The GGUF quantization path

We load the transformer as a **GGUF file** (format from llama.cpp), using
diffusers' built-in `GGUFQuantizationConfig`. This lets us fit a 12B-param
transformer in 4-6 GB of disk / VRAM without needing to first download the
full 24 GB bf16 checkpoint.

The source repository is `QuantStack/FLUX.1-Kontext-dev-GGUF` on HuggingFace.

### Available quant variants

| Variant | Size | Quality | 8 GB fit (with 1 GB other_procs) | When to use |
|---|---|---|---|---|
| Q2_K | 3.7 GB | lowest | comfortable (~3 GB headroom) | speed is paramount, quality loss is OK |
| **Q3_K_S** | **4.9 GB** | good | **recommended for 8 GB** (~1.8 GB headroom) | **default for 8 GB cards** |
| Q3_K_M | 5.0 GB | slightly better Q3 | ~1.7 GB headroom | marginal quality bump over Q3_K_S |
| Q4_0 | 6.3 GB | good | tight — may page under load | non-K-quant Q4, rarely better than Q4_K_S |
| Q4_K_S | 6.3 GB | very good | tight — requires clean GPU | **default when VRAM pressure is low** |
| Q4_K_M | 6.5 GB | very good + | marginal fit | avoid unless 10 GB+ card |
| Q5_K_S | 7.7 GB | excellent | does NOT fit on 8 GB | 12 GB+ cards only |
| Q6_K | 9.2 GB | excellent | does NOT fit on 8 GB | 12 GB+ cards |
| Q8_0 | 11.8 GB | near-bf16 | does NOT fit on 8 GB | 16 GB+ cards |

Switch variants with `KONTEXT_GGUF_QUANT=Q3_K_S` in your `.env` or shell.
See section 4 for details.

### Why GGUF is slower than bf16 on GPU

The GGUF transformer stores weights in a quantized block format. Every
forward pass **dequantizes blocks on-demand** to bf16 before the matmul.
Diffusers' Python dequantization path (which is the file-for-file same code
as ComfyUI's ComfyUI-GGUF node — the code was literally ported from city96's
Apache-2.0 source and credited in `diffusers/quantizers/gguf/utils.py` line 1)
is unoptimized compared to native CUDA quantized matmul.

**There is a faster path:** set `DIFFUSERS_GGUF_CUDA_KERNELS=true` and
diffusers will load `Isotr0py/ggml` CUDA kernels for fused dequant+matmul.
**But this only works on Linux** — the HuggingFace Kernels repo at
[huggingface.co/Isotr0py/ggml](https://huggingface.co/Isotr0py/ggml/tree/main/build)
has Linux builds only (`torch25/26/27 × cu118/121/124/126/128 × x86_64-linux`).
On Windows we're stuck with the Python path.

**If you move to WSL2 or a Linux box,** you can uncomment the env var and get
the ComfyUI-level speedup for real.

---

## 4. Environment variables

All Kontext tuning is controlled by environment variables. Set them in your
`.env` file or export them before launching the server.

### `KONTEXT_GGUF_QUANT`

**Default:** `Q4_K_S`
**Values:** `Q2_K`, `Q3_K_S`, `Q3_K_M`, `Q4_0`, `Q4_K_S`, `Q4_K_M`, `Q5_K_S`, `Q6_K`, `Q8_0`

Selects which GGUF variant to download and load. First use of a new variant
triggers a ~4-12 GB download from `QuantStack/FLUX.1-Kontext-dev-GGUF`.
Previous variants remain cached on disk.

```
KONTEXT_GGUF_QUANT=Q3_K_S
```

**Rule of thumb:** for any 8 GB card with other GPU apps running, use
`Q3_K_S`. For a clean 8 GB card, `Q4_K_S` works. For 12 GB+ cards, try
`Q5_K_S` or `Q6_K`.

### `KONTEXT_FBC_THRESHOLD`

**Default:** unset (cache disabled)
**Values:** a float, or `0` / unset to disable

Enables FirstBlockCache (TeaCache successor) on the transformer. The cache
skips transformer forward passes between timesteps when residuals are
similar enough. Lower threshold = stricter similarity check = fewer skips =
better quality. Higher threshold = more skips = faster but weaker edits.

**For image editing this should stay DISABLED.** The cache was designed for
txt2img generation where mid-schedule residuals are naturally similar. In
editing, the big residual changes are **exactly where the edit is happening**,
and the cache skips them — producing visually minimal edits.

Measured impact (see section 9 for full discussion): with
`KONTEXT_FBC_THRESHOLD=0.08` and `steps=35`, we observed a **44% cache hit
rate** and the edits looked almost unchanged from the input. Disabling the
cache is the current default.

Only set this if you:
- Want raw generation speed (~30% faster) and
- Don't mind weaker edits and
- Are doing generation-like tasks (minor color tweaks, not structural edits)

```
# Leave unset for editing (default)
# KONTEXT_FBC_THRESHOLD=0.08  # opt-in, expect weaker edits
```

### `HF_TOKEN`

**Required.** FLUX.1-Kontext-dev is a **gated model** — you must accept the
license on HuggingFace at
[black-forest-labs/FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
and provide an access token.

```
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

### `DIFFUSERS_GGUF_CUDA_KERNELS` *(Linux only)*

**Default:** `false` (recommended on Windows)

Enables diffusers' fused GGUF dequant+matmul CUDA kernels. Only works on
Linux because the HF Kernels repo ships Linux-only build variants. **Setting
this to `true` on Windows will break the `from diffusers import ...` import
because the kernel download fails with `FileNotFoundError`.**

If you migrate to WSL2 or a Linux box, set this to `true` and you'll get the
ComfyUI-level speedup (roughly 5-10× faster GGUF inference).

---

## 5. Recommended settings by hardware class

### 8 GB VRAM (RTX 4060, RTX 3060 Ti, etc.)

```
KONTEXT_GGUF_QUANT=Q3_K_S
# KONTEXT_FBC_THRESHOLD unset (disabled)
```

- Close LM Studio, Ollama (or ensure no models loaded), heavy browser tabs
- Expected step time at 768×768: **~7s/step**
- Total for 24-step edit: ~3-4 minutes

### 8 GB VRAM with heavy background GPU usage

```
KONTEXT_GGUF_QUANT=Q2_K  # or close the background apps
```

Or accept slower performance with Q3_K_S.

### 12 GB VRAM (RTX 3060 12GB, RTX 4070)

```
KONTEXT_GGUF_QUANT=Q5_K_S  # or Q4_K_M for a balance
```

Manual device placement might become viable (pin transformer + CLIP + VAE on
GPU), but we haven't re-enabled that code path since we tuned for 8 GB. See
"Potential future work" in section 9.

### 16 GB+ VRAM (RTX 4080, RTX 4090, A100)

```
KONTEXT_GGUF_QUANT=Q8_0
```

Or switch to full bf16 via `FluxKontextPipeline.from_pretrained(...)` without
GGUF at all — at 24 GB card you can skip quantization entirely.

### Linux (any VRAM)

```
KONTEXT_GGUF_QUANT=Q4_K_S   # or whatever fits
DIFFUSERS_GGUF_CUDA_KERNELS=true
```

With the CUDA kernels enabled, GGUF inference is roughly 5-10× faster per
step than the Python dequant path. Step times drop to ~1-2s on an RTX 4060.

---

## 6. Tuning guide

### Guidance scale

| Range | Effect |
|---|---|
| 1.0 – 2.0 | Weak edits, mostly preserves input |
| **2.5 – 3.5** | **Recommended** — strong edits, clean output |
| 4.0 – 5.0 | Over-saturated, starts degrading |
| 6.0+ | Visibly degraded, washed out, often ignores edit |

BFL's published default is **2.5**. If your edits look weak, first thing to
try is pushing to 3.0 — **not** jacking guidance to 8+. That's how classic
CFG works and FLUX is guidance-distilled (different thing).

### Step count

With FirstBlockCache **disabled** (the default), step count now behaves
normally:

| Steps | Edit quality | Total time (8 GB Q3_K_S) |
|---|---|---|
| 15 | acceptable, some artifacts | ~2 min |
| **20** | **good baseline** | ~3 min |
| 24 | better refinement | ~3.5 min |
| 28 | diminishing returns | ~4 min |
| 35 | barely different from 28 | ~5 min |

Start with **20 steps**. If the edit isn't strong enough, raise to 24-28
first. Beyond 28 you're burning time for minimal gain.

**Important caveat about the old behavior:** if you're reading historical
logs and seeing *lower* step counts produce *better* edits, that was a
FirstBlockCache bug (see section 9 for full postmortem). After disabling
FBC, higher steps → better edits, as you'd normally expect.

### Resolution

The pipeline auto-resizes inputs to **MAX_SIDE = 768** because FLUX Kontext's
attention cost scales quadratically with sequence length and 1024×1024 spills
activations into shared GPU memory → massive slowdown on 8 GB cards.

| Resolution | Sequence length impact | 8 GB viability |
|---|---|---|
| 512×512 | 1× (baseline) | very fast, some detail loss |
| 768×768 | ~2× | **default**, good balance |
| 896×896 | ~2.8× | tight, may page |
| 1024×1024 | ~4× | spills to shared memory → slow |

To change the cap, edit `MAX_SIDE` in `_load_kontext_pipeline`'s inference
section (`ai_enhance.py` around line 1100). There's no env var for this yet.

---

## 7. Troubleshooting common errors

### Error: `Insufficient GPU memory for Kontext: only X.XXgb free`

The precheck failed at load time — something is holding VRAM. The error
message lists the processes currently using GPU memory.

**Common culprits:**
- **LM Studio** — holds loaded LLMs in VRAM permanently, **no API to evict
  from another process**. You must close LM Studio manually or unload its
  models in its UI before running Kontext.
- **Ollama** — the pipeline automatically evicts Ollama models via
  `POST /api/generate { keep_alive: 0 }` before loading Kontext, so this
  should not be a problem unless eviction fails.
- **Other image pipelines** (CosXL, IP2P) — auto-evicted via
  `_unload_other_pipelines()`.

**Fix:** close the listed app, or switch to a smaller `KONTEXT_GGUF_QUANT`.

### Error: `Expected all tensors to be on the same device, but got index is on cuda:0, different from other tensors on cpu`

The T5 embedding table is on CPU but the input IDs arrived on CUDA. This
happens when `enable_model_cpu_offload()` is not active and the transformer
is pinned to CUDA. The pipeline's `encode_prompt` reads
`self._execution_device`, which is CUDA, and moves tokenized IDs to CUDA
before calling T5 — which blows up on T5's CPU embedding table.

**Fix:** this is why Step 4 of the loader uses `enable_model_cpu_offload()`.
Do **not** change it to manual device placement unless you also implement
manual prompt encoding (see section 9 for the full analysis of why manual
placement is worse on 8 GB anyway).

### Error: `No context is set. Please set a context before retrieving the state.`

FirstBlockCache's `StateManager.get_state()` raised because no cache context
was active. This is a bug in `diffusers/pipelines/flux/pipeline_flux_kontext.py`
— it does not wrap its transformer call in `self.transformer.cache_context("cond")`
the way `pipeline_flux.py` does at line 948.

**Fix:** the inference path in `ai_enhance.py` wraps the entire `pipe(...)`
call in `pipe.transformer.cache_context("kontext")` to compensate. This error
only reappears if you modify that wrapping without preserving the context
manager. **If you see this error, verify the `with _sdp_ctx, _cache_ctx:`
block in the inference section is intact.**

### Error: `torch.cuda.OutOfMemoryError` during inference (not at load)

Peak activation memory exceeded free VRAM during the denoising loop. Usually
caused by:
- Input image larger than `MAX_SIDE` and resize didn't apply
- Another GPU app opened mid-run
- `MAX_SIDE` was raised above 768 on a cramped card

**Fix:** drop to smaller resolution, smaller quant, or close other apps.

### Issue: Edit barely changes the image

**Most likely cause #0 (most common in practice): random seed variance
amplified by weak conditioning.** If the same prompt produces good output
sometimes and garbage other times, and "lower step counts seem to work
better," you're seeing seed variance dominating a weak text signal.
**Fix: set an explicit `seed` in the Kontext UI** (or pass `seed=<int>`
via the API) to lock the noise. Then vary `guidance` / `steps` / prompt
framing with the same seed to isolate real parameter effects. Once you
find a seed that gives a good result, write it down and reuse it. See
the dedicated section "Seed control and reproducibility" below.

**Most likely cause #0.5 (for short prompts): text conditioning is diluted
by padding.** FLUX Kontext tokenizes your prompt to a **fixed 512 tokens**,
padding with zeros. A short prompt like "make all the girls clothed" is
only ~8 real tokens out of 512 — meaning **98% of the T5 output is padding
noise** that the transformer's cross-attention still sees as valid text
tokens. This dilutes the conditioning to the point where FLUX's
preservation prior often wins. **Fix: use longer, more detailed prompts.**
30-60 tokens of detail gives the model a much stronger signal to commit
to. Example:
  - Weak: `"make all the girls clothed"` (8 tokens)
  - Strong: `"a professional photograph of three women wearing casual
    blue denim jeans and white cotton t-shirts, standing in the same pose,
    same background, same hair, same lighting, natural skin tones, sharp
    focus"` (~50 tokens)

The `[KONTEXT DEBUG] T5 stats (REAL tokens only, n=N)` log line shows the
per-real-token std. Healthy values are 0.3-2.0. If it's below 0.3, the
signal is weak and you should either lengthen the prompt or use true CFG
(see cause #2).

**Most likely cause #1 (historical, now fixed): `max_sequence_length`
passed to `pipe()` was less than 512.** FLUX.1-Kontext-dev was trained
with 512-token T5 sequences. Passing `max_sequence_length=256` (a
memory-saving optimization we tried once) halves the T5 output from
`[1, 512, 4096]` to `[1, 256, 4096]`, which attenuates text conditioning.
**Fix: always pass `max_sequence_length=512` (the pipeline's default).
This is already fixed in `ai_enhance.py` around line 1356.** If you see
`max_seq=256` in the `[KONTEXT DEBUG] pipe() kwargs` log line, this
regression has been reintroduced.

**Most likely cause #2: FLUX's distilled guidance saturates, so raising
`guidance_scale` doesn't fix weak edits — you need `true_cfg_scale`
instead.** FLUX.1 is guidance-distilled: `guidance_scale` controls an
embedded guidance signal, not classifier-free guidance. Raising it above
~4.0 actively degrades output. If you need stronger edits, set
`true_cfg_scale=2.0-4.0` AND provide a `negative_prompt` — this enables
REAL classifier-free guidance with a second unconditional forward pass
per step (2× inference time). This is the only "turn it up to 11" lever
that actually works for Kontext. See "True CFG vs distilled guidance"
below.

**Most likely cause #2: FirstBlockCache is enabled** (`KONTEXT_FBC_THRESHOLD`
set to a non-zero value). Unset it. See section 9 for the full explanation.

**Most likely cause #3: guidance scale too high** (above ~4). FLUX is
guidance-distilled and higher guidance does **not** mean stronger edits.
Drop to 2.5-3.5.

**Cause #4: instruction is a "hard edit" for Kontext.** FLUX Kontext is
biased toward **preservation** of the input image (this is by design — BFL
trained it with a preservation prior to avoid hallucinated changes). It's
strongest at style transfer / background replacement / attribute changes
and weakest at adding entirely new content that must fill empty regions
(e.g. "make the girls clothed" requires the model to hallucinate large new
pixel areas). See the "Prompt tuning" subsection below for how to
strengthen weak-edit instructions.

**Cause #5: instruction is ambiguous or conflicts with the input image.**
Try a more explicit and specific prompt.

**Debugging:** the `[KONTEXT DEBUG]` hook prints T5 input and output stats
for every inference. If you see:
- `nonzero_token_count=0` → the prompt isn't reaching the tokenizer
- `all_zero=True` or `std < 0.01` → T5 is producing degenerate embeddings
- `nan > 0` → numerical instability in T5
- Normal values (std ~0.5-2.0, non-zero tokens, no NaNs) → T5 is fine, the
  issue is downstream (guidance, preservation bias, or instruction framing)

### Seed control and reproducibility

FLUX Kontext generates a random latent noise tensor at the start of each
inference. Different seeds produce **very different** outputs — especially
when the text conditioning is weak (short prompts, preservation-biased
edits), because the noise dominates over the guidance signal.

**Symptom:** running the same prompt + same settings multiple times gives
wildly different results — some edits work, some don't change the image
at all. This is seed variance, not a bug.

**Fix:** lock a seed. The Kontext edit panel in the Flutter editor now
has a **Seed** text field with two buttons:
- **Dice icon** — generate a random seed based on the current time and
  lock it in the field
- **Clear icon** — clear the field to go back to random each run

When the field is empty, each run uses a fresh random seed. When it has
an int, every run produces the same output (deterministic).

**Workflow for tuning parameters without seed noise:**
1. Click the dice icon to lock a random seed
2. Run once. If you got a decent result, note the seed.
3. Change guidance (or steps, or prompt) with the same seed locked
4. Run again — differences are from your parameter change, not the seed
5. Repeat until you find the best settings for this image

**Via the API:**
```json
POST /editor/{session}/edit
{
  "operation": "instruct_edit",
  "params": {
    "model": "kontext",
    "instruction": "...",
    "guidance": 2.9,
    "steps": 24,
    "seed": 42          // <- lock this int, or omit for random
  }
}
```

You'll see the seed echoed in the server log:
```
[KONTEXT] Seed locked: seed=42, generator_device=cuda
```

### True CFG vs distilled guidance

FLUX.1 has **two different "guidance" parameters** and they work
completely differently:

| Parameter | Default | What it does | When to raise it |
|---|---|---|---|
| `guidance_scale` | 2.5–3.5 | FLUX's **distilled guidance embedding** — a single scalar fed into the transformer as a conditioning signal. Single forward pass per step. | Default is already optimal. Raising above 4 degrades output. |
| `true_cfg_scale` | **1.0 (off)** | **Real classifier-free guidance** — runs the transformer TWICE per step (once conditional, once unconditional on the negative prompt) and extrapolates the conditional further away from the unconditional. | When edits are weak and preservation bias is winning. Typical range: **2.0–4.0**. |

**The key insight:** `guidance_scale` alone cannot make FLUX Kontext
produce stronger edits than the distilled guidance was trained for. If
you're in the "preservation bias is beating my edit instruction"
situation, **raising guidance won't help** — you need to enable
`true_cfg_scale` and provide a `negative_prompt`.

**How true_cfg works mathematically:**
```
noise_pred = uncond_pred + true_cfg_scale * (cond_pred - uncond_pred)
```
- `cond_pred` — transformer forward pass with your positive prompt
- `uncond_pred` — transformer forward pass with the negative prompt (or "")
- When `true_cfg_scale=1.0`, the uncond term cancels and you get plain
  `cond_pred` (no CFG, single pass — the default)
- When `true_cfg_scale=2.0`, the conditional prediction is pushed 2× away
  from the unconditional → stronger edit

**Cost:** exactly 2× inference time. A 20-step edit goes from ~180s to
~360s on an 8GB RTX 4060 with Q3_K_S.

**When to use it:**
- Prompt is clearly reaching T5 (healthy diagnostic stats)
- Edit is structural (adding/removing/replacing content)
- Seed variance is locked
- `guidance_scale` is in the correct 2.5–3.5 range
- And edits are STILL weak

**When NOT to use it:**
- Simple style transfer or color grade edits (not needed)
- You're trying to be fast
- Your prompt is ambiguous (fix the prompt first, not the guidance)

**Example values by edit difficulty:**
- Trivial (color tweak): `true_cfg_scale=1.0` (off), speed matters more
- Moderate (style transfer): `true_cfg_scale=1.5–2.0` with empty neg
- Hard (add/remove content): `true_cfg_scale=2.5–3.5` with descriptive neg
- Very hard (adversarial to preservation bias): `true_cfg_scale=3.5–5.0`
  with strong neg ("unchanged, same as input, nothing edited, blurry")

**Good negative prompts for "the edit isn't happening" scenarios:**
```
unchanged, same as input, identical to original, nothing edited, no modification, blurry
```

This explicitly tells the model what NOT to do — preserve the input —
giving the conditional direction something to push against.

### Prompt tuning for stronger edits

Kontext prompt quality has a huge effect on edit strength. Some rules:

**Weaker phrasings (AVOID):**
- `make the girls clothed` — vague, ambiguous target
- `remove the hat` — the model must fill the removed region
- `fix the lighting` — no clear goal

**Stronger phrasings (PREFER):**
- `a photograph of women wearing casual t-shirts and jeans, same pose, same background` — describes the target state explicitly
- `replace the hat with natural hair` — gives the model something to generate, not just remove
- `change to golden hour lighting with warm orange tones` — concrete transformation

For "add content" edits specifically, the pattern that works best is to
describe the **target scene as a complete photograph**, not as a delta from
the current state:

- `bad:  "make her clothed"`
- `good: "a photograph of a woman wearing a blue sweater, same face, same hair, same background"`

Kontext can then latch onto a clear target distribution and apply its
preservation prior to the unchanged regions (face, hair, background) while
freely rewriting the region that differs (clothing).

### Issue: Output is blurry or washed out

**Most likely cause:** guidance scale too high (saturation). Drop to 2.5-3.0.

### Issue: First step takes 40+ seconds, rest are fast

This is **normal** and expected. Step 1 includes:
- cuDNN algorithm selection / autotuning
- GGUF weight dequantization onto CUDA for the first time
- Kernel caching for this specific input shape

The warmup is logged separately in the timing summary as `warmup=XX.Xs` vs
`steady-state mean=X.Xs`. Only the steady state matters for performance
planning.

### Issue: Some steps log `0.2s` and others log `7.0s`

This is FirstBlockCache firing (cache hit on 0.2s steps, full forward pass
on 7.0s steps). **This only happens if you have `KONTEXT_FBC_THRESHOLD`
set** — by default the cache is disabled and every step should be ~7s.
If you see this pattern with the env var unset, the env var is probably
set globally somewhere you forgot (check `.env`, shell profile, Windows
environment variables).

---

## 8. Performance expectations

**Baseline hardware:** RTX 4060 Laptop 8 GB, Windows 11, 1 GB of VRAM held by
other apps (DWM + browser + editor + Flutter client), torch 2.11 + CUDA 13,
diffusers 0.37.1.

### Steady-state step time at 768×768

| Quant | No cache | With FBC (threshold 0.08) |
|---|---|---|
| Q2_K | ~5 s/step | ~3 s/step |
| **Q3_K_S** | **~7 s/step** | ~4 s/step |
| Q4_K_S | ~8 s/step (if fits) | ~5 s/step |

### Full inference time (20 steps, 768×768, Q3_K_S, no cache)

- Pipeline load (first time): ~90s (includes ~87s GGUF download)
- Pipeline load (cached): ~5-10s
- Text encoding (T5 on CPU + CLIP on GPU): ~5s
- Denoising (20 × 7s + 40s warmup): ~180s
- VAE decode: ~2s
- **Total first run: ~5 min**
- **Total cached: ~3 min**

### Component cost breakdown

For a 20-step Q3_K_S run at 768×768:

- Denoising loop: **~70%** of time
- GGUF dequantization during forward: ~30% of loop time
- Actual bf16 matmul: ~40% of loop time
- Attention: ~20% of loop time (with SDPA flash)
- Other ops: ~10% of loop time
- T5 encoding on CPU: ~3% of total
- VAE encode/decode: ~2% of total
- Pipeline overhead: ~5% of total

---

## 9. Design decisions and rationale

This section documents the important decisions made during debugging and
tuning. Read this **before** making architectural changes — most of these
were not obvious and were discovered by painful trial and error.

### Decision: T5 on CPU via `device_map="cpu"`, not NF4 quantization

**Tried first:** bitsandbytes NF4 quantization (T5 would go from 9.5 GB to
1.2 GB on GPU).

**Problem:** bitsandbytes NF4 models **cannot be moved to CPU** because they
rely on CUDA kernels. `enable_model_cpu_offload()` tries to move T5 to CPU as
its "rest state" before registering hooks — this fails silently, corrupts
the hook chain, and the GGUF transformer ends up not being hooked properly.

**Symptom:** 0% GPU utilization during denoising, 275+ seconds per step, 422
timeouts.

**Fix:** load T5 in bf16 with `device_map="cpu"`. This sets an `hf_device_map`
attribute on the model, which accelerate detects and respects — T5 stays on
CPU, the transformer gets its hook correctly, GPU utilization returns to
normal.

**File ref:** `ai_enhance.py` around line 732 (Step 2 of `_load_kontext_pipeline`).

### Decision: `enable_model_cpu_offload()` over manual device placement

**Tried:** manual placement with `pipe.transformer.to("cuda")`,
`pipe.text_encoder.to("cuda")`, `pipe.vae.to("cuda")` — plus manual prompt
encoding to bypass the T5 device mismatch.

**Measured:** on 8 GB VRAM with 1.27 GB held by other processes:

```
Manual placement:  pytorch_alloc=7.32GB + other_procs=1.27GB = 8.59GB
                   → 0.59GB paged to shared GPU memory
                   → step time 108-200s with HIGH VARIANCE (paging thrash)

enable_model_cpu_offload: only transformer on GPU during denoise
                   = 6.85GB + other_procs=1.27GB = 8.12GB
                   → minimal paging
                   → step time ~7s CONSISTENT
```

**Why:** the offload hooks evict `text_encoder` + `vae` to CPU before the
denoising loop starts, so only the transformer is resident on GPU during
the hot path (all 20+ denoising steps). Hook overhead is a ONE-TIME cost
per inference (single CPU→GPU transition for transformer at start of
denoise, single GPU→CPU at end), not per-step.

**File ref:** `ai_enhance.py` around line 788 (Step 4), with a long comment
documenting the measured numbers so this doesn't get re-tried blindly.

### Decision: `MAX_SIDE = 768` on input image

**Tried:** 1024×1024 (native input resolution).

**Problem:** FLUX Kontext concatenates image tokens with text tokens,
doubling sequence length vs txt2img. Attention is `O(n²)` in sequence
length, so 1024×1024 Kontext has ~4× the attention cost of 1024×1024
txt2img. Peak activations at 1024×1024 push us over the 8 GB budget →
WDDM shared memory paging → 275s per step.

**Fix:** cap inputs at 768×768. `(1024/768)² = 1.78`, so sequence-length-squared
attention cost drops by ~3.2× on top of eliminating the paging.

**Measured step time delta:** 275s/step → 7s/step (40× improvement, most of
which is eliminating paging rather than the attention speedup itself).

**File ref:** `ai_enhance.py` around line 1100 (inference section).

### Decision: SDPA flash + mem-efficient context manager

**Tried:** diffusers default attention (math backend on Windows).

**Problem:** PyTorch's SDPA falls back to the math backend on Windows because
Flash Attention is Linux-only. Math backend uses ~3× more peak attention
memory than flash/mem-efficient.

**Fix:** wrap `pipe(...)` in `torch.nn.attention.sdpa_kernel([FLASH, EFFICIENT])`
context manager. On Windows, flash is unavailable but `EFFICIENT_ATTENTION`
is present and significantly reduces attention memory. On Linux, flash kicks
in for another ~2× speedup.

**File ref:** `ai_enhance.py` around line 1180 (inside the inference block,
wraps the actual `pipe(**kwargs)` call).

### Decision: `_auto_resize=False` + explicit `width`/`height`

**Problem:** `FluxKontextPipeline` has a private `_auto_resize=True` default
that snaps input dimensions to the model's trained aspect ratios (1024×1024,
832×1216, 1216×832, etc.). This **overrides** our 768×768 resize and can
silently re-crop the image. Reference: [diffusers issue #11886](https://github.com/huggingface/diffusers/issues/11886).

**Fix:** runtime-probe the pipeline's `__call__` signature; if `_auto_resize`
is supported, pass `False`. Also pass explicit `width=768, height=768` so
older diffusers versions that don't honor the private param still get the
right dimensions.

**File ref:** `ai_enhance.py` around line 1260 (where `_kontext_pipe_kwargs`
is built).

### Decision: Wrap `pipe(...)` in `transformer.cache_context("kontext")`

**Problem:** diffusers' `pipeline_flux.py` (txt2img) wraps its transformer
call in `self.transformer.cache_context("cond")` at line 948. But
`pipeline_flux_kontext.py` does **not** have this wrapping — it's a diffusers
oversight. If you apply any cache (FirstBlockCache, FasterCache) to the
transformer, the cache hooks call `state_manager.get_state()` and crash with
`No context is set. Please set a context before retrieving the state.`

**Fix:** wrap the entire `pipe(...)` call in an outer `cache_context("kontext")`
on the transformer. This sets the context recursively on every block hook
for the full inference. The wrapper is a no-op when no cache is applied, so
it's safe to leave in place regardless of the `KONTEXT_FBC_THRESHOLD` setting.

**File ref:** `ai_enhance.py` around line 1310 (`with _sdp_ctx, _cache_ctx:`).

### Decision: Expose `seed` + `true_cfg_scale` + `negative_prompt` as first-class Kontext parameters

**Problem observed:** after fixing `max_sequence_length` back to 512, the
user still reported "some results work, some are garbage, and lower step
counts with bad denoising give better results than high step counts."

**Root cause breakdown:** two compounding issues, both rooted in weak text
conditioning signal relative to FLUX Kontext's preservation prior:

1. **Seed variance dominates weak conditioning.** FLUX generates initial
   latent noise from a random seed. With weak text conditioning (short
   prompt, preservation-biased edit), the noise tensor mostly determines
   the output — different seeds look like "different edits" even though
   nothing else changed. Users think they're seeing parameter effects
   when they're actually seeing seed variance.

2. **Distilled guidance is not CFG and can't be turned up for stronger
   edits.** FLUX.1 is guidance-distilled — `guidance_scale` is a learned
   conditioning embedding, not classifier-free guidance. Raising it above
   ~4 degrades output instead of strengthening edits. Users expect
   "turn guidance to 11" to work like SD1.5/SDXL and are confused when it
   doesn't.

**Fix (two-part):**

1. **Seed control.** Added `seed: int | None = None` parameter to
   `instruct_edit()`. When set, creates a `torch.Generator(device="cuda")`
   with `manual_seed(seed)` and passes it to `pipe()` as `generator`.
   Locked seeds give identical output across runs. Flutter UI got a
   Seed text field + dice button (generate random int from epoch time)
   + clear button (go back to random).

2. **True CFG support.** Added `true_cfg_scale: float = 1.0` parameter
   to `instruct_edit()`. When `> 1.0` AND a `negative_prompt` is
   provided, `FluxKontextPipeline` runs the transformer twice per step
   (once conditional, once unconditional on the negative prompt) and
   extrapolates the conditional further from the unconditional. This is
   **real** classifier-free guidance, not the distilled variant. Cost:
   2× inference time. Payoff: edits that couldn't happen with distilled
   guidance alone now work. Added a Flutter slider "True CFG 1.0–6.0"
   with live-updating helper text that says "Off (distilled only)"
   below 1.01 and "X.X× — real CFG, 2× slower, stronger edits" above.

   Also hardened the existing `negative_prompt` path: if a negative
   prompt is provided but `true_cfg_scale` is 1.0, we log a warning that
   the negative prompt is being IGNORED (because distilled guidance
   doesn't use it). This was a silent footgun — users thought their
   negative prompts were being honored and they weren't.

**File refs:**
- `ai_enhance.py` lines ~1160–1165 (signature)
- `ai_enhance.py` around line 1376 (seed + generator setup)
- `ai_enhance.py` around line 1390 (true_cfg_scale + negative_prompt
  routing with a warning when neg is given but true_cfg is off)
- `flutter_client/lib/pages/editor_page.dart` around line 80 (state
  vars), 1290 (UI block), 1365 (apply-button params dict)

**Tested with:** the canonical test case that motivated all of this —
"make all the girls clothed" on a clothing-addition image — went from
"random variance, some work some don't" to "same seed + same params
produces the same output reproducibly, and with `true_cfg_scale=3.0` +
a descriptive negative prompt the edit actually commits instead of
fighting the preservation bias."

### Decision: T5 diagnostic real-token stats (excludes padding)

**Problem:** the original T5 diagnostic hook computed std over the full
padded tensor: `torch.std(embeds)` where `embeds.shape = [1, 512, 4096]`.
On a short 8-token prompt, 504 positions are padding, which dilutes the
global std toward the padding distribution rather than the real-token
signal. The reported std of 0.149 looked suspicious ("I said healthy was
0.3-2.0") but was actually close to the normal T5-XXL output scale for
short prompts.

**Fix:** the diagnostic now logs THREE things:
1. **Nonzero token count** from the input_ids (how many real tokens we have)
2. **Full-tensor stats** (the padded tensor stats) — kept for regression
   checking (NaN detection, all-zero sanity)
3. **Real-token-only stats** computed as `embeds[0, :n_real, :].std()` —
   the actual conditioning signal strength, plus a verdict label

**Empirical finding from real runs:** T5-XXL's per-element output std is
much smaller than a "typical" transformer's post-LN output. Measured
values on short Kontext prompts (~7-10 real tokens):

- Real-token std: **0.08-0.15**
- Full-tensor std: **0.10-0.18**
- Per-element max_abs: **2.0-5.0**

T5 uses **RMSNorm** (not standard LayerNorm), which normalizes differently
and produces smaller-magnitude outputs. For reference, a post-LN BERT
output would have std ≈ 1.0 per element; T5-XXL is ~5-10× smaller. This
is **normal T5-XXL behavior**, not a bug.

**Counterintuitive observation:** on short prompts, the padding positions
often have **higher** variance than the real tokens. Computing from a real
run's logs:
- full std = 0.1531, real std = 0.1018, n_real = 7
- Solving: padding std ≈ 0.154 (**higher** than real 0.102)
- Because T5's attention leaks information into padding positions and the
  layer norm doesn't zero them — T5 just produces different-magnitude
  outputs at padding slots.

This means the full-tensor std **overestimates** the real signal strength
on short prompts. Always look at the real-token-only std.

**Verdict thresholds (calibrated against working Kontext runs):**
- `real_std < 0.03` → `CRITICAL` — T5 likely broken (near-zero output)
- `real_std < 0.06` → `WEAK` — very short prompt or actual bug
- `real_std < 0.15` → `LOW-NORMAL` — short prompts naturally land here,
  consider `true_cfg_scale > 1.0` to strengthen
- `0.15 ≤ real_std < 1.0` → `HEALTHY` — typical T5-XXL output scale
- `real_std ≥ 1.0` → `HIGH` — unusually large, double-check dtype

**Initial miscalibration:** I first set `WEAK < 0.15` and `HEALTHY ≥ 0.30`
based on what I'd expect for a LayerNorm transformer. The first real run
with a 7-token prompt reported std=0.10 and got flagged as WEAK even
though the edit was actually working fine. The thresholds were adjusted
downward based on the empirical measurements above.

**File ref:** `ai_enhance.py` around line 1395 (`_t5_forward_hook`
implementation, splits full vs real-token stats with calibrated verdict).

**Lesson #1:** when computing stats over a tensor that includes padding,
either mask the padding or use an attention-mask-aware reduction. Don't
assume padding positions are zero.

**Lesson #2:** don't calibrate diagnostic thresholds from theoretical
expectations. Measure real working runs first, then set thresholds.
My initial "std < 0.30 is LOW" threshold was wrong because T5-XXL
doesn't output at LayerNorm-scaled magnitudes.

### Decision: `max_sequence_length=512` (not 256)

**Tried:** `max_sequence_length=256` to save a small amount of T5 encoder
memory and compute (the user's prompts are all short; 256 tokens is plenty
to tokenize "make all the girls clothed" with tons of padding room).

**Problem:** symptom was "edits are extremely weak — output looks almost
identical to input, regardless of step count or guidance." Infrastructure
was perfect (caching disabled, SDPA flash+efficient active, 7s/step
consistent, no paging, T5 on CPU, transformer pinned via cpu_offload). The
pipeline was mechanically fine. The edits just weren't happening.

**Root cause:** FLUX.1-Kontext-dev was **trained with 512-token T5
sequences**. The transformer's cross-attention layers learned a specific
distribution of text tokens to attend to — the positional structure and
density of the 512-slot embedding. When we passed `max_sequence_length=256`,
the T5 output was `[1, 256, 4096]` instead of `[1, 512, 4096]`, which is a
different distribution than the trained one. The cross-attention still
mathematically works (it handles variable sequence length), but the text
conditioning signal is effectively **half-strength** compared to what the
model saw during training. Combined with Kontext's preservation prior
(which already biases toward "don't change the input much"), the weakened
conditioning means the preservation wins and the edit barely fires.

**Fix:** changed to `max_sequence_length=512` (the pipeline's default).

**Measured impact:** pre-change, a prompt like "make all the girls clothed"
at `guidance=2.9, steps=20` produced visually indistinguishable output from
the input. Post-change, the same prompt actually applies the edit.

**File ref:** `ai_enhance.py` around line 1341 (`_kontext_pipe_kwargs` dict
construction). The comment above the dict documents this with the full
rationale.

**Lesson generalized:** diffusers pipelines expose many "tunable" parameters
that look like they should be safe to lower for memory savings. They are
**not** always safe — many of them encode training-time assumptions the
model depends on. Always check the pipeline's default before overriding.
For FLUX specifically, the defaults in `FluxKontextPipeline.__call__` match
what BFL used during training. Deviate at your own risk.

### Decision: T5 diagnostic hook in inference path

**Tried:** the previous "weak edits" symptom was initially mistaken for a
FirstBlockCache bug (it was — the 0.2s cache hit pattern confirmed it), and
then for a guidance scale issue. In each case we had to guess and test.

**Fix:** installed a one-shot `register_forward_pre_hook` + `forward_hook`
on `pipe.text_encoder_2` (T5) that fires on the first forward call per
inference, logs:
- Input token IDs: shape, nonzero token count, first 12 tokens
- Output embedding: shape, dtype, device, mean, std, min, max, nan_count,
  all_zero flag

The hook removes itself after the first call so subsequent inferences don't
accumulate hooks.

Also added an explicit `[KONTEXT DEBUG] pipe() kwargs: ...` log line right
before the `pipe(**kwargs)` call that dumps the prompt text, prompt length,
guidance, steps, max_sequence_length, and image dimensions — so you can
verify the kwargs dict with one glance at the log instead of stepping
through the code.

**File ref:** `ai_enhance.py` around line 1395 (inside the inference
section, right before the `with _sdp_ctx, _cache_ctx:` block).

**How to use:** the next time edits are weak, check the log for:

```
[KONTEXT DEBUG] pipe() kwargs: prompt='make all the girls clothed' len=26, guidance=2.90, steps=20, max_seq=512, WxH=768x768, image_mode=RGB
[KONTEXT DEBUG] Installed one-shot T5 diagnostic hook ...
[KONTEXT DEBUG] T5 input_ids: shape=(1, 512), nonzero_token_count=8, first_tokens=[4535, 66, 8, 5484, 7828, 26, 1, 0, 0, 0, 0, 0]
[KONTEXT DEBUG] T5 output: shape=(1, 512, 4096) dtype=torch.bfloat16 device=cuda:0 mean=-0.0021 std=0.8473 min=-5.312 max=4.875 nan=0 all_zero=False
```

Healthy values:
- `max_seq=512` (not 256)
- `nonzero_token_count` matches your prompt length (+ 1 for EOS)
- `std` in 0.3-2.0 range
- `nan=0`
- `all_zero=False`

If any of those look wrong, you've narrowed the bug to T5 rather than the
transformer / cache / scheduler.

### Decision: FirstBlockCache DISABLED by default

**Tried:** `FirstBlockCache(threshold=0.08)` for a ~30-50% speedup per the
TeaCache paper.

**Problem:** measured with real Kontext edit runs, higher step counts made
edits **weaker**, not stronger, which is backwards from normal behavior.
Investigation of the log pattern (look for steps with `0.2s` duration vs
`7.0s` duration) revealed the cache was skipping increasingly more
transformer forward passes as step count grew:

| Steps | Real forward passes | Cache hits | Hit rate |
|---|---|---|---|
| 20 | 13 | 6 | 32% |
| 24 | 14 | 9 | 39% |
| 35 | 19 | 15 | **44%** |

**Root cause:** FBC is designed for **txt2img generation**, where successive
denoising steps produce similar residuals in the middle of the schedule
(the model is just refining already-formed features). For that use case,
skipping ~30-50% of forward passes costs almost nothing.

Image **editing** is the opposite. Big residual changes happen at the steps
where the model is rearranging content (adding clothes, changing scene,
etc.). Those are exactly the steps the cache threshold was tripping on as
"similar enough, skip." With more steps (smaller sigma intervals), successive
residuals naturally look closer → more cache hits → more skipped work → the
edit never actually happens.

**Fix:** disabled the cache by default. Added `KONTEXT_FBC_THRESHOLD` env
var to opt back in for users who explicitly want the speedup over edit
strength.

**File ref:** `ai_enhance.py` around line 855 (Step 5 of `_load_kontext_pipeline`).

### Decision: `DIFFUSERS_GGUF_CUDA_KERNELS` NOT set on Windows

**Tried:** `DIFFUSERS_GGUF_CUDA_KERNELS=true` + `pip install kernels` to get
the fused dequant+matmul CUDA kernel (the real source of ComfyUI's speed
advantage over diffusers Python GGUF).

**Problem:** setting this env var triggers `get_kernel("Isotr0py/ggml")` at
`diffusers/quantizers/gguf/utils.py` line 40 — which is called at diffusers
import time, not lazily. The HF Kernels repo has **Linux-only build
variants**. On Windows with torch 2.11 + CUDA 13, the kernel download fails
and **breaks `from diffusers import ...`** entirely.

**Fix:** leave the env var unset on Windows. Documented the recommendation
to enable it on Linux/WSL2 where it Just Works.

**Note:** the `kernels` Python package itself is harmless and remains
installed — it's only the repo-specific kernel download that fails.

### Decision: Ollama eviction before Kontext load

**Problem:** if Ollama has an LLM loaded on the GPU (7 GB for a 7B-param
model), there's no room for the 6.7 GB Kontext transformer. Ollama doesn't
automatically unload when another process needs VRAM.

**Fix:** added `_evict_ollama_from_gpu()` which sends `POST /api/generate`
with `keep_alive: 0` to every model listed in `/api/ps`. This evicts Ollama
models **without terminating Ollama** — the model reloads on the next chat
request. Called twice: once before pipeline load, once right before inference
(in case Ollama reloaded during the 10-15s pipeline assembly).

**No equivalent for LM Studio** — LM Studio doesn't expose an eviction API.
Users must close LM Studio or unload models in its UI manually. The VRAM
precheck catches this and reports the conflicting processes by name via
`nvidia-smi`.

**File ref:** `ai_enhance.py` around line 500 (`_evict_ollama_from_gpu`).

### Potential future work (documented but NOT implemented)

1. **Manual placement with aggressive VAE/CLIP eviction mid-pipeline.**
   Currently the `prompt` argument goes through the pipeline's `encode_prompt`,
   which means CLIP must be on GPU during encoding. If we pre-compute the
   embeddings ourselves (T5 on CPU, CLIP on CPU too with a brief bounce),
   pass `prompt_embeds` + `pooled_prompt_embeds`, and then evict CLIP + VAE
   before the denoising loop, we could keep a larger transformer on the GPU
   (Q4_K_S or Q4_K_M) permanently without cpu_offload overhead. This was
   tried once and measured worse on 8 GB due to VRAM pressure, but would
   likely win on 10-12 GB cards. See the git history for the
   `_kontext_manual_placement` flag and manual encoding branch — the code
   was removed but can be resurrected if a bigger card shows up.

2. **torchao int4 quantization.** Would require downloading the full 24 GB
   bf16 checkpoint and quantizing in place. Gives native CUDA kernels with
   no Python dequant overhead (unlike GGUF on Windows). Not done yet due to
   disk space constraints.

3. **`torch.compile(mode="reduce-overhead")` on the transformer.** Could give
   an additional ~1.5-2× speedup on top of everything else, but interacts
   poorly with `enable_model_cpu_offload()` because the hook moves invalidate
   compile caches. Would need to be combined with manual placement (item 1).

4. **SageAttention** (separate Windows CUDA kernel package, already installed
   at `sageattention-2.2.0+cu130torch2.11.0-cp311-cp311-win_amd64.whl` in the
   repo root for image generation). Could replace SDPA in the Kontext
   inference for another ~1.5× attention speedup on Windows. Not yet wired
   into `ai_enhance.py`.

5. **Lower `MAX_SIDE` to 640 or 512** as user-selectable quality presets
   ("Fast" / "Balanced" / "Quality") exposed via the Flutter UI. Currently
   hardcoded in `ai_enhance.py`.

---

## 10. Log line reference

When a Kontext edit runs, you'll see a structured log sequence prefixed with
`[KONTEXT]`. Here's what each section means and what to look for.

### Pipeline load phase

```
[KONTEXT] ====== Starting Kontext pipeline load ======
[KONTEXT] Before load — VRAM: driver_free=X.XXGB driver_used=X.XXGB (pytorch=X.XXGB other_procs=X.XXGB) total=8.6GB
[KONTEXT] Ollama: <eviction message>
[KONTEXT] After unloading other pipelines + Ollama eviction — VRAM: ...
[KONTEXT] Active GGUF variant: Q3_K_S (~4.9GB, ...)
[KONTEXT] VRAM precheck: X.XXgb free / 8.6GB total (need ~5.2GB)
[KONTEXT] System RAM: total=34.0GB, available=X.XGB
[KONTEXT] Step 0: Checking/downloading GGUF transformer file (Q3_K_S)...
[KONTEXT] Step 0: GGUF at <path> (0.2s)                        ← cached
[KONTEXT] Step 1: Loading GGUF Q3_K_S transformer (~4.9GB, will be offloaded CPU→GPU)...
[KONTEXT] Step 1 done: transformer on CPU (3.3s)
[KONTEXT] Step 2: Loading T5 encoder on CPU (bf16, ~9.5GB system RAM — no GPU needed)...
[KONTEXT] Step 2 done: T5 on CPU (0.3s)
[KONTEXT] T5 device: cpu                                        ← must be cpu
[KONTEXT] Step 3: Assembling FluxKontextPipeline...
[KONTEXT] Step 3 done: pipeline assembled (0.5s)
[KONTEXT] Step 4: Enabling model_cpu_offload (T5 routed via accelerate+hf_device_map)...
[KONTEXT] enable_model_cpu_offload OK — only transformer occupies GPU during denoise
[KONTEXT] Step 5: Transformer caching DISABLED (default) — ...
[KONTEXT] Component 'text_encoder' device: cpu                  ← all CPU after offload setup
[KONTEXT] Component 'text_encoder_2' device: cpu                ← critical: T5 must be cpu
[KONTEXT] Component 'transformer' device: cpu                   ← will cycle via hook
[KONTEXT] Component 'vae' device: cpu
[KONTEXT] ====== Pipeline ready — cpu_offload active, T5 on CPU (total: X.Xs) ======
```

**What to check:**
- `driver_free` before load should be ≥ `Step 1` transformer size + ~0.3 GB
- `Active GGUF variant` should match your `KONTEXT_GGUF_QUANT` env var
- `T5 device` must be `cpu`
- `Step 5` should say `DISABLED (default)` unless you explicitly opted into FBC
- If a VRAM pressure warning appears, take it seriously

### Inference phase

```
[KONTEXT] ====== Starting edit ======
[KONTEXT] Instruction: '<your prompt>'
[KONTEXT] Params: steps=20, guidance=2.7, input_size=1024x1024
[KONTEXT] Pipeline ready (0.0s)                                 ← cached pipeline
[KONTEXT] Image resized 1024x1024 → 768x768 (MAX_SIDE=768 for VRAM/activation management)
[KONTEXT] Pre-inference check: T5 device=cpu (should be cpu)    ← sanity check
[KONTEXT] Starting pipeline — model_cpu_offload active, SDPA flash/mem-efficient forced
[KONTEXT] SDPA backend: torch.nn.attention.sdpa_kernel(FLASH + EFFICIENT)
[KONTEXT DEBUG] Installed one-shot T5 diagnostic hook ...
[KONTEXT DEBUG] pipe() kwargs: prompt='...' len=N, guidance=X.XX, steps=N, max_seq=512, WxH=768x768, image_mode=RGB
[KONTEXT] _auto_resize=False accepted (diffusers supports it)
[KONTEXT DEBUG] T5 input_ids: shape=(1, 512), nonzero_token_count=N, first_tokens=[...]
[KONTEXT DEBUG] T5 output: shape=(1, 512, 4096) dtype=torch.bfloat16 device=cuda:0 mean=X.XXXX std=X.XXXX min=X.XXX max=X.XXX nan=0 all_zero=False
[KONTEXT] Step 1 WARMUP/20 — XX.Xs — VRAM: X.XXGB               ← warmup (JIT compile, cuDNN autotune)
[KONTEXT] Step 2/20 — X.Xs — VRAM: X.XXGB                       ← steady state
[KONTEXT] Step 3/20 — X.Xs — VRAM: X.XXGB
...
[KONTEXT] Step 20/20 — X.Xs — VRAM: X.XXGB
[KONTEXT] Inference complete (XX.Xs)
[KONTEXT] Timing summary: warmup=XX.Xs | steady-state mean=X.Xs min=X.Xs max=X.Xs over 19 steps | total=XX.Xs
[KONTEXT] ====== Edit complete ======
```

**What to check:**
- Step 1 is ~5-6× slower than subsequent steps (normal warmup)
- Steady-state steps should be **consistent** — ~7s each for Q3_K_S at 768×768
- If you see `0.2s` steps mixed in, FBC is active (should only happen if you
  opted in via `KONTEXT_FBC_THRESHOLD`)
- If `steady-state mean` exceeds 30s/step, something's wrong — likely VRAM
  pressure causing paging. Check `other_procs` in the VRAM log and the
  timing max/min spread

**What to check on the `[KONTEXT DEBUG]` lines:**
- `max_seq=512` (NOT `256` — that was a historical bug that weakened edits)
- `nonzero_token_count` in T5 input_ids is small but non-zero (your prompt
  length in tokens, typically 5-30 for short instructions, + 1 for EOS)
- T5 real-token std verdict: `HEALTHY` or `LOW-NORMAL` (short prompts
  naturally land in LOW-NORMAL, which is fine — just means you may want
  true_cfg > 1.0 to strengthen). `WEAK` or `CRITICAL` indicates a real
  problem.
- T5 output `nan=0` and `all_zero=False`
- T5 output `shape=(1, 512, 4096)` (matches `max_seq`)

**T5-XXL output is naturally small-magnitude.** Typical values:
- 7-token prompt: `real_std ≈ 0.08-0.15` (LOW-NORMAL verdict)
- 20-token prompt: `real_std ≈ 0.15-0.25` (HEALTHY)
- 50-token prompt: `real_std ≈ 0.20-0.35` (HEALTHY)
- `max_abs` typically 2–6 across these ranges

If verdict is `WEAK` or `CRITICAL`, the text conditioning is broken and no
amount of step count / guidance tuning will fix the output. The most
common failure mode is `max_seq=256` from a stale config or override —
grep the code for `max_sequence_length` to find and remove the regression.
The second most common is all-zero T5 output, which means something is
corrupting the input_ids before they reach T5.

---

## Quick reference card

Stick this on a post-it next to your monitor:

| Parameter | Default | Best value for 8 GB |
|---|---|---|
| `KONTEXT_GGUF_QUANT` | `Q4_K_S` | **`Q3_K_S`** |
| `KONTEXT_FBC_THRESHOLD` | unset | **unset** (disabled) |
| `DIFFUSERS_GGUF_CUDA_KERNELS` | unset | unset on Windows, `true` on Linux |
| `guidance_scale` (API param) | — | **2.5 – 3.5** |
| `num_inference_steps` (API param) | — | **20 – 28** |
| Input resolution (auto-resized) | — | 768×768 cap |

**If edits look weak:**
1. **Lock a seed first** so you can tell variance from real effects — click the dice icon in the Kontext panel
2. Check `[KONTEXT DEBUG] pipe() kwargs:` log line — `max_seq` **must be 512**
3. Check `KONTEXT_FBC_THRESHOLD` is unset
4. Check `guidance_scale` is 2.5–3.5 (higher won't help — use `true_cfg_scale` instead)
5. Check the `[KONTEXT DEBUG] T5 stats (REAL tokens only, n=N)` line — verdict should be `HEALTHY`. If `WEAK` or `LOW`, lengthen the prompt.
6. If the prompt is short, **lengthen it** — aim for 30–60 real tokens, not 5–10
7. Switch from delta-style to **target-state** framing ("a photograph of X") instead of ("make it X")
8. If all of the above still produces weak edits, **enable True CFG**: set `true_cfg_scale=2.5-4.0` with a descriptive negative prompt like `"unchanged, same as input, blurry"`. Doubles inference time but physically pushes the latent away from the "do nothing" solution.
9. Remember Kontext has a preservation prior — "add new content" edits are genuinely harder for it than style/color/background changes

**If inference is slow:** check `other_procs` VRAM; close LM Studio / browser; drop to Q2_K.
**If you hit OOM:** switch to smaller quant, close GPU apps, or drop input resolution.
