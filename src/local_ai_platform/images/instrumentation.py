"""Shared instrumentation helpers for image pipelines.

Every generative pipeline and AI image-edit model in the platform goes
through the helpers in this module so that logs are directly comparable
across:

  - in-process diffusers runs            (service.py _run_diffusers)
  - subprocess diffusers runs            (service.py _diffusers_worker)
  - ControlNet subprocess                (service.py _controlnet_worker)
  - OpenVINO subprocess                  (service.py _openvino_worker)
  - stable-diffusion.cpp subprocess      (service.py _sdcpp_worker)
  - FLUX.1-Kontext manual loop           (ai_enhance.py _run_kontext_denoising)
  - CosXL k-diffusion manual loop        (ai_enhance.py _run_cosxl_kdiff)
  - InstructPix2Pix stock diffusers      (ai_enhance.py instruct_edit)
  - ControlNet edit stock diffusers      (ai_enhance.py instruct_edit)
  - GFPGAN / CodeFormer face restore     (ai_enhance.py)
  - RealESRGAN upscale                   (ai_enhance.py)
  - rembg / BiRefNet / U2-Net            (ai_enhance.py)
  - 4 ONNX Runtime ops                   (ai_models.py)

Two kinds of helpers:

1. **Loop instrumentation** — for pipelines that have a denoising step
   callback. Reports pre-inference audit (scheduler, component
   placements, VRAM), per-step stats (latent std/mean/min/max/NaN/Inf,
   step_dt, warmup vs steady), post-inference timing summary, and
   output coherence.

2. **Single-shot instrumentation** — for models that do a single forward
   pass (ONNX, GFPGAN, RealESRGAN, etc.). Provides a context manager
   that logs start/elapsed/vram_delta and tensor-stats helpers.

Healthy latent stats by model family (at CFG=7, sigma_data=1.0):
  - SD 1.5:   initial std ≈ 14,  final std ≈ 0.8–1.3
  - SDXL:     initial std ≈ 14,  final std ≈ 0.8–1.3
  - Flux:     initial std ≈ 1.0  (flow matching), final std ≈ 0.8–1.2
  - Z-Image:  initial std ≈ 1.0  (flow matching), final std ≈ 0.8–1.2

Failure signatures:
  - std GROWING after mid-steps        → wrong prediction_type
  - std plateauing early (step ~5)     → scheduler timestep range wrong
  - NaN/Inf                            → fp16 overflow (VAE or attention)
  - final std ≈ initial std            → denoising not happening
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator


# ── Loop instrumentation (shared with CosXL edit pipeline) ──────────


def format_latent_stats(latents: Any) -> str:
    """Format latent tensor stats as a string suitable for logging.

    Returns something like:
        latent[shape=(1, 4, 128, 128) std=1.234 mean=-0.056 min=-3.21 max=3.45 nan=0 inf=0]

    Returns an empty string if stats can't be computed (wrong tensor type,
    CUDA OOM during the stat computation, etc.).
    """
    try:
        import torch as _torch
    except Exception:
        return ""
    if latents is None or not hasattr(latents, "shape"):
        return ""
    try:
        _f = latents.detach().float()
        _nan = int(_torch.isnan(latents).sum().item())
        _inf = int(_torch.isinf(latents).sum().item())
        return (
            f" latent[shape={tuple(latents.shape)} "
            f"std={float(_f.std()):.3f} mean={float(_f.mean()):.3f} "
            f"min={float(_f.min()):.2f} max={float(_f.max()):.2f} "
            f"nan={_nan} inf={_inf}]"
        )
    except Exception:
        return ""


def format_vram_snapshot(label: str) -> str | None:
    """Return a one-line VRAM snapshot for logging, or None if no CUDA.

    Format mirrors the CosXL edit pipeline:
        Before inference — VRAM: driver_free=7.44GB driver_used=1.14GB
        (pytorch=0.00GB other_procs=1.14GB) total=8.6GB
    """
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            return None
        free, total = _torch.cuda.mem_get_info()
        alloc = _torch.cuda.memory_allocated()
        other = max(0, (total - free) - alloc)
        return (
            f"{label} — VRAM: driver_free={free / 1e9:.2f}GB "
            f"driver_used={(total - free) / 1e9:.2f}GB "
            f"(pytorch={alloc / 1e9:.2f}GB other_procs={other / 1e9:.2f}GB) "
            f"total={total / 1e9:.1f}GB"
        )
    except Exception:
        return None


def get_vram_allocated_gb() -> float | None:
    """Return torch.cuda.memory_allocated in GB, or None if no CUDA.

    Fast path — doesn't call mem_get_info (no driver round-trip). Safe
    to call inside a step callback without adding measurable overhead.
    """
    try:
        import torch as _torch
        if not _torch.cuda.is_available():
            return None
        return _torch.cuda.memory_allocated() / 1e9
    except Exception:
        return None


def format_component_placement(pipe: Any) -> list[str]:
    """Return one audit line per pipeline component showing device+dtype.

    Used for debugging "is the model actually on the device I expect"
    and "is CPU offload installing hooks on the components I think".
    """
    lines: list[str] = []
    for _name in ("text_encoder", "text_encoder_2", "text_encoder_3",
                  "tokenizer", "tokenizer_2", "tokenizer_3",
                  "unet", "transformer", "vae", "controlnet"):
        _comp = getattr(pipe, _name, None)
        if _comp is None:
            continue
        if "tokenizer" in _name:
            continue
        try:
            _p0 = next(iter(_comp.parameters()))
            _hook = getattr(_comp, "_hf_hook", None)
            _hook_str = f" offload_hook={type(_hook).__name__}" if _hook else ""
            lines.append(
                f"Component {_name}: device={_p0.device} dtype={_p0.dtype}{_hook_str}"
            )
        except StopIteration:
            continue
        except Exception:
            continue
    return lines


def summarize_scheduler(pipe: Any) -> str:
    """One-line scheduler description: class + critical config fields.

    Different families care about different fields, so we include the
    superset and let the reader filter:
      - prediction_type (epsilon / v_prediction / sample / flow)
      - beta_schedule (for beta-based schedulers)
      - timestep_spacing
      - num_train_timesteps
      - sigma_min / sigma_max / sigma_data (for EDM-family)
      - use_karras_sigmas / use_exponential_sigmas / use_beta_sigmas
    """
    _sched = getattr(pipe, "scheduler", None)
    if _sched is None:
        return "scheduler=<none>"
    _cls = type(_sched).__name__
    try:
        _cfg = dict(_sched.config)
    except Exception:
        return f"scheduler={_cls} config=<unavailable>"
    _interesting = [
        "prediction_type", "beta_schedule", "timestep_spacing",
        "num_train_timesteps", "sigma_min", "sigma_max", "sigma_data",
        "use_karras_sigmas", "use_exponential_sigmas", "use_beta_sigmas",
    ]
    _parts = [f"{_k}={_cfg[_k]}" for _k in _interesting if _k in _cfg]
    return f"scheduler={_cls} " + " ".join(_parts)


def analyze_output_coherence(image: Any) -> tuple[bool, str]:
    """Analyze a decoded PIL.Image for signs of pipeline failure.

    Returns (is_coherent, description). Description includes per-channel
    stds, overall px_mean/px_std, range, and a PASSED/FAILED verdict.

    Failure modes detected:
      - All-black or all-white (pixel range <= 2)
      - Monochrome / uniform noise (per-channel spread < 3)
      - Very low contrast (px_std < 8)
      - NaN pixels
    """
    try:
        import numpy as _np
        _arr = _np.array(image)
        if _arr.size == 0:
            return False, "Output stats: <empty array>"
        _mn = int(_arr.min())
        _mx = int(_arr.max())
        _range = _mx - _mn
        _px_mean = float(_arr.mean())
        _px_std = float(_arr.std())
        _has_nan = bool(_np.isnan(_arr.astype(_np.float32)).any())

        _per_ch_stds: list[float] = []
        _spread: float = 0.0
        if _arr.ndim >= 3 and _arr.shape[-1] >= 3:
            _per_ch_stds = [float(_arr[..., c].std()) for c in range(_arr.shape[-1])]
            _spread = max(_per_ch_stds) - min(_per_ch_stds)

        _size = getattr(image, "size", "?")
        _mode = getattr(image, "mode", "?")
        _ch_str = [f"{s:.1f}" for s in _per_ch_stds] if _per_ch_stds else "n/a"

        _stats = (
            f"Output stats: size={_size} mode={_mode} "
            f"px_mean={_px_mean:.1f} px_std={_px_std:.1f} "
            f"range=[{_mn}..{_mx}] per_channel_std={_ch_str}"
        )

        _failures: list[str] = []
        if _has_nan:
            _failures.append("NaN pixels")
        if _range <= 2:
            _failures.append(f"pixel range={_range} (<=2: blank/solid)")
        if _per_ch_stds and _spread < 3:
            _failures.append(f"per-channel spread={_spread:.1f} (<3: monochrome/noise)")
        if _px_std < 8:
            _failures.append(f"px_std={_px_std:.1f} (<8: low contrast/blank)")

        if _failures:
            return False, f"{_stats} | COHERENCE FAILED: {'; '.join(_failures)}"
        return True, f"{_stats} | coherence PASSED"
    except Exception as _err:
        return True, f"Output stats: <analysis failed: {_err}>"


# ── Loop instrumentation convenience wrappers ──────────────────────


def log_pre_inference_audit(pipe: Any, log_fn: Callable[[str], None],
                            prefix: str = "Audit") -> None:
    """Emit the pre-inference audit block (pipeline class, scheduler,
    component placements, VRAM snapshot) via `log_fn`.

    `log_fn` should accept a single string. Use a lambda that wraps
    `logger.info("[IMG] %s", m)` or the subprocess `_log(msg)` style.
    """
    try:
        log_fn(f"{prefix}: pipeline_class={type(pipe).__name__}")
        log_fn(f"{prefix}: {summarize_scheduler(pipe)}")
        for _line in format_component_placement(pipe):
            log_fn(f"{prefix}: {_line}")
        _vram = format_vram_snapshot("Before inference")
        if _vram:
            log_fn(_vram)
    except Exception as _err:
        log_fn(f"{prefix}: pre-inference audit failed: {_err}")


def log_post_inference_summary(
    warmup_sec: float,
    steady_times: list[float],
    total_sec: float,
    log_fn: Callable[[str], None],
) -> None:
    """Emit the post-inference timing summary line."""
    try:
        if steady_times:
            ss_mean = sum(steady_times) / len(steady_times)
            ss_min = min(steady_times)
            ss_max = max(steady_times)
            log_fn(
                f"Timing: warmup={warmup_sec:.2f}s steady_mean={ss_mean:.2f}s "
                f"steady_min={ss_min:.2f}s steady_max={ss_max:.2f}s "
                f"steady_n={len(steady_times)} total={total_sec:.1f}s"
            )
        else:
            log_fn(f"Timing: warmup={warmup_sec:.2f}s total={total_sec:.1f}s (no steady steps)")
        _vram = format_vram_snapshot("After inference")
        if _vram:
            log_fn(_vram)
    except Exception as _err:
        log_fn(f"Timing summary failed: {_err}")


class StepInstrumentation:
    """Accumulator used by step callbacks to record per-step timing.

    Usage:
        si = StepInstrumentation()
        def step_cb(pipe, step, timestep, cb_kwargs):
            line = si.observe(step, timestep, cb_kwargs)
            log_fn(line)
            return cb_kwargs
        # ...after inference:
        log_post_inference_summary(si.warmup_sec, si.steady_times, total, log_fn)

    The returned log line already includes `step_dt`, `WARMUP`/`STEADY`
    label, VRAM allocated, and latent stats. Pass `total_steps` to get
    the clamped "Step X/Y" prefix.
    """

    def __init__(self, total_steps: int) -> None:
        self.total_steps = int(total_steps)
        self.steady_times: list[float] = []
        self.warmup_sec: float = 0.0
        self._last_ts: float = time.time()

    def reset_clock(self) -> None:
        """Call immediately before `pipe(...)` to exclude audit overhead."""
        self._last_ts = time.time()

    def observe(self, step: int, timestep: Any, cb_kwargs: Any) -> str:
        clamped = min(step + 1, self.total_steps)
        now = time.time()
        step_dt = now - self._last_ts
        self._last_ts = now

        latents = None
        if isinstance(cb_kwargs, dict):
            latents = cb_kwargs.get("latents")
        lat_str = format_latent_stats(latents)

        vram_gb = get_vram_allocated_gb()
        vram_str = f" vram_alloc={vram_gb:.2f}GB" if vram_gb is not None else ""

        if clamped == 1:
            self.warmup_sec = step_dt
            label = "WARMUP"
        else:
            self.steady_times.append(step_dt)
            label = "STEADY"

        ts_val = float(timestep) if timestep is not None else 0.0
        return (
            f"Step {clamped}/{self.total_steps} [{label}] timestep={ts_val:.1f} "
            f"step_dt={step_dt:.2f}s{vram_str}{lat_str}"
        )


# ── Single-shot instrumentation (ONNX / GFPGAN / etc.) ──────────────


@contextmanager
def instrument_op(
    label: str,
    log_fn: Callable[[str], None],
    *,
    check_coherence_for: Any = None,
) -> Iterator[dict[str, Any]]:
    """Context manager for single-shot inference operations.

    Emits:
      - "<label>: START vram_alloc=X.XXGB"
      - "<label>: DONE elapsed=X.XXs vram_delta=+X.XXGB peak=X.XXGB"
      - "<label>: <coherence line>"  (if check_coherence_for is passed
        OR if the caller puts an 'output_image' key into the yielded dict)

    The yielded dict lets the caller stash an output image for automatic
    coherence checking on exit:

        with instrument_op("GFPGAN", _log) as ctx:
            result = model.enhance(image)
            ctx["output_image"] = result

    VRAM peak is tracked via torch.cuda.max_memory_allocated reset; it
    measures the delta over just this op's lifetime.
    """
    ctx: dict[str, Any] = {"output_image": None}
    start = time.time()

    try:
        import torch as _torch  # type: ignore
        _has_cuda = bool(_torch.cuda.is_available())
    except Exception:
        _torch = None  # type: ignore
        _has_cuda = False

    vram_before_gb: float | None = None
    if _has_cuda and _torch is not None:
        try:
            _torch.cuda.reset_peak_memory_stats()
            vram_before_gb = _torch.cuda.memory_allocated() / 1e9
        except Exception:
            vram_before_gb = None

    _vb_str = f" vram_alloc={vram_before_gb:.2f}GB" if vram_before_gb is not None else ""
    log_fn(f"{label}: START{_vb_str}")

    try:
        yield ctx
    finally:
        elapsed = time.time() - start

        vram_after_gb: float | None = None
        peak_gb: float | None = None
        if _has_cuda and _torch is not None:
            try:
                vram_after_gb = _torch.cuda.memory_allocated() / 1e9
                peak_gb = _torch.cuda.max_memory_allocated() / 1e9
            except Exception:
                pass

        parts = [f"{label}: DONE elapsed={elapsed:.2f}s"]
        if vram_before_gb is not None and vram_after_gb is not None:
            delta = vram_after_gb - vram_before_gb
            parts.append(f"vram_delta={delta:+.2f}GB")
        if peak_gb is not None:
            parts.append(f"peak={peak_gb:.2f}GB")
        log_fn(" ".join(parts))

        target = check_coherence_for if check_coherence_for is not None else ctx.get("output_image")
        if target is not None:
            try:
                _ok, _desc = analyze_output_coherence(target)
                log_fn(f"{label}: {_desc}")
            except Exception as _err:
                log_fn(f"{label}: coherence check failed: {_err}")


def format_tensor_stats(label: str, tensor: Any) -> str:
    """Format a torch tensor or numpy array for diagnostic logging.

    Returns a line like:
        label[shape=(1, 3, 512, 512) dtype=float32 device=cuda:0
        std=0.312 mean=0.487 min=0.00 max=1.00 nan=0 inf=0]
    """
    try:
        import numpy as _np
    except Exception:
        _np = None  # type: ignore

    if tensor is None:
        return f"{label}=<none>"

    # Torch tensor path
    try:
        import torch as _torch  # type: ignore
        if isinstance(tensor, _torch.Tensor):
            _f = tensor.detach().float()
            _nan = int(_torch.isnan(tensor).sum().item()) if tensor.is_floating_point() else 0
            _inf = int(_torch.isinf(tensor).sum().item()) if tensor.is_floating_point() else 0
            return (
                f"{label}[shape={tuple(tensor.shape)} dtype={tensor.dtype} "
                f"device={tensor.device} std={float(_f.std()):.3f} "
                f"mean={float(_f.mean()):.3f} min={float(_f.min()):.2f} "
                f"max={float(_f.max()):.2f} nan={_nan} inf={_inf}]"
            )
    except Exception:
        pass

    # Numpy path
    if _np is not None:
        try:
            if isinstance(tensor, _np.ndarray):
                _f32 = tensor.astype(_np.float32, copy=False)
                _nan = int(_np.isnan(_f32).sum())
                _inf = int(_np.isinf(_f32).sum())
                return (
                    f"{label}[shape={tensor.shape} dtype={tensor.dtype} "
                    f"std={float(_f32.std()):.3f} mean={float(_f32.mean()):.3f} "
                    f"min={float(_f32.min()):.2f} max={float(_f32.max()):.2f} "
                    f"nan={_nan} inf={_inf}]"
                )
        except Exception:
            pass

    return f"{label}=<unreadable type={type(tensor).__name__}>"


def format_onnx_session_info(session: Any, label: str = "onnx_session") -> list[str]:
    """Return a few diagnostic lines about an onnxruntime.InferenceSession.

    Reports: providers, input names+shapes+types, output names+shapes+types.
    One list element per line so the caller can prefix and log each.
    """
    lines: list[str] = []
    try:
        providers = session.get_providers()
        lines.append(f"{label}: providers={providers}")
    except Exception:
        pass
    try:
        for inp in session.get_inputs():
            lines.append(f"{label}: input {inp.name} shape={inp.shape} type={inp.type}")
    except Exception:
        pass
    try:
        for out in session.get_outputs():
            lines.append(f"{label}: output {out.name} shape={out.shape} type={out.type}")
    except Exception:
        pass
    return lines
