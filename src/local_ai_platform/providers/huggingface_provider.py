from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Generator

from ..config import get_settings
from .base import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    GenerationSettings,
    ModelCapabilities,
    ModelInfo,
)

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Transformers provider with quantization, FlashAttention,
    TurboQuant KV cache compression, and speculative decoding.

    Optimizations based on:
        "Algorithms that make powerful local LLMs feasible on 16-32 GB laptops
         with 4-8 GB GPUs (2022-2026)"

    Key improvements over baseline:
    - NF4/GPTQ/AWQ auto-detection and loading (4x weight memory reduction)
    - TurboQuant KV cache compression: 3-bit, ~6x KV memory reduction (ICLR 2026)
    - FlashAttention-2 / SDPA auto-enablement (2-4x attention speedup)
    - Speculative decoding via assistant_model (2-3x decode speedup)
    - Fixed: cache key no longer includes settings (prevents duplicate model loads)
    - Model unloading support for memory management
    - System-aware defaults from hardware detection
    """

    provider_name = "huggingface"

    def __init__(
        self,
        default_model: str = "google/flan-t5-base",
        model_catalog: str = "",
        device: str = "auto",
        low_memory: bool = True,
        cpu_offload: bool = True,
        cache_dir: str = "",
        api_token: str = "",
    ) -> None:
        self.default_model = default_model
        self.model_catalog_str = model_catalog
        self.device = device
        self.low_memory = low_memory
        self.cpu_offload = cpu_offload
        self.cache_dir = cache_dir or ""
        self.api_token = api_token
        # ── Caches keyed by model ID only (NOT by settings) ──
        self._pipeline_cache: dict[str, Any] = {}
        self._model_cache: dict[str, Any] = {}      # Raw model objects for streaming
        self._tokenizer_cache: dict[str, Any] = {}
        self._metadata_cache: dict[str, dict[str, Any]] = {}
        self._draft_model_cache: dict[str, Any] = {}  # For speculative decoding
        self._turboquant_applied: dict[str, bool] = {}  # Track TurboQuant state per model

    def configured_models(self) -> list[str]:
        models = [p.strip() for p in self.model_catalog_str.split(",") if p.strip()]
        if self.default_model and self.default_model not in models:
            models.insert(0, self.default_model)
        return models

    # ── Model Loading with Optimizations ──────────────────────────

    def _detect_quantization_config(self, model: str) -> dict[str, Any] | None:
        """Auto-detect optimal quantization for model loading.

        Priority chain (tries best → falls back):
        1. Pre-quantized GPTQ models → GPTQConfig
        2. Pre-quantized AWQ models → AwqConfig
        3. TorchAO INT4 (PyTorch-native, composable with torch.compile) → TorchAoConfig
        4. bitsandbytes NF4 (widest compatibility) → BitsAndBytesConfig
        """
        model_lower = model.lower()

        # Pre-quantized GPTQ models
        if "gptq" in model_lower:
            try:
                from transformers import GPTQConfig
                return {"quantization_config": GPTQConfig(bits=4, disable_exllama=True)}
            except ImportError:
                logger.info("auto-gptq not installed — loading GPTQ model without optimization")
                return None

        # Pre-quantized AWQ models
        if "awq" in model_lower:
            try:
                from transformers import AwqConfig
                return {"quantization_config": AwqConfig(bits=4)}
            except ImportError:
                logger.info("autoawq not installed — loading AWQ model without optimization")
                return None

        # For large models on constrained hardware: auto-quantize to INT4
        try:
            from local_ai_platform.system_info import get_cached_hardware
            hw = get_cached_hardware()
            gpu_vram_gb = hw.best_gpu_vram_mb / 1024 if hw.best_gpu_vram_mb else 0
        except Exception:
            gpu_vram_gb = 0

        if gpu_vram_gb <= 0 or gpu_vram_gb > 8:
            return None  # No GPU or plenty of VRAM — no quantization needed

        # Check if model is large enough to benefit from quantization
        if not any(s in model_lower for s in ("7b", "8b", "13b", "14b", "70b")):
            return None

        # Try TorchAO INT4 first (PyTorch-native, ~1.7x speedup, composable with torch.compile)
        try:
            from transformers import TorchAoConfig
            from torchao.quantization import Int4WeightOnlyConfig
            logger.info("Applying TorchAO INT4 quantization for %s (%.1f GB VRAM)", model, gpu_vram_gb)
            return {
                "quantization_config": TorchAoConfig(
                    quant_type=Int4WeightOnlyConfig(group_size=128),
                )
            }
        except ImportError:
            pass

        # Fallback: bitsandbytes NF4 (widest compatibility)
        try:
            import bitsandbytes  # noqa: F401
            from transformers import BitsAndBytesConfig
            logger.info("Applying NF4 quantization for %s (%.1f GB VRAM)", model, gpu_vram_gb)
            return {
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=self._best_compute_dtype(),
                    bnb_4bit_use_double_quant=True,
                )
            }
        except ImportError:
            pass

        return None

    @staticmethod
    def _best_compute_dtype():
        """Return the best compute dtype for the current GPU."""
        try:
            import torch
            if torch.cuda.is_available():
                cap = torch.cuda.get_device_capability()
                if cap[0] >= 8:  # Ampere+ supports bf16 natively
                    return torch.bfloat16
                return torch.float16
        except Exception:
            pass
        return None  # Let transformers decide

    def _select_attn_implementation(self, model: str) -> str | None:
        """Select the best attention implementation for the hardware.

        Priority: flash_attention_2 > sdpa > eager (default)
        Based on: FlashAttention-2 paper (Dao, 2023) — 2-4x attention speedup
        """
        try:
            import torch
            if not torch.cuda.is_available():
                return "sdpa"  # PyTorch native SDPA works on CPU too

            cap = torch.cuda.get_device_capability()
            if cap[0] >= 8:  # Ampere+ (compute capability 8.0+)
                try:
                    import flash_attn  # noqa: F401
                    logger.info("Using FlashAttention-2 for %s", model)
                    return "flash_attention_2"
                except ImportError:
                    pass

            # SDPA (Scaled Dot-Product Attention) — built into PyTorch 2.0+
            if hasattr(torch.nn.functional, "scaled_dot_product_attention"):
                return "sdpa"
        except Exception:
            pass

        return None  # Use model default

    def _get_pipeline(self, model: str, settings: GenerationSettings) -> Any:
        """Load model pipeline with quantization and attention optimizations.

        FIXED: Cache key is model ID only — different settings reuse same model.
        """
        if model in self._pipeline_cache:
            logger.debug("HF pipeline cache hit: %s", model)
            return self._pipeline_cache[model]

        import time as _time
        from transformers import pipeline, AutoTokenizer

        logger.info("Loading HF model: %s (device=%s, low_memory=%s)", model, self.device, self.low_memory)
        t_load_start = _time.monotonic()

        # Measure VRAM before load
        vram_before = 0
        try:
            import torch
            if torch.cuda.is_available():
                vram_before = torch.cuda.memory_allocated() // (1024 * 1024)
        except Exception:
            pass

        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._tokenizer_cache[model] = tokenizer

        task_order = ["text-generation", "text2text-generation"]
        pipe = None
        last_error = None

        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": self.device,
        }
        if self.low_memory:
            model_kwargs["low_cpu_mem_usage"] = True

        # ── Quantization (GPTQ/AWQ/TorchAO/NF4) ──
        quant_config = self._detect_quantization_config(model)
        if quant_config:
            model_kwargs.update(quant_config)
            logger.info("HF quantization: %s", type(quant_config.get("quantization_config")).__name__)

        # ── Attention implementation (FlashAttention-2 / SDPA) ──
        attn_impl = self._select_attn_implementation(model)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl
            logger.info("HF attention: %s", attn_impl)

        for task in task_order:
            try:
                pipe = pipeline(
                    task=task,
                    model=model,
                    tokenizer=tokenizer,
                    model_kwargs=model_kwargs,
                )
                break
            except Exception as exc:
                last_error = exc

        if pipe is None:
            raise RuntimeError(f"Cannot load HF model `{model}`: {last_error}")

        # Log load metrics
        t_load_elapsed = _time.monotonic() - t_load_start
        vram_after = 0
        try:
            import torch
            if torch.cuda.is_available():
                vram_after = torch.cuda.memory_allocated() // (1024 * 1024)
        except Exception:
            pass
        logger.info("HF model loaded: %s in %.1fs (VRAM: %d→%d MB, +%d MB)",
                     model, t_load_elapsed, vram_before, vram_after, vram_after - vram_before)

        # ── TurboQuant KV cache compression (ICLR 2026) ──
        # Compresses KV cache to 3-4 bits via PolarQuant + QJL residual.
        # ~6x KV memory reduction, zero calibration needed, works on any model.
        # Applied once at model load time based on system hardware detection.
        pipe = self._apply_turboquant(pipe, model)
        self._turboquant_applied[model] = True

        self._pipeline_cache[model] = pipe
        return pipe

    def _apply_turboquant(self, pipe_or_model: Any, model_name: str) -> Any:
        """Apply TurboQuant KV cache compression if available.

        TurboQuant (Google, ICLR 2026) compresses KV cache to 3-4 bits using
        PolarQuant (Walsh-Hadamard rotation) + QJL (1-bit residual correction).
        ~6x KV memory reduction with near-zero quality loss.

        Bit width is auto-detected from system hardware:
        - ≤8 GB VRAM or ≤12 GB RAM → 3-bit (max compression)
        - ≥8 GB VRAM or ≥24 GB RAM → 4-bit (best quality)
        """
        try:
            import turboquant
        except ImportError:
            return pipe_or_model  # Not installed, skip silently

        try:
            # Auto-detect optimal bit width from hardware
            bit_width = 3  # Default: aggressive compression
            try:
                from local_ai_platform.system_info import get_cached_hardware
                hw = get_cached_hardware()
                gpu_vram_gb = hw.best_gpu_vram_mb / 1024 if hw.best_gpu_vram_mb else 0
                ram_gb = hw.ram_total_mb / 1024
                if gpu_vram_gb >= 8 or ram_gb >= 24:
                    bit_width = 4  # More VRAM → higher quality 4-bit
            except Exception:
                pass

            # For pipeline objects, wrap the underlying model
            target = pipe_or_model
            if hasattr(pipe_or_model, "model"):
                target = pipe_or_model.model

            wrapped = turboquant.wrap(
                target,
                bit_width=bit_width,
                n_outlier_channels=8,  # Preserve high-magnitude KV channels
            )

            # Put wrapped model back into pipeline
            if hasattr(pipe_or_model, "model") and wrapped is not target:
                pipe_or_model.model = wrapped
                logger.info(
                    "TurboQuant KV cache compression enabled for %s (%d-bit, ~%.0fx reduction)",
                    model_name, bit_width, 16 / bit_width,
                )
            elif wrapped is not target:
                logger.info(
                    "TurboQuant KV cache compression enabled for %s (%d-bit)",
                    model_name, bit_width,
                )
                return wrapped

        except Exception as exc:
            logger.debug("TurboQuant wrapping failed for %s: %s", model_name, exc)

        return pipe_or_model

    # (instance-level _turboquant_applied is set in __init__)

    def _get_model_for_streaming(self, model: str) -> Any:
        """Get or load model for streaming generation. Reuses pipeline model if available."""
        if model in self._model_cache:
            return self._model_cache[model]

        # Reuse from pipeline cache if already loaded (TurboQuant already applied)
        if model in self._pipeline_cache:
            gen_model = self._pipeline_cache[model].model
            self._model_cache[model] = gen_model
            return gen_model

        # Load fresh with same optimizations as pipeline
        from transformers import AutoModelForCausalLM

        model_kwargs: dict[str, Any] = {
            "torch_dtype": "auto",
            "device_map": self.device,
            "low_cpu_mem_usage": self.low_memory,
        }
        quant_config = self._detect_quantization_config(model)
        if quant_config:
            model_kwargs.update(quant_config)
        attn_impl = self._select_attn_implementation(model)
        if attn_impl:
            model_kwargs["attn_implementation"] = attn_impl

        gen_model = AutoModelForCausalLM.from_pretrained(model, **model_kwargs)
        # Apply TurboQuant KV cache compression for streaming path too
        gen_model = self._apply_turboquant(gen_model, model)
        self._turboquant_applied[model] = True
        self._model_cache[model] = gen_model
        return gen_model

    # ── Speculative Decoding ──────────────────────────────────────

    # Draft models for speculative decoding: {main_model_pattern: draft_model_id}
    _DRAFT_MODEL_MAP: dict[str, str] = {
        "llama-3": "meta-llama/Llama-3.2-1B",
        "llama-2": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "mistral": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "gemma": "google/gemma-2-2b",
        "phi": "microsoft/phi-2",
        "qwen": "Qwen/Qwen2.5-0.5B",
    }

    def _get_draft_model(self, main_model: str) -> Any | None:
        """Load a small draft model for speculative decoding.

        Speculative decoding uses a fast "draft" model to propose tokens,
        then the main model verifies in fewer passes → 2-3x decode speedup.
        Paper: "Accelerating LLM Decoding with Speculative Sampling" (Chen et al., 2023)
        """
        if main_model in self._draft_model_cache:
            return self._draft_model_cache[main_model]

        # Find matching draft model
        main_lower = main_model.lower()
        draft_id = None
        for pattern, did in self._DRAFT_MODEL_MAP.items():
            if pattern in main_lower:
                # Don't use speculative decoding if main model is already small
                if any(s in main_lower for s in ("1b", "2b", "0.5b", "tiny", "mini", "small")):
                    return None
                draft_id = did
                break

        if not draft_id:
            return None

        try:
            from transformers import AutoModelForCausalLM
            logger.info("Loading draft model %s for speculative decoding of %s", draft_id, main_model)
            draft = AutoModelForCausalLM.from_pretrained(
                draft_id,
                torch_dtype="auto",
                device_map=self.device,
                low_cpu_mem_usage=True,
            )
            self._draft_model_cache[main_model] = draft
            return draft
        except Exception as exc:
            logger.debug("Failed to load draft model %s: %s", draft_id, exc)
            return None

    # ── Tokenizer & Prompt Building ───────────────────────────────

    def _get_tokenizer(self, model: str) -> Any:
        if model in self._tokenizer_cache:
            return self._tokenizer_cache[model]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self._tokenizer_cache[model] = tokenizer
        return tokenizer

    @staticmethod
    def _messages_to_hf_format(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _build_prompt(self, model: str, messages: list[ChatMessage]) -> str:
        """Build prompt using the model's native chat template when available."""
        tokenizer = self._get_tokenizer(model)
        hf_messages = self._messages_to_hf_format(messages)

        # Try chat template first (correct for instruction-tuned models)
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                prompt = tokenizer.apply_chat_template(
                    hf_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception:
                pass

        # Try chat_template attribute
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
            try:
                prompt = tokenizer.apply_chat_template(
                    hf_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return prompt
            except Exception:
                pass

        # Fallback: build a simple prompt for base models
        parts = []
        for msg in messages:
            if msg.role == "system":
                parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")
        parts.append("Assistant:")
        return "\n\n".join(parts)

    def _supports_chat_template(self, model: str) -> bool:
        try:
            tokenizer = self._get_tokenizer(model)
            return hasattr(tokenizer, "apply_chat_template") and (
                hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
            )
        except Exception:
            return False

    # ── Chat & Streaming ──────────────────────────────────────────

    def chat(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        settings = settings or GenerationSettings()
        pipe = self._get_pipeline(model, settings)
        prompt = self._build_prompt(model, messages)

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_tokens,
            "do_sample": settings.temperature > 0,
            "temperature": max(settings.temperature, 0.01),
            "top_p": settings.top_p,
            "top_k": settings.top_k,
            "repetition_penalty": settings.repetition_penalty,
            "return_full_text": False,
        }
        if settings.seed is not None:
            import torch
            torch.manual_seed(settings.seed)

        result = pipe(prompt, **gen_kwargs)

        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                text = first.get("generated_text", "")
            elif isinstance(first, list) and first:
                text = first[0].get("generated_text", "")
            else:
                text = str(first)
        else:
            text = str(result)

        text = text.strip()

        return ChatResponse(
            content=text,
            model=model,
            provider=self.provider_name,
            usage=None,
            finish_reason="stop",
        )

    def stream(
        self,
        model: str,
        messages: list[ChatMessage],
        settings: GenerationSettings | None = None,
    ) -> Generator[str, None, None]:
        """Stream with speculative decoding support for 2-3x decode speedup."""
        settings = settings or GenerationSettings()

        try:
            from transformers import TextIteratorStreamer
            import torch
            import threading

            tokenizer = self._get_tokenizer(model)
            prompt = self._build_prompt(model, messages)
            inputs = tokenizer(prompt, return_tensors="pt")

            # Reuse model from pipeline/model cache (FIXED: no duplicate loading)
            gen_model = self._get_model_for_streaming(model)

            if hasattr(gen_model, "device"):
                inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs: dict[str, Any] = {
                **inputs,
                "max_new_tokens": settings.max_tokens,
                "do_sample": settings.temperature > 0,
                "temperature": max(settings.temperature, 0.01),
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "repetition_penalty": settings.repetition_penalty,
                "streamer": streamer,
            }

            # ── Speculative decoding: use draft model for 2-3x speedup ──
            draft_model = self._get_draft_model(model)
            if draft_model is not None:
                gen_kwargs["assistant_model"] = draft_model

            thread = threading.Thread(target=gen_model.generate, kwargs=gen_kwargs)
            thread.start()

            for text in streamer:
                if text:
                    yield text

            thread.join()

        except ImportError:
            # Fallback: no streaming, yield full response
            response = self.chat(model, messages, settings)
            yield response.content

    # ── Model Unloading ───────────────────────────────────────────

    def unload_model(self, model: str) -> None:
        """Unload a model and free GPU/CPU memory."""
        import gc

        if model in self._pipeline_cache:
            del self._pipeline_cache[model]
        if model in self._model_cache:
            del self._model_cache[model]
        if model in self._draft_model_cache:
            del self._draft_model_cache[model]
        if model in self._tokenizer_cache:
            del self._tokenizer_cache[model]

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload_all(self) -> None:
        """Unload all cached models."""
        import gc
        self._pipeline_cache.clear()
        self._model_cache.clear()
        self._draft_model_cache.clear()
        self._tokenizer_cache.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    # ── Model Listing ─────────────────────────────────────────────

    # Minimum cache size to consider a model "installed" (not just metadata).
    _MIN_INSTALLED_SIZE = 50 * 1024 * 1024  # 50 MB

    def _scan_installed_models(self) -> list[str]:
        """Scan HuggingFace cache to discover all locally installed models."""
        installed: list[str] = []
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir(cache_dir=self._hf_hub_cache())
            repos = getattr(cache_info, "repos", None) or []
            for repo in repos:
                if getattr(repo, "repo_type", "model") != "model":
                    continue
                repo_id = getattr(repo, "repo_id", None)
                if not repo_id:
                    continue
                size = getattr(repo, "size_on_disk", 0) or 0
                if size < self._MIN_INSTALLED_SIZE:
                    continue
                installed.append(str(repo_id))
        except Exception:
            try:
                hub_dir = self._hf_hub_cache()
                if hub_dir.exists():
                    for d in hub_dir.iterdir():
                        if d.is_dir() and d.name.startswith("models--"):
                            snapshots = d / "snapshots"
                            if not snapshots.exists():
                                continue
                            total = sum(f.stat().st_size for f in snapshots.rglob("*") if f.is_file())
                            if total < self._MIN_INSTALLED_SIZE:
                                continue
                            model_id = d.name.replace("models--", "").replace("--", "/", 1)
                            installed.append(model_id)
            except Exception:
                pass
        return installed

    def list_models(self) -> list[ModelInfo]:
        known_ids = set(self.configured_models())
        all_model_ids = list(known_ids)

        try:
            for model_id in self._scan_installed_models():
                if model_id not in known_ids:
                    all_model_ids.append(model_id)
                    known_ids.add(model_id)
        except Exception:
            pass

        infos = []
        for model_id in all_model_ids:
            meta = self.model_metadata(model_id)
            infos.append(ModelInfo(
                name=model_id,
                provider=self.provider_name,
                size_bytes=meta.get("size_bytes"),
                capabilities=ModelCapabilities(
                    supports_chat=True,
                    supports_tools=False,
                    supports_vision=meta.get("supports", {}).get("vision", False),
                    supports_streaming=True,
                    supports_embeddings=meta.get("supports", {}).get("embeddings", False),
                    context_length=meta.get("context_length"),
                    parameter_size=meta.get("parameters"),
                    quantization=meta.get("quantization"),
                ),
                metadata=meta,
            ))
        return infos

    def is_available(self) -> bool:
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    # ── Metadata ──────────────────────────────────────────────────

    @staticmethod
    def _hf_root() -> Path:
        # [IMPROVE-69] Read HF_HOME via AppSettings for .env parity.
        return Path(get_settings().hf_home or (Path.home() / ".cache" / "huggingface"))

    @staticmethod
    def _hf_hub_cache() -> Path:
        # [IMPROVE-69] Same for HF_HUB_CACHE.
        return Path(get_settings().hf_hub_cache or (HuggingFaceProvider._hf_root() / "hub"))

    def _scan_cache_repo(self, model_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {"installed": False, "location": None, "size_bytes": None, "resolved_snapshot_path": None}
        try:
            from huggingface_hub import scan_cache_dir
            cache_info = scan_cache_dir(cache_dir=self._hf_hub_cache())
            repos = getattr(cache_info, "repos", None) or []
            target = next((r for r in repos if getattr(r, "repo_type", "model") == "model" and getattr(r, "repo_id", None) == model_id), None)
            if target:
                out["installed"] = True
                out["location"] = str(getattr(target, "repo_path", ""))
                out["size_bytes"] = int(getattr(target, "size_on_disk", 0) or 0) or None
                snapshots = {str(getattr(r, "commit_hash", "")): str(getattr(r, "snapshot_path", "")) for r in (getattr(target, "revisions", None) or [])}
                refs = getattr(target, "refs", None) or set()
                chosen = None
                for ref in refs:
                    if getattr(ref, "ref_name", "") == "main" and getattr(ref, "commit_hash", None):
                        chosen = snapshots.get(str(getattr(ref, "commit_hash")))
                        break
                if not chosen and snapshots:
                    chosen = next(iter(snapshots.values()))
                out["resolved_snapshot_path"] = chosen
        except Exception:
            models_dir = self._hf_root() / "hub"
            safe = f"models--{model_id.replace('/', '--')}"
            candidate = models_dir / safe
            if candidate.exists():
                out["installed"] = True
                out["location"] = str(candidate)
                snapshots = candidate / "snapshots"
                if snapshots.exists():
                    all_snaps = [p for p in snapshots.iterdir() if p.is_dir()]
                    if all_snaps:
                        out["resolved_snapshot_path"] = str(all_snaps[0])
        return out

    def model_metadata(self, model_id: str, refresh: bool = False) -> dict[str, Any]:
        if not refresh and model_id in self._metadata_cache:
            return self._metadata_cache[model_id]

        info: dict[str, Any] = {
            "runtime": "transformers_local",
            "context_length": None,
            "parameters": None,
            "quantization": None,
            "size_bytes": None,
            "installed": False,
            "location": None,
            "resolved_snapshot_path": None,
            "pipeline_tag": None,
            "downloads": None,
            "likes": None,
            "library_name": None,
            "license": None,
            "source_url": f"https://huggingface.co/{model_id}",
            "supports": {
                "chat": True,
                "tools": False,
                "vision": False,
                "embeddings": "embed" in model_id.lower() or "sentence-transformers" in model_id.lower(),
                "streaming": True,
            },
            "has_chat_template": False,
            "updated_at": int(time.time()),
        }

        cache = self._scan_cache_repo(model_id)
        info.update({k: v for k, v in cache.items() if v is not None})

        snapshot = info.get("resolved_snapshot_path")
        if snapshot:
            config_path = Path(str(snapshot)) / "config.json"
            if config_path.exists():
                try:
                    cfg = json.loads(config_path.read_text(encoding="utf-8"))
                    info["context_length"] = cfg.get("max_position_embeddings") or cfg.get("n_positions") or cfg.get("seq_length")
                    hidden = cfg.get("hidden_size") or cfg.get("d_model")
                    layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
                    vocab = cfg.get("vocab_size")
                    if hidden and layers and vocab:
                        info["parameters"] = f"~{int(hidden) * int(layers) * 12 + int(vocab) * int(hidden)}"
                    q = cfg.get("quantization_config")
                    if q:
                        info["quantization"] = q.get("quant_method") or str(q)
                    model_type = str(cfg.get("model_type", "")).lower()
                    if "vision" in model_type or "llava" in model_type:
                        info["supports"]["vision"] = True
                except Exception:
                    pass

            tok_config = Path(str(snapshot)) / "tokenizer_config.json"
            if tok_config.exists():
                try:
                    tok_cfg = json.loads(tok_config.read_text(encoding="utf-8"))
                    if tok_cfg.get("chat_template"):
                        info["has_chat_template"] = True
                except Exception:
                    pass

        try:
            from huggingface_hub import model_info
            remote = model_info(model_id)
            info["pipeline_tag"] = getattr(remote, "pipeline_tag", None)
            info["downloads"] = getattr(remote, "downloads", None)
            info["likes"] = getattr(remote, "likes", None)
            info["library_name"] = getattr(remote, "library_name", None)
            info["license"] = getattr(remote, "license", None)
            if info["pipeline_tag"] in {"image-text-to-text", "image-to-text"}:
                info["supports"]["vision"] = True
            if info["pipeline_tag"] in {"feature-extraction", "sentence-similarity"}:
                info["supports"]["embeddings"] = True
        except Exception:
            pass

        self._metadata_cache[model_id] = info
        return info
