from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Generator

from .base import (
    BaseProvider,
    ChatMessage,
    ChatResponse,
    GenerationSettings,
    ModelCapabilities,
    ModelInfo,
)


class HuggingFaceProvider(BaseProvider):
    """HuggingFace Transformers provider with proper chat template support."""

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
        self._pipeline_cache: dict[str, Any] = {}
        self._tokenizer_cache: dict[str, Any] = {}
        self._metadata_cache: dict[str, dict[str, Any]] = {}

    def configured_models(self) -> list[str]:
        models = [p.strip() for p in self.model_catalog_str.split(",") if p.strip()]
        if self.default_model and self.default_model not in models:
            models.insert(0, self.default_model)
        return models

    def _get_pipeline(self, model: str, settings: GenerationSettings) -> Any:
        cache_key = f"{model}:{settings.temperature}:{settings.max_tokens}"
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]

        from transformers import pipeline, AutoTokenizer

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

        self._pipeline_cache[cache_key] = pipe
        return pipe

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
        """Stream using TextIteratorStreamer for real token-by-token output."""
        settings = settings or GenerationSettings()

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
            import torch
            import threading

            tokenizer = self._get_tokenizer(model)
            prompt = self._build_prompt(model, messages)
            inputs = tokenizer(prompt, return_tensors="pt")

            # Load model (reuse from pipeline cache if available)
            cache_key = f"{model}:{settings.temperature}:{settings.max_tokens}"
            if cache_key in self._pipeline_cache:
                pipe = self._pipeline_cache[cache_key]
                gen_model = pipe.model
            else:
                gen_model = AutoModelForCausalLM.from_pretrained(
                    model,
                    torch_dtype="auto",
                    device_map=self.device,
                    low_cpu_mem_usage=self.low_memory,
                )

            if hasattr(gen_model, "device"):
                inputs = {k: v.to(gen_model.device) for k, v in inputs.items()}

            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            gen_kwargs = {
                **inputs,
                "max_new_tokens": settings.max_tokens,
                "do_sample": settings.temperature > 0,
                "temperature": max(settings.temperature, 0.01),
                "top_p": settings.top_p,
                "top_k": settings.top_k,
                "repetition_penalty": settings.repetition_penalty,
                "streamer": streamer,
            }

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

    # Minimum cache size to consider a model "installed" (not just metadata).
    # Real models are 100+ MB; metadata-only entries are a few KB.
    _MIN_INSTALLED_SIZE = 50 * 1024 * 1024  # 50 MB

    def _scan_installed_models(self) -> list[str]:
        """Scan HuggingFace cache to discover all locally installed models.

        Filters out metadata-only entries (model cards, config files) that
        are cached when browsing models but don't contain actual weights.
        """
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
                # Filter out metadata-only entries (a few KB of config/README)
                size = getattr(repo, "size_on_disk", 0) or 0
                if size < self._MIN_INSTALLED_SIZE:
                    continue
                installed.append(str(repo_id))
        except Exception:
            # Fallback: scan filesystem directly
            try:
                hub_dir = self._hf_hub_cache()
                if hub_dir.exists():
                    for d in hub_dir.iterdir():
                        if d.is_dir() and d.name.startswith("models--"):
                            # Skip incomplete downloads (no snapshots)
                            snapshots = d / "snapshots"
                            if not snapshots.exists():
                                continue
                            # Check total size — skip metadata-only
                            total = sum(f.stat().st_size for f in snapshots.rglob("*") if f.is_file())
                            if total < self._MIN_INSTALLED_SIZE:
                                continue
                            # Convert "models--org--name" back to "org/name"
                            model_id = d.name.replace("models--", "").replace("--", "/", 1)
                            installed.append(model_id)
            except Exception:
                pass
        return installed

    def list_models(self) -> list[ModelInfo]:
        # Start with configured models
        known_ids = set(self.configured_models())
        all_model_ids = list(known_ids)

        # Add any locally installed models not already in the catalog
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

    # ── Metadata (kept from your original, cleaned up) ─────────────

    @staticmethod
    def _hf_root() -> Path:
        return Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))

    @staticmethod
    def _hf_hub_cache() -> Path:
        return Path(os.getenv("HF_HUB_CACHE") or (HuggingFaceProvider._hf_root() / "hub"))

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
            # Fallback: check filesystem directly
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

        # Read config.json from snapshot
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

            # Check for chat template in tokenizer config
            tok_config = Path(str(snapshot)) / "tokenizer_config.json"
            if tok_config.exists():
                try:
                    tok_cfg = json.loads(tok_config.read_text(encoding="utf-8"))
                    if tok_cfg.get("chat_template"):
                        info["has_chat_template"] = True
                except Exception:
                    pass

        # HuggingFace Hub API metadata
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
