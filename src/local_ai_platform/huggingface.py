from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .config import AppConfig


class HuggingFaceController:
    """Lazy Hugging Face local pipeline integration for LangChain."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._llm_cache: dict[str, Any] = {}
        self._metadata_cache: dict[str, dict[str, Any]] = {}

    def configured_models(self) -> list[str]:
        models = [part.strip() for part in self.config.hf_model_catalog.split(",") if part.strip()]
        if self.config.hf_default_model and self.config.hf_default_model not in models:
            models.insert(0, self.config.hf_default_model)
        return models

    def _settings(self, settings: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = settings or {}
        return {
            "temperature": float(raw.get("temperature", 0.2)),
            "top_p": float(raw.get("top_p", 0.9)),
            "top_k": int(raw.get("top_k", 50)),
            "max_new_tokens": int(raw.get("max_new_tokens", raw.get("max_tokens", 256))),
            "repetition_penalty": float(raw.get("repetition_penalty", 1.05)),
            "seed": int(raw.get("seed", 42)),
        }

    def get_llm(self, model_name: str, settings: dict[str, Any] | None = None) -> Any:
        clean = model_name.strip()
        cfg = self._settings(settings)
        cache_key = f"{clean}:{json.dumps(cfg, sort_keys=True)}"
        if cache_key in self._llm_cache:
            return self._llm_cache[cache_key]

        from langchain_huggingface import HuggingFacePipeline
        from transformers import pipeline

        task_order = ["text-generation", "text2text-generation"]
        last_error: Exception | None = None
        pipe = None

        for task in task_order:
            try:
                pipe = pipeline(
                    task=task,
                    model=clean,
                    device_map=self.config.hf_model_device,
                    model_kwargs={"torch_dtype": "auto", "low_cpu_mem_usage": bool(self.config.hf_low_memory_mode)},
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if pipe is None:
            raise RuntimeError(f"Unable to load Hugging Face model `{clean}`: {last_error}")

        llm = HuggingFacePipeline(
            pipeline=pipe,
            pipeline_kwargs={
                "max_new_tokens": cfg["max_new_tokens"],
                "do_sample": cfg["temperature"] > 0,
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "top_k": cfg["top_k"],
                "repetition_penalty": cfg["repetition_penalty"],
                "return_full_text": False,
            },
        )
        self._llm_cache[cache_key] = llm
        return llm

    @staticmethod
    def _hf_root() -> Path:
        return Path(os.getenv("HF_HOME") or (Path.home() / ".cache" / "huggingface"))

    @staticmethod
    def _hf_hub_cache() -> Path:
        return Path(os.getenv("HF_HUB_CACHE") or (HuggingFaceController._hf_root() / "hub"))

    def _model_cache_dir(self, model_id: str) -> Path | None:
        models_dir = self._hf_root() / "hub"
        if not models_dir.exists():
            return None
        safe = f"models--{model_id.replace('/', '--')}"
        candidate = models_dir / safe
        if candidate.exists():
            return candidate
        return None

    def _scan_cache_repo(self, model_id: str) -> dict[str, Any]:
        out: dict[str, Any] = {
            "installed": False,
            "location": None,
            "resolved_snapshot_path": None,
            "size_bytes": None,
            "cached_files_count": None,
            "last_seen": None,
            "metadata_source": "config",
        }
        try:
            from huggingface_hub import scan_cache_dir

            cache_info = scan_cache_dir(cache_dir=self._hf_hub_cache())
            repos = getattr(cache_info, "repos", None) or []
            target = next((r for r in repos if getattr(r, "repo_type", "model") == "model" and getattr(r, "repo_id", None) == model_id), None)
            if target is not None:
                out["installed"] = True
                out["location"] = str(getattr(target, "repo_path", None) or "") or None
                out["size_bytes"] = int(getattr(target, "size_on_disk", 0) or 0) or None
                out["cached_files_count"] = len(getattr(target, "cached_files", None) or [])
                refs = getattr(target, "refs", None) or set()
                snapshots = {str(getattr(r, "commit_hash", "")): str(getattr(r, "snapshot_path", "")) for r in (getattr(target, "revisions", None) or []) if getattr(r, "commit_hash", None)}
                chosen = None
                for ref in refs:
                    if getattr(ref, "ref_name", "") == "main" and getattr(ref, "commit_hash", None):
                        chosen = snapshots.get(str(getattr(ref, "commit_hash")))
                        break
                if not chosen and snapshots:
                    chosen = next(iter(snapshots.values()))
                out["resolved_snapshot_path"] = chosen
                last_accessed = [getattr(r, "last_accessed", None) for r in (getattr(target, "revisions", None) or []) if getattr(r, "last_accessed", None)]
                if last_accessed:
                    out["last_seen"] = max(last_accessed)
                out["metadata_source"] = "hf_cache_scan"
                if out["resolved_snapshot_path"] is None:
                    out["snapshot_reason"] = "snapshot_ref_missing"
                return out
        except Exception:
            out["cache_scan_reason"] = "cache_metadata_unavailable"

        cache_dir = self._model_cache_dir(model_id)
        if not cache_dir:
            return out
        out["installed"] = True
        out["location"] = str(cache_dir)
        try:
            files = [p for p in cache_dir.rglob("*") if p.is_file()]
            out["size_bytes"] = sum(p.stat().st_size for p in files)
            out["cached_files_count"] = len(files)
            out["last_seen"] = int(max((p.stat().st_mtime for p in files), default=time.time())) if files else None
        except Exception:
            pass

        snapshots = cache_dir / "snapshots"
        if snapshots.exists():
            all_snaps = [p for p in snapshots.iterdir() if p.is_dir()]
            if all_snaps:
                main_ref = cache_dir / "refs" / "main"
                if main_ref.exists():
                    try:
                        rev = main_ref.read_text(encoding="utf-8").strip()
                        resolved = snapshots / rev
                        if resolved.exists():
                            out["resolved_snapshot_path"] = str(resolved)
                    except Exception:
                        pass
                if not out["resolved_snapshot_path"]:
                    out["resolved_snapshot_path"] = str(all_snaps[0])
        if not out.get("resolved_snapshot_path") and out.get("installed"):
            out["snapshot_reason"] = "snapshot_ref_missing"
        out["metadata_source"] = "local_cache"
        return out

    @staticmethod
    def _estimate_params(cfg: dict[str, Any]) -> str | None:
        n_params = cfg.get("num_parameters")
        if n_params:
            return str(n_params)
        hidden = cfg.get("hidden_size") or cfg.get("d_model")
        layers = cfg.get("num_hidden_layers") or cfg.get("n_layer")
        vocab = cfg.get("vocab_size")
        if hidden and layers and vocab:
            estimate = int(hidden) * int(layers) * 12 + int(vocab) * int(hidden)
            return f"~{estimate} (estimated)"
        return None

    @staticmethod
    def _safe_getattr(obj: Any, *names: str) -> Any:
        for name in names:
            if hasattr(obj, name):
                return getattr(obj, name)
        return None

    def model_metadata(self, model_id: str, refresh: bool = False) -> dict[str, Any]:
        key = model_id.strip()
        if not refresh and key in self._metadata_cache:
            return self._metadata_cache[key]

        info: dict[str, Any] = {
            "runtime": "transformers_local",
            "runtimes": ["transformers_local"],
            "context_length": None,
            "parameters": None,
            "quantization": None,
            "size_bytes": None,
            "installed": False,
            "location": None,
            "local_path": None,
            "resolved_snapshot_path": None,
            "cached_files_count": None,
            "last_seen": None,
            "snapshot_reason": None,
            "cache_scan_reason": None,
            "pipeline_tag": None,
            "downloads": None,
            "likes": None,
            "last_modified": None,
            "library_name": None,
            "license": None,
            "source_url": f"https://huggingface.co/{key}",
            "supports": {
                "chat": True,
                "tools": False,
                "vision": False,
                "json_mode": False,
                "embeddings": "embed" in key.lower() or "sentence-transformers" in key.lower(),
                "streaming": False,
            },
            "tags": ["configured"],
            "metadata_source": "config",
            "metadata_completeness": "partial",
            "estimated_fields": [],
            "updated_at": int(time.time()),
        }

        cache = self._scan_cache_repo(key)
        info.update({
            "installed": bool(cache.get("installed")),
            "location": cache.get("location"),
            "local_path": cache.get("location"),
            "resolved_snapshot_path": cache.get("resolved_snapshot_path"),
            "size_bytes": cache.get("size_bytes"),
            "cached_files_count": cache.get("cached_files_count"),
            "last_seen": cache.get("last_seen"),
            "snapshot_reason": cache.get("snapshot_reason"),
            "cache_scan_reason": cache.get("cache_scan_reason"),
            "metadata_source": cache.get("metadata_source") or info["metadata_source"],
        })

        chosen_cfg = None
        snapshot = info.get("resolved_snapshot_path")
        if snapshot:
            candidate = Path(str(snapshot)) / "config.json"
            if candidate.exists():
                chosen_cfg = candidate
        if chosen_cfg is None and info.get("location"):
            cfg_path = Path(str(info["location"])) / "snapshots"
            if cfg_path.exists():
                for snap in cfg_path.iterdir():
                    candidate = snap / "config.json"
                    if candidate.exists():
                        chosen_cfg = candidate
                        break
        if chosen_cfg and chosen_cfg.exists():
            try:
                cfg = json.loads(chosen_cfg.read_text(encoding="utf-8"))
                info["context_length"] = cfg.get("max_position_embeddings") or cfg.get("n_positions") or cfg.get("seq_length")
                info["parameters"] = self._estimate_params(cfg)
                if isinstance(info["parameters"], str) and info["parameters"].startswith("~"):
                    info["estimated_fields"].append("parameters")
                q = cfg.get("quantization_config")
                if q:
                    info["quantization"] = q.get("quant_method") or str(q)
                model_type = str(cfg.get("model_type", "")).lower()
                if "vision" in model_type or "llava" in model_type:
                    info["supports"]["vision"] = True
            except Exception:
                pass

        try:
            from huggingface_hub import model_info

            remote = model_info(key)
            pipeline_tag = self._safe_getattr(remote, "pipeline_tag")
            info["pipeline_tag"] = pipeline_tag
            if pipeline_tag in {"image-text-to-text", "image-to-text", "text-to-image"}:
                info["supports"]["vision"] = True
            if pipeline_tag in {"feature-extraction", "sentence-similarity"}:
                info["supports"]["embeddings"] = True
            info["downloads"] = self._safe_getattr(remote, "downloads")
            info["likes"] = self._safe_getattr(remote, "likes")
            info["library_name"] = self._safe_getattr(remote, "library_name")
            info["license"] = self._safe_getattr(remote, "license")
            info["last_modified"] = str(self._safe_getattr(remote, "last_modified", "lastModified") or "") or None
            remote_tags = self._safe_getattr(remote, "tags")
            if remote_tags:
                info["tags"] = sorted(set(info["tags"] + list(remote_tags)))
            info["tags"] = sorted(set(info["tags"] + ["hub"]))
            if info["installed"]:
                info["metadata_source"] = "hub+cache"
            else:
                info["metadata_source"] = "hub"
        except Exception:
            pass

        known = [
            info.get("size_bytes"),
            info.get("parameters"),
            info.get("context_length"),
            info.get("pipeline_tag"),
            info.get("downloads"),
            info.get("cached_files_count"),
            info.get("resolved_snapshot_path"),
        ]
        if sum(1 for x in known if x not in {None, ""}) >= 4:
            info["metadata_completeness"] = "good"

        self._metadata_cache[key] = info
        return info

    def chat(self, model_name: str, system_prompt: str, history: list[tuple[str, str]], user_input: str, settings: dict[str, Any] | None = None) -> str:
        llm = self.get_llm(model_name, settings=settings)
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])
        prompt = (
            f"System: {system_prompt}\n\n"
            f"Conversation so far:\n{history_text if history_text else '(none)'}\n\n"
            f"User: {user_input}\nAssistant:"
        )
        result = llm.invoke(prompt)
        return str(result).strip()
