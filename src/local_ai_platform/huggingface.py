"""Backward-compatible HuggingFace controller.

Wraps the new HuggingFaceProvider to maintain API compatibility
with existing code (API routes, Gradio UI, etc).
"""
from __future__ import annotations

from typing import Any

from .config import AppConfig
from .providers.huggingface_provider import HuggingFaceProvider


class HuggingFaceController:
    """Backward-compatible wrapper around HuggingFaceProvider."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._provider = HuggingFaceProvider(
            default_model=config.hf_default_model,
            model_catalog=config.hf_model_catalog,
            device=config.hf_model_device,
            low_memory=config.hf_low_memory_mode,
            cpu_offload=config.hf_enable_cpu_offload,
            cache_dir=config.hf_cache_dir,
            api_token=config.hf_api_token,
        )

    @property
    def provider(self) -> HuggingFaceProvider:
        return self._provider

    def configured_models(self) -> list[str]:
        return self._provider.configured_models()

    def get_llm(self, model_name: str, settings: dict[str, Any] | None = None) -> Any:
        """Get the underlying pipeline (for LangChain HuggingFacePipeline compatibility)."""
        from .providers.base import GenerationSettings
        s = GenerationSettings.from_dict(settings)
        return self._provider._get_pipeline(model_name, s)

    def model_metadata(self, model_id: str, refresh: bool = False) -> dict[str, Any]:
        return self._provider.model_metadata(model_id, refresh=refresh)

    def chat(
        self,
        model_name: str,
        system_prompt: str,
        history: list[tuple[str, str]],
        user_input: str,
        settings: dict[str, Any] | None = None,
    ) -> str:
        """Chat using proper chat templates (fixed from v1 raw string prompt)."""
        from .providers.base import ChatMessage, GenerationSettings

        messages = [ChatMessage(role="system", content=system_prompt)]
        for user_msg, assistant_msg in history:
            messages.append(ChatMessage(role="user", content=user_msg))
            messages.append(ChatMessage(role="assistant", content=assistant_msg))
        messages.append(ChatMessage(role="user", content=user_input))

        response = self._provider.chat(
            model_name,
            messages,
            GenerationSettings.from_dict(settings),
        )
        return response.content
