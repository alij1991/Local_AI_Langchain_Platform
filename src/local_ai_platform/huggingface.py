from __future__ import annotations

from typing import Any

from .config import AppConfig


class HuggingFaceController:
    """Lazy Hugging Face local pipeline integration for LangChain."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._llm_cache: dict[str, Any] = {}

    def configured_models(self) -> list[str]:
        models = [part.strip() for part in self.config.hf_model_catalog.split(",") if part.strip()]
        if self.config.hf_default_model and self.config.hf_default_model not in models:
            models.insert(0, self.config.hf_default_model)
        return models

    def get_llm(self, model_name: str) -> Any:
        clean = model_name.strip()
        if clean in self._llm_cache:
            return self._llm_cache[clean]

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
                    device_map=self.config.hf_device,
                    model_kwargs={"torch_dtype": "auto"},
                )
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        if pipe is None:
            raise RuntimeError(f"Unable to load Hugging Face model `{clean}`: {last_error}")

        llm = HuggingFacePipeline(
            pipeline=pipe,
            pipeline_kwargs={"max_new_tokens": 256, "do_sample": False, "return_full_text": False},
        )
        self._llm_cache[clean] = llm
        return llm

    def chat(self, model_name: str, system_prompt: str, history: list[tuple[str, str]], user_input: str) -> str:
        llm = self.get_llm(model_name)
        history_text = "\n".join([f"User: {u}\nAssistant: {a}" for u, a in history])
        prompt = (
            f"System: {system_prompt}\n\n"
            f"Conversation so far:\n{history_text if history_text else '(none)'}\n\n"
            f"User: {user_input}\nAssistant:"
        )
        result = llm.invoke(prompt)
        return str(result).strip()
