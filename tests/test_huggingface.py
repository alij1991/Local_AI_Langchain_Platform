from local_ai_platform.config import AppConfig
from local_ai_platform.huggingface import HuggingFaceController


def _config() -> AppConfig:
    return AppConfig(
        ollama_base_url="http://127.0.0.1:11434",
        default_model="gemma3:1b",
        prompt_builder_model="gemma3:1b",
        hf_default_model="google/flan-t5-base",
        hf_model_catalog="google/flan-t5-base,tiiuae/falcon-rw-1b",
        hf_device="auto",
        gradio_share=False,
        gradio_server_port=7860,
    )


def test_configured_models_includes_default():
    ctl = HuggingFaceController(_config())
    models = ctl.configured_models()

    assert models[0] == "google/flan-t5-base"
    assert "tiiuae/falcon-rw-1b" in models
