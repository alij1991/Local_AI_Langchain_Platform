import json
from pathlib import Path

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
        api_server_port=8000,
    )


def test_configured_models_includes_default():
    ctl = HuggingFaceController(_config())
    models = ctl.configured_models()

    assert models[0] == "google/flan-t5-base"
    assert "tiiuae/falcon-rw-1b" in models


def test_model_metadata_from_local_config(monkeypatch, tmp_path):
    ctl = HuggingFaceController(_config())
    model_id = "acme/test-model"
    root = tmp_path / "hub" / "models--acme--test-model" / "snapshots" / "abc"
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"hidden_size": 1024, "num_hidden_layers": 24, "vocab_size": 32000, "max_position_embeddings": 8192}), encoding="utf-8")

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    meta = ctl.model_metadata(model_id, refresh=True)

    assert meta["installed"] is True
    assert meta["context_length"] == 8192
    assert str(meta["parameters"]).startswith("~")
