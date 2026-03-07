from pathlib import Path

from local_ai_platform.config import AppConfig
from local_ai_platform.images.service import ImageGenerationService


def _cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(
        ollama_base_url='http://127.0.0.1:11434',
        default_model='gemma3:1b',
        prompt_builder_model='gemma3:1b',
        hf_default_model='google/flan-t5-base',
        hf_model_catalog='google/flan-t5-base',
        hf_device='auto',
        gradio_share=False,
        gradio_server_port=7860,
        api_server_port=8000,
        local_models_dir=str(tmp_path / 'models'),
    )


def test_local_model_scan_detects_diffusers_and_transformers(tmp_path):
    cfg = _cfg(tmp_path)
    models = Path(cfg.local_models_dir)
    (models / 'img-model').mkdir(parents=True)
    (models / 'img-model' / 'model_index.json').write_text('{}', encoding='utf-8')

    (models / 'text-model').mkdir(parents=True)
    (models / 'text-model' / 'config.json').write_text('{"model_type":"llama","max_position_embeddings":4096}', encoding='utf-8')

    svc = ImageGenerationService(cfg)
    body = svc.refresh_models()

    assert any(m['model_id'] == 'local:img-model' for m in body['items'])
    assert any(m['model_id'] == 'local:text-model' for m in body['local_text_models'])


def test_doctor_reports_local_models_missing(tmp_path):
    cfg = _cfg(tmp_path)
    svc = ImageGenerationService(cfg)

    report = svc.doctor()

    assert 'checks' in report
    local_check = next(c for c in report['checks'] if c['name'] == 'local_models')
    assert local_check['ok'] is False
