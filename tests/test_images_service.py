from pathlib import Path

from local_ai_platform.config import AppConfig
from local_ai_platform.images.service import ImageGenerationService, ImageRuntimeResult


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


def test_generate_uses_cpu_fallback_when_gpu_required_but_unavailable(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.hf_image_runtime = 'diffusers_local'
    cfg.hf_image_require_gpu = True
    cfg.hf_image_allow_cpu_fallback = True
    svc = ImageGenerationService(cfg)

    monkeypatch.setattr(svc, '_resolve_model_source', lambda model_id: ('remote', model_id))
    monkeypatch.setattr(svc, 'get_device_status', lambda: {
        'torch_installed': True,
        'cuda_available': False,
        'cuda_version': None,
        'effective_device': 'cpu',
        'torch_version': '2.10.0+cpu',
    })

    def _fake_run_diffusers(**kwargs):
        assert kwargs['device'] == 'cpu'
        return ImageRuntimeResult(
            ok=True,
            image_bytes=b'x',
            metadata={'runtime': 'diffusers_local', 'device_used': 'cpu'},
        )

    monkeypatch.setattr(svc, '_cache_dir', lambda model_id: Path('/tmp'))
    monkeypatch.setattr(svc, '_run_diffusers_isolated', _fake_run_diffusers)

    result = svc.generate(model_id='google/flan-t5-base', prompt='test')
    assert result.ok is True
    assert result.metadata and 'warning' in result.metadata


def test_validate_model_reports_missing_files(tmp_path):
    cfg = _cfg(tmp_path)
    svc = ImageGenerationService(cfg)

    models = Path(cfg.local_models_dir)
    (models / 'broken-model').mkdir(parents=True)

    report = svc.validate_model('local:broken-model')
    assert report['loadable'] is False
    assert 'invalid_model_format' in report['errors']


def test_validate_model_includes_memory_estimates(tmp_path):
    cfg = _cfg(tmp_path)
    svc = ImageGenerationService(cfg)

    models = Path(cfg.local_models_dir)
    mdir = models / 'ok-model'
    mdir.mkdir(parents=True)
    (mdir / 'model_index.json').write_text('{}', encoding='utf-8')
    (mdir / 'weights.safetensors').write_bytes(b'1234')

    report = svc.validate_model('local:ok-model')
    assert report['folder_size_bytes'] is not None
    assert report['estimated_ram_required_bytes'] is not None
    assert report['device_candidate'] in {'cpu', 'cuda'}


def test_recommended_settings_returns_defaults(tmp_path):
    cfg = _cfg(tmp_path)
    svc = ImageGenerationService(cfg)

    rec = svc.recommended_settings('google/flan-t5-base')
    assert 'recommended_width' in rec
    assert 'recommended_height' in rec
    assert 'recommended_steps' in rec


def test_doctor_contains_human_memory_fields(tmp_path):
    cfg = _cfg(tmp_path)
    svc = ImageGenerationService(cfg)
    rep = svc.doctor()
    mem = rep.get('memory', {})
    assert 'available_ram_human' in mem
    assert 'available_virtual_memory_human' in mem


def test_execution_plan_contains_expected_fields(tmp_path):
    cfg = _cfg(tmp_path)
    svc = ImageGenerationService(cfg)
    plan = svc.build_image_execution_plan('google/flan-t5-base', requested={'width': 640, 'height': 640, 'steps': 12})
    assert 'device_plan' in plan
    assert 'torch_dtype' in plan
    assert 'recommended_width' in plan
    assert 'warnings' in plan


def test_generate_uses_timeout_and_returns_effective_settings(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    cfg.hf_image_runtime = 'diffusers_local'
    svc = ImageGenerationService(cfg)

    monkeypatch.setattr(svc, '_resolve_model_source', lambda model_id: ('remote', model_id))
    monkeypatch.setattr(svc, '_cache_dir', lambda model_id: Path('/tmp'))
    monkeypatch.setattr(svc, 'build_image_execution_plan', lambda model_id, requested=None: {
        'device_plan': 'cpu_low_memory',
        'torch_dtype': 'float32',
        'use_attention_slicing': True,
        'use_vae_tiling': True,
        'use_model_cpu_offload': False,
        'use_sequential_cpu_offload': False,
        'recommended_width': 640,
        'recommended_height': 640,
        'recommended_steps': 12,
        'expected_timeout_sec': 333,
        'warnings': [],
        'reason': 'test',
    })

    observed = {}

    def _fake_run_diffusers(**kwargs):
        observed.update(kwargs)
        return ImageRuntimeResult(ok=True, image_bytes=b'x', metadata={'runtime': 'diffusers_local', 'device_used': kwargs['device'], 'runtime_strategy': kwargs['execution_plan'].get('device_plan')})

    monkeypatch.setattr(svc, '_run_diffusers_isolated', _fake_run_diffusers)

    res = svc.generate(model_id='Tongyi-MAI/Z-Image-Turbo', prompt='p', params_json={'timeout_sec': 444, 'width': 512, 'height': 512, 'steps': 16, 'guidance_scale': 5.5})
    assert res.ok is True
    assert observed['timeout_s'] == 444
    eff = (res.metadata or {}).get('effective_settings') or {}
    assert eff.get('width') == 512
    assert eff.get('height') == 512
    assert eff.get('steps') == 16
    assert float(eff.get('guidance_scale')) == 5.5
    assert eff.get('timeout_s') == 444
