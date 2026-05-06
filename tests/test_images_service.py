from pathlib import Path

from local_ai_platform.config import AppConfig
from local_ai_platform.images.service import ImageGenerationService, ImageRuntimeResult


def _cfg() -> AppConfig:
    return AppConfig(
        ollama_base_url='http://127.0.0.1:11434',
        default_model='gemma3:1b',
        prompt_builder_model='gemma3:1b',
        hf_default_model='',
        hf_model_catalog='',
        hf_device='auto',
        gradio_share=False,
        gradio_server_port=7860,
        api_server_port=8000,
    )


def _setup_hf_cache(tmp_path: Path, model_id: str, files: dict[str, str] | None = None) -> Path:
    """Create a fake HF cache structure for a model.

    Returns the snapshot directory path.
    Structure: tmp_path/hub/models--org--name/snapshots/abc123/
    """
    safe_name = model_id.replace('/', '--')
    snapshot = tmp_path / 'hub' / f'models--{safe_name}' / 'snapshots' / 'abc123'
    snapshot.mkdir(parents=True)
    if files:
        for name, content in files.items():
            (snapshot / name).write_text(content, encoding='utf-8')
    return snapshot


def test_hf_cache_scan_detects_diffusers_model(tmp_path, monkeypatch, reset_settings_cache):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))

    # Create a diffusers image model (has model_index.json)
    _setup_hf_cache(tmp_path, 'test-org/img-model', {
        'model_index.json': '{}',
    })

    svc = ImageGenerationService(cfg)
    # [IMPROVE-173] Wave 37 — _scan_hf_cache_models filters out repos
    # < 50 MB (metadata-only HF cache entries: model cards, config
    # files cached when browsing). The test fixture writes ~2 bytes
    # of model_index.json which trips this filter; bypass it for
    # the test by returning a large fake size.
    monkeypatch.setattr(svc, '_dir_size', lambda path: 100 * 1024 * 1024)
    body = svc.refresh_models()

    assert any(m['model_id'] == 'test-org/img-model' for m in body['items'])


def test_hf_cache_scan_ignores_text_models(tmp_path, monkeypatch):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))

    # Create a text model (no model_index.json, has config.json + tokenizer)
    snap = _setup_hf_cache(tmp_path, 'test-org/text-model', {
        'config.json': '{"model_type":"llama","max_position_embeddings":4096}',
        'tokenizer.json': '{}',
    })
    (snap / 'model.safetensors').write_bytes(b'\x00' * 100)

    svc = ImageGenerationService(cfg)
    body = svc.refresh_models()

    # Text model should NOT appear in image models
    assert not any(m['model_id'] == 'test-org/text-model' for m in body['items'])


def test_doctor_reports_local_models_missing(tmp_path, monkeypatch, reset_settings_cache):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    # Create empty hub dir so scan runs but finds nothing
    (tmp_path / 'hub').mkdir(parents=True)

    svc = ImageGenerationService(cfg)
    report = svc.doctor()

    assert 'checks' in report
    local_check = next(c for c in report['checks'] if c['name'] == 'local_models')
    assert local_check['ok'] is False


def test_generate_uses_cpu_fallback_when_gpu_required_but_unavailable(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg.hf_image_runtime = 'diffusers_local'
    cfg.hf_image_require_gpu = True
    cfg.hf_image_allow_cpu_fallback = True
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    svc = ImageGenerationService(cfg)

    monkeypatch.setattr(svc, '_resolve_model_source', lambda model_id: ('remote', model_id))
    monkeypatch.setattr(svc, 'get_device_status', lambda: {
        'torch_installed': True,
        'cuda_available': False,
        'cuda_version': None,
        'effective_device': 'cpu',
        'torch_version': '2.10.0+cpu',
    })
    # [IMPROVE-174] Wave 37 — patch the execution plan to return a CPU
    # plan so the CPU-fallback gate at service.py:10315 fires
    # (`require_gpu and not _any_gpu and preferred == 'cpu'`). Without
    # this patch, build_image_execution_plan uses the real
    # _get_hardware_profile and returns a GPU plan on machines with a
    # CUDA card, _any_gpu=True, and the gate doesn't trigger — so the
    # test fails on developer machines with a GPU even though the
    # production logic is correct. The companion IMPROVE-174 patch
    # below points the patch target at `_run_diffusers` (in-process)
    # rather than `_run_diffusers_isolated` (subprocess fallback).
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
        'expected_timeout_sec': 60,
        'warnings': [],
        'reason': 'test-cpu-fallback',
    })

    def _fake_run_diffusers(**kwargs):
        assert kwargs['device'] == 'cpu'
        return ImageRuntimeResult(
            ok=True,
            image_bytes=b'x',
            metadata={'runtime': 'diffusers_local', 'device_used': 'cpu'},
        )

    monkeypatch.setattr(svc, '_cache_dir', lambda model_id: Path('/tmp'))
    # [IMPROVE-174] Wave 37 — patch the in-process `_run_diffusers`
    # (service.py:8849), NOT the subprocess-isolated
    # `_run_diffusers_isolated` (service.py:8603). `generate()` calls
    # `self._run_diffusers` directly at 5 call sites (line 9914 retry
    # / 10361 main path / 10408 post-failure retry / 10507 hires-fix
    # / 10539 refine). `_run_diffusers_isolated` was the primary path
    # during the [IMPROVE-44] OOM ladder introduction but became a
    # fallback after the persistent-worker-pool refactor lifted
    # in-process pipeline caching as the default. A future call-site
    # move would need to update this patch target again.
    monkeypatch.setattr(svc, '_run_diffusers', _fake_run_diffusers)

    result = svc.generate(model_id='some-org/test-model', prompt='test')
    assert result.ok is True
    assert result.metadata and 'warning' in result.metadata


def test_validate_model_reports_missing_files(tmp_path, monkeypatch, reset_settings_cache):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))

    # Create a model with no model_index.json (broken for images)
    _setup_hf_cache(tmp_path, 'test-org/broken-model')

    svc = ImageGenerationService(cfg)
    report = svc.validate_model('test-org/broken-model')
    assert report['loadable'] is False


def test_validate_model_includes_memory_estimates(tmp_path, monkeypatch, reset_settings_cache):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))

    # Create a valid diffusers model with model_index.json + weights
    snap = _setup_hf_cache(tmp_path, 'test-org/ok-model', {
        'model_index.json': '{}',
    })
    (snap / 'weights.safetensors').write_bytes(b'1234')

    svc = ImageGenerationService(cfg)
    report = svc.validate_model('test-org/ok-model')
    assert report['folder_size_bytes'] is not None
    assert report['estimated_ram_required_bytes'] is not None
    # [IMPROVE-173] Wave 37 — `device_candidate` reads
    # `effective_device` from `get_device_status()` which now includes
    # the device index ('cuda:0' / 'cuda:1') rather than the bare
    # family name. Check the device family prefix to accept either
    # shape.
    assert report['device_candidate'].split(':')[0] in {'cpu', 'cuda'}


def test_recommended_settings_returns_defaults(tmp_path, monkeypatch):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    svc = ImageGenerationService(cfg)

    rec = svc.recommended_settings('some-org/test-model')
    assert 'recommended_width' in rec
    assert 'recommended_height' in rec
    assert 'recommended_steps' in rec


def test_doctor_contains_human_memory_fields(tmp_path, monkeypatch):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    (tmp_path / 'hub').mkdir(parents=True)
    svc = ImageGenerationService(cfg)
    rep = svc.doctor()
    mem = rep.get('memory', {})
    assert 'available_ram_human' in mem
    assert 'available_virtual_memory_human' in mem


def test_execution_plan_contains_expected_fields(tmp_path, monkeypatch):
    cfg = _cfg()
    monkeypatch.setenv('HF_HOME', str(tmp_path))
    svc = ImageGenerationService(cfg)
    plan = svc.build_image_execution_plan('some-org/test-model', requested={'width': 640, 'height': 640, 'steps': 12})
    assert 'device_plan' in plan
    assert 'torch_dtype' in plan
    assert 'recommended_width' in plan
    assert 'warnings' in plan


def test_generate_uses_timeout_and_returns_effective_settings(tmp_path, monkeypatch):
    cfg = _cfg()
    cfg.hf_image_runtime = 'diffusers_local'
    monkeypatch.setenv('HF_HOME', str(tmp_path))
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

    # [IMPROVE-174] Wave 37 — patch the in-process `_run_diffusers`
    # (service.py:8849), NOT the subprocess-isolated
    # `_run_diffusers_isolated` (service.py:8603). See the same
    # comment block in test_generate_uses_cpu_fallback_when_gpu_required_but_unavailable
    # above for the full rationale.
    monkeypatch.setattr(svc, '_run_diffusers', _fake_run_diffusers)

    res = svc.generate(model_id='some-org/test-model', prompt='p', params_json={'timeout_sec': 444, 'width': 512, 'height': 512, 'steps': 16, 'guidance_scale': 5.5})
    assert res.ok is True
    assert observed['timeout_s'] == 444
    eff = (res.metadata or {}).get('effective_settings') or {}
    assert eff.get('width') == 512
    assert eff.get('height') == 512
    assert eff.get('steps') == 16
    assert float(eff.get('guidance_scale')) == 5.5
    assert eff.get('timeout_s') == 444
