from local_ai_platform.lmstudio import LMStudioController


class _DummySDK:
    def list_downloaded_models(self):
        return [{"id": "qwen/qwen3-4b"}]

    def start_server(self):
        return "ok"


def test_extract_model_names_from_mixed_payload():
    payload = {
        "data": [
            {"id": "qwen/qwen3-4b"},
            {"name": "liquid/lfm2.5-1.2b"},
            "other/model",
        ]
    }

    names = LMStudioController._extract_model_names(payload)

    assert names == ["qwen/qwen3-4b", "liquid/lfm2.5-1.2b", "other/model"]


def test_extract_model_names_empty_payload():
    assert LMStudioController._extract_model_names(None) == []


class _DownloadedModel:
    def __init__(self, model_key: str):
        self.model_key = model_key


def test_extract_model_names_from_sdk_objects_uses_model_key():
    payload = [_DownloadedModel("liquid/lfm2.5-1.2b"), _DownloadedModel("qwen/qwen3-4b")]
    assert LMStudioController._extract_model_names(payload) == ["liquid/lfm2.5-1.2b", "qwen/qwen3-4b"]


def test_call_candidates_supports_exact_and_fuzzy():
    sdk = _DummySDK()

    ok_exact, payload = LMStudioController._call_candidates(sdk, ["list_downloaded_models"])
    assert ok_exact is True
    assert payload == [{"id": "qwen/qwen3-4b"}]

    ok_fuzzy, payload_fuzzy = LMStudioController._call_candidates(sdk, ["list_local_models"])
    assert ok_fuzzy is True
    assert payload_fuzzy == [{"id": "qwen/qwen3-4b"}]


def test_start_stop_server_graceful_when_sdk_has_no_server_methods():
    class _Cfg:
        lm_studio_base_url = "http://127.0.0.1:1234/v1"
        lm_studio_api_key = "lm-studio"

    controller = LMStudioController(_Cfg())
    controller._call_sdk = lambda *args, **kwargs: (_ for _ in ()).throw(AttributeError("missing"))

    start = controller.start_server()
    stop = controller.stop_server()

    assert start.ok is True
    assert "does not expose server start" in start.output
    assert stop.ok is True
    assert "does not expose server stop" in stop.output
