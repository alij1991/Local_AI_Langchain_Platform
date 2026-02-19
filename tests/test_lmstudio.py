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


def test_call_candidates_supports_exact_and_fuzzy():
    sdk = _DummySDK()

    ok_exact, payload = LMStudioController._call_candidates(sdk, ["list_downloaded_models"])
    assert ok_exact is True
    assert payload == [{"id": "qwen/qwen3-4b"}]

    ok_fuzzy, payload_fuzzy = LMStudioController._call_candidates(sdk, ["list_local_models"])
    assert ok_fuzzy is True
    assert payload_fuzzy == [{"id": "qwen/qwen3-4b"}]
