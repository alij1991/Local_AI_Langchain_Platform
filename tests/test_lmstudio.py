from local_ai_platform.lmstudio import LMStudioController


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
