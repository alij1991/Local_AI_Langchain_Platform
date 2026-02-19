from local_ai_platform.lmstudio import LMStudioController


def test_parse_model_lines_filters_cli_table_noise():
    output = """You have 5 models, taking up 16.02 GB of disk space.
LLM PARAMS ARCH SIZE
qwen/qwen3-4b (1 variant) 4B qwen3 2.50 GB LOADED
liquid/lfm2.5-1.2b (1 variant) 1.2B lfm2 1.25 GB LOADED
"""

    models = LMStudioController.parse_model_lines(output)

    assert models == ["qwen/qwen3-4b", "liquid/lfm2.5-1.2b"]


def test_normalize_model_name_extracts_id():
    raw = "qwen/qwen3-4b (1 variant) 4B qwen3 2.50 GB LOADED"
    assert LMStudioController.normalize_model_name(raw) == "qwen/qwen3-4b"
