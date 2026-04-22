# Local AI Platform

Windows desktop AI platform. FastAPI backend + Flutter client. Image editing/generation, chat, voice, agents.

## Env (don't re-check)

- Python: `C:/AI/Local_AI_Langchain_Platform/.venv/Scripts/python` — 3.11.4, PyTorch 2.11.0+cu130
- CUDA toolkits: 12.4 + 13.0 (both needed — see images/CLAUDE.md)
- GPU: 8GB NVIDIA. 1.14GB baseline WDDM overhead is NOT reducible.
- HF token: `HF_TOKEN` in `.env` (some repos are gated)
- Shell: bash, Unix paths (forward slashes)

## Where things live

- `api_server.py` — HTTP endpoints. **Large — grep, never read whole.**
- `src/local_ai_platform/images/` — image edit + generation. See `images/CLAUDE.md` for landmines.
- `src/local_ai_platform/providers/` — LLM provider adapters (Ollama, HF, OpenAI-compat)
- `src/local_ai_platform/agents.py` + `tools/` — agent runtime + tool registry
- `src/local_ai_platform/partner/` — voice/persona partner
- `src/local_ai_platform/memory.py` + `repositories/` — memory store, SQLite at `data/app.db`
- `flutter_client/lib/pages/*_page.dart` — one file per screen

## Conventions

- Module logger: `logger = logging.getLogger(__name__)`.
- Log tags to grep flows: `[KONTEXT]`, `[KONTEXT-NUNCHAKU]`, `[COSXL]`, `[IMG]`, `[IMG-CN]`, `[ENHANCE]`, `[PARTNER]`, `[AGENT]`.
- Comments explain *why*, not *what*.
- No emojis in files.
- Fix only what was asked. Don't refactor adjacent code.

## Do-not

- Read `api_server.py` top-to-bottom. Grep first, Read with offset.
- Trust the `*_PROMPT.md` / `*_AUDIT.md` files at repo root — historical dumps, may contradict current code.
- Touch `.venv`, `data/`, or anything in `.claudeignore`.
- Commit `.env` (contains HF token).

## Section-specific context

Before touching a section, read its landmines file. Auto-loaded files are noted; others: **Read first, don't guess.**

- **Images** (edit/generation) — `src/local_ai_platform/images/CLAUDE.md` *(auto-loads when working in that dir)*
- **Systems** (agent DAGs, templates) — Read `src/local_ai_platform/CLAUDE_SYSTEMS.md` before editing `system_templates.py`, `system_info.py`, `repositories/systems.py`, or `systems_page.dart`.
- **Models** (HF discovery, download, detection) — Read `src/local_ai_platform/CLAUDE_MODELS.md` before editing `huggingface.py`, `repositories/models.py`, `ai_models.py`, `/models/*` endpoints, or `models_page.dart`.
