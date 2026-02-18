# Local AI LangChain Platform

Starter project for a **Python UI app** backed by **LangChain** using **LM Studio models**.

This scaffold gives you:
- A Streamlit interface for chatting with agents.
- LM Studio integration (OpenAI-compatible API endpoint).
- Two built-in agents using different lightweight models (`planner` and `worker`).
- Conversation memory per agent.
- Tool support (example tools included).
- Combined mode to run multiple agents on the same prompt.

> ✅ The code has been adjusted for modern LangChain package layout (including the
> `AgentExecutor` import path) and uses a message-history memory pattern that is
> compatible with current LangChain agent prompts.

## Project Structure

```text
.
├── app.py
├── pyproject.toml
├── .env.example
├── src/local_ai_platform
│   ├── __init__.py
│   ├── agents.py
│   ├── config.py
│   └── tools.py
└── tests
    └── test_config.py
```

## Recommended Models for 16GB RAM Laptops

The defaults are intentionally set to smaller models for better local speed:
- `qwen2.5-3b-instruct` (good general purpose default/planner)
- `phi-3.5-mini-instruct` (fast worker model)

If your laptop is slower or has less VRAM/shared memory, try:
- `llama-3.2-3b-instruct`
- `gemma-2-2b-it`

Tip: in LM Studio, prefer **4-bit quantized variants** (for example `Q4_K_M`) for much better performance.

## Quick Start

1. Create and activate a virtual environment.
2. Upgrade packaging tools first:

```bash
python -m pip install --upgrade pip setuptools wheel
```

3. Install dependencies:

```bash
pip install -e .[dev]
```

To force-upgrade to the newest releases in your environment:

```bash
python -m pip install -U streamlit langchain langchain-openai langchain-community langgraph pydantic pytest ruff
```

4. Copy env template:

```bash
cp .env.example .env

# load the values for this shell session
set -a
source .env
set +a
```

5. In LM Studio:
   - Start the local server.
   - Load the models referenced in your `.env`.

6. Run the UI:

```bash
streamlit run app.py
```

## Environment Variables

- `LM_STUDIO_BASE_URL`: LM Studio OpenAI-compatible base URL (default `http://127.0.0.1:1234/v1`).
- `LM_STUDIO_API_KEY`: Any value accepted by your local setup (default `lm-studio`).
- `LM_STUDIO_DEFAULT_MODEL`: Default model name (default `qwen2.5-3b-instruct`).
- `LM_STUDIO_PLANNER_MODEL`: Model for planning agent (default `qwen2.5-3b-instruct`).
- `LM_STUDIO_WORKER_MODEL`: Model for worker agent (default `phi-3.5-mini-instruct`).

## Add Your Own Tools

Edit `src/local_ai_platform/tools.py` and register more tools in `build_default_tools()`.

## Combine More Agents

Add definitions in `AgentOrchestrator.definitions` and expand routing logic in `combined_response()`.

## Fixing Common Testing/Setup Issues

If you saw install/test failures like I did in a restricted environment, use this checklist:

1. **Proxy/network issues while installing**
   - Ensure your shell has proper internet access.
   - If you are behind a proxy, set:

```bash
export HTTPS_PROXY=http://<proxy-host>:<proxy-port>
export HTTP_PROXY=http://<proxy-host>:<proxy-port>
```

2. **Missing packaging/build tools**

```bash
python -m pip install --upgrade pip setuptools wheel
```

3. **Run validation locally**

```bash
python -m compileall app.py src tests
pytest
```

4. **Streamlit not found**
   - Install dependencies in the active venv first:

```bash
pip install -e .[dev]
```

Then launch:

```bash
streamlit run app.py
```

5. **`ImportError: cannot import name 'AgentExecutor' from 'langchain.agents'`**
   - You likely have a newer LangChain build where `AgentExecutor` moved.
   - This project now imports it from `langchain.agents.agent` for compatibility with current releases.
   - Then reinstall/upgrade dependencies:

```bash
python -m pip install -U langchain langchain-openai langchain-community
pip install -e .[dev]
```

6. **Have we updated the code for latest libraries?**
   - Yes. This repo now uses:
     - `AgentExecutor` from `langchain.agents.agent`
     - prompt-driven `chat_history` with `HumanMessage`/`AIMessage` instead of
       legacy memory wiring in the executor
   - Keep your environment up to date with:

```bash
python -m pip install -U streamlit langchain langchain-openai langchain-community langgraph pydantic pytest ruff
pip install -e .[dev]
```

## Create a New GitHub Repository

From your local machine (with `gh` CLI authenticated):

```bash
git init
git add .
git commit -m "Initial scaffold: Streamlit + LangChain + LM Studio multi-agent app"
gh repo create <your-org-or-user>/<repo-name> --private --source=. --push
```

Or create an empty repo on GitHub first and then:

```bash
git remote add origin git@github.com:<your-org-or-user>/<repo-name>.git
git push -u origin main
```
