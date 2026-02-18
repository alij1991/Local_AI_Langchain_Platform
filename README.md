# Local AI LangChain Platform

Starter project for a **Python UI app** backed by **LangChain** using **LM Studio models**.

This scaffold gives you:
- A Streamlit interface for chatting with agents.
- LM Studio integration (OpenAI-compatible API endpoint).
- Two built-in agents using different models (`planner` and `worker`).
- Conversation memory per agent.
- Tool support (example tools included).
- Combined mode to run multiple agents on the same prompt.

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

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -e .[dev]
```

3. Copy env template:

```bash
cp .env.example .env
```

4. In LM Studio:
   - Start the local server.
   - Load the models referenced in your `.env`.

5. Run the UI:

```bash
streamlit run app.py
```

## Environment Variables

- `LM_STUDIO_BASE_URL`: LM Studio OpenAI-compatible base URL (default `http://127.0.0.1:1234/v1`).
- `LM_STUDIO_API_KEY`: Any value accepted by your local setup (default `lm-studio`).
- `LM_STUDIO_DEFAULT_MODEL`: Default model name.
- `LM_STUDIO_PLANNER_MODEL`: Model for planning agent.
- `LM_STUDIO_WORKER_MODEL`: Model for worker agent.

## Add Your Own Tools

Edit `src/local_ai_platform/tools.py` and register more tools in `build_default_tools()`.

## Combine More Agents

Add definitions in `AgentOrchestrator.definitions` and expand routing logic in `combined_response()`.

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
