# Migration Guide: Local AI Platform v2

## What changed

### 1. Unified Provider System (`providers/`)

All model backends now implement `BaseProvider` with a consistent interface:

```python
from local_ai_platform.providers import (
    ProviderRouter, build_router_from_config,
    ChatMessage, GenerationSettings,
)

# Build router from config (auto-registers all providers)
router = build_router_from_config(config)

# Chat with any provider using "provider:model" syntax
response = router.chat(
    "ollama:llama3.2",
    [ChatMessage(role="user", content="Hello!")],
    GenerationSettings(temperature=0.3),
)

# Auto-detection works too:
router.chat("llama3.2", [...])            # → Ollama (default)
router.chat("microsoft/Phi-3-mini", [...]) # → HuggingFace (has '/')
router.chat("mistral-7b.gguf", [...])      # → llama.cpp (ends in .gguf)
```

**Supported providers:**
- `ollama` — Ollama server (default)
- `huggingface` — HF Transformers with proper chat templates
- `llamacpp` — Direct GGUF via llama-cpp-python
- `lmstudio` — LM Studio (OpenAI-compatible)
- `vllm` — vLLM server (OpenAI-compatible)

### 2. Fixed HuggingFace Chat

**Before (v1):** Raw string concatenation → poor quality on instruction-tuned models
```python
# OLD: Built a raw "System: ...\nUser: ...\nAssistant:" prompt
prompt = f"System: {system_prompt}\n\nUser: {user_input}\nAssistant:"
```

**After (v2):** Uses `apply_chat_template()` → correct format per model
```python
# NEW: Uses the model's native chat template
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
```

Also added: real streaming via `TextIteratorStreamer`, metadata with chat template detection.

### 3. LangGraph ReAct Agents

**Before (v1):** `from langchain.agents import create_agent` (deprecated/broken)

**After (v2):** `from langgraph.prebuilt import create_react_agent` (current)

```python
from langgraph.prebuilt import create_react_agent

agent = create_react_agent(model=llm, tools=tools)
result = agent.invoke({"messages": [SystemMessage(...), HumanMessage(...)]})
```

### 4. Supervisor Agent Pattern

New multi-agent routing where a supervisor decides which specialist handles each request:

```python
orchestrator.add_agent("coder", "qwen2.5-coder:7b", "You are a coding expert...", role="specialist")
orchestrator.add_agent("researcher", "llama3.2:3b", "You are a research agent...", role="specialist")

orchestrator.create_supervisor(
    name="supervisor",
    model_name="llama3.2:3b",
    specialist_agents=["coder", "researcher"],
)

result = orchestrator.chat_with_supervisor("supervisor", "Write a sorting algorithm")
# → Routes to "coder" agent automatically
```

### 5. Smart Memory

Token-aware history management that prevents context overflow:

```python
from local_ai_platform.memory import SmartMemory

memory = SmartMemory(max_context_tokens=4096)
messages = memory.prepare_messages(
    system_prompt="You are helpful.",
    history=long_chat_history,
    user_input="What were we talking about?",
)
# → Truncates or summarizes old messages to fit within budget
```

Optional vector store for semantic search across past conversations:

```python
from local_ai_platform.memory import VectorMemory

vmem = VectorMemory()
vmem.store("conv_123", messages)
context = vmem.get_relevant_context("How do I use Docker?")
```

### 6. Async Support

All providers support async operations:

```python
# Async chat
response = await router.achat("ollama:llama3.2", messages)

# Async streaming
async for chunk in router.astream("ollama:llama3.2", messages):
    print(chunk, end="")

# Async agent chat
response = await orchestrator.achat_with_agent("my_agent", "Hello")
```

DB also has async support:
```python
from local_ai_platform.db import AsyncDB

db = AsyncDB()
await db.init()
rows = await db.execute("SELECT * FROM conversations")
```

### 7. lmstudio.py → Deleted

LM Studio is now handled by `OpenAICompatibleProvider`. No separate controller needed.

```python
# Configure via env var or config
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1

# Use via router
router.chat("lmstudio:my-model", messages)
```

---

## File changes

| File | Status | Notes |
|------|--------|-------|
| `providers/__init__.py` | **NEW** | Package exports |
| `providers/base.py` | **NEW** | BaseProvider, ChatMessage, GenerationSettings |
| `providers/ollama_provider.py` | **NEW** | Ollama via official SDK |
| `providers/huggingface_provider.py` | **NEW** | Fixed HF with chat templates + streaming |
| `providers/llamacpp_provider.py` | **NEW** | Direct GGUF inference |
| `providers/openai_compatible_provider.py` | **NEW** | LM Studio, vLLM, any OpenAI-compat |
| `providers/router.py` | **NEW** | Unified routing + auto-detection |
| `config.py` | **UPDATED** | Added llama.cpp, LM Studio, vLLM, memory settings |
| `agents.py` | **REWRITTEN** | Provider router, create_react_agent, supervisor, async |
| `memory.py` | **REWRITTEN** | Smart memory, token counting, vector store |
| `db.py` | **UPDATED** | Added AsyncDB wrapper |
| `huggingface.py` | **UPDATED** | Backward-compatible wrapper around new provider |
| `ollama.py` | **UPDATED** | Backward-compatible wrapper around new provider |
| `lmstudio.py` | **DELETED** | Replaced by OpenAICompatibleProvider |
| All repo files | **UNCHANGED** | conversations.py, agents_repo.py, etc. |

---

## Backward compatibility

The wrapper files (`huggingface.py`, `ollama.py`) maintain the old API surface:

```python
# These still work exactly as before:
from local_ai_platform.ollama import OllamaController, CommandResult, ModelInfo
from local_ai_platform.huggingface import HuggingFaceController

ctrl = OllamaController(config)
result = ctrl.list_local_models()
```

But you can also access the underlying new provider:
```python
ctrl.provider  # → OllamaProvider instance
ctrl.provider.chat(model, messages)  # New unified API
```

---

## New environment variables

```bash
# llama.cpp
LOCAL_MODELS_DIR=./models
LLAMACPP_N_GPU_LAYERS=-1     # -1 = all layers on GPU
LLAMACPP_N_CTX=4096

# LM Studio
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1

# vLLM
VLLM_BASE_URL=http://127.0.0.1:8080/v1

# Memory
VECTOR_STORE_DIR=./data/vectorstore
SMART_MEMORY_ENABLED=true
MAX_CONTEXT_TOKENS=4096
```

---

## New pip dependencies

```bash
pip install aiosqlite aiohttp tiktoken

# Optional:
pip install llama-cpp-python   # For direct GGUF
pip install chromadb            # For vector memory
pip install langchain-openai    # For LM Studio/vLLM LangChain agents
```
