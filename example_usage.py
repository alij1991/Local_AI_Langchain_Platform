"""
Example: How to set up and use the Local AI Platform v2.

Run this after installing dependencies:
    pip install -r requirements.txt
"""
from __future__ import annotations


def example_basic_chat():
    """Basic chat through the provider router."""
    from local_ai_platform.config import load_config
    from local_ai_platform.providers import (
        ChatMessage,
        GenerationSettings,
        build_router_from_config,
    )

    config = load_config()
    router = build_router_from_config(config)

    # Check which providers are available
    print("Provider status:", router.available_providers)

    # Chat with Ollama (default)
    response = router.chat(
        "gemma3:1b",
        [
            ChatMessage(role="system", content="You are a helpful assistant."),
            ChatMessage(role="user", content="What is Python?"),
        ],
        GenerationSettings(temperature=0.3, max_tokens=512),
    )
    print(f"[{response.provider}:{response.model}] {response.content}")


def example_multi_provider():
    """Chat with different providers using prefix syntax."""
    from local_ai_platform.config import load_config
    from local_ai_platform.providers import ChatMessage, build_router_from_config

    config = load_config()
    router = build_router_from_config(config)

    messages = [ChatMessage(role="user", content="Explain recursion briefly.")]

    # Ollama
    r1 = router.chat("ollama:gemma3:1b", messages)
    print(f"Ollama: {r1.content[:100]}...")

    # HuggingFace (auto-detected by '/' in name)
    r2 = router.chat("microsoft/Phi-3-mini-4k-instruct", messages)
    print(f"HuggingFace: {r2.content[:100]}...")

    # LM Studio (if running)
    try:
        r3 = router.chat("lmstudio:default", messages)
        print(f"LM Studio: {r3.content[:100]}...")
    except Exception as e:
        print(f"LM Studio not available: {e}")

    # llama.cpp (if you have a .gguf file in HF cache)
    try:
        r4 = router.chat("llamacpp:mistral-7b-q4.gguf", messages)
        print(f"llama.cpp: {r4.content[:100]}...")
    except Exception as e:
        print(f"llama.cpp not available: {e}")


def example_streaming():
    """Streaming chat responses."""
    from local_ai_platform.config import load_config
    from local_ai_platform.providers import ChatMessage, build_router_from_config

    config = load_config()
    router = build_router_from_config(config)

    messages = [ChatMessage(role="user", content="Write a haiku about code.")]

    print("Streaming: ", end="", flush=True)
    for chunk in router.stream("ollama:gemma3:1b", messages):
        print(chunk, end="", flush=True)
    print()


def example_agents():
    """Set up agents with different models and providers."""
    from local_ai_platform.config import load_config
    from local_ai_platform.agents import AgentOrchestrator

    config = load_config()
    orch = AgentOrchestrator(config)

    # Add agents with different providers
    orch.add_agent(
        name="general",
        model_name="gemma3:1b",
        system_prompt="You are a helpful general assistant. Be concise.",
        provider="ollama",
    )

    orch.add_agent(
        name="coder",
        model_name="qwen2.5-coder:7b",
        system_prompt="You are an expert programmer. Write clean, well-commented code.",
        provider="ollama",
        role="specialist",
    )

    orch.add_agent(
        name="researcher",
        model_name="llama3.2:3b",
        system_prompt="You are a research analyst. Provide thorough, well-sourced answers.",
        provider="ollama",
        role="specialist",
    )

    # Direct chat with a specific agent
    response = orch.chat_with_agent("coder", "Write a Python function to find prime numbers.")
    print(f"Coder: {response[:200]}...")

    # Streaming chat
    print("\nStreaming from general agent:")
    for partial in orch.stream_chat_with_agent("general", "What's 2+2?"):
        print(f"\r{partial}", end="", flush=True)
    print()


def example_supervisor():
    """Supervisor agent that routes to specialists."""
    from local_ai_platform.config import load_config
    from local_ai_platform.agents import AgentOrchestrator

    config = load_config()
    orch = AgentOrchestrator(config)

    # Add specialist agents
    orch.add_agent(
        name="coder",
        model_name="qwen2.5-coder:7b",
        system_prompt="You are an expert programmer. Write clean code with explanations.",
        provider="ollama",
        role="specialist",
    )

    orch.add_agent(
        name="writer",
        model_name="gemma3:1b",
        system_prompt="You are a professional writer. Create clear, engaging content.",
        provider="ollama",
        role="specialist",
    )

    # Create supervisor that routes between them
    orch.create_supervisor(
        name="supervisor",
        model_name="llama3.2:3b",
        specialist_agents=["coder", "writer"],
    )

    # The supervisor decides which agent handles the request
    result = orch.chat_with_supervisor("supervisor", "Write a bubble sort in Python")
    print(f"Agent used: {result['agent_used']}")
    print(f"Response: {result['response'][:200]}...")

    result = orch.chat_with_supervisor("supervisor", "Write a blog post about AI")
    print(f"\nAgent used: {result['agent_used']}")
    print(f"Response: {result['response'][:200]}...")


def example_workflow():
    """Sequential agent workflow where each agent builds on prior work."""
    from local_ai_platform.config import load_config
    from local_ai_platform.agents import AgentOrchestrator

    config = load_config()
    orch = AgentOrchestrator(config)

    orch.add_agent("researcher", "gemma3:1b", "Research the topic and provide key facts.", provider="ollama")
    orch.add_agent("writer", "gemma3:1b", "Write a well-structured article from the research.", provider="ollama")
    orch.add_agent("editor", "gemma3:1b", "Edit and polish the article. Fix any issues.", provider="ollama")

    # Pipeline: researcher → writer → editor
    outputs = orch.run_agent_workflow(
        "Explain how solar panels work",
        sequence=["researcher", "writer", "editor"],
    )
    for agent_name, output in outputs.items():
        print(f"\n{'='*60}")
        print(f"Agent: {agent_name}")
        print(f"{'='*60}")
        print(output[:300] + "...")


def example_smart_memory():
    """Token-aware memory management."""
    from local_ai_platform.memory import SmartMemory
    from local_ai_platform.providers import ChatMessage

    memory = SmartMemory(max_context_tokens=2048, reserved_for_response=512)

    # Simulate a long conversation history
    history = []
    for i in range(50):
        history.append(ChatMessage(role="user", content=f"This is message {i} with some content that takes up tokens."))
        history.append(ChatMessage(role="assistant", content=f"This is response {i} with detailed information."))

    # Smart memory truncates/summarizes to fit budget
    messages = memory.prepare_messages(
        system_prompt="You are a helpful assistant.",
        history=history,
        user_input="What were we discussing earlier?",
    )

    print(f"Original history: {len(history)} messages")
    print(f"After smart memory: {len(messages)} messages")
    print(f"Token budget: {memory.budget}")
    print(f"Estimated tokens: {memory.counter.count_messages(messages)}")


def example_list_all_models():
    """List models from all available providers."""
    from local_ai_platform.config import load_config
    from local_ai_platform.providers import build_router_from_config
    from local_ai_platform.formatting import format_bytes_human

    config = load_config()
    router = build_router_from_config(config)

    print("Available providers:")
    for name, available in router.available_providers.items():
        status = "available" if available else "not running"
        print(f"  {name}: {status}")

    print("\nAll models:")
    for model in router.list_all_models():
        size = format_bytes_human(model.size_bytes) or "unknown size"
        caps = []
        if model.capabilities.supports_tools:
            caps.append("tools")
        if model.capabilities.supports_vision:
            caps.append("vision")
        if model.capabilities.supports_streaming:
            caps.append("stream")
        cap_str = ", ".join(caps) if caps else "basic"
        print(f"  [{model.provider}] {model.name} ({size}) [{cap_str}]")


async def example_async_chat():
    """Async chat and streaming."""
    from local_ai_platform.config import load_config
    from local_ai_platform.providers import ChatMessage, build_router_from_config
    from local_ai_platform.agents import AgentOrchestrator

    config = load_config()
    router = build_router_from_config(config)
    orch = AgentOrchestrator(config, router=router)

    orch.add_agent("assistant", "gemma3:1b", "You are helpful.", provider="ollama")

    # Async chat
    response = await router.achat(
        "ollama:gemma3:1b",
        [ChatMessage(role="user", content="Hello!")],
    )
    print(f"Async: {response.content}")

    # Async agent chat
    response = await orch.achat_with_agent("assistant", "What's the weather like?")
    print(f"Async agent: {response[:100]}")

    # Async streaming
    print("Async stream: ", end="")
    async for chunk in router.astream("ollama:gemma3:1b", [ChatMessage(role="user", content="Count to 5")]):
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    import sys

    examples = {
        "basic": example_basic_chat,
        "multi": example_multi_provider,
        "stream": example_streaming,
        "agents": example_agents,
        "supervisor": example_supervisor,
        "workflow": example_workflow,
        "memory": example_smart_memory,
        "models": example_list_all_models,
    }

    if len(sys.argv) > 1 and sys.argv[1] in examples:
        examples[sys.argv[1]]()
    elif len(sys.argv) > 1 and sys.argv[1] == "async":
        import asyncio
        asyncio.run(example_async_chat())
    else:
        print("Usage: python example_usage.py <example>")
        print(f"Available: {', '.join(list(examples.keys()) + ['async'])}")
