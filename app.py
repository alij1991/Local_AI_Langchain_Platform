from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.ollama import ModelInfo, OllamaController

CSS = """
.container {max-width: 1600px !important; margin: 0 auto;}
#left-pane, #right-pane {border: 1px solid #2c3550; border-radius: 14px; padding: 12px; background: #121827;}
#header-title h1 {font-size: 1.9rem; margin-bottom: 0.2rem;}
#header-title p {opacity: 0.88;}
"""


def build_chat_handler(orchestrator: AgentOrchestrator, messages_format: bool) -> Callable:
    def chat(message: str, history, agent_name: str):
        clean = (message or "").strip()
        if not clean:
            return "", history or []

        try:
            response = "❌ Create/select an agent first." if not agent_name else orchestrator.chat_with_agent(agent_name, clean)
        except Exception as exc:  # noqa: BLE001
            response = f"❌ Agent error: {exc}"

        turns = list(history or [])
        if messages_format:
            turns.extend([{"role": "user", "content": clean}, {"role": "assistant", "content": response}])
        else:
            turns.append((clean, response))
        return "", turns

    return chat


def _chatbot_uses_messages_format() -> bool:
    data_model_name = getattr(getattr(gr.Chatbot(), "data_model", None), "__name__", "")
    return "Messages" in data_model_name


def _human_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    size = float(size_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "unknown"


def _format_bool(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _agent_model_choices(infos: list[ModelInfo]) -> list[str]:
    normal = [info.name for info in infos if "embedding" not in info.name.lower()]
    embeddings = [info.name for info in infos if "embedding" in info.name.lower()]
    return normal + embeddings


def _models_markdown(infos: list[ModelInfo]) -> str:
    if not infos:
        return "No local models found."

    header = "| Model | Size | Family | Params | Quantization | Generate | Tool support |\n|---|---:|---|---|---|---|---|"
    rows = [
        (
            f"| `{info.name}` | {_human_size(info.size_bytes)} | {info.family} | {info.parameter_size} | "
            f"{info.quantization} | {_format_bool(info.supports_generate)} | {_format_bool(info.supports_tools)} |"
        )
        for info in infos
    ]
    return "\n".join([header, *rows])


def build_app() -> gr.Blocks:
    config = load_config()
    orchestrator = AgentOrchestrator(config)
    controller = OllamaController(config)
    messages_format = _chatbot_uses_messages_format()
    chat_fn = build_chat_handler(orchestrator, messages_format=messages_format)

    def _local_infos() -> list[ModelInfo]:
        ok, infos, _ = controller.list_local_models_detailed()
        return infos if ok else []

    def _pick_startup_model() -> str:
        models = [m.name for m in _local_infos()]
        if not models:
            return config.default_model
        for preferred in ["gemma3:1b", "qwen2.5:1.5b", "llama3.2:3b", config.default_model]:
            if preferred in models:
                return preferred
        return models[0]

    startup_model = _pick_startup_model()
    initial_infos = _local_infos()
    initial_model_choices = _agent_model_choices(initial_infos) if initial_infos else [startup_model]
    if startup_model not in initial_model_choices:
        initial_model_choices = [startup_model, *initial_model_choices]
    if config.prompt_builder_model not in [m.name for m in initial_infos]:
        config.prompt_builder_model = startup_model

    orchestrator.add_agent(
        name="assistant",
        model_name=startup_model,
        system_prompt="You are a practical AI assistant. Be concise, accurate, and tool-aware.",
    )

    def _agent_choices(default: str | None = None) -> dict[str, Any]:
        agents = orchestrator.list_agents()
        return gr.update(choices=agents, value=(default if default in agents else agents[0]))

    def _agent_map_text() -> str:
        rows = [f"- `{name}` → `{model}`" for name, model in orchestrator.get_agent_models().items()]
        return "\n".join(rows) if rows else "No agents configured yet."

    def _tool_map_text() -> str:
        rows = [f"- `{name}`" for name in orchestrator.get_tool_names()]
        return "\n".join(rows) if rows else "No custom tools yet."

    def refresh_models() -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        ok, infos, error = controller.list_local_models_detailed()
        fallback = gr.update(choices=initial_model_choices, value=startup_model)
        if not ok:
            return f"❌ {error}", "", fallback, fallback, fallback
        if not infos:
            return "No models returned by Ollama SDK.", "", fallback, fallback, fallback

        load_choices = [info.name for info in infos]
        agent_choices = _agent_model_choices(infos)
        selected = startup_model if startup_model in agent_choices else agent_choices[0]
        load_selected = startup_model if startup_model in load_choices else load_choices[0]

        return (
            f"✅ Found {len(infos)} model(s).",
            _models_markdown(infos),
            gr.update(choices=load_choices, value=load_selected),
            gr.update(choices=agent_choices, value=selected),
            gr.update(choices=agent_choices, value=selected),
        )

    def load_selected_model(model_name: str | None) -> str:
        selected = (model_name or "").strip()
        if not selected:
            return "❌ Select a model first."
        result = controller.load_model(selected)
        return result.output if result.ok else f"❌ {result.output}"

    def list_loaded_models() -> str:
        result = controller.list_loaded_models()
        return result.output if result.ok else f"❌ {result.output}"

    def create_agent(name: str, model_name: str | None, system_prompt: str):
        clean = name.strip().lower().replace(" ", "-")
        selected = (model_name or "").strip()
        if not clean:
            return "❌ Agent name is required.", gr.update(), gr.update(), gr.update(), gr.update()
        if clean in orchestrator.definitions:
            return "❌ Agent already exists.", gr.update(), gr.update(), gr.update(), gr.update()
        if not selected:
            return "❌ Select or type a model name.", gr.update(), gr.update(), gr.update(), gr.update()

        orchestrator.add_agent(clean, selected, system_prompt.strip())
        return (
            f"✅ Agent `{clean}` created.",
            gr.update(value=_agent_map_text()),
            _agent_choices(clean),
            _agent_choices(clean),
            _agent_choices(clean),
        )

    def update_agent_model(agent_name: str, model_name: str | None):
        if not agent_name:
            return "❌ Select an agent.", gr.update()
        selected = (model_name or "").strip()
        if not selected:
            return "❌ Select or type a model name.", gr.update()
        orchestrator.set_agent_model(agent_name, selected)
        return f"✅ Updated `{agent_name}` model to `{selected}`.", gr.update(value=_agent_map_text())

    def draft_prompt(description: str) -> str:
        if not description.strip():
            return ""
        try:
            return orchestrator.generate_system_prompt(description)
        except Exception as exc:  # noqa: BLE001
            candidates = [m.name for m in _local_infos()]
            hint = f"Try one of: {', '.join(candidates[:5])}" if candidates else "No local Ollama models found."
            return f"❌ {exc}\n\n{hint}"

    def apply_tool_template(tool_mode: str):
        if tool_mode == "instruction":
            return "summarize_for_exec", "Summarize output into 5 bullets + 3 action items."
        return "delegate_to_assistant", "Ask another specialist agent to handle this task."

    def add_tool(tool_name: str, tool_type: str, instructions: str, target_agent: str):
        clean = tool_name.strip().lower().replace(" ", "_")
        if not clean:
            return "❌ Tool name is required.", gr.update()
        if clean in orchestrator.get_tool_names():
            return "❌ Tool name already exists.", gr.update()

        if tool_type == "instruction":
            orchestrator.add_instruction_tool(clean, instructions.strip() or "General helper tool")
            message = f"✅ Added instruction tool `{clean}`."
        else:
            if target_agent not in orchestrator.definitions:
                return "❌ Select a valid target agent.", gr.update()
            orchestrator.add_agent_delegate_tool(clean, target_agent)
            message = f"✅ Added delegate tool `{clean}` -> `{target_agent}`."
        return message, gr.update(value=_tool_map_text())

    def run_workflow(prompt: str, sequence_csv: str) -> str:
        sequence = [part.strip() for part in sequence_csv.split(",") if part.strip()]
        outputs = orchestrator.run_agent_workflow(prompt, sequence)
        if not outputs:
            return "❌ No valid agents in sequence."
        return "\n\n".join([f"## {name}\n{text}" for name, text in outputs.items()])

    with gr.Blocks(title="Local AI Studio") as demo:
        with gr.Column(elem_id="header-title"):
            gr.Markdown("# 🤖 Local AI Studio")
            gr.Markdown("A cleaner workspace for model management, agents, tools, and workflows.")

        with gr.Row(equal_height=True):
            with gr.Column(scale=7, elem_id="left-pane"):
                gr.Markdown("### Chat")
                active_agent = gr.Dropdown(label="Active agent", choices=orchestrator.list_agents(), value="assistant")
                chat = gr.Chatbot(label="Conversation", height=560)
                prompt = gr.Textbox(label="Message", placeholder="Ask your active agent...")
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

            with gr.Column(scale=5, elem_id="right-pane"):
                with gr.Tabs():
                    with gr.Tab("Models"):
                        status = gr.Markdown("Refreshing models...")
                        model_table = gr.Markdown()
                        with gr.Row():
                            list_models_btn = gr.Button("Refresh Models", variant="primary")
                            list_loaded_btn = gr.Button("List Loaded / Running")
                        sdk_model_dropdown = gr.Dropdown(label="Load model", choices=initial_model_choices, value=startup_model)
                        load_btn = gr.Button("Load Selected Model")
                        lm_output = gr.Textbox(label="Ollama output", lines=7)

                    with gr.Tab("Agents"):
                        agent_map = gr.Markdown(value=_agent_map_text())
                        gr.Markdown("Create role-specific agents and map each to a model.")
                        new_agent_name = gr.Textbox(label="New agent name", placeholder="legal-reviewer")
                        new_agent_model = gr.Dropdown(label="Model", choices=initial_model_choices, value=startup_model, allow_custom_value=True)
                        new_agent_prompt = gr.Textbox(label="System prompt", lines=5)
                        create_agent_btn = gr.Button("Create Agent", variant="primary")
                        create_agent_status = gr.Markdown()
                        gr.Markdown("---")
                        update_agent_name = gr.Dropdown(label="Agent", choices=orchestrator.list_agents(), value="assistant")
                        update_agent_model_name = gr.Dropdown(label="New model", choices=initial_model_choices, value=startup_model, allow_custom_value=True)
                        update_agent_btn = gr.Button("Apply Model Update")
                        update_agent_status = gr.Markdown()

                    with gr.Tab("Tools"):
                        gr.Markdown(
                            "**Instruction tool** = reusable guidance snippet.\n\n"
                            "**Delegate tool** = forwards a task to another configured agent."
                        )
                        tool_map = gr.Markdown(value=_tool_map_text())
                        tool_type = gr.Radio(label="Tool type", choices=["instruction", "delegate_agent"], value="instruction")
                        tool_name = gr.Textbox(label="Tool name", placeholder="summarize_for_exec")
                        tool_instructions = gr.Textbox(label="Instructions", lines=4)
                        delegate_target = gr.Dropdown(label="Delegate target agent", choices=orchestrator.list_agents(), value="assistant")
                        with gr.Row():
                            tool_template_btn = gr.Button("Use Suggested Template")
                            create_tool_btn = gr.Button("Create Tool", variant="primary")
                        create_tool_status = gr.Markdown()

                    with gr.Tab("Workflow + Prompt Builder"):
                        wf_prompt = gr.Textbox(label="Workflow prompt", lines=4)
                        wf_sequence = gr.Textbox(label="Agent sequence (comma-separated)", value="assistant")
                        run_wf_btn = gr.Button("Run Workflow", variant="primary")
                        wf_output = gr.Markdown()
                        gr.Markdown("---")
                        gr.Markdown(f"Prompt-builder model: `{config.prompt_builder_model}`")
                        prompt_desc = gr.Textbox(label="Agent description", lines=4)
                        draft_btn = gr.Button("Generate System Prompt")
                        drafted_prompt = gr.Textbox(label="Generated system prompt", lines=8)

        send_btn.click(chat_fn, inputs=[prompt, chat, active_agent], outputs=[prompt, chat])
        prompt.submit(chat_fn, inputs=[prompt, chat, active_agent], outputs=[prompt, chat])
        clear_btn.click(lambda: [], outputs=chat)

        list_models_btn.click(refresh_models, outputs=[status, model_table, sdk_model_dropdown, new_agent_model, update_agent_model_name])
        list_loaded_btn.click(list_loaded_models, outputs=lm_output)
        load_btn.click(load_selected_model, inputs=sdk_model_dropdown, outputs=lm_output)

        create_agent_btn.click(
            create_agent,
            inputs=[new_agent_name, new_agent_model, new_agent_prompt],
            outputs=[create_agent_status, agent_map, active_agent, delegate_target, update_agent_name],
        )
        update_agent_btn.click(update_agent_model, inputs=[update_agent_name, update_agent_model_name], outputs=[update_agent_status, agent_map])

        tool_template_btn.click(apply_tool_template, inputs=tool_type, outputs=[tool_name, tool_instructions])
        create_tool_btn.click(add_tool, inputs=[tool_name, tool_type, tool_instructions, delegate_target], outputs=[create_tool_status, tool_map])

        run_wf_btn.click(run_workflow, inputs=[wf_prompt, wf_sequence], outputs=wf_output)
        draft_btn.click(draft_prompt, inputs=prompt_desc, outputs=drafted_prompt)

        demo.load(refresh_models, outputs=[status, model_table, sdk_model_dropdown, new_agent_model, update_agent_model_name])

    return demo


if __name__ == "__main__":
    app_config = load_config()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=app_config.gradio_server_port, share=app_config.gradio_share, theme=gr.themes.Soft(), css=CSS)
