from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.ollama import ModelInfo, OllamaController


def build_chat_handler(orchestrator: AgentOrchestrator, messages_format: bool) -> Callable:
    def chat(message: str, history, agent_name: str):
        clean = message.strip()
        if not clean:
            return "", history or []

        try:
            if not agent_name:
                response = "❌ Create/select an agent first."
            else:
                response = orchestrator.chat_with_agent(agent_name, clean)
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
    chatbot = gr.Chatbot()
    data_model_name = getattr(getattr(chatbot, "data_model", None), "__name__", "")
    return "Messages" in data_model_name

def _human_size(size_bytes: int | None) -> str:
    if size_bytes is None:
        return "unknown"
    size = float(size_bytes)
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return "unknown"


def _format_tool_support(value: bool | None) -> str:
    if value is True:
        return "yes"
    if value is False:
        return "no"
    return "unknown"


def _model_label(info: ModelInfo) -> str:
    return (
        f"{info.name} · {_human_size(info.size_bytes)} · {info.parameter_size} · "
        f"{info.quantization} · tools:{_format_tool_support(info.supports_tools)}"
    )


def _load_model_choices(infos: list[ModelInfo]) -> list[str]:
    return [info.name for info in infos]


def _agent_model_choices(infos: list[ModelInfo]) -> list[str]:
    chat_models = [
        info.name
        for info in infos
        if info.supports_generate is not False and "embedding" not in info.name.lower()
    ]
    return chat_models or [info.name for info in infos]


def _models_markdown(infos: list[ModelInfo]) -> str:
    if not infos:
        return "No local models found."

    header = "| Model | Size | Family | Params | Quantization | Generate | Tool support |\n|---|---:|---|---|---|---|---|"
    rows = [
        (
            f"| `{info.name}` | {_human_size(info.size_bytes)} | {info.family} | "
            f"{info.parameter_size} | {info.quantization} | {_format_tool_support(info.supports_generate)} | {_format_tool_support(info.supports_tools)} |"
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

    def _local_model_infos() -> list[ModelInfo]:
        ok, infos, _ = controller.list_local_models_detailed()
        return infos if ok else []

    def _pick_startup_model() -> str:
        infos = _local_model_infos()
        models = [info.name for info in infos]
        if not models:
            return config.default_model

        preferred = ["gemma3:1b", "qwen2.5:1.5b", "llama3.2:3b", config.default_model]
        for model in preferred:
            if model in models:
                return model
        return models[0]

    startup_model = _pick_startup_model()
    if config.prompt_builder_model not in [info.name for info in _local_model_infos()] and startup_model:
        config.prompt_builder_model = startup_model

    orchestrator.add_agent(
        name="assistant",
        model_name=startup_model,
        system_prompt="You are a practical AI assistant. Be concise, accurate, and tool-aware.",
    )

    def _agent_choices(default: str | None = None) -> dict[str, Any]:
        agents = orchestrator.list_agents()
        value = default if default in agents else (agents[0] if agents else None)
        return gr.update(choices=agents, value=value)

    def _agent_map_text() -> str:
        rows = [f"- `{name}` → `{model}`" for name, model in orchestrator.get_agent_models().items()]
        return "\n".join(rows) if rows else "No agents configured yet."

    def _tool_map_text() -> str:
        rows = [f"- `{name}`" for name in orchestrator.get_tool_names()]
        return "\n".join(rows) if rows else "No custom tools yet."

    def refresh_models() -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
        ok, infos, error = controller.list_local_models_detailed()
        if not ok:
            return (
                f"❌ {error}",
                "",
                gr.update(),
                gr.update(choices=[startup_model], value=startup_model),
                gr.update(choices=[startup_model], value=startup_model),
            )
        if not infos:
            return (
                "No models returned by Ollama SDK.",
                "",
                gr.update(),
                gr.update(choices=[startup_model], value=startup_model),
                gr.update(choices=[startup_model], value=startup_model),
            )

        load_choices = _load_model_choices(infos)
        agent_choices = _agent_model_choices(infos)
        chosen = startup_model if startup_model in agent_choices else agent_choices[0]
        load_chosen = startup_model if startup_model in load_choices else load_choices[0]

        return (
            f"✅ Found {len(infos)} model(s).",
            _models_markdown(infos),
            gr.update(choices=load_choices, value=load_chosen),
            gr.update(choices=agent_choices, value=chosen),
            gr.update(choices=agent_choices, value=chosen),
        )

    def load_selected_model(model_name: str) -> str:
        result = controller.load_model(model_name)
        if result.ok:
            return result.output
        return f"❌ {result.output}"

    def list_loaded_models() -> str:
        result = controller.list_loaded_models()
        return result.output if result.ok else f"❌ {result.output}"

    def create_agent(name: str, model_name: str | None, system_prompt: str):
        clean = name.strip().lower().replace(" ", "-")
        selected_model = (model_name or "").strip()
        if not clean:
            return "❌ Agent name is required.", gr.update(), gr.update(), gr.update(), gr.update()
        if clean in orchestrator.definitions:
            return "❌ Agent already exists.", gr.update(), gr.update(), gr.update(), gr.update()
        if not selected_model:
            return "❌ Select or type a model name.", gr.update(), gr.update(), gr.update(), gr.update()

        orchestrator.add_agent(clean, selected_model, system_prompt.strip())
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
        selected_model = (model_name or "").strip()
        if not selected_model:
            return "❌ Select or type a model name.", gr.update()
        orchestrator.set_agent_model(agent_name, selected_model)
        return f"✅ Updated `{agent_name}` model to `{selected_model}`.", gr.update(value=_agent_map_text())

    def draft_prompt(description: str) -> str:
        if not description.strip():
            return ""
        try:
            return orchestrator.generate_system_prompt(description)
        except Exception as exc:  # noqa: BLE001
            candidates = [info.name for info in _local_model_infos()]
            hint = (
                f"Prompt-builder model `{config.prompt_builder_model}` is unavailable. "
                f"Try one of: {', '.join(candidates[:5])}" if candidates else "No local Ollama models found."
            )
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
            msg = f"✅ Added instruction tool `{clean}`."
        else:
            if target_agent not in orchestrator.definitions:
                return "❌ Select a valid target agent.", gr.update()
            orchestrator.add_agent_delegate_tool(clean, target_agent)
            msg = f"✅ Added delegate tool `{clean}` -> `{target_agent}`."

        return msg, gr.update(value=_tool_map_text())

    def run_workflow(prompt: str, sequence_csv: str) -> str:
        sequence = [part.strip() for part in sequence_csv.split(",") if part.strip()]
        outputs = orchestrator.run_agent_workflow(prompt, sequence)
        if not outputs:
            return "❌ No valid agents in sequence."
        return "\n\n".join([f"## {name}\n{text}" for name, text in outputs.items()])

    with gr.Blocks(title="Local AI LangChain Platform") as demo:
        gr.Markdown("# 🤖 Local AI Studio")
        gr.Markdown("A ChatGPT-style workspace: chat on the left, build agents/tools/workflows on the right.")

        with gr.Row():
            with gr.Column(scale=7):
                with gr.Group():
                    gr.Markdown("### Chat")
                    active_agent = gr.Dropdown(
                        label="Active agent",
                        choices=orchestrator.list_agents(),
                        value=orchestrator.list_agents()[0],
                    )
                    chat = gr.Chatbot(label="Conversation", height=520)
                    prompt = gr.Textbox(label="Message", placeholder="Ask your assistant...")
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear Chat")

            with gr.Column(scale=5):
                with gr.Tab("Model Catalog"):
                    status = gr.Markdown("Click refresh to load local model metadata.")
                    model_table = gr.Markdown()
                    with gr.Row():
                        list_models_btn = gr.Button("Refresh Models", variant="primary")
                        list_loaded_btn = gr.Button("List Loaded")
                    sdk_model_dropdown = gr.Dropdown(label="Load model", choices=[startup_model], value=startup_model)
                    load_btn = gr.Button("Load Selected Model")
                    lm_output = gr.Textbox(label="Ollama output", lines=6)

                with gr.Tab("Agents"):
                    agent_map = gr.Markdown(value=_agent_map_text(), label="Agent map")
                    gr.Markdown("Create role-specific agents and map each to a model.")
                    new_agent_name = gr.Textbox(label="New agent name", placeholder="legal-reviewer")
                    new_agent_model = gr.Dropdown(
                        label="Model",
                        choices=[startup_model],
                        value=startup_model,
                        allow_custom_value=True,
                    )
                    new_agent_prompt = gr.Textbox(label="System prompt", lines=5)
                    create_agent_btn = gr.Button("Create Agent", variant="primary")
                    create_agent_status = gr.Markdown()

                    gr.Markdown("---\nUpdate existing agent model")
                    update_agent_name = gr.Dropdown(
                        label="Agent",
                        choices=orchestrator.list_agents(),
                        value=orchestrator.list_agents()[0],
                    )
                    update_agent_model_name = gr.Dropdown(
                        label="New model",
                        choices=[startup_model],
                        value=startup_model,
                        allow_custom_value=True,
                    )
                    update_agent_btn = gr.Button("Apply Model Update")
                    update_agent_status = gr.Markdown()

                with gr.Tab("Tools"):
                    gr.Markdown(
                        "### How tools work\n"
                        "- **Instruction tool**: reusable policy snippet injected when an agent calls the tool.\n"
                        "- **Delegate tool**: forwards sub-tasks to another configured agent.\n"
                        "\nTip: give tools very specific names and instructions so the model knows when to call them."
                    )
                    tool_map = gr.Markdown(value=_tool_map_text(), label="Registered tools")
                    tool_type = gr.Radio(label="Tool type", choices=["instruction", "delegate_agent"], value="instruction")
                    tool_name = gr.Textbox(label="Tool name", placeholder="summarize_for_exec")
                    tool_instructions = gr.Textbox(label="Instructions", lines=4)
                    delegate_target = gr.Dropdown(
                        label="Delegate target agent",
                        choices=orchestrator.list_agents(),
                        value=orchestrator.list_agents()[0],
                    )
                    with gr.Row():
                        tool_template_btn = gr.Button("Use Suggested Template")
                        create_tool_btn = gr.Button("Create Tool", variant="primary")
                    create_tool_status = gr.Markdown()

                with gr.Tab("Workflow + Prompt Builder"):
                    gr.Markdown("Run sequential LangGraph workflows and draft system prompts.")
                    wf_prompt = gr.Textbox(label="Workflow prompt", lines=4)
                    wf_sequence = gr.Textbox(label="Agent sequence (comma-separated)", value="assistant")
                    run_wf_btn = gr.Button("Run Workflow", variant="primary")
                    wf_output = gr.Markdown()

                    gr.Markdown("---\nPrompt Builder")
                    gr.Markdown(f"Prompt-builder model: `{config.prompt_builder_model}`")
                    prompt_desc = gr.Textbox(label="Agent description", lines=4)
                    draft_btn = gr.Button("Generate System Prompt")
                    drafted_prompt = gr.Textbox(label="Generated system prompt", lines=8)

        send_btn.click(chat_fn, inputs=[prompt, chat, active_agent], outputs=[prompt, chat])
        prompt.submit(chat_fn, inputs=[prompt, chat, active_agent], outputs=[prompt, chat])
        clear_btn.click(lambda: [], outputs=chat)

        list_models_btn.click(
            refresh_models,
            outputs=[status, model_table, sdk_model_dropdown, new_agent_model, update_agent_model_name],
        )
        list_loaded_btn.click(list_loaded_models, outputs=lm_output)
        load_btn.click(load_selected_model, inputs=sdk_model_dropdown, outputs=lm_output)

        create_agent_btn.click(
            create_agent,
            inputs=[new_agent_name, new_agent_model, new_agent_prompt],
            outputs=[create_agent_status, agent_map, active_agent, delegate_target, update_agent_name],
        )
        update_agent_btn.click(
            update_agent_model,
            inputs=[update_agent_name, update_agent_model_name],
            outputs=[update_agent_status, agent_map],
        )

        tool_template_btn.click(apply_tool_template, inputs=[tool_type], outputs=[tool_name, tool_instructions])
        create_tool_btn.click(
            add_tool,
            inputs=[tool_name, tool_type, tool_instructions, delegate_target],
            outputs=[create_tool_status, tool_map],
        )

        run_wf_btn.click(run_workflow, inputs=[wf_prompt, wf_sequence], outputs=wf_output)
        draft_btn.click(draft_prompt, inputs=prompt_desc, outputs=drafted_prompt)

        demo.load(
            refresh_models,
            outputs=[status, model_table, sdk_model_dropdown, new_agent_model, update_agent_model_name],
        )

    return demo


if __name__ == "__main__":
    app_config = load_config()
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=app_config.gradio_server_port,
        share=app_config.gradio_share,
        theme=gr.themes.Soft(),
    )
