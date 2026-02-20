from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.ollama import ModelInfo, OllamaController

CSS = """
.container {max-width: 1700px !important; margin: 0 auto;}
#left-pane, #right-pane {border: 1px solid #2c3550; border-radius: 14px; padding: 12px; background: #121827;}
"""


def _chatbot_uses_messages_format() -> bool:
    return "Messages" in getattr(getattr(gr.Chatbot(), "data_model", None), "__name__", "")


def _append_turn(turns: list, user: str, assistant: str, messages_format: bool):
    if messages_format:
        turns.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    else:
        turns.append((user, assistant))


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
    header = "| Model | Size | Family | Params | Quantization | Generate | Vision | Tool support |\n|---|---:|---|---|---|---|---|---|"
    rows = [
        (
            f"| `{info.name}` | {_human_size(info.size_bytes)} | {info.family} | {info.parameter_size} | {info.quantization} | "
            f"{_format_bool(info.supports_generate)} | {_format_bool(info.supports_vision)} | {_format_bool(info.supports_tools)} |"
        )
        for info in infos
    ]
    return "\n".join([header, *rows])


def _attachment_context(file_paths: list[str] | None) -> tuple[str, list[str]]:
    if not file_paths:
        return "", []

    text_parts: list[str] = []
    image_paths: list[str] = []
    for file_path in file_paths:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}:
            image_paths.append(str(path))
            continue

        try:
            if suffix in {".txt", ".md", ".csv", ".json", ".py", ".log"}:
                content = path.read_text(encoding="utf-8", errors="ignore")
                text_parts.append(f"[From {path.name}]\n{content[:3500]}")
            else:
                text_parts.append(f"[Attached file: {path.name} ({path.stat().st_size} bytes)]")
        except Exception as exc:  # noqa: BLE001
            text_parts.append(f"[Attached file unreadable: {path.name} ({exc})]")

    return "\n\n".join(text_parts), image_paths


def build_chat_handler(orchestrator: AgentOrchestrator, messages_format: bool) -> Callable:
    def chat(message: str, history, agent_name: str, attachments) -> Generator[tuple[str, list], None, None]:
        clean = (message or "").strip()
        turns = list(history or [])

        if not clean and not attachments:
            yield "", turns
            return

        if not agent_name:
            _append_turn(turns, clean or "(empty)", "❌ Create/select an agent first.", messages_format)
            yield "", turns
            return

        attachment_text, image_paths = _attachment_context(attachments)
        composed = clean
        if attachment_text:
            composed = f"{clean}\n\nAttachment context:\n{attachment_text}" if clean else attachment_text

        definition = orchestrator.definitions.get(agent_name)
        can_stream = bool(definition and definition.provider == "ollama" and not image_paths)

        if can_stream:
            partial = ""
            try:
                for chunk in orchestrator.stream_chat_with_agent(agent_name, composed):
                    partial = chunk
                    interim = list(turns)
                    _append_turn(interim, clean or "(attachment)", partial, messages_format)
                    yield "", interim
            except Exception as exc:  # noqa: BLE001
                _append_turn(turns, clean or "(attachment)", f"❌ Agent error: {exc}", messages_format)
                yield "", turns
                return

            _append_turn(turns, clean or "(attachment)", partial, messages_format)
            yield "", turns
            return

        try:
            response = orchestrator.chat_with_agent(agent_name, composed, image_paths=image_paths)
        except Exception as exc:  # noqa: BLE001
            response = f"❌ Agent error: {exc}"

        _append_turn(turns, clean or "(attachment)", response, messages_format)
        yield "", turns

    return chat


def build_app() -> gr.Blocks:
    config = load_config()
    orchestrator = AgentOrchestrator(config)
    controller = OllamaController(config)
    messages_format = _chatbot_uses_messages_format()
    chat_fn = build_chat_handler(orchestrator, messages_format=messages_format)

    def _local_infos() -> list[ModelInfo]:
        ok, infos, _ = controller.list_local_models_detailed()
        return infos if ok else []

    def _hf_models() -> list[str]:
        return orchestrator.hf.configured_models()

    def _pick_startup_model() -> str:
        models = [m.name for m in _local_infos()]
        if not models:
            return config.default_model
        return models[0]

    startup_model = _pick_startup_model()
    orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.", provider="ollama")

    def _agent_choices(default: str | None = None) -> dict[str, Any]:
        agents = orchestrator.list_agents()
        return gr.update(choices=agents, value=(default if default in agents else agents[0]))

    def _agent_map_text() -> str:
        rows = [f"- `{name}` → `{model}`" for name, model in orchestrator.get_agent_models().items()]
        return "\n".join(rows) if rows else "No agents configured yet."

    def _tool_map_text() -> str:
        rows = [f"- `{name}`" for name in orchestrator.get_tool_names()]
        return "\n".join(rows) if rows else "No custom tools yet."

    def _provider_models(provider: str) -> list[str]:
        if provider == "huggingface":
            return _hf_models() or [config.hf_default_model]
        local = _agent_model_choices(_local_infos())
        return local or [startup_model]

    def refresh_models(provider_create: str = "ollama", provider_update: str = "ollama"):
        ok, infos, error = controller.list_local_models_detailed()
        load_choices = [i.name for i in infos] if ok and infos else [startup_model]
        create_choices = _provider_models(provider_create)
        update_choices = _provider_models(provider_update)
        hf_md = "\n".join(["### Hugging Face catalog", *[f"- `{m}`" for m in _hf_models()]])

        status = f"✅ Found {len(infos)} Ollama model(s)." if ok else f"❌ {error}"
        model_md = _models_markdown(infos) if ok else ""
        return (
            status,
            model_md,
            hf_md,
            gr.update(choices=load_choices, value=load_choices[0]),
            gr.update(choices=create_choices, value=create_choices[0]),
            gr.update(choices=update_choices, value=update_choices[0]),
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

    def create_agent(name: str, provider: str, model_name: str | None, system_prompt: str):
        clean = name.strip().lower().replace(" ", "-")
        model = (model_name or "").strip()
        if not clean:
            return "❌ Agent name is required.", gr.update(), gr.update(), gr.update(), gr.update()
        if not model:
            return "❌ Select a model.", gr.update(), gr.update(), gr.update(), gr.update()
        if clean in orchestrator.definitions:
            return "❌ Agent already exists.", gr.update(), gr.update(), gr.update(), gr.update()
        orchestrator.add_agent(clean, model, system_prompt.strip(), provider=provider)
        return f"✅ Agent `{clean}` created.", gr.update(value=_agent_map_text()), _agent_choices(clean), _agent_choices(clean), _agent_choices(clean)

    def update_agent_model(agent_name: str, provider: str, model_name: str | None):
        selected = (model_name or "").strip()
        if not agent_name:
            return "❌ Select an agent.", gr.update()
        if not selected:
            return "❌ Select a model.", gr.update()
        orchestrator.set_agent_model(agent_name, selected, provider=provider)
        return f"✅ Updated `{agent_name}` to `{provider}:{selected}`.", gr.update(value=_agent_map_text())

    def draft_prompt(description: str) -> str:
        if not description.strip():
            return ""
        try:
            return orchestrator.generate_system_prompt(description)
        except Exception as exc:  # noqa: BLE001
            return f"❌ {exc}"

    def apply_tool_template(tool_mode: str):
        return ("summarize_for_exec", "Summarize output into 5 bullets.") if tool_mode == "instruction" else ("delegate_to_assistant", "Delegate task.")

    def add_tool(tool_name: str, tool_type: str, instructions: str, target_agent: str):
        clean = tool_name.strip().lower().replace(" ", "_")
        if not clean:
            return "❌ Tool name is required.", gr.update()
        if tool_type == "instruction":
            orchestrator.add_instruction_tool(clean, instructions.strip() or "General helper tool")
            return f"✅ Added `{clean}`.", gr.update(value=_tool_map_text())
        if target_agent not in orchestrator.definitions:
            return "❌ Select valid target agent.", gr.update()
        orchestrator.add_agent_delegate_tool(clean, target_agent)
        return f"✅ Added delegate `{clean}`.", gr.update(value=_tool_map_text())

    def run_workflow(prompt: str, sequence_csv: str) -> str:
        outputs = orchestrator.run_agent_workflow(prompt, [s.strip() for s in sequence_csv.split(",") if s.strip()])
        return "\n\n".join([f"## {k}\n{v}" for k, v in outputs.items()]) if outputs else "❌ No valid agents in sequence."

    with gr.Blocks(title="Local AI Studio") as demo:
        gr.Markdown("# 🤖 Local AI Studio — Ollama + Hugging Face + Tools")
        with gr.Row(equal_height=True):
            with gr.Column(scale=5, elem_id="left-pane"):
                with gr.Tabs():
                    with gr.Tab("Models"):
                        status = gr.Markdown("Refreshing models...")
                        model_table = gr.Markdown()
                        hf_table = gr.Markdown()
                        with gr.Row():
                            list_models_btn = gr.Button("Refresh Providers", variant="primary")
                            list_loaded_btn = gr.Button("List Loaded / Running")
                        sdk_model_dropdown = gr.Dropdown(label="Load Ollama model", choices=[startup_model], value=startup_model)
                        load_btn = gr.Button("Load Selected Ollama Model")
                        lm_output = gr.Textbox(label="Provider output", lines=7)

                    with gr.Tab("Agents"):
                        agent_map = gr.Markdown(value=_agent_map_text())
                        new_agent_name = gr.Textbox(label="New agent name")
                        new_agent_provider = gr.Dropdown(label="Provider", choices=["ollama", "huggingface"], value="ollama")
                        new_agent_model = gr.Dropdown(label="Model", choices=[startup_model], value=startup_model, allow_custom_value=True)
                        new_agent_prompt = gr.Textbox(label="System prompt", lines=4)
                        create_agent_btn = gr.Button("Create Agent", variant="primary")
                        create_agent_status = gr.Markdown()
                        gr.Markdown("---")
                        update_agent_name = gr.Dropdown(label="Agent", choices=orchestrator.list_agents(), value="assistant")
                        update_agent_provider = gr.Dropdown(label="Provider", choices=["ollama", "huggingface"], value="ollama")
                        update_agent_model_name = gr.Dropdown(label="New model", choices=[startup_model], value=startup_model, allow_custom_value=True)
                        update_agent_btn = gr.Button("Apply Model Update")
                        update_agent_status = gr.Markdown()

                    with gr.Tab("Tools"):
                        gr.Markdown("Built-ins include `tavily_web_search` and `mcp_query` now.")
                        tool_map = gr.Markdown(value=_tool_map_text())
                        tool_type = gr.Radio(label="Tool type", choices=["instruction", "delegate_agent"], value="instruction")
                        tool_name = gr.Textbox(label="Tool name")
                        tool_instructions = gr.Textbox(label="Instructions", lines=3)
                        delegate_target = gr.Dropdown(label="Delegate target", choices=orchestrator.list_agents(), value="assistant")
                        with gr.Row():
                            tool_template_btn = gr.Button("Use Template")
                            create_tool_btn = gr.Button("Create Tool", variant="primary")
                        create_tool_status = gr.Markdown()

                    with gr.Tab("Workflow + Prompt Builder"):
                        wf_prompt = gr.Textbox(label="Workflow prompt", lines=3)
                        wf_sequence = gr.Textbox(label="Agent sequence", value="assistant")
                        run_wf_btn = gr.Button("Run Workflow", variant="primary")
                        wf_output = gr.Markdown()
                        gr.Markdown("---")
                        prompt_desc = gr.Textbox(label="Agent description", lines=3)
                        draft_btn = gr.Button("Generate System Prompt")
                        drafted_prompt = gr.Textbox(label="Generated system prompt", lines=6)

            with gr.Column(scale=7, elem_id="right-pane"):
                gr.Markdown("### Chat")
                active_agent = gr.Dropdown(label="Active agent", choices=orchestrator.list_agents(), value="assistant")
                attachments = gr.File(label="Attach image or document", file_count="multiple", type="filepath")
                chat = gr.Chatbot(label="Conversation", height=620)
                prompt = gr.Textbox(label="Message", placeholder="Type a prompt and optionally attach image/document")
                with gr.Row():
                    send_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear Chat")

        send_btn.click(chat_fn, inputs=[prompt, chat, active_agent, attachments], outputs=[prompt, chat])
        prompt.submit(chat_fn, inputs=[prompt, chat, active_agent, attachments], outputs=[prompt, chat])
        clear_btn.click(lambda: [], outputs=chat)

        list_models_btn.click(refresh_models, inputs=[new_agent_provider, update_agent_provider], outputs=[status, model_table, hf_table, sdk_model_dropdown, new_agent_model, update_agent_model_name])
        list_loaded_btn.click(list_loaded_models, outputs=lm_output)
        load_btn.click(load_selected_model, inputs=sdk_model_dropdown, outputs=lm_output)

        new_agent_provider.change(lambda p: gr.update(choices=_provider_models(p), value=_provider_models(p)[0]), inputs=new_agent_provider, outputs=new_agent_model)
        update_agent_provider.change(lambda p: gr.update(choices=_provider_models(p), value=_provider_models(p)[0]), inputs=update_agent_provider, outputs=update_agent_model_name)

        create_agent_btn.click(create_agent, inputs=[new_agent_name, new_agent_provider, new_agent_model, new_agent_prompt], outputs=[create_agent_status, agent_map, active_agent, delegate_target, update_agent_name])
        update_agent_btn.click(update_agent_model, inputs=[update_agent_name, update_agent_provider, update_agent_model_name], outputs=[update_agent_status, agent_map])

        tool_template_btn.click(apply_tool_template, inputs=tool_type, outputs=[tool_name, tool_instructions])
        create_tool_btn.click(add_tool, inputs=[tool_name, tool_type, tool_instructions, delegate_target], outputs=[create_tool_status, tool_map])

        run_wf_btn.click(run_workflow, inputs=[wf_prompt, wf_sequence], outputs=wf_output)
        draft_btn.click(draft_prompt, inputs=prompt_desc, outputs=drafted_prompt)

        demo.load(refresh_models, inputs=[new_agent_provider, update_agent_provider], outputs=[status, model_table, hf_table, sdk_model_dropdown, new_agent_model, update_agent_model_name])

    return demo


if __name__ == "__main__":
    app_config = load_config()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=app_config.gradio_server_port, share=app_config.gradio_share, theme=gr.themes.Soft(), css=CSS)
