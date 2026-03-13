from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.ollama import ModelInfo, OllamaController

CSS = """
.container {max-width: 1580px !important; margin: 0 auto;}
body, .gradio-container {background: #090b12 !important; color: #e9edf8;}
#app-shell {min-height: 92vh; gap: 14px; align-items: stretch;}
#sidebar {background: #121724; border: 1px solid #283149; border-radius: 16px; padding: 12px;}
.nav-btn button {
  width: 100%; text-align: left; justify-content: flex-start;
  border-radius: 12px; border: 1px solid #2e3750; background: #151b2a; min-height: 44px;
}
#main-panel {
  background: radial-gradient(1200px 500px at 20% 10%, #121a32 0%, #0f1323 60%, #0d1020 100%);
  border: 1px solid #2b3551; border-radius: 24px; padding: 18px;
}
#chat-hero {display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 80vh;}
#chat-title {text-align: center; margin: 8px 0 16px 0 !important; font-size: 42px; font-weight: 700; color: #f5f8ff;}
#integration-row {display: flex; gap: 18px; margin-bottom: 18px; align-items: center; justify-content: center;}
#integration-row .icon {
  width: 28px; height: 28px; border-radius: 999px; display: inline-flex; align-items: center; justify-content: center;
  background: #101627; border: 1px solid #2a3450; color: #e8ecff; font-size: 14px;
}
#chat-wrap {
  width: min(860px, 92%); background: #0c111e; border: 1px solid #242e47;
  border-radius: 16px; padding: 8px; box-shadow: 0 12px 40px rgba(0,0,0,.35);
}
#chat-history {border: none !important; background: transparent !important;}
#chat-history .bubble-wrap {font-size: 15px;}
#composer-shell {border-top: 1px solid #212a41; padding: 10px 8px 6px 8px;}
#askbox textarea {
  border: none !important; background: transparent !important; box-shadow: none !important;
  min-height: 56px !important; color: #f0f4ff !important; font-size: 30px;
}
#askbox textarea::placeholder {color: #8f9ab8 !important;}
#composer-bottom {align-items: center; gap: 8px;}
#plus-mini button, #mic-btn button {
  border-radius: 10px; min-width: 36px; width: 36px; min-height: 36px; padding: 0;
  background: transparent; border: 1px solid #34405e; color: #dfe6ff;
}
#agent-mini {max-width: 180px;}
#agent-mini .wrap {border-radius: 10px !important; border: 1px solid #34405e !important; background: #11182b !important;}
#agent-mini input {color: #edf2ff !important;}
#send-mini {margin-left: auto;}
#send-mini button {
  border-radius: 999px; min-width: 38px; width: 38px; min-height: 38px; padding: 0;
  background: #00bfa5; border: none; font-size: 16px; color: #05231f;
}
#hidden-upload {display: none;}
#mic-status {font-size: 12px; opacity: 0.85; color: #8f9ab8; margin-top: 2px; min-height: 16px;}
#chat-clear button {margin-top: 8px; opacity: .75;}
"""

MIC_SCRIPT = """
<script>
(() => {
  const setup = () => {
    const micButton = document.querySelector('#mic-btn button');
    const promptArea = document.querySelector('#askbox textarea');
    const statusNode = document.querySelector('#mic-status p');
    if (!micButton || !promptArea || !statusNode || micButton.dataset.bound === '1') return;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      statusNode.textContent = '⚠️ Browser speech recognition is unavailable.';
      micButton.disabled = true;
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = false;
    let active = false;
    let finalText = '';
    recognition.onstart = () => { active = true; micButton.textContent = '■'; statusNode.textContent = 'Listening...'; };
    recognition.onresult = (event) => {
      let interim = '';
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) finalText += transcript + ' '; else interim += transcript;
      }
      promptArea.value = `${finalText}${interim}`.trim();
      promptArea.dispatchEvent(new Event('input', { bubbles: true }));
    };
    recognition.onend = () => { active = false; micButton.textContent = '🎙'; statusNode.textContent = finalText ? 'Speech captured.' : 'mic ready'; finalText = ''; };
    recognition.onerror = (event) => { active = false; micButton.textContent = '🎙'; statusNode.textContent = `Mic error: ${event.error}`; };
    micButton.addEventListener('click', () => {
      if (active) return recognition.stop();
      try { recognition.start(); } catch { statusNode.textContent = 'Microphone already active.'; }
    });
    micButton.dataset.bound = '1';
  };
  window.addEventListener('load', setup);
  setTimeout(setup, 900);
})();
</script>
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
            if suffix in {".txt", ".md", ".csv", ".json", ".py", ".log", ".html"}:
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
    systems_registry: dict[str, dict[str, str]] = {}

    def _local_infos() -> list[ModelInfo]:
        ok, infos, _ = controller.list_local_models_detailed()
        return infos if ok else []

    def _hf_models() -> list[str]:
        return orchestrator.hf.configured_models()

    startup_model = ([m.name for m in _local_infos()] or [config.default_model])[0]
    orchestrator.add_agent("assistant", startup_model, "You are a practical AI assistant.", provider="ollama")

    def _agent_choices(default: str | None = None) -> dict[str, Any]:
        agents = orchestrator.list_agents()
        return gr.update(choices=agents, value=(default if default in agents else agents[0]))

    def _agent_map_text() -> str:
        rows = [f"- `{n}` → `{m}`" for n, m in orchestrator.get_agent_models().items()]
        return "\n".join(rows) if rows else "No agents configured yet."

    def _tool_map_text() -> str:
        rows = [f"- `{name}`" for name in orchestrator.get_tool_names()]
        return "\n".join(rows) if rows else "No custom tools yet."

    def _provider_models(provider: str) -> list[str]:
        if provider == "huggingface":
            return _hf_models() or [config.hf_default_model]
        local = _agent_model_choices(_local_infos())
        return local or [startup_model]

    def _hf_markdown() -> str:
        return "\n".join(["### Hugging Face catalog", *[f"- `{m}`" for m in _hf_models()]])

    def refresh_models(provider_create: str = "ollama", provider_update: str = "ollama"):
        ok, infos, error = controller.list_local_models_detailed()
        load_choices = [i.name for i in infos] if ok and infos else [startup_model]
        create_choices = _provider_models(provider_create)
        update_choices = _provider_models(provider_update)
        return (
            (f"✅ Found {len(infos)} Ollama model(s)." if ok else f"❌ {error}"),
            (_models_markdown(infos) if ok else ""),
            _hf_markdown(),
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
        if clean in orchestrator.definitions:
            return "❌ Agent already exists.", gr.update(), gr.update(), gr.update(), gr.update()
        if not model:
            return "❌ Select a model.", gr.update(), gr.update(), gr.update(), gr.update()
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
        seq = [s.strip() for s in sequence_csv.split(",") if s.strip()]
        outputs = orchestrator.run_agent_workflow(prompt, seq)
        return "\n\n".join([f"## {k}\n{v}" for k, v in outputs.items()]) if outputs else "❌ No valid agents in sequence."

    def save_system(name: str, objective: str, sequence_csv: str, tools_csv: str, notes: str):
        clean = name.strip().lower().replace(" ", "-")
        if not clean:
            return "❌ System name is required.", gr.update(), gr.update()
        systems_registry[clean] = {
            "objective": objective.strip(),
            "sequence": sequence_csv.strip(),
            "tools": tools_csv.strip(),
            "notes": notes.strip(),
        }
        listing = "\n".join([f"- `{k}`: {v['objective'] or 'No objective'}" for k, v in systems_registry.items()])
        return f"✅ System `{clean}` saved.", gr.update(value=listing), gr.update(choices=list(systems_registry.keys()), value=clean)

    def run_system(system_name: str, prompt: str) -> str:
        if system_name not in systems_registry:
            return "❌ Select a valid saved system."
        seq = [s.strip() for s in systems_registry[system_name]["sequence"].split(",") if s.strip()]
        outputs = orchestrator.run_agent_workflow(prompt, seq)
        if not outputs:
            return "❌ System run produced no outputs (check sequence agents)."
        return "\n\n".join([f"## {k}\n{v}" for k, v in outputs.items()])

    def switch_section(section: str):
        return (
            gr.update(visible=section == "chat"),
            gr.update(visible=section == "models"),
            gr.update(visible=section == "agents"),
            gr.update(visible=section == "tools"),
            gr.update(visible=section == "systems"),
        )

    with gr.Blocks(title="Local AI Studio") as demo:
        with gr.Row(elem_id="app-shell"):
            with gr.Column(scale=1, min_width=220, elem_id="sidebar"):
                gr.Markdown("### Sections")
                nav_chat = gr.Button("💬 Chat", elem_classes=["nav-btn"], variant="secondary")
                nav_models = gr.Button("🧠 Models", elem_classes=["nav-btn"], variant="secondary")
                nav_agents = gr.Button("🤖 Agents", elem_classes=["nav-btn"], variant="secondary")
                nav_tools = gr.Button("🧰 Tools", elem_classes=["nav-btn"], variant="secondary")
                nav_systems = gr.Button("🧩 Systems", elem_classes=["nav-btn"], variant="secondary")

            with gr.Column(scale=6, elem_id="main-panel"):
                with gr.Column(visible=True) as section_chat:
                    with gr.Column(elem_id="chat-hero"):
                        gr.Markdown("# Ask anything", elem_id="chat-title")
                        gr.HTML(
                            """
                            <div id='integration-row'>
                              <span class='icon'>◌</span><span class='icon'>𝕏</span><span class='icon'>M</span>
                              <span class='icon'>31</span><span class='icon'>▦</span><span class='icon'>📄</span>
                              <span class='icon'>◔</span><span class='icon'>✣</span><span class='icon'>◍</span>
                            </div>
                            """
                        )
                        with gr.Column(elem_id="chat-wrap"):
                            chat = gr.Chatbot(label="", height=260, elem_id="chat-history")
                            with gr.Column(elem_id="composer-shell"):
                                prompt = gr.Textbox(label="", placeholder="Write your message...", elem_id="askbox")
                                with gr.Row(elem_id="composer-bottom"):
                                    plus_btn = gr.Button("+", elem_id="plus-mini", variant="secondary")
                                    active_agent = gr.Dropdown(label="", choices=orchestrator.list_agents(), value="assistant", elem_id="agent-mini")
                                    mic_btn = gr.Button("🎙", elem_id="mic-btn", variant="secondary")
                                    send_btn = gr.Button("↑", elem_id="send-mini", variant="primary")
                                mic_status = gr.Markdown("mic ready", elem_id="mic-status")
                                attachments = gr.File(label="", value=None, file_count="multiple", type="filepath", elem_id="hidden-upload", file_types=None, interactive=True)
                            clear_btn = gr.Button("Clear", variant="secondary", elem_id="chat-clear")
                            gr.HTML(MIC_SCRIPT)

                with gr.Column(visible=False) as section_models:
                    gr.Markdown("### Models", elem_id="section-title")
                    status = gr.Markdown("Refreshing models...")
                    model_table = gr.Markdown()
                    hf_table = gr.Markdown()
                    with gr.Row():
                        list_models_btn = gr.Button("Refresh Providers", variant="primary")
                        list_loaded_btn = gr.Button("List Loaded / Running")
                    sdk_model_dropdown = gr.Dropdown(label="Load Ollama model", choices=[startup_model], value=startup_model)
                    load_btn = gr.Button("Load Selected Ollama Model")
                    lm_output = gr.Textbox(label="Provider output", lines=7)

                with gr.Column(visible=False) as section_agents:
                    gr.Markdown("### Agents", elem_id="section-title")
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

                with gr.Column(visible=False) as section_tools:
                    gr.Markdown("### Tools", elem_id="section-title")
                    gr.Markdown("Built-ins: `multiply_numbers`, `utc_now`, `tavily_web_search`, `mcp_query`.")
                    tool_map = gr.Markdown(value=_tool_map_text())
                    tool_type = gr.Radio(label="Tool type", choices=["instruction", "delegate_agent"], value="instruction")
                    tool_name = gr.Textbox(label="Tool name")
                    tool_instructions = gr.Textbox(label="Instructions", lines=3)
                    delegate_target = gr.Dropdown(label="Delegate target", choices=orchestrator.list_agents(), value="assistant")
                    with gr.Row():
                        tool_template_btn = gr.Button("Use Template")
                        create_tool_btn = gr.Button("Create Tool", variant="primary")
                    create_tool_status = gr.Markdown()

                with gr.Column(visible=False) as section_systems:
                    gr.Markdown("### Systems", elem_id="section-title")
                    gr.Markdown("Design multi-agent systems (email responder, news analyzer, etc.).")
                    system_name = gr.Textbox(label="System name", placeholder="news-analysis-pipeline")
                    system_objective = gr.Textbox(label="Objective", lines=2)
                    system_sequence = gr.Textbox(label="Agent sequence (comma-separated)", value="assistant")
                    system_tools = gr.Textbox(label="Preferred tools (comma-separated)", placeholder="tavily_web_search,utc_now")
                    system_notes = gr.Textbox(label="Notes", lines=3)
                    with gr.Row():
                        save_system_btn = gr.Button("Save System", variant="primary")
                        run_system_btn = gr.Button("Run System")
                    save_system_status = gr.Markdown()
                    systems_list = gr.Markdown(value="No systems saved yet.")
                    systems_selector = gr.Dropdown(label="Saved systems", choices=[])
                    systems_prompt = gr.Textbox(label="System run prompt", lines=2)
                    systems_output = gr.Markdown()

        nav_chat.click(lambda: switch_section("chat"), outputs=[section_chat, section_models, section_agents, section_tools, section_systems])
        nav_models.click(lambda: switch_section("models"), outputs=[section_chat, section_models, section_agents, section_tools, section_systems])
        nav_agents.click(lambda: switch_section("agents"), outputs=[section_chat, section_models, section_agents, section_tools, section_systems])
        nav_tools.click(lambda: switch_section("tools"), outputs=[section_chat, section_models, section_agents, section_tools, section_systems])
        nav_systems.click(lambda: switch_section("systems"), outputs=[section_chat, section_models, section_agents, section_tools, section_systems])

        send_btn.click(chat_fn, inputs=[prompt, chat, active_agent, attachments], outputs=[prompt, chat])
        prompt.submit(chat_fn, inputs=[prompt, chat, active_agent, attachments], outputs=[prompt, chat])
        clear_btn.click(lambda: [], outputs=chat)

        plus_btn.click(None, None, None, js="""() => { const input = document.querySelector('#hidden-upload input[type=file]'); if (input) input.click(); }""")

        list_models_btn.click(refresh_models, inputs=[new_agent_provider, update_agent_provider], outputs=[status, model_table, hf_table, sdk_model_dropdown, new_agent_model, update_agent_model_name])
        list_loaded_btn.click(list_loaded_models, outputs=lm_output)
        load_btn.click(load_selected_model, inputs=sdk_model_dropdown, outputs=lm_output)

        new_agent_provider.change(lambda p: gr.update(choices=_provider_models(p), value=_provider_models(p)[0]), inputs=new_agent_provider, outputs=new_agent_model)
        update_agent_provider.change(lambda p: gr.update(choices=_provider_models(p), value=_provider_models(p)[0]), inputs=update_agent_provider, outputs=update_agent_model_name)

        create_agent_btn.click(create_agent, inputs=[new_agent_name, new_agent_provider, new_agent_model, new_agent_prompt], outputs=[create_agent_status, agent_map, active_agent, delegate_target, update_agent_name])
        update_agent_btn.click(update_agent_model, inputs=[update_agent_name, update_agent_provider, update_agent_model_name], outputs=[update_agent_status, agent_map])

        tool_template_btn.click(apply_tool_template, inputs=tool_type, outputs=[tool_name, tool_instructions])
        create_tool_btn.click(add_tool, inputs=[tool_name, tool_type, tool_instructions, delegate_target], outputs=[create_tool_status, tool_map])

        save_system_btn.click(save_system, inputs=[system_name, system_objective, system_sequence, system_tools, system_notes], outputs=[save_system_status, systems_list, systems_selector])
        run_system_btn.click(run_system, inputs=[systems_selector, systems_prompt], outputs=systems_output)

        demo.load(refresh_models, inputs=[new_agent_provider, update_agent_provider], outputs=[status, model_table, hf_table, sdk_model_dropdown, new_agent_model, update_agent_model_name])

    return demo


if __name__ == "__main__":
    app_config = load_config()
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=app_config.gradio_server_port, share=app_config.gradio_share, theme=gr.themes.Soft(), css=CSS)
