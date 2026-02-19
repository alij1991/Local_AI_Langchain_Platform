from __future__ import annotations

from collections.abc import Callable

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.lmstudio import OllamaController


def build_chat_handler(orchestrator: AgentOrchestrator) -> Callable:
    def chat(message: str, history: list[tuple[str, str]] | None, agent_name: str):
        if not message.strip():
            return "", history or []
        if not agent_name:
            return "", (history or []) + [(message, "❌ Create/select an agent first.")]

        response = orchestrator.chat_with_agent(agent_name, message)
        return "", (history or []) + [(message, response)]

    return chat


def build_app() -> gr.Blocks:
    config = load_config()
    orchestrator = AgentOrchestrator(config)
    controller = OllamaController(config)
    chat_fn = build_chat_handler(orchestrator)

    # starter agent for immediate use (not planner/worker pattern)
    orchestrator.add_agent(
        name="assistant",
        model_name=config.default_model,
        system_prompt="You are a practical AI assistant. Be concise, accurate, and tool-aware.",
    )

    def _agent_choices(default: str | None = None) -> dict:
        agents = orchestrator.list_agents()
        value = default if default in agents else (agents[0] if agents else None)
        return gr.update(choices=agents, value=value)

    def _agent_map_text() -> str:
        rows = [f"- `{name}` → `{model}`" for name, model in orchestrator.get_agent_models().items()]
        return "\n".join(rows) if rows else "No agents configured yet."

    def _tool_map_text() -> str:
        rows = [f"- `{name}`" for name in orchestrator.get_tool_names()]
        return "\n".join(rows) if rows else "No custom tools yet."

    def list_models() -> tuple[str, dict, dict]:
        result = controller.list_local_models()
        if not result.ok:
            return f"❌ {result.output}", gr.update(), gr.update()
        models = [line for line in result.output.splitlines() if line.strip()]
        if not models:
            return "No models returned by Ollama SDK.", gr.update(), gr.update()
        return (
            f"✅ Found {len(models)} model(s).",
            gr.update(choices=models, value=models[0]),
            gr.update(choices=models, value=models[0]),
        )

    def load_selected_model(model_name: str) -> str:
        result = controller.load_model(model_name)
        return result.output if result.ok else f"❌ {result.output}"

    def start_server() -> str:
        result = controller.start_server()
        return result.output if result.ok else f"❌ {result.output}"

    def stop_server() -> str:
        result = controller.stop_server()
        return result.output if result.ok else f"❌ {result.output}"

    def list_loaded_models() -> str:
        result = controller.list_loaded_models()
        return result.output if result.ok else f"❌ {result.output}"

    def create_agent(name: str, model_name: str, system_prompt: str):
        clean = name.strip().lower().replace(" ", "-")
        if not clean:
            return "❌ Agent name is required.", gr.update(), gr.update(), gr.update(), gr.update()
        if clean in orchestrator.definitions:
            return "❌ Agent already exists.", gr.update(), gr.update(), gr.update(), gr.update()
        orchestrator.add_agent(clean, model_name.strip(), system_prompt.strip())
        return (
            f"✅ Agent `{clean}` created.",
            gr.update(value=_agent_map_text()),
            _agent_choices(clean),
            _agent_choices(clean),
            _agent_choices(clean),
        )

    def update_agent_model(agent_name: str, model_name: str):
        if not agent_name:
            return "❌ Select an agent.", gr.update()
        orchestrator.set_agent_model(agent_name, model_name)
        return f"✅ Updated `{agent_name}` model to `{model_name}`.", gr.update(value=_agent_map_text())

    def draft_prompt(description: str) -> str:
        if not description.strip():
            return ""
        return orchestrator.generate_system_prompt(description)

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
        gr.Markdown("# 🤖 Local AI System Builder")
        gr.Markdown("Create agents, generate prompts, register tools, and run graph workflows.")

        with gr.Row():
            with gr.Column(scale=2):
                agent_map = gr.Markdown(value=_agent_map_text(), label="Agents")
            with gr.Column(scale=1):
                tool_map = gr.Markdown(value=_tool_map_text(), label="Tools")

        with gr.Tabs():
            with gr.Tab("1) Ollama"):
                status = gr.Markdown("Use SDK actions below.")
                with gr.Row():
                    start_btn = gr.Button("Start Server")
                    stop_btn = gr.Button("Stop Server")
                    list_loaded_btn = gr.Button("List Loaded Models")
                    list_models_btn = gr.Button("List Local Models")
                with gr.Row():
                    sdk_model_dropdown = gr.Dropdown(label="Local models", choices=[])
                    load_btn = gr.Button("Load Selected Model", variant="primary")
                lm_output = gr.Textbox(label="Ollama output", lines=8)

            with gr.Tab("2) Prompt Builder (built-in)"):
                gr.Markdown(
                    f"Built-in prompt-crafter model: `{config.prompt_builder_model}`"
                )
                prompt_desc = gr.Textbox(
                    label="Agent description",
                    lines=4,
                    placeholder="Example: Agent that reviews contract clauses and highlights legal risk.",
                )
                draft_btn = gr.Button("Generate System Prompt", variant="primary")
                drafted_prompt = gr.Textbox(label="Generated system prompt", lines=8)

            with gr.Tab("3) Agent Builder"):
                with gr.Row():
                    new_agent_name = gr.Textbox(label="Agent name", placeholder="legal-reviewer")
                    new_agent_model = gr.Dropdown(
                        label="Model",
                        choices=[config.default_model],
                        value=config.default_model,
                        allow_custom_value=True,
                    )
                new_agent_prompt = gr.Textbox(label="System prompt", lines=6)
                create_agent_btn = gr.Button("Create Agent", variant="primary")
                create_agent_status = gr.Markdown()

                gr.Markdown("### Update existing agent model")
                with gr.Row():
                    update_agent_name = gr.Dropdown(
                        label="Agent",
                        choices=orchestrator.list_agents(),
                        value=orchestrator.list_agents()[0],
                    )
                    update_agent_model_name = gr.Dropdown(
                        label="New model",
                        choices=[config.default_model],
                        value=config.default_model,
                        allow_custom_value=True,
                    )
                update_agent_btn = gr.Button("Apply Model Update")
                update_agent_status = gr.Markdown()

            with gr.Tab("4) Tool Builder"):
                with gr.Row():
                    tool_name = gr.Textbox(label="Tool name", placeholder="delegate_to_legal_reviewer")
                    tool_type = gr.Radio(
                        label="Tool type",
                        choices=["instruction", "delegate_agent"],
                        value="instruction",
                    )
                tool_instructions = gr.Textbox(label="Instructions", lines=4)
                delegate_target = gr.Dropdown(
                    label="Delegate target agent",
                    choices=orchestrator.list_agents(),
                    value=orchestrator.list_agents()[0],
                )
                create_tool_btn = gr.Button("Create Tool", variant="primary")
                create_tool_status = gr.Markdown()

            with gr.Tab("5) Chat"):
                active_agent = gr.Dropdown(
                    label="Chat with agent",
                    choices=orchestrator.list_agents(),
                    value=orchestrator.list_agents()[0],
                )
                chat = gr.Chatbot(label="Conversation")
                prompt = gr.Textbox(label="Message")
                send_btn = gr.Button("Send", variant="primary")

            with gr.Tab("6) Graph Workflow"):
                wf_prompt = gr.Textbox(label="Workflow prompt", lines=4)
                wf_sequence = gr.Textbox(label="Agent sequence (comma-separated)", value="assistant")
                run_wf_btn = gr.Button("Run Workflow", variant="primary")
                wf_output = gr.Markdown()

        start_btn.click(start_server, outputs=lm_output)
        stop_btn.click(stop_server, outputs=lm_output)
        list_loaded_btn.click(list_loaded_models, outputs=lm_output)
        list_models_btn.click(list_models, outputs=[status, sdk_model_dropdown, new_agent_model])
        load_btn.click(load_selected_model, inputs=sdk_model_dropdown, outputs=lm_output)

        draft_btn.click(draft_prompt, inputs=prompt_desc, outputs=drafted_prompt)

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

        create_tool_btn.click(
            add_tool,
            inputs=[tool_name, tool_type, tool_instructions, delegate_target],
            outputs=[create_tool_status, tool_map],
        )

        send_btn.click(chat_fn, inputs=[prompt, chat, active_agent], outputs=[prompt, chat])
        prompt.submit(chat_fn, inputs=[prompt, chat, active_agent], outputs=[prompt, chat])

        run_wf_btn.click(run_workflow, inputs=[wf_prompt, wf_sequence], outputs=wf_output)

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
