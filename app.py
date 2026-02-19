from __future__ import annotations

from collections.abc import Callable

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.lmstudio import LMStudioController


APP_CSS = """
#status-chip {padding: 8px 12px; border-radius: 10px; background: #1f2937;}
#panel-hint {color: #9ca3af; font-size: 0.95rem;}
"""


def build_chat_handler(orchestrator: AgentOrchestrator) -> Callable:
    def chat(message: str, history: list[tuple[str, str]] | None, mode: str, single_agent: str):
        if not message.strip():
            return "", history or []

        history = history or []
        if mode == "Single Agent":
            response = orchestrator.chat_with_agent(single_agent, message)
        else:
            output = orchestrator.combined_response(message)
            response = "\n\n".join([f"### {name.title()}\n{text}" for name, text in output.items()])

        return "", history + [(message, response)]

    return chat


def build_app() -> gr.Blocks:
    config = load_config()
    orchestrator = AgentOrchestrator(config)
    controller = LMStudioController(config)
    chat_fn = build_chat_handler(orchestrator)

    def _agent_choices(default: str | None = None) -> dict:
        agents = orchestrator.list_agents()
        value = default if default in agents else (agents[0] if agents else None)
        return gr.update(choices=agents, value=value)

    def _agent_map_text() -> str:
        return "\n".join([f"- `{name}` → `{model}`" for name, model in orchestrator.get_agent_models().items()])

    def _tool_map_text() -> str:
        return "\n".join([f"- `{name}`" for name in orchestrator.get_tool_names()])

    def apply_agent_models(planner_model_name: str, worker_model_name: str) -> tuple[str, dict]:
        orchestrator.set_agent_model("planner", planner_model_name)
        orchestrator.set_agent_model("worker", worker_model_name)
        return "✅ Planner/Worker model assignments updated.", gr.update(value=_agent_map_text())

    def add_agent(name: str, model_name: str, system_prompt: str):
        clean_name = name.strip().lower().replace(" ", "-")
        if not clean_name:
            return "❌ Agent name is required.", gr.update(), gr.update(), gr.update(), gr.update()
        if clean_name in orchestrator.definitions:
            return "❌ Agent name already exists.", gr.update(), gr.update(), gr.update(), gr.update()

        orchestrator.add_agent(clean_name, model_name.strip(), system_prompt.strip())
        agents = orchestrator.list_agents()
        return (
            f"✅ Added agent `{clean_name}`.",
            gr.update(value=_agent_map_text()),
            _agent_choices(clean_name),
            _agent_choices(clean_name),
            gr.update(choices=agents, value=model_name.strip() if model_name.strip() else agents[0]),
        )

    def add_tool(tool_name: str, tool_type: str, tool_instructions: str, target_agent: str):
        clean_name = tool_name.strip().lower().replace(" ", "_")
        if not clean_name:
            return "❌ Tool name is required.", gr.update()
        if clean_name in orchestrator.get_tool_names():
            return "❌ Tool name already exists.", gr.update()

        if tool_type == "instruction":
            orchestrator.add_instruction_tool(clean_name, tool_instructions.strip() or "General helper tool")
            msg = f"✅ Added instruction tool `{clean_name}`."
        else:
            if target_agent not in orchestrator.definitions:
                return "❌ Pick a valid target agent.", gr.update()
            orchestrator.add_agent_delegate_tool(clean_name, target_agent)
            msg = f"✅ Added delegate tool `{clean_name}` → `{target_agent}`."

        return msg, gr.update(value=_tool_map_text())

    def run_workflow(workflow_prompt: str, sequence_csv: str) -> str:
        sequence = [part.strip() for part in sequence_csv.split(",") if part.strip()]
        outputs = orchestrator.run_agent_workflow(workflow_prompt, sequence)
        if not outputs:
            return "❌ Workflow could not run. Check sequence agent names."
        return "\n\n".join([f"## {name}\n{text}" for name, text in outputs.items()])

    def start_server() -> str:
        result = controller.start_server()
        return result.output if result.ok else f"❌ {result.output}"

    def stop_server() -> str:
        result = controller.stop_server()
        return result.output if result.ok else f"❌ {result.output}"

    def list_loaded_models() -> str:
        result = controller.list_loaded_models()
        return result.output if result.ok else f"❌ {result.output}"

    def load_selected_model(model_name: str) -> str:
        result = controller.load_model(model_name)
        return result.output if result.ok else f"❌ {result.output}"

    def list_local_models() -> tuple[str, dict, dict, dict, dict]:
        cli_result = controller.list_local_models()
        if not cli_result.ok:
            return f"❌ {cli_result.output}", gr.update(), gr.update(), gr.update(), gr.update()

        models = controller.parse_model_lines(cli_result.output)
        if not models:
            return "No models returned by CLI.", gr.update(), gr.update(), gr.update(), gr.update()

        current = orchestrator.get_agent_models()
        planner_selected = current.get("planner", models[0])
        worker_selected = current.get("worker", models[0])
        if planner_selected not in models:
            planner_selected = models[0]
        if worker_selected not in models:
            worker_selected = models[0]

        return (
            f"✅ Found {len(models)} model(s) from LM Studio CLI.",
            gr.update(choices=models, value=models[0]),
            gr.update(choices=models, value=planner_selected),
            gr.update(choices=models, value=worker_selected),
            gr.update(choices=models, value=models[0]),
        )

    theme = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate")
    with gr.Blocks(title="Local AI LangChain Platform", theme=theme, css=APP_CSS) as demo:
        gr.Markdown("# 🤖 Local AI LangChain Platform")
        gr.Markdown(
            "Design, wire, and run agentic AI systems with LM Studio + LangChain + LangGraph."
        )

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Active agent/model map")
                agent_map = gr.Markdown(value=_agent_map_text(), elem_id="status-chip")
            with gr.Column(scale=1):
                gr.Markdown("### Tool registry")
                tool_map = gr.Markdown(value=_tool_map_text(), elem_id="status-chip")

        with gr.Tabs():
            with gr.Tab("Chat Workspace"):
                gr.Markdown("Use this tab for regular interactions with one agent or planner+worker.")
                with gr.Row():
                    mode = gr.Radio(
                        label="Run mode",
                        choices=["Single Agent", "Combined (Planner + Worker)"],
                        value="Single Agent",
                    )
                    single_agent = gr.Dropdown(
                        label="Agent",
                        choices=orchestrator.list_agents(),
                        value="planner",
                    )
                with gr.Row():
                    planner_model = gr.Dropdown(
                        label="Planner model",
                        choices=[config.planner_model, config.worker_model],
                        value=config.planner_model,
                        allow_custom_value=True,
                    )
                    worker_model = gr.Dropdown(
                        label="Worker model",
                        choices=[config.planner_model, config.worker_model],
                        value=config.worker_model,
                        allow_custom_value=True,
                    )
                    apply_models_btn = gr.Button("Apply Models", variant="primary")
                model_status = gr.Markdown("Model assignments ready.")
                chatbot = gr.Chatbot(label="Conversation")
                prompt = gr.Textbox(label="Prompt", placeholder="Ask your agent system...")
                send = gr.Button("Send", variant="primary")

            with gr.Tab("LM Studio Control"):
                gr.Markdown("Run LM Studio commands directly and keep model lifecycle inside the app.")
                control_status = gr.Markdown("Click **List Local Models (CLI)** before loading models.")
                with gr.Row():
                    start_btn = gr.Button("Start Server")
                    stop_btn = gr.Button("Stop Server")
                    list_cli_btn = gr.Button("List Local Models (CLI)")
                    list_loaded_btn = gr.Button("List Loaded Models (API)")
                with gr.Row():
                    model_to_load = gr.Dropdown(label="Model to load", choices=[])
                    load_btn = gr.Button("Load Selected Model", variant="primary")
                lm_output = gr.Textbox(label="LM Studio output", lines=10)

            with gr.Tab("Agent Builder"):
                gr.Markdown("Create specialized agents with custom prompts and models.")
                with gr.Row():
                    agent_name = gr.Textbox(label="Agent name", placeholder="researcher")
                    agent_model = gr.Dropdown(
                        label="Agent model",
                        choices=[config.planner_model, config.worker_model],
                        value=config.planner_model,
                        allow_custom_value=True,
                    )
                agent_prompt = gr.Textbox(label="System prompt", lines=5)
                create_agent_btn = gr.Button("Create Agent", variant="primary")
                create_agent_status = gr.Markdown()

            with gr.Tab("Tool Builder"):
                gr.Markdown("Create tools for instructions or to delegate work to other agents.")
                with gr.Row():
                    tool_name = gr.Textbox(label="Tool name", placeholder="delegate_to_researcher")
                    tool_type = gr.Radio(
                        label="Tool type",
                        choices=["instruction", "delegate_agent"],
                        value="instruction",
                    )
                tool_instructions = gr.Textbox(label="Tool instructions", lines=4)
                delegate_target = gr.Dropdown(
                    label="Delegate target agent",
                    choices=orchestrator.list_agents(),
                    value="planner",
                )
                create_tool_btn = gr.Button("Create Tool", variant="primary")
                create_tool_status = gr.Markdown()

            with gr.Tab("Graph Workflow"):
                gr.Markdown(
                    "Run a LangGraph workflow by chaining agents in a comma-separated sequence."
                )
                workflow_prompt = gr.Textbox(label="Workflow prompt", lines=4)
                workflow_sequence = gr.Textbox(
                    label="Sequence",
                    value="planner,worker",
                    info="Example: planner,researcher,worker",
                )
                run_workflow_btn = gr.Button("Run Workflow", variant="primary")
                workflow_output = gr.Markdown()

        apply_models_btn.click(
            apply_agent_models,
            inputs=[planner_model, worker_model],
            outputs=[model_status, agent_map],
        )
        send.click(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])
        prompt.submit(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])

        start_btn.click(start_server, outputs=lm_output)
        stop_btn.click(stop_server, outputs=lm_output)
        list_loaded_btn.click(list_loaded_models, outputs=lm_output)
        list_cli_btn.click(
            list_local_models,
            outputs=[control_status, model_to_load, planner_model, worker_model, agent_model],
        )
        load_btn.click(load_selected_model, inputs=model_to_load, outputs=lm_output)

        create_agent_btn.click(
            add_agent,
            inputs=[agent_name, agent_model, agent_prompt],
            outputs=[create_agent_status, agent_map, single_agent, delegate_target, agent_model],
        )
        create_tool_btn.click(
            add_tool,
            inputs=[tool_name, tool_type, tool_instructions, delegate_target],
            outputs=[create_tool_status, tool_map],
        )
        run_workflow_btn.click(
            run_workflow,
            inputs=[workflow_prompt, workflow_sequence],
            outputs=workflow_output,
        )

    return demo


if __name__ == "__main__":
    app_config = load_config()
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=app_config.gradio_server_port,
        share=app_config.gradio_share,
    )
