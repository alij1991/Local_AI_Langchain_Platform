from __future__ import annotations

from collections.abc import Callable

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator
from local_ai_platform.lmstudio import LMStudioController


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

        history = history + [(message, response)]
        return "", history

    return chat


def build_app() -> gr.Blocks:
    config = load_config()
    orchestrator = AgentOrchestrator(config)
    controller = LMStudioController(config)
    chat_fn = build_chat_handler(orchestrator)

    def _agent_dropdown_updates(default: str | None = None) -> dict:
        agents = orchestrator.list_agents()
        value = default if default in agents else (agents[0] if agents else None)
        return gr.update(choices=agents, value=value)

    def apply_agent_models(planner_model_name: str, worker_model_name: str) -> tuple[str, dict]:
        orchestrator.set_agent_model("planner", planner_model_name)
        orchestrator.set_agent_model("worker", worker_model_name)
        return (
            f"Planner model set to `{planner_model_name}`. Worker model set to `{worker_model_name}`.",
            gr.update(value="\n".join([f"- {k}: `{v}`" for k, v in orchestrator.get_agent_models().items()])),
        )

    def add_agent(name: str, model_name: str, system_prompt: str):
        clean_name = name.strip().lower().replace(" ", "-")
        if not clean_name:
            return "❌ Agent name is required.", gr.update(), gr.update(), gr.update(), gr.update()
        if clean_name in orchestrator.definitions:
            return "❌ Agent name already exists.", gr.update(), gr.update(), gr.update(), gr.update()

        orchestrator.add_agent(clean_name, model_name.strip(), system_prompt.strip())
        agents = orchestrator.list_agents()
        status = f"✅ Added agent `{clean_name}` using model `{model_name}`."
        agent_lines = "\n".join([f"- `{a}` -> `{orchestrator.definitions[a].model_name}`" for a in agents])
        return status, gr.update(value=agent_lines), _agent_dropdown_updates(clean_name), _agent_dropdown_updates(clean_name), gr.update(choices=agents, value=agents)

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
            msg = f"✅ Added delegate tool `{clean_name}` targeting `{target_agent}`."

        tool_lines = "\n".join([f"- `{name}`" for name in orchestrator.get_tool_names()])
        return msg, gr.update(value=tool_lines)

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
            "✅ Local model list loaded from LM Studio CLI.",
            gr.update(choices=models, value=models[0]),
            gr.update(choices=models, value=planner_selected),
            gr.update(choices=models, value=worker_selected),
            gr.update(choices=models, value=models[0]),
        )

    with gr.Blocks(title="Local AI LangChain Platform") as demo:
        gr.Markdown("# 🤖 Local AI LangChain Platform")
        gr.Markdown("Build and run customizable agentic systems with LM Studio + LangChain + LangGraph.")

        with gr.Tabs():
            with gr.Tab("Chat"):
                with gr.Row():
                    mode = gr.Radio(
                        label="Agent mode",
                        choices=["Single Agent", "Combined (Planner + Worker)"],
                        value="Single Agent",
                    )
                    single_agent = gr.Dropdown(
                        label="Single agent",
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
                    apply_models_btn = gr.Button("Apply Agent Models")

                model_status = gr.Markdown(
                    f"Planner: `{config.planner_model}` | Worker: `{config.worker_model}`"
                )
                model_map = gr.Markdown(
                    value="\n".join([f"- {k}: `{v}`" for k, v in orchestrator.get_agent_models().items()]),
                    label="Active agent->model map",
                )

                chatbot = gr.Chatbot(label="Conversation")
                prompt = gr.Textbox(label="Your prompt", placeholder="Ask your agents something...")
                send = gr.Button("Send", variant="primary")

            with gr.Tab("LM Studio"):
                control_status = gr.Markdown("Use these controls to manage LM Studio and model loading.")
                with gr.Row():
                    start_btn = gr.Button("Start LM Studio Server")
                    stop_btn = gr.Button("Stop LM Studio Server")
                    list_cli_btn = gr.Button("List Local Models (CLI)")
                    list_loaded_btn = gr.Button("List Loaded Models (Server API)")
                with gr.Row():
                    model_to_load = gr.Dropdown(label="Model to load via CLI", choices=[])
                    load_btn = gr.Button("Load Selected Model")
                loaded_models_output = gr.Textbox(label="LM Studio command output", lines=10)

            with gr.Tab("Agent Builder"):
                agent_name = gr.Textbox(label="Agent name", placeholder="researcher")
                agent_model = gr.Dropdown(
                    label="Agent model",
                    choices=[config.planner_model, config.worker_model],
                    value=config.planner_model,
                    allow_custom_value=True,
                )
                agent_prompt = gr.Textbox(
                    label="System prompt",
                    lines=5,
                    placeholder="You are a research agent that summarizes references...",
                )
                create_agent_btn = gr.Button("Create Agent")
                create_agent_status = gr.Markdown()
                agents_list = gr.Markdown(
                    value="\n".join([f"- `{a}` -> `{orchestrator.definitions[a].model_name}`" for a in orchestrator.list_agents()])
                )

            with gr.Tab("Tool Builder"):
                tool_name = gr.Textbox(label="Tool name", placeholder="delegate_to_researcher")
                tool_type = gr.Radio(
                    label="Tool type",
                    choices=["instruction", "delegate_agent"],
                    value="instruction",
                )
                tool_instructions = gr.Textbox(
                    label="Tool instructions",
                    lines=4,
                    placeholder="Use this tool when user asks for math validation.",
                )
                delegate_target = gr.Dropdown(
                    label="Delegate target agent",
                    choices=orchestrator.list_agents(),
                    value="planner",
                )
                create_tool_btn = gr.Button("Create Tool")
                create_tool_status = gr.Markdown()
                tools_list = gr.Markdown(
                    value="\n".join([f"- `{name}`" for name in orchestrator.get_tool_names()])
                )

            with gr.Tab("Graph Workflow"):
                workflow_prompt = gr.Textbox(
                    label="Workflow prompt",
                    lines=4,
                    placeholder="Plan, validate, and summarize a solution for deploying this system.",
                )
                workflow_sequence = gr.Textbox(
                    label="Agent sequence (comma-separated)",
                    value="planner,worker",
                )
                run_workflow_btn = gr.Button("Run Graph Workflow")
                workflow_output = gr.Markdown()

        apply_models_btn.click(
            apply_agent_models,
            inputs=[planner_model, worker_model],
            outputs=[model_status, model_map],
        )
        send.click(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])
        prompt.submit(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])

        start_btn.click(start_server, outputs=loaded_models_output)
        stop_btn.click(stop_server, outputs=loaded_models_output)
        list_loaded_btn.click(list_loaded_models, outputs=loaded_models_output)
        list_cli_btn.click(
            list_local_models,
            outputs=[control_status, model_to_load, planner_model, worker_model, agent_model],
        )
        load_btn.click(load_selected_model, inputs=model_to_load, outputs=loaded_models_output)

        create_agent_btn.click(
            add_agent,
            inputs=[agent_name, agent_model, agent_prompt],
            outputs=[create_agent_status, agents_list, single_agent, delegate_target, agent_model],
        )
        create_tool_btn.click(
            add_tool,
            inputs=[tool_name, tool_type, tool_instructions, delegate_target],
            outputs=[create_tool_status, tools_list],
        )
        run_workflow_btn.click(
            run_workflow,
            inputs=[workflow_prompt, workflow_sequence],
            outputs=workflow_output,
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
