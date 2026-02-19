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
            response = (
                "### Planner\n"
                f"{output['planner']}\n\n"
                "### Worker\n"
                f"{output['worker']}"
            )

        history = history + [(message, response)]
        return "", history

    return chat


def build_app() -> gr.Blocks:
    config = load_config()
    orchestrator = AgentOrchestrator(config)
    controller = LMStudioController(config)
    chat_fn = build_chat_handler(orchestrator)

    def apply_agent_models(planner_model_name: str, worker_model_name: str) -> str:
        orchestrator.set_agent_model("planner", planner_model_name)
        orchestrator.set_agent_model("worker", worker_model_name)
        return (
            f"Planner model set to `{planner_model_name}`. "
            f"Worker model set to `{worker_model_name}`."
        )

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

    def list_local_models() -> tuple[str, dict, dict, dict]:
        cli_result = controller.list_local_models()
        if not cli_result.ok:
            return f"❌ {cli_result.output}", gr.update(), gr.update(), gr.update()

        models = controller.parse_model_lines(cli_result.output)
        if not models:
            return "No models returned by CLI.", gr.update(), gr.update(), gr.update()

        current = orchestrator.get_agent_models()
        planner_selected = current["planner"] if current["planner"] in models else models[0]
        worker_selected = current["worker"] if current["worker"] in models else models[0]

        return (
            "✅ Local model list loaded from LM Studio CLI.",
            gr.update(choices=models, value=models[0]),
            gr.update(choices=models, value=planner_selected),
            gr.update(choices=models, value=worker_selected),
        )

    with gr.Blocks(title="Local AI LangChain Platform") as demo:
        gr.Markdown("# 🤖 Local AI LangChain Platform")
        gr.Markdown(
            "Run multi-agent conversations with LM Studio + LangChain in a self-hosted Gradio UI."
        )

        with gr.Row():
            mode = gr.Radio(
                label="Agent mode",
                choices=["Single Agent", "Combined (Planner + Worker)"],
                value="Single Agent",
            )
            single_agent = gr.Dropdown(
                label="Single agent",
                choices=["planner", "worker"],
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

        with gr.Accordion("LM Studio Control Panel", open=False):
            control_status = gr.Markdown("Use these controls to manage LM Studio and models.")
            with gr.Row():
                start_btn = gr.Button("Start LM Studio Server")
                stop_btn = gr.Button("Stop LM Studio Server")
                list_cli_btn = gr.Button("List Local Models (CLI)")
                list_loaded_btn = gr.Button("List Loaded Models (Server API)")
            with gr.Row():
                model_to_load = gr.Dropdown(label="Model to load via CLI", choices=[])
                load_btn = gr.Button("Load Selected Model")
            loaded_models_output = gr.Textbox(label="LM Studio command output", lines=8)

        chatbot = gr.Chatbot(label="Conversation")
        prompt = gr.Textbox(label="Your prompt", placeholder="Ask your agents something...")
        send = gr.Button("Send", variant="primary")

        apply_models_btn.click(
            apply_agent_models,
            inputs=[planner_model, worker_model],
            outputs=model_status,
        )
        start_btn.click(start_server, outputs=loaded_models_output)
        stop_btn.click(stop_server, outputs=loaded_models_output)
        list_loaded_btn.click(list_loaded_models, outputs=loaded_models_output)
        list_cli_btn.click(
            list_local_models,
            outputs=[
                control_status,
                model_to_load,
                planner_model,
                worker_model,
            ],
        )
        load_btn.click(load_selected_model, inputs=model_to_load, outputs=loaded_models_output)

        send.click(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])
        prompt.submit(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
