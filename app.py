from __future__ import annotations

from collections.abc import Callable

import gradio as gr

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator


def build_chat_handler(orchestrator: AgentOrchestrator) -> Callable:
    def chat(message: str, history: list[dict], mode: str, single_agent: str):
        if not message.strip():
            return "", history

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

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": response},
        ]
        return "", history

    return chat


def build_app() -> gr.Blocks:
    orchestrator = AgentOrchestrator(load_config())
    chat_fn = build_chat_handler(orchestrator)

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

        chatbot = gr.Chatbot(type="messages", label="Conversation")
        prompt = gr.Textbox(label="Your prompt", placeholder="Ask your agents something...")
        send = gr.Button("Send", variant="primary")

        send.click(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])
        prompt.submit(chat_fn, inputs=[prompt, chatbot, mode, single_agent], outputs=[prompt, chatbot])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
