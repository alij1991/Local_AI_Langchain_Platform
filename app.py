from __future__ import annotations

import streamlit as st

from local_ai_platform import load_config
from local_ai_platform.agents import AgentOrchestrator


st.set_page_config(page_title="Local AI LangChain Platform", page_icon="🤖", layout="wide")

st.title("🤖 Local AI LangChain Platform")
st.caption("Run multi-agent conversations with LM Studio + LangChain.")

if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = AgentOrchestrator(load_config())
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

orchestrator: AgentOrchestrator = st.session_state.orchestrator

with st.sidebar:
    st.header("Agent mode")
    mode = st.radio(
        "Choose how to run",
        options=["Single Agent", "Combined (Planner + Worker)"],
        index=0,
    )
    single_agent = st.selectbox("Single agent", options=["planner", "worker"])

prompt = st.chat_input("Ask your agents something...")

for entry in st.session_state.chat_log:
    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])

if prompt:
    st.session_state.chat_log.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Running agents..."):
            if mode == "Single Agent":
                response = orchestrator.chat_with_agent(single_agent, prompt)
            else:
                output = orchestrator.combined_response(prompt)
                response = (
                    "### Planner\n"
                    f"{output['planner']}\n\n"
                    "### Worker\n"
                    f"{output['worker']}"
                )
            st.markdown(response)
            st.session_state.chat_log.append({"role": "assistant", "content": response})
