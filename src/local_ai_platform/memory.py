from __future__ import annotations

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


def db_messages_to_langchain(rows: list[dict]) -> list[BaseMessage]:
    out: list[BaseMessage] = []
    for row in rows:
        role = row.get("role")
        content = row.get("content", "")
        if role == "assistant":
            out.append(AIMessage(content=content))
        elif role == "system":
            out.append(SystemMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    return out
