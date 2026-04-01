from __future__ import annotations

from local_ai_platform.db import get_conn


def set_agent_tools(agent_name: str, tool_ids: list[str]) -> None:
    """Replace all tool bindings for an agent."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM agent_tools WHERE agent_name = ?", (agent_name,))
        for i, tid in enumerate(tool_ids):
            conn.execute(
                "INSERT OR IGNORE INTO agent_tools (agent_name, tool_id, sort_order) VALUES (?, ?, ?)",
                (agent_name, tid, i),
            )
        conn.commit()
    finally:
        conn.close()


def get_agent_tool_ids(agent_name: str) -> list[str]:
    """Return tool IDs bound to an agent, ordered by sort_order."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT tool_id FROM agent_tools WHERE agent_name = ? ORDER BY sort_order",
            (agent_name,),
        ).fetchall()
        return [r["tool_id"] for r in rows]
    finally:
        conn.close()


def clear_agent_tools(agent_name: str) -> None:
    """Remove all tool bindings for an agent."""
    conn = get_conn()
    try:
        conn.execute("DELETE FROM agent_tools WHERE agent_name = ?", (agent_name,))
        conn.commit()
    finally:
        conn.close()


def list_all_agent_tools() -> dict[str, list[str]]:
    """Return all agent-tool bindings as {agent_name: [tool_ids]}."""
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT agent_name, tool_id FROM agent_tools ORDER BY agent_name, sort_order"
        ).fetchall()
        result: dict[str, list[str]] = {}
        for r in rows:
            result.setdefault(r["agent_name"], []).append(r["tool_id"])
        return result
    finally:
        conn.close()
