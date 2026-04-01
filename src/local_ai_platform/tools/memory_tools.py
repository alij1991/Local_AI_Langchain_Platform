"""Long-term memory tools: save and recall information across conversations."""
from __future__ import annotations

import json
from datetime import datetime, timezone

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class SaveMemoryInput(BaseModel):
    key: str = Field(..., description="Short descriptive key for the memory, e.g. 'user_preference_theme' or 'project_deadline'")
    value: str = Field(..., description="The information to remember")
    namespace: str = Field("default", description="Category namespace, e.g. 'preferences', 'facts', 'notes'")


class RecallMemoryInput(BaseModel):
    query: str = Field(..., description="What to search for in memory")
    namespace: str = Field("default", description="Category namespace to search in, or 'default' for all")
    max_results: int = Field(5, description="Maximum memories to return")


class ListMemoriesInput(BaseModel):
    namespace: str = Field("default", description="Namespace to list, or 'all' for everything")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_memory(key: str, value: str, namespace: str = "default") -> str:
    """Save a piece of information to long-term memory."""
    from local_ai_platform.db import get_conn

    conn = get_conn()
    try:
        now = _now()
        conn.execute(
            "INSERT INTO memory_store (namespace, key, value_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(namespace, key) DO UPDATE SET value_json=excluded.value_json, updated_at=excluded.updated_at",
            (namespace, key, json.dumps({"text": value}), now, now),
        )
        conn.commit()
        return f"Saved to memory: [{namespace}] {key}"
    except Exception as e:
        return f"Failed to save memory: {e}"
    finally:
        conn.close()


def recall_memory(query: str, namespace: str = "default", max_results: int = 5) -> str:
    """Search long-term memory for relevant information."""
    from local_ai_platform.db import get_conn

    conn = get_conn()
    try:
        query_lower = query.lower()
        if namespace == "all" or namespace == "default":
            rows = conn.execute(
                "SELECT namespace, key, value_json, updated_at FROM memory_store ORDER BY updated_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT namespace, key, value_json, updated_at FROM memory_store WHERE namespace = ? ORDER BY updated_at DESC",
                (namespace,),
            ).fetchall()

        # Simple keyword matching (semantic search would require embeddings)
        scored = []
        for r in rows:
            val = json.loads(r["value_json"]).get("text", "")
            key = r["key"]
            # Score by keyword overlap
            score = sum(1 for word in query_lower.split() if word in key.lower() or word in val.lower())
            if score > 0 or not query.strip():
                scored.append((score, r, val))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:max_results]

        if not top:
            return f"No memories found matching '{query}'"

        lines = []
        for score, r, val in top:
            lines.append(f"[{r['namespace']}] {r['key']}: {val}")
        return "Found memories:\n" + "\n".join(lines)
    except Exception as e:
        return f"Memory recall failed: {e}"
    finally:
        conn.close()


def list_memories(namespace: str = "default") -> str:
    """List all saved memories in a namespace."""
    from local_ai_platform.db import get_conn

    conn = get_conn()
    try:
        if namespace == "all":
            rows = conn.execute(
                "SELECT namespace, key, value_json, updated_at FROM memory_store ORDER BY namespace, updated_at DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT namespace, key, value_json, updated_at FROM memory_store WHERE namespace = ? ORDER BY updated_at DESC",
                (namespace,),
            ).fetchall()

        if not rows:
            return f"No memories in namespace '{namespace}'"

        lines = []
        for r in rows:
            val = json.loads(r["value_json"]).get("text", "")
            preview = val[:100] + "..." if len(val) > 100 else val
            lines.append(f"[{r['namespace']}] {r['key']}: {preview}")
        return f"{len(rows)} memories:\n" + "\n".join(lines)
    except Exception as e:
        return f"Failed to list memories: {e}"
    finally:
        conn.close()


def get_memory_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=save_memory,
            name="save_memory",
            description="Save information to long-term memory for recall in future conversations. Use for facts, preferences, notes, and important details.",
            args_schema=SaveMemoryInput,
        ),
        StructuredTool.from_function(
            func=recall_memory,
            name="recall_memory",
            description="Search long-term memory for previously saved information. Use when the user references something from a past conversation.",
            args_schema=RecallMemoryInput,
        ),
        StructuredTool.from_function(
            func=list_memories,
            name="list_memories",
            description="List all saved memories, optionally filtered by namespace.",
            args_schema=ListMemoriesInput,
        ),
    ]
