from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from .providers.base import ChatMessage

logger = logging.getLogger(__name__)


# ── LangChain <-> ChatMessage converters ───────────────────────────

def db_messages_to_langchain(rows: list[dict]) -> list[BaseMessage]:
    """Convert DB rows to LangChain message objects."""
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


def langchain_to_chat_messages(messages: list[BaseMessage]) -> list[ChatMessage]:
    """Convert LangChain messages to unified ChatMessage format."""
    out = []
    for msg in messages:
        if isinstance(msg, SystemMessage):
            out.append(ChatMessage(role="system", content=str(msg.content)))
        elif isinstance(msg, AIMessage):
            out.append(ChatMessage(role="assistant", content=str(msg.content)))
        elif isinstance(msg, HumanMessage):
            out.append(ChatMessage(role="user", content=str(msg.content)))
        else:
            out.append(ChatMessage(role="user", content=str(msg.content)))
    return out


def chat_messages_to_langchain(messages: list[ChatMessage]) -> list[BaseMessage]:
    """Convert unified ChatMessage format to LangChain messages."""
    out: list[BaseMessage] = []
    for msg in messages:
        if msg.role == "system":
            out.append(SystemMessage(content=msg.content))
        elif msg.role == "assistant":
            out.append(AIMessage(content=msg.content))
        else:
            out.append(HumanMessage(content=msg.content))
    return out


# ── Token counting ────────────────────────────────────────────────

class TokenCounter:
    """Approximate token counter with optional tiktoken or transformers tokenizer."""

    def __init__(self, model: str | None = None) -> None:
        self._tokenizer: Any = None
        self._model = model
        self._init_tokenizer()

    def _init_tokenizer(self) -> None:
        # Try tiktoken first (fast, works offline for common models)
        try:
            import tiktoken
            try:
                self._tokenizer = tiktoken.encoding_for_model(self._model or "gpt-4")
            except KeyError:
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            return
        except ImportError:
            pass

        # Try transformers tokenizer
        if self._model:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._model)
                return
            except Exception:
                pass

        # Fallback: character-based estimation (4 chars ≈ 1 token)
        self._tokenizer = None

    def count(self, text: str) -> int:
        if self._tokenizer is None:
            return len(text) // 4

        if hasattr(self._tokenizer, "encode"):
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                return len(text) // 4

        return len(text) // 4

    def count_messages(self, messages: list[ChatMessage]) -> int:
        total = 0
        for msg in messages:
            total += self.count(msg.content) + 4  # role + formatting overhead
        return total


# ── Smart memory manager ──────────────────────────────────────────

class SmartMemory:
    """Token-aware memory manager that handles history truncation and summarization.

    Strategies (in order of preference):
    1. If history fits in budget → use as-is
    2. If slightly over → truncate oldest messages
    3. If way over → summarize older messages into a context block
    """

    def __init__(
        self,
        max_context_tokens: int = 4096,
        reserved_for_response: int = 1024,
        summarize_threshold: float = 0.7,
        model: str | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.reserved_for_response = reserved_for_response
        self.summarize_threshold = summarize_threshold
        self.counter = TokenCounter(model)
        self._summary_cache: dict[str, str] = {}

    @property
    def budget(self) -> int:
        return self.max_context_tokens - self.reserved_for_response

    def prepare_messages(
        self,
        system_prompt: str,
        history: list[ChatMessage],
        user_input: str,
    ) -> list[ChatMessage]:
        """Build a token-budgeted message list.

        Returns [system, ...history..., user] that fits within the token budget.
        """
        system_msg = ChatMessage(role="system", content=system_prompt)
        user_msg = ChatMessage(role="user", content=user_input)

        fixed_cost = self.counter.count(system_prompt) + self.counter.count(user_input) + 12
        remaining = self.budget - fixed_cost

        if remaining <= 0:
            return [system_msg, user_msg]

        # Check if full history fits
        history_tokens = self.counter.count_messages(history)
        if history_tokens <= remaining:
            return [system_msg, *history, user_msg]

        # Try truncation: keep recent messages, drop oldest
        truncated = self._truncate_history(history, remaining)
        truncated_tokens = self.counter.count_messages(truncated)

        if truncated_tokens <= remaining:
            return [system_msg, *truncated, user_msg]

        # Summarize old history, keep recent
        summary, recent = self._summarize_and_keep_recent(history, remaining)
        result = [system_msg]
        if summary:
            result.append(ChatMessage(
                role="system",
                content=f"Summary of earlier conversation:\n{summary}",
            ))
        result.extend(recent)
        result.append(user_msg)
        return result

    def _truncate_history(self, history: list[ChatMessage], budget: int) -> list[ChatMessage]:
        """Keep the most recent messages that fit within the token budget."""
        kept: list[ChatMessage] = []
        running = 0

        for msg in reversed(history):
            cost = self.counter.count(msg.content) + 4
            if running + cost > budget:
                break
            kept.insert(0, msg)
            running += cost

        return kept

    def _summarize_and_keep_recent(
        self,
        history: list[ChatMessage],
        budget: int,
    ) -> tuple[str, list[ChatMessage]]:
        """Split history into summarized old + kept recent messages."""
        # Keep the most recent 30% of budget for verbatim messages
        recent_budget = int(budget * 0.3)
        summary_budget = budget - recent_budget

        # Find split point
        recent: list[ChatMessage] = []
        recent_tokens = 0
        split_idx = len(history)

        for i in range(len(history) - 1, -1, -1):
            cost = self.counter.count(history[i].content) + 4
            if recent_tokens + cost > recent_budget:
                split_idx = i + 1
                break
            recent.insert(0, history[i])
            recent_tokens += cost

        # Summarize older messages
        old = history[:split_idx]
        if not old:
            return "", recent

        cache_key = self._history_hash(old)
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key], recent

        summary = self._create_summary(old, max_tokens=min(summary_budget, 300))
        self._summary_cache[cache_key] = summary
        return summary, recent

    @staticmethod
    def _create_summary(messages: list[ChatMessage], max_tokens: int = 300) -> str:
        """Create a concise summary of messages.

        This is a simple extractive summary. For better results,
        use the LLM itself to summarize (see summarize_with_llm below).
        """
        lines = []
        for msg in messages:
            prefix = "User" if msg.role == "user" else "Assistant"
            content = msg.content[:200] + "..." if len(msg.content) > 200 else msg.content
            lines.append(f"- {prefix}: {content}")

        full = "\n".join(lines)
        # Rough token limit: 4 chars per token
        char_limit = max_tokens * 4
        if len(full) > char_limit:
            full = full[:char_limit] + "\n[...truncated]"
        return full

    @staticmethod
    def _history_hash(messages: list[ChatMessage]) -> str:
        raw = "|".join(f"{m.role}:{m.content[:100]}" for m in messages)
        return hashlib.md5(raw.encode()).hexdigest()

    async def summarize_with_llm(
        self,
        messages: list[ChatMessage],
        provider_chat_fn: Any,
        model: str,
    ) -> str:
        """Use the LLM itself to summarize conversation history."""
        conversation_text = "\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in messages
        )
        summary_prompt = [
            ChatMessage(
                role="system",
                content="Summarize the following conversation concisely. Focus on key topics, decisions, and context that would be useful for continuing the conversation. Keep it under 200 words.",
            ),
            ChatMessage(role="user", content=conversation_text),
        ]
        from .providers.base import GenerationSettings
        response = await provider_chat_fn(model, summary_prompt, GenerationSettings(max_tokens=300))
        return response.content


# ── Vector store memory (optional, for retrieval-augmented memory) ──

class VectorMemory:
    """ChromaDB-backed semantic memory for long-term recall.

    Stores conversation chunks and retrieves relevant context
    for the current query.
    """

    def __init__(self, collection_name: str = "conversations", persist_dir: str = "./data/vectorstore") -> None:
        self.collection_name = collection_name
        self.persist_dir = persist_dir
        self._collection: Any = None

    def _get_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.persist_dir)
            self._collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._collection
        except ImportError:
            logger.warning("chromadb not installed — vector memory disabled")
            return None

    def store(
        self,
        conversation_id: str,
        messages: list[ChatMessage],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store conversation messages as searchable chunks."""
        collection = self._get_collection()
        if collection is None:
            return

        docs = []
        ids = []
        metadatas = []

        for i, msg in enumerate(messages):
            if not msg.content.strip():
                continue
            doc_id = f"{conversation_id}_{i}"
            docs.append(msg.content)
            ids.append(doc_id)
            metadatas.append({
                "conversation_id": conversation_id,
                "role": msg.role,
                "index": i,
                **(metadata or {}),
            })

        if docs:
            collection.upsert(documents=docs, ids=ids, metadatas=metadatas)

    def search(
        self,
        query: str,
        n_results: int = 5,
        conversation_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant conversation chunks."""
        collection = self._get_collection()
        if collection is None:
            return []

        where = {"conversation_id": conversation_id} if conversation_id else None
        try:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
            )
            # Flatten results
            out = []
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(documents, metadatas, distances):
                out.append({
                    "content": doc,
                    "metadata": meta,
                    "relevance": 1.0 - dist,  # cosine similarity
                })
            return out
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)
            return []

    def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """Get relevant past context as a formatted string for injection."""
        results = self.search(query, n_results=n_results)
        if not results:
            return ""

        parts = []
        for r in results:
            role = r["metadata"].get("role", "unknown")
            parts.append(f"[{role}]: {r['content'][:300]}")
        return "Relevant past context:\n" + "\n".join(parts)
