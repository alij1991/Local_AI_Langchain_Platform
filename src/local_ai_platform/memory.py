from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
from datetime import datetime, timezone
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
        # [IMPROVE-184] Wave 44 — delegate the tiktoken load to the
        # shared `_get_tiktoken_encoding()` helper in
        # `token_counting.py`. Pre-W44 this method had its own
        # `tiktoken.encoding_for_model(...)` + cl100k_base fallback
        # that duplicated the loader logic in
        # `_tiktoken_count`. The helper unifies both. Default model
        # for the encoding lookup is `gpt-4` (matching the pre-W44
        # behaviour: TokenCounter() with no model passes "gpt-4" to
        # encoding_for_model; the helper does the same model-aware
        # lookup with cl100k_base fallback on KeyError).
        from .token_counting import _get_tiktoken_encoding

        encoding = _get_tiktoken_encoding(self._model or "gpt-4")
        if encoding is not None:
            self._tokenizer = encoding
            return

        # Try transformers tokenizer (separate family of tokenizers,
        # not unified with the tiktoken path; stays inline). The
        # AutoTokenizer.from_pretrained() call is for HuggingFace
        # model IDs that don't have a tiktoken-registered encoding.
        if self._model:
            try:
                from transformers import AutoTokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(self._model)
                return
            except Exception:
                pass

        # Fallback: character-based estimation (4 chars ≈ 1 token).
        # This tier is also distinct from the executor's
        # `_estimate_tokens` 4-char heuristic — that one is a
        # hot-path optimization for DAG inter-node context that the
        # closure plan deliberately preserves untouched.
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
        *,
        conv_id: str | None = None,
        agent_name: str | None = None,
        compactor: Any = None,  # ContextCompactor — late-bound to avoid forward ref
    ) -> list[ChatMessage]:
        """Build a token-budgeted message list.

        Returns [system, ...history..., user] that fits within the token budget.

        [IMPROVE-15] When ``compactor`` + ``conv_id`` + ``agent_name`` are
        supplied AND the persisted summary / facts exist, builds the
        compacted form: system + summary block + facts block + anchor +
        user. Falls through to the existing budget-tier chain (full /
        truncate / extractive) when no summary has been generated yet
        (early in the conversation, or before the background job has
        run). The new branch is purely additive — callers that don't
        pass the compactor see identical pre-IMPROVE-15 behavior.
        """
        system_msg = ChatMessage(role="system", content=system_prompt)
        user_msg = ChatMessage(role="user", content=user_input)

        fixed_cost = self.counter.count(system_prompt) + self.counter.count(user_input) + 12
        remaining = self.budget - fixed_cost

        if remaining <= 0:
            return [system_msg, user_msg]

        # [IMPROVE-15] Compacted-context branch. Engages only when:
        #   * caller supplied a compactor + conv_id + agent_name, AND
        #   * the persisted summary and/or facts exist (the read returns
        #     useful content), AND
        #   * the history is long enough that compaction is meaningful
        #     (otherwise the "full history fits" tier is cheaper).
        if compactor is not None and conv_id and agent_name and len(history) > compactor.anchor_count + 5:
            try:
                summary_text, facts, anchor = compactor.get_compacted_context(
                    conv_id, agent_name, history,
                )
            except Exception as exc:
                # Defensive: a misbehaving compactor must not block the
                # chat path. Fall through to the legacy budget tiers.
                logger.debug(
                    "[IMPROVE-15] compactor.get_compacted_context raised "
                    "(%s) — falling back to legacy tiers", exc,
                )
                summary_text, facts, anchor = None, {}, history
            if summary_text or facts:
                return self._build_compacted_message_list(
                    system_msg=system_msg,
                    summary_text=summary_text,
                    facts=facts,
                    anchor=anchor,
                    user_msg=user_msg,
                )

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

    @staticmethod
    def _build_compacted_message_list(
        *,
        system_msg: ChatMessage,
        summary_text: str | None,
        facts: dict[str, str],
        anchor: list[ChatMessage],
        user_msg: ChatMessage,
    ) -> list[ChatMessage]:
        """[IMPROVE-15] Assemble the compacted message list:
            [system, summary?, facts?, *anchor, user]

        Both ``summary_text`` and ``facts`` are optional — when one is
        absent the corresponding system message is omitted entirely
        (rather than emitted as an empty placeholder, which would
        confuse the LLM).
        """
        result: list[ChatMessage] = [system_msg]
        if summary_text:
            result.append(ChatMessage(
                role="system",
                content=f"Summary of earlier conversation:\n{summary_text}",
            ))
        if facts:
            # Stable ordering for reproducibility in tests + caches.
            fact_lines = [f"- {k}: {v}" for k, v in sorted(facts.items())]
            result.append(ChatMessage(
                role="system",
                content="Known facts about this conversation:\n" + "\n".join(fact_lines),
            ))
        result.extend(anchor)
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


# ── [IMPROVE-15] Hybrid context compression ──────────────────────


_FACTS_NAMESPACE_PREFIX = "facts"


# Module-level in-flight set for background summarization tasks.
# A second concurrent ``summarize_in_background`` call for the same
# ``conv_id`` short-circuits — the first task will persist the result
# the second would have produced anyway. Lock guards mutation; reads
# under contention are best-effort (False-positive duplicate runs are
# harmless, just wasted work).
_SUMMARIZE_IN_FLIGHT: set[str] = set()
_SUMMARIZE_IN_FLIGHT_LOCK = threading.Lock()


# Prompts. Kept short so a 1B-class summarizer (gemma3:1b) handles
# them comfortably and the per-call latency stays low. Output shape
# is what we parse downstream — pin via tests.
_SUMMARIZER_PROMPT = """\
You are a conversation summarizer. Read the conversation below and \
produce a concise 2-3 sentence summary capturing key topics, decisions, \
and context useful for continuing the conversation. Focus on facts and \
intent. Skip greetings and pleasantries. Output ONLY the summary text.

Conversation:
{conversation_text}

Summary:"""


_FACT_EXTRACTOR_PROMPT = """\
Extract durable facts from the conversation below. Output ONLY a JSON \
object mapping snake_case keys to short string values. Include names, \
dates, projects, preferences, and decisions. Skip greetings, opinions, \
and ephemeral exchanges. Output {{}} when no durable facts are present.

Example output:
{{"user_name": "Ali", "project_deadline": "2026-04-30", "preferred_language": "Python"}}

Conversation:
{conversation_text}

JSON:"""


def _format_messages_for_prompt(messages: list[ChatMessage]) -> str:
    """Render a list of ChatMessages as plain text for the summarizer/
    extractor prompts. Truncates each message to a 500-char preview so
    a single huge user message can't blow the LLM context budget.
    """
    lines: list[str] = []
    for m in messages:
        role = "User" if m.role == "user" else (
            "Assistant" if m.role == "assistant" else m.role.capitalize()
        )
        text = m.content
        if len(text) > 500:
            text = text[:500] + "…"
        lines.append(f"{role}: {text}")
    return "\n".join(lines)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ContextCompactor:
    """[IMPROVE-15] Hybrid context compression: anchor + summary + facts.

    Pre-IMPROVE-15 ``SmartMemory.prepare_messages`` was effectively a
    head-trim — when budget was exceeded it dropped the oldest
    messages. Long conversations lost early context entirely; the
    user's name, project deadlines, decisions made 30 turns ago all
    vanished without trace.

    This class implements the standard 2025-2026 hybrid pattern:

      * **Anchor**: keep system prompt + last N messages verbatim.
      * **Summarize**: middle messages compressed into a 2-3 sentence
        running summary, persisted to ``conversation_summaries``.
      * **Key facts**: durable KV facts extracted via the LLM and
        stored in the existing ``memory_store`` table under namespace
        ``facts:{agent_name}:{conv_id}``. Re-injected on every turn
        regardless of summary state.

    Q21=B (background job): summarization runs out-of-band via
    ``asyncio.create_task`` from the chat router AFTER the response
    is returned. ``get_compacted_context`` reads the persisted
    summary, never blocks on LLM calls. When no summary exists yet
    (early in a conversation, or before the background job has run),
    callers fall back to ``SmartMemory``'s existing extractive
    truncation tier.

    Thread-safety: the in-flight dedup set is protected by a module-
    level ``threading.Lock``. Multiple concurrent chat turns trigger
    at most one summarization per conv_id at a time.

    Sources (2025-2026):
      * mem0.ai — LLM Chat History Summarization (Oct 2025):
        https://mem0.ai/blog/llm-chat-history-summarization-guide-2025
      * Zylos Research — AI Agent Context Compression (Feb 2026):
        https://zylos.ai/research/2026-02-28-ai-agent-context-compression-strategies
      * Redis — Context Window Overflow in 2026:
        https://redis.io/blog/context-window-overflow/
    """

    def __init__(
        self,
        *,
        anchor_count: int = 10,
        summary_threshold: int = 20,
        summary_max_tokens: int = 300,
        summarizer_model: str = "ollama:gemma3:1b",
        router: Any = None,  # ProviderRouter — late-bound to avoid circular import
    ) -> None:
        # The "anchor": last N messages kept verbatim. 10 is the
        # 2025-2026 enterprise default — small enough that even a
        # 4k-context model has room, large enough to preserve
        # immediate-turn coherence (clarifications, follow-ups).
        self.anchor_count = anchor_count
        # Re-summarize after this many NEW messages since the last
        # summary's ``summarized_through_message_id``. Plus the anchor
        # below the threshold gives a rolling-window effect.
        self.summary_threshold = summary_threshold
        self.summary_max_tokens = summary_max_tokens
        self.summarizer_model = summarizer_model
        self.router = router

    # ── Trigger decision ──────────────────────────────────────────

    def should_summarize(
        self, conv_id: str, current_message_count: int,
    ) -> bool:
        """Decide whether to fire a background summarization task.

        True iff:
          * ``current_message_count > anchor_count + 5`` (have enough
            messages older than the anchor to be worth summarizing),
            AND
          * Either no summary exists, OR the existing summary is
            stale by ``>= summary_threshold`` messages.

        Cheap — a single PK lookup on ``conversation_summaries``.
        Called from the trigger path so we don't queue work that
        would no-op anyway.
        """
        # Need enough older history to summarize beyond the anchor.
        if current_message_count <= self.anchor_count + 5:
            return False
        from .repositories.summaries import get_summary
        try:
            existing = get_summary(conv_id)
        except Exception as exc:
            # DB read failure is NOT a reason to skip summarization;
            # the background task will hit the same DB and might
            # succeed when transient connection issues resolve.
            logger.debug(
                "[IMPROVE-15] should_summarize: get_summary failed (%s) — "
                "treating as no existing summary",
                exc,
            )
            existing = None
        if existing is None:
            return True
        # Stale check: how many new messages since last summary?
        last_count = int(existing.get("summarized_message_count") or 0)
        new_messages_since = current_message_count - last_count
        return new_messages_since >= self.summary_threshold

    # ── Background summarization (Q21=B) ──────────────────────────

    async def summarize_in_background(
        self, conv_id: str, agent_name: str,
    ) -> bool:
        """Fire-and-forget background summarization task.

        Idempotent — the in-flight dedup set short-circuits a second
        concurrent invocation for the same ``conv_id``. Returns True
        on success, False on any failure (logs + swallows; chat path
        stays alive).

        Sequence:
          1. Acquire in-flight slot for ``conv_id``; bail if held.
          2. Verify the summarizer model is available (cheap call).
          3. Read the conversation messages.
          4. Slice older history (everything before the anchor).
          5. Call the LLM with the summarizer prompt.
          6. Persist via ``upsert_summary``.
          7. Best-effort fact extraction (failure here doesn't block
             summary persistence).
          8. Release the in-flight slot.
        """
        # 1. In-flight dedup. Quick acquire-or-bail; the second call
        #    returns immediately rather than waiting for the first.
        with _SUMMARIZE_IN_FLIGHT_LOCK:
            if conv_id in _SUMMARIZE_IN_FLIGHT:
                logger.debug(
                    "[IMPROVE-15] summarize_in_background: conv %s "
                    "already in flight — skipping duplicate trigger",
                    conv_id,
                )
                return False
            _SUMMARIZE_IN_FLIGHT.add(conv_id)
        try:
            return await self._summarize_impl(conv_id, agent_name)
        finally:
            with _SUMMARIZE_IN_FLIGHT_LOCK:
                _SUMMARIZE_IN_FLIGHT.discard(conv_id)

    async def _summarize_impl(
        self, conv_id: str, agent_name: str,
    ) -> bool:
        # 2. Verify the summarizer model exists. Skip silently when
        #    not pulled — the user can install gemma3:1b later and
        #    the next trigger will succeed. Better than crashing the
        #    background task pool with a missing-model error.
        if self.router is None:
            logger.debug(
                "[IMPROVE-15] summarize_in_background: no router bound — "
                "skipping",
            )
            return False
        try:
            info = self.router.get_model_info(self.summarizer_model)
        except Exception as exc:
            logger.debug(
                "[IMPROVE-15] summarize_in_background: get_model_info "
                "raised (%s) — skipping",
                exc,
            )
            return False
        if info is None:
            logger.info(
                "[IMPROVE-15] Summarizer model %s not available — "
                "skipping. Pull it via Ollama or set "
                "ContextCompactor.summarizer_model to an available "
                "model to enable.",
                self.summarizer_model,
            )
            return False

        # 3-4. Read messages, slice older-than-anchor.
        from .repositories.conversations import list_messages
        try:
            db_messages = list_messages(conv_id, limit=10_000)
        except Exception as exc:
            logger.warning(
                "[IMPROVE-15] summarize_in_background: list_messages "
                "failed (%s) — aborting", exc,
            )
            return False
        if len(db_messages) <= self.anchor_count + 5:
            # Race: between trigger-decision and now, the threshold
            # check no longer holds. Bail cleanly.
            return False
        # Older messages = everything except the last anchor_count.
        # The "5" buffer matches should_summarize's gate.
        older = db_messages[: -self.anchor_count] if self.anchor_count else db_messages
        if not older:
            return False
        last_message_id = str(older[-1].get("id"))
        older_count = len(db_messages)  # total (used for staleness check)

        chat_messages = [
            ChatMessage(
                role=str(m.get("role") or "user"),
                content=str(m.get("content") or ""),
            )
            for m in older
        ]
        conversation_text = _format_messages_for_prompt(chat_messages)

        # 5. Call the LLM via the router (sync-via-thread to keep this
        #    method async-clean from any caller).
        from .providers.base import GenerationSettings
        summary_prompt_messages = [
            ChatMessage(
                role="system",
                content=_SUMMARIZER_PROMPT.format(
                    conversation_text=conversation_text,
                ),
            ),
            ChatMessage(role="user", content="Generate the summary now."),
        ]
        try:
            response = await asyncio.to_thread(
                self.router.chat,
                self.summarizer_model,
                summary_prompt_messages,
                GenerationSettings(max_tokens=self.summary_max_tokens),
            )
            summary_text = (response.content or "").strip()
        except Exception as exc:
            logger.warning(
                "[IMPROVE-15] summarize_in_background: LLM call failed "
                "(%s) — aborting summary for conv %s",
                exc, conv_id,
            )
            return False

        if not summary_text:
            logger.info(
                "[IMPROVE-15] summarize_in_background: empty summary for "
                "conv %s — skipping persistence",
                conv_id,
            )
            return False

        # 6. Persist summary.
        from .repositories.summaries import upsert_summary
        try:
            upsert_summary(
                conversation_id=conv_id,
                summary_text=summary_text,
                summarized_through_message_id=last_message_id,
                summarized_message_count=older_count,
                summarizer_model=self.summarizer_model,
            )
        except Exception as exc:
            logger.warning(
                "[IMPROVE-15] summarize_in_background: upsert_summary "
                "failed (%s) — aborting", exc,
            )
            return False

        # 7. Best-effort fact extraction. Failure here is logged but
        #    doesn't undo the summary persistence — facts and summary
        #    are independent value adds.
        try:
            await self._extract_and_store_facts(
                conv_id, agent_name, chat_messages,
            )
        except Exception as exc:
            logger.info(
                "[IMPROVE-15] summarize_in_background: fact extraction "
                "failed (%s) — summary persisted regardless",
                exc,
            )

        logger.info(
            "[IMPROVE-15] Summarized conv %s through msg %s (%d msgs total)",
            conv_id, last_message_id, older_count,
        )
        return True

    async def _extract_and_store_facts(
        self, conv_id: str, agent_name: str, chat_messages: list[ChatMessage],
    ) -> None:
        """Ask the LLM to extract durable facts as JSON; persist any
        valid keys into the ``memory_store`` table under namespace
        ``facts:{agent_name}:{conv_id}``.

        Bad JSON is silently skipped (logged at debug). The summary
        still persists. Per-fact errors don't abort the loop.
        """
        from .providers.base import GenerationSettings
        if self.router is None:
            return
        conversation_text = _format_messages_for_prompt(chat_messages)
        fact_prompt_messages = [
            ChatMessage(
                role="system",
                content=_FACT_EXTRACTOR_PROMPT.format(
                    conversation_text=conversation_text,
                ),
            ),
            ChatMessage(role="user", content="Extract facts as JSON now."),
        ]
        response = await asyncio.to_thread(
            self.router.chat,
            self.summarizer_model,
            fact_prompt_messages,
            GenerationSettings(max_tokens=200),
        )
        raw = (response.content or "").strip()
        # The LLM sometimes wraps JSON in ```json ... ``` fences or
        # adds a leading sentence. Strip both before parsing.
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        # Find the first { and the last } — defensive against trailing
        # commentary the model may emit despite the "ONLY JSON"
        # instruction.
        first = raw.find("{")
        last = raw.rfind("}")
        if first < 0 or last <= first:
            logger.debug(
                "[IMPROVE-15] fact extraction: no JSON in LLM output "
                "for conv %s — skipping", conv_id,
            )
            return
        try:
            facts = json.loads(raw[first : last + 1])
        except json.JSONDecodeError as exc:
            logger.debug(
                "[IMPROVE-15] fact extraction: JSON parse failed (%s) — "
                "skipping for conv %s",
                exc, conv_id,
            )
            return
        if not isinstance(facts, dict):
            return

        namespace = f"{_FACTS_NAMESPACE_PREFIX}:{agent_name}:{conv_id}"
        from .db import get_conn
        now = _now()
        conn = get_conn()
        try:
            for key, value in facts.items():
                if not isinstance(key, str) or not key:
                    continue
                # Reject empties / placeholders / overly long values.
                value_str = str(value).strip()
                if not value_str or len(value_str) > 500:
                    continue
                try:
                    conn.execute(
                        "INSERT INTO memory_store (namespace, key, value_json, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?) "
                        "ON CONFLICT(namespace, key) DO UPDATE SET "
                        "value_json=excluded.value_json, updated_at=excluded.updated_at",
                        (
                            namespace, key,
                            json.dumps({"value": value_str}),
                            now, now,
                        ),
                    )
                except Exception as exc:
                    logger.debug(
                        "[IMPROVE-15] fact extraction: per-fact write "
                        "failed (%s) — continuing", exc,
                    )
            conn.commit()
        finally:
            conn.close()

    # ── Read path (synchronous) ──────────────────────────────────

    def get_compacted_context(
        self, conv_id: str, agent_name: str, history: list[ChatMessage],
    ) -> tuple[str | None, dict[str, str], list[ChatMessage]]:
        """Read persisted summary + facts; return
        ``(summary_text, facts_dict, anchor_history)``.

        Synchronous — no LLM calls. Called once per chat turn from
        ``SmartMemory.prepare_messages``. When no summary exists yet
        the first slot is None and the anchor is the FULL history;
        ``SmartMemory`` falls through to its extractive truncation
        tier.
        """
        from .repositories.summaries import get_summary

        try:
            summary_row = get_summary(conv_id)
        except Exception as exc:
            logger.debug(
                "[IMPROVE-15] get_compacted_context: get_summary failed "
                "(%s) — treating as no summary", exc,
            )
            summary_row = None
        summary_text: str | None = None
        if summary_row is not None:
            summary_text = str(summary_row.get("summary_text") or "") or None

        # Anchor = last N messages verbatim. When summary is absent,
        # we still split (caller decides what to do with each piece).
        anchor = (
            list(history[-self.anchor_count:])
            if self.anchor_count > 0 else []
        )

        facts = self._read_facts(agent_name, conv_id)
        return summary_text, facts, anchor

    def _read_facts(
        self, agent_name: str, conv_id: str,
    ) -> dict[str, str]:
        """Read all facts for this (agent, conv). Returns dict mapping
        ``key`` → ``value`` (string). Empty when no facts persisted.

        Tolerant of malformed value_json — silently drops bad rows.
        """
        namespace = f"{_FACTS_NAMESPACE_PREFIX}:{agent_name}:{conv_id}"
        from .db import get_conn
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT key, value_json FROM memory_store WHERE namespace = ? "
                "ORDER BY updated_at DESC",
                (namespace,),
            ).fetchall()
        except Exception as exc:
            logger.debug(
                "[IMPROVE-15] _read_facts: query failed (%s) — "
                "returning empty", exc,
            )
            return {}
        finally:
            conn.close()

        out: dict[str, str] = {}
        for r in rows:
            key = r["key"]
            try:
                payload = json.loads(r["value_json"])
                value = payload.get("value") if isinstance(payload, dict) else payload
            except Exception:
                continue
            if isinstance(value, str) and value:
                out[key] = value
        return out


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
