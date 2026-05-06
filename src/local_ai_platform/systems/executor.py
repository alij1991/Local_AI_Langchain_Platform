"""[IMPROVE-NEW-4] System DAG executor — extracted from agents.py.

Pre-this-commit, ``AgentOrchestrator.execute_system_graph`` (sync) and
``AgentOrchestrator.astream_system_graph`` (streaming) were 378 and
341 LoC respectively, sharing edge-routing logic, telemetry shape,
DAG-build code, and the classify_llm_router_edges helper. The
duplication had real cost — the [IMPROVE-35] commit (bd8b4d7) fixed
a 3-tuple/4-tuple edge_map drift between the two paths that had
gone unnoticed because every streaming test stubbed the executor
itself. Single source of truth here.

Public functions take an ``AgentOrchestrator`` as the first
positional parameter (``orch``) — this gives the executor access
to:

  * ``orch.definitions``   — dict[name, AgentDefinition]
  * ``orch.config``        — for prompt_builder_model
  * ``orch.router``        — for classify_llm_router_edges
  * ``orch.chat_with_agent(agent, prompt)``      — sync exec
  * ``orch.astream_chat_with_agent(agent, prompt)`` — streaming
                                                     async generator

Behavioural fidelity: byte-for-byte preserved relative to the
pre-extraction agents.py. Existing test suites
(test_systems_*.py, test_systems_chat_stream.py,
test_systems_parallel_waves.py, test_systems_llm_router_edges.py)
verify.
"""
from __future__ import annotations

import asyncio
import logging
import math
import re
import time as _time
import uuid
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable

from ..observability_events import emit_typed

if TYPE_CHECKING:
    from ..agents import AgentOrchestrator

logger = logging.getLogger(__name__)


# ── [IMPROVE-33 / IMPROVE-84] Bounded inter-node context ─────────
#
# Pre-IMPROVE-33 ``execute_system_graph`` (and its streaming twin)
# accumulated every prior node's output into a single string that
# was prepended to the next node's prompt. With 5 nodes producing
# 2k tokens each, node 5 saw ~10k tokens of context BEFORE the
# user input — enough to bust the context window of any small
# local model.
#
# Doc rationale at docs/features/05-systems.md:403-415: replace the
# unbounded string-concat with a token-budgeted, structured
# context builder. Newest outputs win; older ones get elided when
# the budget runs out.
#
# Originally lived in agents.py; Wave 7 [IMPROVE-75] extracted the
# executor to this module and left a lazy ``from ..agents import``
# inside the executor's function bodies as a temporary shim.
# IMPROVE-84 (this wave) finishes the migration by moving the
# four primitives here — the only callers are in this file. The
# lazy imports are gone.
#
# This is a simpler primitive than IMPROVE-15's full
# ``ContextCompactor`` (which does LLM-based summarization +
# key-fact extraction). For DAG runs the typical depth is ~5-10
# nodes, so a recency-based truncation already buys enough
# headroom. LLM-summarized inter-node context is a follow-up.

# Default budget targets a comfortable upper bound for most local
# 3-7B models (gemma3:4b ~4k context after system prompt and tools;
# qwen2.5:7b 32k but we don't want to spend it all on backref). Per
# system override via ``definition.context_budget_tokens``.
_INTER_NODE_CONTEXT_BUDGET_TOKENS = 4000

# Tokens-per-character heuristic for English. Avoids pulling
# tiktoken into the hot path of every node call — the actual model
# tokenizer would give a more precise count, but this is a budget
# guard, not a billing meter, so a 4-char rule of thumb is fine.
# Pinned by ``test_estimate_tokens_uses_4chars_per_token``.
_INTER_NODE_CHARS_PER_TOKEN = 4

# [IMPROVE-166] Wave 32 — per-edge "pass" config (Tranche D piece
# 2). Edge.rule.pass values control which prior outputs the
# downstream agent sees in its context block. Default "all"
# preserves pre-Wave-32 behaviour; "source_only" + "none" are the
# scoped variants. An invalid pass value silently falls back to
# "all" — same forward-compat semantics as the existing
# ``_evaluate_edge_rule`` "unknown rule_type = always follow"
# branch (rule schema additions in newer builds shouldn't crash
# older builds).
#
# Multi-incoming-edge policy: when more than one edge fires into
# the same target Y, the LAST-fired edge's pass + source wins
# (overwrite-on-fire). Deliberate simplification — alternative
# policies (intersection / most-restrictive) are queued for a
# future wave if the multi-incoming case becomes load-bearing.
_VALID_PASS_MODES = ("all", "source_only", "none")
_DEFAULT_PASS_MODE = "all"


def _estimate_tokens(text: str) -> int:
    """Rough token-count estimate without a real tokenizer.

    Returns ``max(1, len(text) // 4)`` so any non-empty string
    contributes at least one token to the budget — prevents an
    empty-string entry from "free-riding" the budget.
    """
    if not text:
        return 0
    return max(1, len(text) // _INTER_NODE_CHARS_PER_TOKEN)


def _build_inter_node_context(
    node_outputs: list[dict[str, Any]],
    budget_tokens: int = _INTER_NODE_CONTEXT_BUDGET_TOKENS,
    *,
    summarizer: Callable[[list[dict[str, Any]]], "str | None"] | None = None,
    pass_mode: str = _DEFAULT_PASS_MODE,
    source_node_id: str | None = None,
) -> str:
    """[IMPROVE-33] Build a token-budgeted prior-context block.

    Walks ``node_outputs`` newest-first, packing each ``status:
    "ok"`` entry's text into the result until the token budget is
    exhausted. Older entries that don't fit are summarized as a
    single ``[... N earlier output(s) elided ...]`` marker so the
    downstream agent knows context was truncated.

    Skipped/error entries are dropped — propagating "(agent X not
    found)" or an exception traceback into a downstream prompt
    only confuses the next agent.

    Returns an empty string when there are no usable entries —
    callers can then skip the "Context from prior agents:" prefix
    entirely (matches the legacy "if accumulated_context" branch).

    [IMPROVE-165] Wave 31 — when a ``summarizer`` callable is
    provided AND there are elided entries, the elided entries are
    handed to the summarizer + its non-empty return value replaces
    the legacy ``[... N earlier output(s) elided ...]`` marker
    with ``[Summary of N earlier output(s): {summary}]``. On any
    summarizer failure (raises, returns None, returns empty
    string), the legacy marker is restored so a wedged summarizer
    LLM never makes the executor worse than pre-Wave-31. Default
    ``summarizer=None`` preserves pre-Wave-31 truncation-only
    behaviour.

    [IMPROVE-166] Wave 32 — per-edge pass config:

      * ``pass_mode="all"`` (default) — full inter-node context
        as above; pre-Wave-32 behaviour unchanged.
      * ``pass_mode="none"`` — returns "" immediately. The
        downstream agent sees only the user input.
      * ``pass_mode="source_only"`` — filters ``node_outputs`` to
        entries with ``node == source_node_id`` so the downstream
        agent sees ONLY its immediate predecessor. Used in
        pipeline-style DAGs where node N+1 should consume only
        node N, not earlier history.

    An unknown ``pass_mode`` (typo / future schema addition)
    silently falls back to "all" — forward-compat with rule-shape
    additions in newer builds, mirroring ``_evaluate_edge_rule``'s
    unknown-rule-type semantics.
    """
    if pass_mode == "none":
        return ""

    usable = [r for r in node_outputs if r.get("status") == "ok"]

    if pass_mode == "source_only":
        if source_node_id is None:
            # Without a source, source_only collapses to no
            # context — equivalent to "none" for an edge that
            # didn't track its source (defensive guard).
            return ""
        usable = [r for r in usable if r.get("node") == source_node_id]
    # ``pass_mode`` not in _VALID_PASS_MODES → fall through to the
    # "all" path with the unfiltered usable list.

    if not usable:
        return ""

    # Walk newest-first; the most recent context is always preserved.
    chunks_newest_first: list[str] = []
    used_tokens = 0
    elided_count = 0

    for idx in range(len(usable) - 1, -1, -1):
        rec = usable[idx]
        agent = rec.get("agent", "?")
        role = rec.get("role", "")
        text = rec.get("text") or ""
        chunk = f"\n[{agent} ({role})]: {text}\n"
        chunk_tokens = _estimate_tokens(chunk)

        if used_tokens + chunk_tokens > budget_tokens:
            # This record + everything older gets elided.
            elided_count = idx + 1
            break

        chunks_newest_first.append(chunk)
        used_tokens += chunk_tokens

    chunks = list(reversed(chunks_newest_first))
    if elided_count > 0:
        # [IMPROVE-165] Wave 31 — try summarizer first; fall back
        # to legacy marker on any failure path.
        summary = None
        if summarizer is not None:
            elided_entries = usable[:elided_count]
            try:
                summary = summarizer(elided_entries)
            except Exception as exc:
                logger.warning(
                    "[IMPROVE-165] summarizer raised (fallback to "
                    "legacy marker): %s", exc,
                )
                summary = None
        if summary:
            prefix = (
                f"\n[Summary of {elided_count} earlier output(s): "
                f"{summary}]\n"
            )
        else:
            prefix = (
                f"\n[... {elided_count} earlier output(s) elided to fit "
                f"context budget ...]\n"
            )
        chunks.insert(0, prefix)
    return "".join(chunks)


# ── [IMPROVE-165] Wave 31 — LLM-summarized inter-node context ─────
#
# Tranche D piece 1 from the Wave 18 deferred queue. The IMPROVE-84
# block comment near _INTER_NODE_CONTEXT_BUDGET_TOKENS named "LLM-
# summarized inter-node context" as a follow-up; this implements
# the elision-replacement variant.
#
# The summarizer is opt-in via the
# ``dag_inter_node_summarization_model`` setting (default empty =
# disabled). When set, call sites in this module build a closure
# capturing ``orch`` + the model name + invoke the helper below
# from inside ``_build_inter_node_context``.
#
# Failure modes are graceful: any exception, empty response, or
# missing model identifier returns ``None`` so
# ``_build_inter_node_context`` falls back to the legacy elision
# marker. Pattern: opt-in features fail open to the default,
# rather than fail-loud (the inverse of Wave 28's schema-
# versioning fail-loud — different contracts call for different
# defaults).


def _summarize_elided_outputs(
    orchestrator: Any,
    model: str,
    elided_entries: list[dict[str, Any]],
) -> "str | None":
    """[IMPROVE-165] Summarize a list of elided node outputs into
    1-2 sentences via a one-shot ``orchestrator.router.chat`` call.

    Returns the trimmed summary text on success, or ``None`` on
    any failure (LLM unreachable, empty response, model missing,
    transient timeout). Callers (specifically
    ``_build_inter_node_context``) use the None to fall back to
    the legacy elision marker.

    Why a separate helper rather than inline in
    ``_build_inter_node_context``: the helper's signature
    (orchestrator + model + entries) makes it trivially mockable
    in tests + keeps ``_build_inter_node_context`` agnostic to the
    LLM-call mechanism (any callable matching the
    ``Callable[[list[dict]], str | None]`` summarizer signature
    works).
    """
    if not elided_entries or not model:
        return None
    try:
        from ..providers.base import ChatMessage, GenerationSettings

        joined = "\n\n".join(
            f"[{r.get('agent', '?')} ({r.get('role', '')})]: "
            f"{r.get('text', '')}"
            for r in elided_entries
        )
        messages = [
            ChatMessage(
                role="system",
                content=(
                    "You are a context summarizer for a multi-agent "
                    "DAG runner. Summarize the prior agents' outputs "
                    "in ONE OR TWO sentences. Capture the key "
                    "facts/decisions; drop step-by-step reasoning. "
                    "Reply with the summary text only — no preamble."
                ),
            ),
            ChatMessage(role="user", content=joined),
        ]
        settings = GenerationSettings(
            temperature=0.2, max_tokens=200,
        )
        resp = orchestrator.router.chat(model, messages, settings)
        text = (getattr(resp, "content", "") or "").strip()
        if not text:
            return None
        return text
    except Exception as exc:
        logger.warning(
            "[IMPROVE-165] summarizer LLM call failed (fallback to "
            "legacy marker): %s", exc,
        )
        return None


def _build_summarizer(
    orchestrator: Any, model: str | None,
) -> Callable[[list[dict[str, Any]]], "str | None"] | None:
    """[IMPROVE-165] Convenience builder used by the 3 call sites
    in this module. Returns a closure that captures the
    orchestrator + model name and adapts the
    ``_summarize_elided_outputs`` signature to the
    ``_build_inter_node_context`` ``summarizer`` kwarg shape.

    Returns ``None`` when ``model`` is empty / falsy so call sites
    can pass the result directly without an extra branch.
    """
    if not model:
        return None

    def _summarize(elided: list[dict[str, Any]]) -> "str | None":
        return _summarize_elided_outputs(orchestrator, model, elided)

    return _summarize


# ── Edge classification ──────────────────────────────────────────


def _compute_logprob_confidence(response: Any) -> float | None:
    """[IMPROVE-179] Wave 42 — Extract the first-token logprob
    from the response and map to a confidence in [0, 1].

    Returns ``exp(first_token_logprob)`` when the response
    carries an Ollama-shape ``logprobs`` array (top-level field
    in ``response.raw``); returns ``None`` for every other
    case so callers can fall back to the W33 heuristic
    cleanly:

      * ``response.raw`` is None / not a dict (the W18-era
        ``ChatResponse.raw`` escape hatch's contract — non-
        Ollama providers don't populate it with logprobs).
      * ``logprobs`` key missing (caller didn't pass
        ``GenerationSettings(logprobs=True)``, OR Ollama
        version too old to expose logprobs).
      * ``logprobs`` is empty / not a list (defensive — the
        Ollama Python client v0.6.1's response shape carries
        a list, but a corrupt response should fall back
        gracefully rather than raise).
      * First-entry's ``logprob`` field missing / not numeric
        (defensive against future Ollama API shape changes).

    The simple ``exp(first_token_logprob)`` formulation
    (rather than normalizing across ``top_logprobs``
    alternatives) is sufficient for the W33 IMPROVE-167
    classifier: classifier responses are short option-name
    strings emitted at temperature=0.2, and the first
    content-bearing token's logprob captures the LLM's
    commitment to its answer. Live audit during the W42
    audit confirmed: a high-confidence "Yes" answer had
    first logprob -0.0825 (exp = 0.92 = 92% confidence)
    while the second-best alternative "yes" was -2.54
    (exp = 0.08 = 8%) — a 12x ratio that the heuristic
    1/matched_count couldn't surface.

    Sources (2025-2026):
      * Ollama Python client v0.6.1 — ``logprobs`` /
        ``top_logprobs`` parameters in chat() + generate().
        https://github.com/ollama/ollama-python
      * Ollama HTTP API logprobs — first added in late-
        2024 ollama core releases; verified live during
        W42 audit on a local install (gemma3:1b returned
        structured logprobs array on a real chat call).
    """
    raw = getattr(response, "raw", None)
    if not isinstance(raw, dict):
        return None
    logprobs_list = raw.get("logprobs")
    if not isinstance(logprobs_list, list) or not logprobs_list:
        return None
    first = logprobs_list[0]
    if not isinstance(first, dict):
        return None
    lp = first.get("logprob")
    if not isinstance(lp, (int, float)):
        return None
    try:
        return math.exp(float(lp))
    except (OverflowError, ValueError):
        # Defensive: a logprob beyond float-range shouldn't
        # exist in practice, but never let the helper raise.
        return None


def classify_llm_router_edges(
    orch: "AgentOrchestrator",
    edges: list[tuple[str, str, str, dict[str, Any]]],
    source_output: str,
    visited: set[str],
) -> str | None:
    """[IMPROVE-35] Run ONE LLM classification call covering all
    ``llm_router`` sibling edges out of a source node.

    Returns the chosen option string (an option name from the
    union of edges' ``options`` arrays) or ``None`` when:

      * No llm_router edges exist among the input.
      * The router/config isn't available.
      * The classification call fails or returns junk.

    The caller fires only the edge whose ``target`` matches the
    returned option. ``None`` means NO llm_router edges fire —
    users wanting deadlock-resilience should add an
    ``always`` fallback edge.

    Doc rationale at docs/features/05-systems.md:425-437. The
    single-call shape (vs per-edge classification) is the
    important detail — three llm_router edges out of one node
    cost one LLM round-trip, not three.
    """
    # Gather candidate options + the first non-empty instruction.
    # Sibling edges typically share the same instruction; if they
    # differ, we use the first one (predictable) and union the
    # options.
    instruction = ""
    options: list[str] = []
    relevant_targets: list[str] = []
    for target, rule_type, _rule_notes, rule in edges:
        if rule_type != "llm_router":
            continue
        if target in visited:
            continue
        if not instruction:
            instruction = (
                rule.get("instruction") or "Pick the next branch."
            )
        edge_opts = rule.get("options") or []
        if edge_opts:
            options.extend(edge_opts)
        else:
            # No explicit options — use the edge's own target as
            # the option name. The convention matches the doc's
            # canonical example where ``options`` carries node
            # names that line up with edge targets.
            options.append(target)
        relevant_targets.append(target)

    if not relevant_targets:
        return None

    # Dedupe options preserving order (first occurrence wins).
    options = list(dict.fromkeys(options))

    # Refuse gracefully if the router isn't reachable. The DAG
    # falls through with chosen_option=None; callers can add an
    # always-edge fallback for resilience.
    if orch.router is None:
        logger.info(
            "[IMPROVE-35] llm_router skipped: no router on orchestrator",
        )
        return None

    try:
        from ..providers import ChatMessage, GenerationSettings
        model = f"ollama:{orch.config.prompt_builder_model}"
        options_block = "\n".join(f"- {o}" for o in options)
        classify_prompt = (
            "You are a routing decision agent for a multi-agent "
            "workflow. Pick exactly ONE option for the branch that "
            "should execute next.\n\n"
            f"Source agent's output:\n\"\"\"\n{source_output}\n\"\"\"\n\n"
            f"Decision criterion:\n{instruction}\n\n"
            f"Options (pick exactly one):\n{options_block}\n\n"
            "Reply with ONLY the chosen option's name. No quotes, no "
            "explanation, no preamble."
        )
        # [IMPROVE-179] Wave 42 — Opt-in logprobs request. Read
        # the env-var inline so changes take effect without a
        # restart of the orchestrator (matches the W33
        # IMPROVE-167 threshold-read pattern at the rejection
        # branch below). When disabled (default), no logprobs
        # are requested and the chat call's bandwidth + Ollama
        # work is identical to pre-W42.
        try:
            from ..config import get_settings
            logprobs_enabled = bool(
                get_settings().dag_classifier_logprobs_enabled,
            )
        except Exception:
            logprobs_enabled = False
        classify_settings = GenerationSettings(
            temperature=0.2,
            max_tokens=64,
            logprobs=logprobs_enabled,
        )
        response = orch.router.chat(
            model,
            [ChatMessage(role="user", content=classify_prompt)],
            classify_settings,
        )
        text = (response.content or "").strip()
        # Strip qwen3/r1 thinking tags (same idiom as the prompt
        # enhancer at ai_enhance.py:3437-3438).
        text = re.sub(
            r"<think>.*?</think>", "", text, flags=re.DOTALL,
        ).strip()
        text_lc = text.lower()

        # [IMPROVE-167] Wave 33 — count how many options appear
        # in the response so we can compute a heuristic confidence
        # for the routing decision. Multi-match = ambiguous = low
        # confidence; clean single-match = high confidence.
        matched_options = [
            opt for opt in options if opt.lower() in text_lc
        ]
        if not matched_options:
            logger.warning(
                "[IMPROVE-35] llm_router output didn't match any "
                "option. options=%s response=%r",
                options, text[:120],
            )
            return None

        # First-match is the chosen option (preserves pre-Wave-33
        # selection semantics); confidence captures how many other
        # options ALSO matched (signaling ambiguity).
        chosen = matched_options[0]

        # [IMPROVE-179] Wave 42 — Prefer logprob-derived
        # confidence when available. The helper returns None on
        # any non-Ollama-shape input (logprobs missing, env-var
        # disabled so no logprobs were requested, response.raw
        # not a dict, malformed array, etc.); fall back to the
        # W33 heuristic ``1 / matched_count`` cleanly. The
        # threshold check below works identically with either
        # source — both produce values in [0, 1].
        confidence_source = "heuristic"
        confidence = 1.0 / len(matched_options)
        if logprobs_enabled:
            logprob_conf = _compute_logprob_confidence(response)
            if logprob_conf is not None:
                confidence = logprob_conf
                confidence_source = "logprob"
            else:
                logger.debug(
                    "[IMPROVE-179] logprob confidence unavailable "
                    "(env on, response.raw lacks logprobs); "
                    "falling back to W33 heuristic",
                )

        # [IMPROVE-167] Wave 33 — apply opt-in threshold. Default
        # 0.0 in AppSettings means any match wins (pre-Wave-33
        # behaviour). When the threshold is non-zero, ambiguous
        # responses are rejected so the always-fallback edge
        # fires instead of a low-confidence pick.
        try:
            from ..config import get_settings
            threshold = float(
                get_settings().dag_classifier_confidence_threshold,
            )
        except Exception:
            threshold = 0.0

        if threshold > 0.0 and confidence < threshold:
            logger.warning(
                "[IMPROVE-167] llm_router classification rejected: "
                "confidence %.3f (%s) < threshold %.3f (matched %d "
                "of %d options); chosen=%r response=%r",
                confidence, confidence_source, threshold,
                len(matched_options), len(options), chosen,
                text[:120],
            )
            return None

        logger.info(
            "[IMPROVE-35] llm_router chose %r from %s "
            "(confidence %.3f source=%s, matched %d of %d)",
            chosen, options, confidence, confidence_source,
            len(matched_options), len(options),
        )
        return chosen
    except Exception as exc:
        logger.warning(
            "[IMPROVE-35] llm_router call failed (%s); no edges fire",
            exc,
        )
        return None


# ── Shared helpers ───────────────────────────────────────────────


def _evaluate_edge_rule(
    rule_type: str,
    rule_notes: str,
    output: str,
    chosen_option: str | None,
    target: str,
) -> bool:
    """Per-edge follow check, shared between sync + streaming
    executors. Pre-extraction this logic was duplicated across
    both methods.

    Rule types:
      always / manual_next      → unconditional
      on_keyword_match          → output.lower() contains any of
                                  rule_notes (comma-separated)
      on_tool_result            → output mentions a tool marker
                                  ("Tool", "tool", "Result:", "```")
      llm_router                → chosen_option (from
                                  classify_llm_router_edges) equals
                                  this edge's target. ``None`` →
                                  no llm_router edge fires.
      <unknown>                 → unconditional (back-compat for
                                  rules added without code)
    """
    if rule_type in ("always", "manual_next"):
        return True
    if rule_type == "on_keyword_match":
        keywords = [
            k.strip().lower() for k in rule_notes.split(",")
            if k.strip()
        ]
        output_lower = output.lower()
        if not keywords:
            return True
        return any(kw in output_lower for kw in keywords)
    if rule_type == "on_tool_result":
        return any(
            marker in output
            for marker in ("Tool", "tool", "Result:", "```")
        )
    if rule_type == "llm_router":
        # Edge fires iff its target matches the LLM-classified
        # option. ``chosen_option`` being None means the LLM
        # failed (router unavailable, classification failed) —
        # in that case NO llm_router edges fire so the user can
        # add an "always" fallback edge for resilience rather
        # than guessing wrong.
        return chosen_option is not None and chosen_option == target
    # unknown rule = always follow (back-compat)
    return True


def _build_dag_structures(
    system_definition: dict,
) -> tuple[
    dict[str, dict],
    dict[str, list[tuple[str, str, str, dict[str, Any]]]],
    dict[str, int],
    list[str],
]:
    """Build node_map, edge_map, in_degree, and the initial
    current_nodes (start node) from a system definition. Shared
    by both executors.

    edge_map values are 4-tuples ``(target, rule_type, rule_notes,
    rule_dict)`` so the [IMPROVE-35] llm_router rule_type can read
    its ``options`` / ``instruction`` from the full rule dict.
    """
    nodes = system_definition.get("nodes", [])
    edges = system_definition.get("edges", [])
    start_node_id = (
        system_definition.get("start_node_id")
        or system_definition.get("startNodeId")
    )

    node_map = {n["id"]: n for n in nodes}
    edge_map: dict[
        str, list[tuple[str, str, str, dict[str, Any]]]
    ] = {n["id"]: [] for n in nodes}
    in_degree: dict[str, int] = {n["id"]: 0 for n in nodes}

    for e in edges:
        src, tgt = e.get("source"), e.get("target")
        rule = e.get("rule", {}) if isinstance(e.get("rule"), dict) else {}
        rule_type = rule.get("type", e.get("ruleType", "always"))
        rule_notes = rule.get("notes", e.get("notes", ""))
        if src in edge_map and tgt in node_map:
            edge_map[src].append((tgt, rule_type, rule_notes, rule))
            in_degree[tgt] = in_degree.get(tgt, 0) + 1

    # Find start node
    if start_node_id and start_node_id in node_map:
        current_nodes = [start_node_id]
    else:
        current_nodes = [
            nid for nid, deg in in_degree.items() if deg == 0
        ]
        if not current_nodes and nodes:
            current_nodes = [nodes[0]["id"]]

    return node_map, edge_map, in_degree, current_nodes


# ── [IMPROVE-83] Shared parallel-wave pre-pass ───────────────────


class _ParallelWaveResult:
    """Return value of ``_run_parallel_wave_or_fallback``.

    Both executors call the helper before their per-node loop and
    merge the result back in: ``preloaded_outputs`` short-circuits
    the LLM call inside the loop, the three int counters get added
    to the executor's running totals (surfaced in ``run_done.perf``),
    and ``streaming_event`` carries an optional SSE-shaped dict the
    streaming executor yields to the frontend.

    Plain class rather than ``@dataclass`` to keep the executor
    module's import surface small (no ``dataclasses`` import).
    """

    __slots__ = (
        "preloaded_outputs",
        "parallel_waves_used",
        "concurrent_nodes_total",
        "parallel_waves_skipped",
        "streaming_event",
    )

    def __init__(
        self,
        *,
        preloaded_outputs: dict[str, tuple[str, int, Exception | None]],
        parallel_waves_used: int,
        concurrent_nodes_total: int,
        parallel_waves_skipped: int,
        streaming_event: dict[str, Any] | None = None,
    ) -> None:
        self.preloaded_outputs = preloaded_outputs
        self.parallel_waves_used = parallel_waves_used
        self.concurrent_nodes_total = concurrent_nodes_total
        self.parallel_waves_skipped = parallel_waves_skipped
        self.streaming_event = streaming_event


async def _run_parallel_wave_or_fallback(
    orch: "AgentOrchestrator",
    *,
    runnable_for_parallel: list[str],
    node_map: dict[str, dict],
    node_outputs: list[dict[str, Any]],
    context_budget: int,
    user_input: str,
    parallel_waves_flag: bool,
    run_id: str,
    system_name: str,
    step: int,
    streaming: bool = False,
    last_pass_per_node: "dict[str, str] | None" = None,
    last_source_per_node: "dict[str, str] | None" = None,
) -> _ParallelWaveResult:
    """[IMPROVE-83] Shared parallel-wave pre-pass for both executors.

    Pre-IMPROVE-83 only the sync ``execute_graph`` had the
    [IMPROVE-36] parallel-wave pre-pass. The streaming variant
    ``astream_graph`` ran every wave sequentially even with
    ``parallel_waves: True`` set on the system. Now that both
    executors share ``_build_dag_structures`` and
    ``_evaluate_edge_rule``, lifting the pre-pass into a shared
    helper closes the parity gap — and a future bug-fix lands once,
    not twice.

    Single source of truth here also pre-empts the kind of drift
    that bit IMPROVE-35 (the 3-tuple/4-tuple ``edge_map`` shape
    that diverged between the two paths and went unnoticed for a
    full release because every streaming test stubbed the executor
    itself).

    Behaviour preserved byte-for-byte from the inline form:
      * Only fires when ``parallel_waves_flag is True`` AND the
        wave has 2+ runnable nodes.
      * Falls back to sequential when nodes share the same agent
        (the [IMPROVE-36] safety constraint — protects shared
        in-memory ``_smart_memories[agent]`` state).
      * Each preloaded node runs via ``asyncio.to_thread`` over
        ``orch.chat_with_agent``, with the SAME pre-wave context
        block (siblings see same context — pipelining traded for
        speed).
      * Per-node duration captured and surfaced on the cached entry
        so the per-node loop's downstream emit sees the actual
        chat-with-agent time, not the post-cache lookup time.
      * Errors during the wave still fire the wave_parallel event
        (with ``errors`` count in context) — the event tracks the
        parallel DECISION, not whether the agents succeeded.

    The ``streaming`` flag controls:
      * Whether ``streaming=True`` lands in the wave_parallel /
        wave_parallel_fallback context dict (matches the
        ``run.start`` / ``run_done`` distinction).
      * Whether the result carries a non-None ``streaming_event``
        dict the streaming executor yields back to the SSE
        consumer. Sync callers ignore the field.

    Returns ``_ParallelWaveResult`` with empty preloaded_outputs +
    zero counters when the helper short-circuits (parallel disabled,
    single-node wave, or duplicate-agent fallback). Caller proceeds
    sequentially in those cases.
    """
    if not parallel_waves_flag or len(runnable_for_parallel) <= 1:
        return _ParallelWaveResult(
            preloaded_outputs={},
            parallel_waves_used=0,
            concurrent_nodes_total=0,
            parallel_waves_skipped=0,
        )

    wave_agents = [
        node_map[n].get("agent", "") for n in runnable_for_parallel
    ]

    if len(wave_agents) != len(set(wave_agents)):
        # Wave is unsafe to parallelize — duplicate agents.
        # [IMPROVE-36 telemetry] Track safety-fallbacks too so a
        # user wondering "why didn't parallel engage" can grep for
        # this in run logs.
        logger.info(
            "[IMPROVE-36] parallel_waves on but wave has duplicate "
            "agents (%s); falling back to sequential", wave_agents,
        )
        emit_typed(
            "system", "wave_parallel_fallback", status="ok",
            context={
                "run_id": run_id,
                "system_name": system_name,
                "step": step,
                "node_count": len(runnable_for_parallel),
                "agents": wave_agents,
                "reason": "duplicate_agents",
                "streaming": streaming,
            },
        )
        return _ParallelWaveResult(
            preloaded_outputs={},
            parallel_waves_used=0,
            concurrent_nodes_total=0,
            parallel_waves_skipped=1,
        )

    # Wave is safe — run concurrently.
    # [IMPROVE-165] Wave 31 — opt-in LLM summarizer for elided
    # inter-node context entries. Empty setting (default) yields
    # ``summarizer=None`` so behaviour matches pre-Wave-31.
    from ..config import get_settings
    pre_wave_summarizer = _build_summarizer(
        orch, get_settings().dag_inter_node_summarization_model,
    )
    # [IMPROVE-166] Wave 32 — per-edge pass config means each
    # node in the wave may need a different context block (the
    # edges into different nodes can specify different pass
    # modes). Build context per-node inside ``_preload`` rather
    # than once-per-wave. The dict-lookups are cheap; the heavy
    # cost is the LLM summarizer if it fires, which is independent
    # of how many nodes are in the wave.
    _last_pass = last_pass_per_node or {}
    _last_source = last_source_per_node or {}

    async def _preload(_nid: str):
        _node_def = node_map[_nid]
        _agent = _node_def.get("agent", "")
        _pass_mode = _last_pass.get(_nid, _DEFAULT_PASS_MODE)
        _source_id = _last_source.get(_nid)
        _ctx = _build_inter_node_context(
            node_outputs, budget_tokens=context_budget,
            summarizer=pre_wave_summarizer,
            pass_mode=_pass_mode,
            source_node_id=_source_id,
        )
        if _ctx:
            _prompt = (
                f"{user_input}\n\nContext from prior agents:\n"
                f"{_ctx}"
            )
        else:
            _prompt = user_input
        _t0 = _time.monotonic()
        try:
            _out = await asyncio.to_thread(
                orch.chat_with_agent, _agent, _prompt,
            )
            return _nid, _out, int(
                (_time.monotonic() - _t0) * 1000,
            ), None
        except Exception as _exc:
            return _nid, "", int(
                (_time.monotonic() - _t0) * 1000,
            ), _exc

    _wave_t0 = _time.monotonic()
    results = await asyncio.gather(
        *[_preload(n) for n in runnable_for_parallel],
    )
    _wave_ms = int((_time.monotonic() - _wave_t0) * 1000)
    preloaded_outputs: dict[str, tuple[str, int, Exception | None]] = {}
    for _nid, _out, _dur, _exc in results:
        preloaded_outputs[_nid] = (_out, _dur, _exc)
    logger.info(
        "[IMPROVE-36] parallel wave: %d nodes ran concurrently",
        len(runnable_for_parallel),
    )
    _wave_errors = sum(
        1 for _, _, _, exc in results if exc is not None
    )
    emit_typed(
        "system", "wave_parallel", status="ok",
        duration_ms=_wave_ms,
        context={
            "run_id": run_id,
            "system_name": system_name,
            "step": step,
            "node_count": len(runnable_for_parallel),
            "agents": wave_agents,
            "errors": _wave_errors,
            "streaming": streaming,
        },
        perf={"node_count": len(runnable_for_parallel)},
    )

    streaming_event: dict[str, Any] | None = None
    if streaming:
        # [IMPROVE-83] SSE-shaped event — distinct from the
        # observability ``wave_parallel`` event above. The frontend
        # uses this to render "wave N: K nodes in flight" before the
        # per-node node_start events fire.
        streaming_event = {
            "type": "wave_parallel",
            "step": step,
            "node_count": len(runnable_for_parallel),
            "agents": wave_agents,
            "duration_ms": _wave_ms,
            "errors": _wave_errors,
        }

    return _ParallelWaveResult(
        preloaded_outputs=preloaded_outputs,
        parallel_waves_used=1,
        concurrent_nodes_total=len(runnable_for_parallel),
        parallel_waves_skipped=0,
        streaming_event=streaming_event,
    )


# ── Sync executor ────────────────────────────────────────────────


async def execute_graph(
    orch: "AgentOrchestrator",
    system_definition: dict,
    user_input: str,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Execute a system graph designed in the visual graph editor.

    Supports edge routing rules:
    - "always": always follow this edge
    - "on_keyword_match": follow if output contains a keyword (from edge notes)
    - "on_tool_result": follow if a tool was called (checks for tool markers)
    - "manual_next": always follow (same as always, user controls via graph)

    ``run_id`` is optional — when callers don't pass one, a fresh
    ``uuid4`` is minted as before. [IMPROVE-68] Commit 5/5 wraps
    this method in a ``trace_run`` block at the route layer and
    passes the same ``run_id`` to both, so the trace JSON file
    on disk and the run_id in the response payload match — that's
    what lets operators jump from /runs to the response and back.

    Returns timing data and tool call info in trace.
    """
    nodes = system_definition.get("nodes", [])
    edges = system_definition.get("edges", [])

    if not nodes:
        return {"final_text": "System has no agent nodes.", "node_outputs": []}

    node_map, edge_map, _in_degree, current_nodes = _build_dag_structures(
        system_definition,
    )

    # [IMPROVE-68] Reuse caller-supplied run_id when given (the
    # /systems/{name}/chat route mints one and passes it to BOTH
    # ``trace_run`` and this executor so the on-disk trace JSON
    # matches the response payload's run_id).
    run_id = run_id or str(uuid.uuid4())
    system_name = (
        system_definition.get("name")
        or system_definition.get("id")
        or "unnamed"
    )
    total_start = _time.monotonic()
    node_outputs: list[dict[str, Any]] = []
    # [IMPROVE-33] Per-system budget override; defaults to
    # ``_INTER_NODE_CONTEXT_BUDGET_TOKENS``. Letting users dial
    # this up for big-context models (or down for cheap models)
    # without redeploying.
    context_budget = int(
        system_definition.get(
            "context_budget_tokens", _INTER_NODE_CONTEXT_BUDGET_TOKENS,
        ),
    )
    visited: set[str] = set()
    # [IMPROVE-166] Wave 32 — per-target per-edge pass tracking.
    # Updated in the edge-firing loop below; consulted at the top
    # of each per-node run to decide which prior outputs to
    # include in the inter-node context. Last-fired-edge wins
    # the multi-incoming-edge case.
    last_pass_per_node: dict[str, str] = {}
    last_source_per_node: dict[str, str] = {}
    max_steps = len(nodes) * 2  # prevent infinite loops

    emit_typed("system", "run.start", status="start",
         context={
             "run_id": run_id,
             "system_name": system_name,
             "conversation_id": conversation_id,
             "node_count": len(nodes),
             "edge_count": len(edges),
         })

    # [IMPROVE-36] Read parallel-waves flag once. Default False
    # preserves the pre-IMPROVE-36 sequential semantics (token
    # budget + per-node ordering). When on, sibling nodes in the
    # same wave run concurrently via ``asyncio.to_thread`` over
    # ``chat_with_agent``. Doc rationale at
    # docs/features/05-systems.md:444-455.
    parallel_waves = bool(system_definition.get("parallel_waves", False))

    # [IMPROVE-36 telemetry] Per-run counters surfaced in the
    # run_done perf dict so the weekly review can answer "how
    # often does parallel mode actually engage" and "what's the
    # fan-out". Increments happen INSIDE the parallel pre-pass
    # below — both run only when the wave is safe to parallelize
    # AND has more than one runnable node, matching the user's
    # intuition for "did the speedup fire here".
    parallel_waves_used = 0
    concurrent_nodes_total = 0
    parallel_waves_skipped = 0  # safety-fallback counter

    step = 0
    while current_nodes and step < max_steps:
        step += 1
        next_nodes: list[str] = []

        # [IMPROVE-36 / IMPROVE-83] Pre-pass: when parallel_waves
        # is on AND the wave has 2+ runnable distinct-agent nodes,
        # run all ``chat_with_agent`` calls concurrently and stash
        # outputs. The per-node loop below reuses the stashed
        # result instead of re-running the LLM call. The shared
        # helper drives both this sync executor and ``astream_graph``
        # so the parallel-wave decision lives in one place.
        #
        # Sequential semantics still apply WITHIN a wave when
        # parallel mode is off — node 2 sees node 1's output via
        # the rebuilt context block. Parallel mode intentionally
        # TRADES that pipelining for speed: siblings see the same
        # pre-wave context.
        runnable_for_parallel = [
            n for n in current_nodes
            if n not in visited and n in node_map
            and node_map[n].get("agent", "") in orch.definitions
        ]
        _wave_result = await _run_parallel_wave_or_fallback(
            orch,
            runnable_for_parallel=runnable_for_parallel,
            node_map=node_map,
            node_outputs=node_outputs,
            context_budget=context_budget,
            user_input=user_input,
            parallel_waves_flag=parallel_waves,
            run_id=run_id,
            system_name=system_name,
            step=step,
            streaming=False,
            last_pass_per_node=last_pass_per_node,
            last_source_per_node=last_source_per_node,
        )
        preloaded_outputs = _wave_result.preloaded_outputs
        parallel_waves_used += _wave_result.parallel_waves_used
        concurrent_nodes_total += _wave_result.concurrent_nodes_total
        parallel_waves_skipped += _wave_result.parallel_waves_skipped

        for nid in current_nodes:
            if nid in visited:
                continue
            visited.add(nid)

            node_def = node_map.get(nid)
            if not node_def:
                continue

            agent_name = node_def.get("agent", "")
            role = (node_def.get("config") or {}).get(
                "role", node_def.get("role", ""),
            )

            if not agent_name or agent_name not in orch.definitions:
                node_outputs.append({
                    "node": nid, "agent": agent_name, "role": role,
                    "text": f"(agent '{agent_name}' not found)",
                    "status": "skipped", "duration_ms": 0,
                })
                emit_typed("system", "node_end", status="skipped",
                     duration_ms=0,
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name, "role": role,
                              "reason": "agent_not_found"})
                continue

            # [IMPROVE-33] Build prompt with token-budgeted prior
            # context — newest outputs win, older ones get elided
            # with a marker so the agent knows truncation happened.
            # [IMPROVE-165] Wave 31 — opt-in LLM summarizer
            # replaces the elision marker with a 1-2 sentence
            # digest when the setting is non-empty.
            # [IMPROVE-166] Wave 32 — per-edge pass config
            # filters which prior outputs the current node sees.
            from ..config import get_settings
            _summarizer = _build_summarizer(
                orch, get_settings().dag_inter_node_summarization_model,
            )
            _pass_mode = last_pass_per_node.get(nid, _DEFAULT_PASS_MODE)
            _source_id = last_source_per_node.get(nid)
            ctx_block = _build_inter_node_context(
                node_outputs, budget_tokens=context_budget,
                summarizer=_summarizer,
                pass_mode=_pass_mode,
                source_node_id=_source_id,
            )
            if ctx_block:
                prompt = (
                    f"{user_input}\n\nContext from prior agents:\n"
                    f"{ctx_block}"
                )
            else:
                prompt = user_input

            # Execute
            node_start = _time.monotonic()
            emit_typed("system", "node_start", status="start",
                 context={"run_id": run_id, "system_name": system_name,
                          "node_id": nid, "agent": agent_name, "role": role,
                          "step": step})
            try:
                # [IMPROVE-36] Use preloaded output when this node
                # ran in parallel above. Otherwise call into
                # chat_with_agent normally (sequential path).
                if nid in preloaded_outputs:
                    output, duration_ms, _preload_exc = preloaded_outputs[nid]
                    if _preload_exc is not None:
                        # Re-raise so the existing except handler
                        # below records the error consistently
                        # with the sequential code path.
                        raise _preload_exc
                else:
                    output = orch.chat_with_agent(agent_name, prompt)
                    duration_ms = int(
                        (_time.monotonic() - node_start) * 1000,
                    )
                node_outputs.append({
                    "node": nid, "agent": agent_name, "role": role,
                    "text": output, "status": "ok",
                    "duration_ms": duration_ms,
                })
                emit_typed("system", "node_end", status="ok",
                     duration_ms=duration_ms,
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name,
                              "role": role},
                     perf={"output_length": len(output) if output else 0})
            except Exception as exc:
                duration_ms = int((_time.monotonic() - node_start) * 1000)
                node_outputs.append({
                    "node": nid, "agent": agent_name, "role": role,
                    "text": str(exc), "status": "error",
                    "duration_ms": duration_ms,
                })
                emit_typed("system", "node_end", status="error",
                     duration_ms=duration_ms,
                     error_code=type(exc).__name__,
                     error_message=str(exc),
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name,
                              "role": role})
                output = str(exc)

            # [IMPROVE-35] Evaluate edge routing — llm_router edges
            # share ONE classification call per source node, so a
            # 3-way conditional doesn't cost 3 LLM round-trips.
            # Other rule types still evaluate independently.
            chosen_option = classify_llm_router_edges(
                orch, edge_map.get(nid, []), output, visited,
            )
            for target, rule_type, rule_notes, _rule in edge_map.get(nid, []):
                if target in visited:
                    continue
                if _evaluate_edge_rule(
                    rule_type, rule_notes, output,
                    chosen_option, target,
                ):
                    # [IMPROVE-166] Wave 32 — capture per-edge
                    # pass config for the target. Last-fired-edge
                    # wins multi-incoming case.
                    last_pass_per_node[target] = _rule.get(
                        "pass", _DEFAULT_PASS_MODE,
                    )
                    last_source_per_node[target] = nid
                    if target not in next_nodes:
                        next_nodes.append(target)

        current_nodes = next_nodes

    total_ms = int((_time.monotonic() - total_start) * 1000)
    final_text = (
        node_outputs[-1]["text"] if node_outputs
        else "No output produced."
    )

    errors = sum(1 for n in node_outputs if n.get("status") == "error")
    emit_typed("system", "run_done", status="error" if errors else "ok",
         duration_ms=total_ms,
         context={"run_id": run_id, "system_name": system_name,
                  "conversation_id": conversation_id},
         perf={
             "nodes_executed": len(node_outputs),
             "error_count": errors,
             "final_text_length": len(final_text) if final_text else 0,
             "steps": step,
             # [IMPROVE-36 telemetry] Per-run aggregates of the
             # parallel-wave decision. Useful for the weekly review
             # but cheap to surface — three int fields.
             "parallel_waves_used": parallel_waves_used,
             "concurrent_nodes_total": concurrent_nodes_total,
             "parallel_waves_skipped": parallel_waves_skipped,
         })

    return {
        "final_text": final_text,
        "node_outputs": node_outputs,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "total_duration_ms": total_ms,
        "nodes_executed": len(node_outputs),
    }


# ── Streaming executor ───────────────────────────────────────────


async def astream_graph(
    orch: "AgentOrchestrator",
    system_definition: dict,
    user_input: str,
    conversation_id: str | None = None,
    run_id: str | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """[IMPROVE-32] Streaming variant of ``execute_graph``.

    Walks the same DAG with the same edge-routing semantics — the
    sync executor's ``visited`` + ``max_steps`` cycle guard,
    agent-name-not-found = ``status="skipped"``, accumulated
    context concatenation, edge rules ``always`` / ``manual_next`` /
    ``on_keyword_match`` / ``on_tool_result`` / ``llm_router`` —
    but yields typed events as nodes progress instead of returning
    the full result at the end.

    Per node it calls ``orch.astream_chat_with_agent(agent, prompt)``
    and re-yields each token / tool_call / tool_result tagged with
    the owning ``node`` id, so the SSE consumer can reconstruct
    per-node sub-streams. Final event is ``{"type": "done", ...}``
    carrying the same payload shape as the sync executor's return
    dict — that's what ``/systems/{name}/chat/stream``'s end-frame
    renders.

    The same ``emit_typed("system", ...)`` calls as the sync path stay
    intact so the active TraceRecorder ContextVar (set by
    ``trace_run`` at the route layer per IMPROVE-68) records the
    per-node timeline identically — operators on /runs see the
    same events whether the system was invoked sync or streamed.
    """
    nodes = system_definition.get("nodes", [])
    edges = system_definition.get("edges", [])

    if not nodes:
        yield {
            "type": "done",
            "final_text": "System has no agent nodes.",
            "node_outputs": [],
            "total_duration_ms": 0,
            "nodes_executed": 0,
            "run_id": run_id or str(uuid.uuid4()),
            "conversation_id": conversation_id,
        }
        return

    node_map, edge_map, _in_degree, current_nodes = _build_dag_structures(
        system_definition,
    )

    run_id = run_id or str(uuid.uuid4())
    system_name = (
        system_definition.get("name")
        or system_definition.get("id")
        or "unnamed"
    )
    total_start = _time.monotonic()
    node_outputs: list[dict[str, Any]] = []
    # [IMPROVE-33] same budget contract as the sync executor.
    context_budget = int(
        system_definition.get(
            "context_budget_tokens", _INTER_NODE_CONTEXT_BUDGET_TOKENS,
        ),
    )
    visited: set[str] = set()
    # [IMPROVE-166] Wave 32 — same per-target tracking as the
    # sync executor (last-fired-edge-wins for multi-incoming).
    last_pass_per_node: dict[str, str] = {}
    last_source_per_node: dict[str, str] = {}
    max_steps = len(nodes) * 2

    emit_typed("system", "run.start", status="start",
         context={
             "run_id": run_id,
             "system_name": system_name,
             "conversation_id": conversation_id,
             "node_count": len(nodes),
             "edge_count": len(edges),
             "streaming": True,
         })

    # [IMPROVE-83] Parallel-waves parity with the sync executor.
    # Pre-IMPROVE-83 the streaming path ignored ``parallel_waves`` —
    # a 3-way fan-out streamed one sibling at a time even with the
    # flag on, so users opting into parallel mode for batched LLM
    # work paid the latency on the streaming UI. Sharing the
    # _run_parallel_wave_or_fallback helper closes the parity gap.
    parallel_waves = bool(system_definition.get("parallel_waves", False))
    parallel_waves_used = 0
    concurrent_nodes_total = 0
    parallel_waves_skipped = 0

    step = 0
    while current_nodes and step < max_steps:
        step += 1
        next_nodes: list[str] = []

        # [IMPROVE-83] Same pre-pass as ``execute_graph``. The
        # helper's ``streaming=True`` flag tags the wave_parallel /
        # wave_parallel_fallback observability events with
        # ``streaming: True`` and returns a dict in
        # ``streaming_event`` we can yield to the SSE consumer
        # before the per-node node_start events fire.
        runnable_for_parallel = [
            n for n in current_nodes
            if n not in visited and n in node_map
            and node_map[n].get("agent", "") in orch.definitions
        ]
        _wave_result = await _run_parallel_wave_or_fallback(
            orch,
            runnable_for_parallel=runnable_for_parallel,
            node_map=node_map,
            node_outputs=node_outputs,
            context_budget=context_budget,
            user_input=user_input,
            parallel_waves_flag=parallel_waves,
            run_id=run_id,
            system_name=system_name,
            step=step,
            streaming=True,
            last_pass_per_node=last_pass_per_node,
            last_source_per_node=last_source_per_node,
        )
        preloaded_outputs = _wave_result.preloaded_outputs
        parallel_waves_used += _wave_result.parallel_waves_used
        concurrent_nodes_total += _wave_result.concurrent_nodes_total
        parallel_waves_skipped += _wave_result.parallel_waves_skipped
        if _wave_result.streaming_event is not None:
            yield _wave_result.streaming_event

        for nid in current_nodes:
            if nid in visited:
                continue
            visited.add(nid)

            node_def = node_map.get(nid)
            if not node_def:
                continue

            agent_name = node_def.get("agent", "")
            role = (node_def.get("config") or {}).get(
                "role", node_def.get("role", ""),
            )

            # Agent-name-not-found path mirrors the sync executor:
            # record status="skipped" and continue. Stream consumers
            # see a node_start + node_end pair so the UI can render
            # the skip explicitly rather than dropping the node
            # silently.
            if not agent_name or agent_name not in orch.definitions:
                yield {
                    "type": "node_start",
                    "node": nid, "agent": agent_name, "role": role,
                }
                skipped_text = f"(agent '{agent_name}' not found)"
                node_outputs.append({
                    "node": nid, "agent": agent_name, "role": role,
                    "text": skipped_text, "status": "skipped",
                    "duration_ms": 0,
                })
                emit_typed("system", "node_end", status="skipped",
                     duration_ms=0,
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name,
                              "role": role, "reason": "agent_not_found"})
                yield {
                    "type": "node_end",
                    "node": nid, "agent": agent_name, "role": role,
                    "text": skipped_text, "status": "skipped",
                    "duration_ms": 0,
                }
                continue

            # [IMPROVE-83] When this node ran in the parallel wave
            # above, surface the cached output as a single token
            # event rather than re-running ``astream_chat_with_agent``.
            # The streaming UI sees the final text appear in one
            # frame — the speedup is real even though we lose the
            # per-token streaming illusion (the underlying call
            # already finished). Mirrors the sync executor's
            # cache-or-call branch below at the same line.
            if nid in preloaded_outputs:
                _cached_text, _cached_dur, _cached_exc = (
                    preloaded_outputs[nid]
                )
                emit_typed("system", "node_start", status="start",
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name,
                              "role": role, "step": step,
                              "preloaded": True})
                yield {
                    "type": "node_start",
                    "node": nid, "agent": agent_name, "role": role,
                    "preloaded": True,
                }
                if _cached_exc is not None:
                    duration_ms = _cached_dur
                    output = str(_cached_exc)
                    node_status = "error"
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": output, "status": "error",
                        "duration_ms": duration_ms,
                    })
                    emit_typed("system", "node_end", status="error",
                         duration_ms=duration_ms,
                         error_code=type(_cached_exc).__name__,
                         error_message=str(_cached_exc),
                         context={"run_id": run_id,
                                  "system_name": system_name,
                                  "node_id": nid, "agent": agent_name,
                                  "role": role})
                else:
                    output = _cached_text or ""
                    duration_ms = _cached_dur
                    node_status = "ok"
                    if output:
                        # Deliver the cached text in one token frame
                        # so consumers that count tokens still see a
                        # non-empty body for the node.
                        yield {
                            "type": "token",
                            "node": nid,
                            "text": output,
                        }
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": output, "status": "ok",
                        "duration_ms": duration_ms,
                    })
                    emit_typed("system", "node_end", status="ok",
                         duration_ms=duration_ms,
                         context={"run_id": run_id,
                                  "system_name": system_name,
                                  "node_id": nid, "agent": agent_name,
                                  "role": role},
                         perf={"output_length":
                               len(output) if output else 0})
                yield {
                    "type": "node_end",
                    "node": nid, "agent": agent_name, "role": role,
                    "text": output, "status": node_status,
                    "duration_ms": duration_ms,
                }
                # Edge routing follows below — same as for
                # non-preloaded nodes (control flow falls into the
                # shared edge block via this if/else's else branch
                # being skipped — see structure: routing block is
                # at the ``for nid`` indent, not nested in the
                # streaming branch).
                # Continue to edge routing — restart the per-node
                # loop body's tail by jumping to the routing
                # section. Implemented via shared edge block below
                # the if/else; we set output + visited and fall
                # through to that block.
            else:
                # [IMPROVE-33] same budgeted context builder as sync.
                # [IMPROVE-165] Wave 31 — same opt-in summarizer
                # as the sync sibling above.
                # [IMPROVE-166] Wave 32 — same per-edge pass
                # config as the sync sibling.
                from ..config import get_settings
                _summarizer = _build_summarizer(
                    orch, get_settings().dag_inter_node_summarization_model,
                )
                _pass_mode = last_pass_per_node.get(
                    nid, _DEFAULT_PASS_MODE,
                )
                _source_id = last_source_per_node.get(nid)
                ctx_block = _build_inter_node_context(
                    node_outputs, budget_tokens=context_budget,
                    summarizer=_summarizer,
                    pass_mode=_pass_mode,
                    source_node_id=_source_id,
                )
                if ctx_block:
                    prompt = (
                        f"{user_input}\n\nContext from prior agents:\n"
                        f"{ctx_block}"
                    )
                else:
                    prompt = user_input

                node_start = _time.monotonic()
                emit_typed("system", "node_start", status="start",
                     context={"run_id": run_id, "system_name": system_name,
                              "node_id": nid, "agent": agent_name,
                              "role": role, "step": step})
                yield {
                    "type": "node_start",
                    "node": nid, "agent": agent_name, "role": role,
                }

                # Stream this node via astream_chat_with_agent. Tag
                # each inner event with the owning node id so
                # consumers can reconstruct per-node sub-streams.
                # The agent's own ``done`` event is consumed here
                # (not re-yielded) — the system-level ``done`` only
                # fires after the whole DAG walk completes.
                output = ""
                node_status = "ok"
                try:
                    async for event in orch.astream_chat_with_agent(
                        agent_name, prompt,
                    ):
                        etype = event.get("type", "")
                        if etype == "token":
                            text = event.get("text", "")
                            if text:
                                output += text
                                yield {
                                    "type": "token",
                                    "node": nid,
                                    "text": text,
                                }
                        elif etype == "tool_call":
                            yield {
                                "type": "tool_call",
                                "node": nid,
                                "name": event.get("name", ""),
                                "args": event.get("args", ""),
                                "call_id": event.get("call_id", ""),
                            }
                        elif etype == "tool_result":
                            yield {
                                "type": "tool_result",
                                "node": nid,
                                "name": event.get("name", ""),
                                "content": event.get("content", ""),
                                "call_id": event.get("call_id", ""),
                            }
                        elif etype == "done":
                            # Prefer the inner stream's ``content``
                            # when we collected nothing via tokens
                            # (path A of astream_chat_with_agent
                            # yields tokens naturally; path B may
                            # emit a synthetic "No response
                            # returned." token. Either way,
                            # ``content`` is the canonical full
                            # text).
                            if not output:
                                output = event.get("content", "") or ""
                    duration_ms = int(
                        (_time.monotonic() - node_start) * 1000,
                    )
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": output, "status": "ok",
                        "duration_ms": duration_ms,
                    })
                    emit_typed("system", "node_end", status="ok",
                         duration_ms=duration_ms,
                         context={"run_id": run_id,
                                  "system_name": system_name,
                                  "node_id": nid, "agent": agent_name,
                                  "role": role},
                         perf={"output_length":
                               len(output) if output else 0})
                    # [IMPROVE-33] context block is rebuilt per-node
                    # from node_outputs so we no longer maintain a
                    # parallel accumulator string here.
                except Exception as exc:
                    duration_ms = int(
                        (_time.monotonic() - node_start) * 1000,
                    )
                    node_status = "error"
                    output = str(exc)
                    node_outputs.append({
                        "node": nid, "agent": agent_name, "role": role,
                        "text": output, "status": "error",
                        "duration_ms": duration_ms,
                    })
                    emit_typed("system", "node_end", status="error",
                         duration_ms=duration_ms,
                         error_code=type(exc).__name__,
                         error_message=str(exc),
                         context={"run_id": run_id,
                                  "system_name": system_name,
                                  "node_id": nid, "agent": agent_name,
                                  "role": role})

                yield {
                    "type": "node_end",
                    "node": nid, "agent": agent_name, "role": role,
                    "text": output, "status": node_status,
                    "duration_ms": duration_ms,
                }

            # Edge routing — same rules as execute_graph
            # (IMPROVE-35 added llm_router with shared classification).
            chosen_option = classify_llm_router_edges(
                orch, edge_map.get(nid, []), output, visited,
            )

            # [IMPROVE-35 telemetry] Surface the classifier
            # decision in the SSE stream when at least one
            # llm_router edge exists out of the current node, so
            # Flutter can render "Router chose: writer" alongside
            # the next-node activation. Emit BEFORE the per-edge
            # iteration so the consumer sees the decision before
            # the first next-node node_start.
            _llm_router_targets = [
                tgt for tgt, rt, _, _ in edge_map.get(nid, [])
                if rt == "llm_router"
            ]
            if _llm_router_targets:
                yield {
                    "type": "routing_decision",
                    "node": nid,
                    "chosen_option": chosen_option,
                    "candidates": list(_llm_router_targets),
                    "rule_count": len(_llm_router_targets),
                }
                emit_typed(
                    "system", "routing_decision", status="ok",
                    context={
                        "run_id": run_id,
                        "system_name": system_name,
                        "node_id": nid,
                        "chosen_option": chosen_option,
                        "candidates": list(_llm_router_targets),
                        "rule_count": len(_llm_router_targets),
                    },
                )

            for target, rule_type, rule_notes, _rule in edge_map.get(nid, []):
                if target in visited:
                    continue
                if _evaluate_edge_rule(
                    rule_type, rule_notes, output,
                    chosen_option, target,
                ):
                    # [IMPROVE-166] Wave 32 — same per-target
                    # tracking as the sync executor.
                    last_pass_per_node[target] = _rule.get(
                        "pass", _DEFAULT_PASS_MODE,
                    )
                    last_source_per_node[target] = nid
                    if target not in next_nodes:
                        next_nodes.append(target)

        current_nodes = next_nodes

    total_ms = int((_time.monotonic() - total_start) * 1000)
    final_text = (
        node_outputs[-1]["text"] if node_outputs
        else "No output produced."
    )

    errors = sum(1 for n in node_outputs if n.get("status") == "error")
    emit_typed("system", "run_done", status="error" if errors else "ok",
         duration_ms=total_ms,
         context={"run_id": run_id, "system_name": system_name,
                  "conversation_id": conversation_id,
                  "streaming": True},
         perf={
             "nodes_executed": len(node_outputs),
             "error_count": errors,
             "final_text_length": len(final_text) if final_text else 0,
             "steps": step,
             # [IMPROVE-83] Same per-run aggregates as the sync
             # executor so dashboards charting parallel engagement
             # don't have to special-case the streaming path.
             "parallel_waves_used": parallel_waves_used,
             "concurrent_nodes_total": concurrent_nodes_total,
             "parallel_waves_skipped": parallel_waves_skipped,
         })

    yield {
        "type": "done",
        "final_text": final_text,
        "node_outputs": node_outputs,
        "conversation_id": conversation_id,
        "run_id": run_id,
        "total_duration_ms": total_ms,
        "nodes_executed": len(node_outputs),
    }
