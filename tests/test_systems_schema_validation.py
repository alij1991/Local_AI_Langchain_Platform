r"""[IMPROVE-31] Tests for Pydantic schema validation at /systems boundary.

Pre-IMPROVE-31 ``POST /systems``, ``PUT /systems/{name}``,
``POST /systems/{name}/clone``, and ``POST /systems/import`` accepted
``body: dict[str, Any]`` and only ran cycle detection (IMPROVE-37) at
the route boundary. Malformed definitions slipped through and crashed
deep in ``execute_system_graph`` with ``KeyError`` / ``TypeError``:

  * ``{"nodes": [{"x": 1}]}`` — no ``id`` → KeyError on dict access
  * Duplicate node ids → executor silently overwrote in node_map
  * Edges referencing nonexistent nodes → executor silently dropped
  * Wrong field types (e.g. ``nodes: "not a list"``) → TypeError

Zero pre-IMPROVE-31 schema-validation test coverage on the routes —
verified by ``grep -r "schema_invalid\|orphan_edge" tests/`` returning
empty.

This commit adds:
  * ``SystemDefinition`` Pydantic model in ``systems_validator.py``.
  * ``validate_definition_schema(definition)`` helper that wraps
    ``model_validate`` + maps ``ValidationError`` into the existing
    ``SystemValidationError`` with structured ``errors`` field.
  * Updated ``_validate_system_or_400`` to run schema check FIRST
    (fields-out-of-shape errors are more actionable than cycles).

Tests cover:
  * Schema unit-level (Pydantic-only, no DB)
  * ``validate_definition_schema`` helper round-trip
  * Route integration via TestClient — POST/PUT/clone/import all
    reject schema-invalid definitions with structured 400 responses
  * Backward-compat: extra fields preserved, camelCase aliases
    accepted, unknown rule types accepted
"""
from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

from local_ai_platform.systems_validator import (
    SystemDefinition,
    SystemValidationError,
    _EdgeRule,
    _SystemEdge,
    _SystemNode,
    validate_definition_schema,
)


# ── Schema unit tests (Pydantic-only) ────────────────────────────────


def test_minimal_valid_definition_round_trips():
    """A 2-node 1-edge graph with all required fields validates and
    serializes back."""
    definition = {
        "nodes": [
            {"id": "a", "agent": "writer"},
            {"id": "b", "agent": "reviewer"},
        ],
        "edges": [{"source": "a", "target": "b"}],
    }
    sd = SystemDefinition.model_validate(definition)
    assert len(sd.nodes) == 2
    assert sd.nodes[0].id == "a"
    assert sd.edges[0].source == "a"
    assert sd.edges[0].target == "b"
    # Default rule is "always".
    assert sd.edges[0].rule.type == "always"


def test_empty_definition_accepted():
    """``{"nodes": [], "edges": []}`` is valid — the executor returns
    "System has no agent nodes" gracefully for empty graphs, so we
    accept the saved-stub state too."""
    sd = SystemDefinition.model_validate({"nodes": [], "edges": []})
    assert sd.nodes == []
    assert sd.edges == []


def test_completely_empty_dict_accepted():
    """Even ``{}`` is valid — the model fills in default empty
    arrays. Pin so legacy "send the whole body" callers don't break."""
    sd = SystemDefinition.model_validate({})
    assert sd.nodes == []
    assert sd.edges == []
    assert sd.start_node_id is None


def test_missing_node_id_rejected():
    """Pre-IMPROVE-31 this slipped through and crashed the executor
    with ``KeyError: 'id'`` deep inside the walker."""
    with pytest.raises(Exception) as exc_info:
        SystemDefinition.model_validate({
            "nodes": [{"agent": "writer"}],
            "edges": [],
        })
    assert "id" in str(exc_info.value).lower()


def test_empty_node_id_rejected():
    """Empty-string id is a more subtle case — pre-IMPROVE-31 the
    cycle check happily walked past it and the executor's node_map
    contained a key of "" that masked typos in edges."""
    with pytest.raises(Exception):
        SystemDefinition.model_validate({
            "nodes": [{"id": "", "agent": "writer"}],
            "edges": [],
        })


def test_duplicate_node_ids_rejected():
    """Pre-IMPROVE-31 the executor's ``node_map = {n["id"]: n for n
    in nodes}`` silently dropped the earlier-defined node when a
    duplicate id appeared. Now it's a 400 at save time."""
    with pytest.raises(Exception) as exc_info:
        SystemDefinition.model_validate({
            "nodes": [
                {"id": "a", "agent": "x"},
                {"id": "a", "agent": "y"},  # dup
            ],
            "edges": [],
        })
    assert "duplicate" in str(exc_info.value).lower()


def test_edge_references_unknown_source_rejected():
    """Pre-IMPROVE-31 ``check_no_cycles`` deferred orphan-edge
    detection to IMPROVE-31 explicitly (see systems_validator.py
    line 46-48). Now it's a 400."""
    with pytest.raises(Exception) as exc_info:
        SystemDefinition.model_validate({
            "nodes": [{"id": "a", "agent": "x"}],
            "edges": [{"source": "ghost", "target": "a"}],
        })
    assert "unknown node" in str(exc_info.value).lower()


def test_edge_references_unknown_target_rejected():
    with pytest.raises(Exception) as exc_info:
        SystemDefinition.model_validate({
            "nodes": [{"id": "a", "agent": "x"}],
            "edges": [{"source": "a", "target": "ghost"}],
        })
    assert "unknown node" in str(exc_info.value).lower()


def test_start_node_id_must_reference_existing_node():
    """The executor today reads ``start_node_id`` and falls back to
    in_degree-based start when missing. Pin: a save-time mismatch
    is rejected so the user sees the typo immediately, not as a
    silent fallback to the wrong start node at runtime."""
    with pytest.raises(Exception) as exc_info:
        SystemDefinition.model_validate({
            "nodes": [{"id": "a", "agent": "x"}],
            "edges": [],
            "start_node_id": "ghost",
        })
    assert "start_node_id" in str(exc_info.value).lower()


def test_start_node_id_camelcase_accepted_alias():
    """Pin backward-compat: ``startNodeId`` (camelCase, what the
    executor accepts as a legacy fallback per agents.py:1375) is
    accepted as input. Output normalizes to snake_case."""
    sd = SystemDefinition.model_validate({
        "nodes": [{"id": "a", "agent": "x"}],
        "edges": [],
        "startNodeId": "a",
    })
    assert sd.start_node_id == "a"


def test_unknown_rule_type_accepted_for_forward_compat():
    """The executor falls through to ``always`` for unknown rule
    types (agents.py:1512: "unknown rule = always follow"). Pinning
    ``Literal`` would reject future rule types the executor would
    otherwise honor — so we accept arbitrary strings here. Pin via
    a hypothetical future ``"on_score_above"`` type."""
    sd = SystemDefinition.model_validate({
        "nodes": [
            {"id": "a", "agent": "x"},
            {"id": "b", "agent": "y"},
        ],
        "edges": [{
            "source": "a", "target": "b",
            "rule": {"type": "on_score_above", "notes": "score > 0.8"},
        }],
    })
    assert sd.edges[0].rule.type == "on_score_above"


def test_legacy_top_level_ruletype_accepted():
    """Pre-IMPROVE-31 saved rows used edge.ruleType (camelCase, top
    level) — kept here for round-trip compat. The executor reads
    both shapes via ``rule.get("type", e.get("ruleType", "always"))``
    at agents.py:1388."""
    sd = SystemDefinition.model_validate({
        "nodes": [
            {"id": "a", "agent": "x"},
            {"id": "b", "agent": "y"},
        ],
        "edges": [{
            "source": "a", "target": "b",
            "ruleType": "manual_next",  # legacy
            "notes": "user clicks Next",
        }],
    })
    assert sd.edges[0].ruleType == "manual_next"


def test_extra_fields_preserved_for_layout():
    """Flutter ships extra UI state (``selected``, ``color``,
    ``comment``) on nodes; ``extra="allow"`` lets these pass through
    schema validation without being rejected. Pin via concrete
    fields so a refactor that switches to ``extra="forbid"``
    breaks loudly."""
    sd = SystemDefinition.model_validate({
        "nodes": [{
            "id": "a", "agent": "x",
            "x": 100.0, "y": 200.0,
            "selected": True, "color": "#ff00ff",
        }],
        "edges": [],
    })
    # x, y are first-class fields.
    assert sd.nodes[0].x == 100.0
    assert sd.nodes[0].y == 200.0
    # Extras survive — the model_dump round-trip preserves them.
    dumped = sd.nodes[0].model_dump()
    assert dumped.get("selected") is True
    assert dumped.get("color") == "#ff00ff"


def test_node_role_at_top_level_or_in_config():
    """The executor reads role from ``config.role`` first then from
    top-level ``role`` (agents.py:1439). Both are accepted as input;
    keep both fields explicit on the model so a Flutter client
    using either spelling validates."""
    sd1 = SystemDefinition.model_validate({
        "nodes": [{"id": "a", "agent": "x", "role": "writer"}],
        "edges": [],
    })
    assert sd1.nodes[0].role == "writer"
    sd2 = SystemDefinition.model_validate({
        "nodes": [{"id": "a", "agent": "x", "config": {"role": "writer"}}],
        "edges": [],
    })
    assert sd2.nodes[0].config.get("role") == "writer"


# ── validate_definition_schema helper ────────────────────────────────


def test_validate_helper_accepts_valid_definition():
    """The helper returns silently on valid input."""
    validate_definition_schema({
        "nodes": [{"id": "a", "agent": "x"}],
        "edges": [],
    })  # no raise


def test_validate_helper_raises_with_structured_errors():
    """On failure, the helper raises ``SystemValidationError`` with
    a structured ``errors`` list — each entry has ``field``, ``msg``,
    ``type`` keys ready to ship in a 400 JSON body."""
    with pytest.raises(SystemValidationError) as exc_info:
        validate_definition_schema({
            "nodes": [{"agent": "x"}],  # missing id
            "edges": [],
        })
    err = exc_info.value
    assert isinstance(err.errors, list)
    assert len(err.errors) >= 1
    assert "id" in err.errors[0]["field"]
    assert "field" in err.errors[0]
    assert "msg" in err.errors[0]
    assert "type" in err.errors[0]
    # cyclic_nodes is empty for schema errors (as opposed to cycle
    # errors) — confirming the two failure-mode classes don't
    # collide via the shared exception class.
    assert err.cyclic_nodes == []


def test_validate_helper_accepts_none_as_empty_graph():
    """Legacy callers pass ``None`` when the body has no
    ``definition`` key. The helper treats it as ``{}`` so empty-graph
    semantics flow through cleanly."""
    validate_definition_schema(None)  # no raise


def test_validate_helper_aggregates_multiple_errors():
    """Pydantic aggregates multiple validation errors into one
    ValidationError. The wrapper preserves all of them in
    ``err.errors`` so the user can fix them in one round-trip."""
    with pytest.raises(SystemValidationError) as exc_info:
        validate_definition_schema({
            "nodes": [
                {"agent": "x"},  # missing id
                {"agent": "y"},  # missing id
            ],
            "edges": [],
        })
    assert len(exc_info.value.errors) == 2


# ── Route integration via TestClient ────────────────────────────────


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient against the real api_server.app with a
    tmp DB so cycle/schema rejections don't pollute the dev DB."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()

    import api_server
    with TestClient(api_server.app) as c:
        yield c


def _unique_name(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_post_systems_400_for_missing_node_id(client):
    """Pre-IMPROVE-31 this would 200 then crash the executor on
    first chat. Now it's a 400 with field path."""
    name = _unique_name("missing_id")
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [{"agent": "x"}],  # no id
            "edges": [],
        },
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    assert body["error"] == "schema_invalid"
    # Errors list contains the missing-id pointer.
    assert any("id" in e["field"] for e in body["errors"])


def test_post_systems_400_for_orphan_edge(client):
    """Pre-IMPROVE-31 ``check_no_cycles`` silently dropped orphan
    edges (deferred-to-IMPROVE-31 in systems_validator.py:46-48)."""
    name = _unique_name("orphan_edge")
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [{"id": "a", "agent": "x"}],
            "edges": [{"source": "a", "target": "ghost"}],
        },
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    assert body["error"] == "schema_invalid"
    assert any("unknown node" in e["msg"].lower() for e in body["errors"])


def test_post_systems_200_for_valid_definition(client):
    """Happy path: a valid definition saves successfully."""
    name = _unique_name("valid")
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [
                {"id": "writer", "agent": "writer"},
                {"id": "reviewer", "agent": "reviewer"},
            ],
            "edges": [{"source": "writer", "target": "reviewer"}],
        },
    })
    assert resp.status_code == 200, resp.text


def test_put_systems_400_for_duplicate_node_ids(client):
    """PUT path also schema-validated. Pre-IMPROVE-31 the executor
    silently overwrote the earlier node in node_map."""
    name = _unique_name("dup_ids")
    resp = client.put(f"/systems/{name}", json={
        "definition": {
            "nodes": [
                {"id": "a", "agent": "x"},
                {"id": "a", "agent": "y"},  # dup
            ],
            "edges": [],
        },
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    assert body["error"] == "schema_invalid"


def test_post_systems_import_400_for_schema_invalid(client):
    """Import path also schema-validated. Pin so a malformed export
    file can't sneak past validation when re-imported."""
    resp = client.post("/systems/import", json={
        "name": _unique_name("import_bad"),
        "definition": {
            "nodes": [{"agent": "x"}],  # no id
            "edges": [],
        },
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    assert body["error"] == "schema_invalid"


def test_clone_system_400_when_source_definition_invalid(client):
    """Defensive: a legacy bad row in DB shouldn't propagate via
    clone. The clone path explicitly validates per
    routers/systems.py:634-637 ("a legacy cyclic row in DB should
    not propagate into a new row via clone"). Same principle for
    schema invalidity. We seed a bad row directly via the DB to
    simulate a pre-IMPROVE-31 broken save."""
    from local_ai_platform.db import get_conn

    bad_name = _unique_name("legacy_bad")
    conn = get_conn()
    try:
        # Bypass the validator — directly insert a malformed
        # definition_json. This simulates a row from before
        # IMPROVE-31 shipped that's now dormant in DB.
        bad_def_json = json.dumps({
            "nodes": [{"agent": "x"}],  # no id — schema-invalid
            "edges": [],
        })
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO systems (name, definition_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?)",
            (bad_name, bad_def_json, now, now),
        )
        conn.commit()
    finally:
        conn.close()

    # Clone fails — the source is schema-invalid.
    resp = client.post(f"/systems/{bad_name}/clone", json={
        "new_name": _unique_name("clone_target"),
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    assert body["error"] == "schema_invalid"


def test_schema_check_runs_before_cycle_check(client):
    """Pin order-of-checks: a definition that's BOTH schema-invalid
    AND cyclic surfaces the schema error first (more actionable —
    the user fixes the missing field before thinking about graph
    topology)."""
    name = _unique_name("schema_then_cycle")
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [
                # Missing id on the first node — schema invalid.
                {"agent": "x"},
                {"id": "b", "agent": "y"},
            ],
            # Self-cycle on b — also invalid (would be cycle if shape
            # was fine).
            "edges": [{"source": "b", "target": "b"}],
        },
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    # Schema error wins — NOT cycle.
    assert body["error"] == "schema_invalid"
    assert "cyclic_nodes" not in body  # cycle response shape NOT used


def test_post_systems_still_rejects_cycles(client):
    """Backward-compat pin: the IMPROVE-37 cycle check still fires
    when the schema is valid but the topology has a cycle. This is
    NOT a regression of the pre-IMPROVE-31 cycle-rejection contract."""
    name = _unique_name("cyclic")
    resp = client.post("/systems", json={
        "name": name,
        "definition": {
            "nodes": [
                {"id": "a", "agent": "x"},
                {"id": "b", "agent": "y"},
            ],
            "edges": [
                {"source": "a", "target": "b"},
                {"source": "b", "target": "a"},  # cycle
            ],
        },
    })
    assert resp.status_code == 400
    body = resp.json().get("detail") or resp.json()
    # Cycle error shape — NOT schema_invalid.
    assert body["error"] == "cycle_detected"
    assert "cyclic_nodes" in body
    assert sorted(body["cyclic_nodes"]) == ["a", "b"]
