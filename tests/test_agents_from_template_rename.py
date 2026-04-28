"""[IMPROVE-34] Tests for the /systems/deploy → /agents/from-template rename.

The doc complaint (``docs/features/05-systems.md:417-423``):
``POST /systems/deploy/{template_id}`` creates an *agent*, not a
system. The URL implied a data model truth that wasn't there. The
rename surfaces that the operation produces an agent — anyone reading
the API surface now sees the right object created.

The shape of this commit:

  * Pure helper ``system_templates.deploy_template_as_agent`` —
    accepts the template id, optional override body, and an
    orchestrator. Returns the response payload. Raises ``KeyError``
    on unknown templates so the HTTP layer can map to 404.
  * Canonical route ``POST /agents/from-template/{template_id}`` in
    ``routers/agents.py`` — thin wrapper around the helper.
  * Deprecated alias ``POST /systems/deploy/{template_id}`` in
    ``routers/systems.py`` — same wrapper plus three RFC headers
    (RFC 9745 Deprecation, RFC 8594 Sunset, RFC 8288 Link with
    rel="successor-version").

Both routes call into the SAME helper so behaviour cannot drift —
``test_aliases_produce_equivalent_agent`` enforces this.

The Flutter caller (``flutter_client/lib/pages/systems_page.dart``)
keeps hitting the old URL until a separate Flutter-migration ticket
moves it. The deprecation headers give the migration window
(2026-04-28 → 2026-10-28).

Sources:
  * RFC 9745 (Deprecation HTTP header field, IETF 2024-12) —
    https://www.rfc-editor.org/rfc/rfc9745
  * RFC 8594 (Sunset HTTP header field, IETF 2019-05) —
    https://www.rfc-editor.org/rfc/rfc8594
  * RFC 8288 (Web Linking) ``rel="successor-version"`` —
    https://www.rfc-editor.org/rfc/rfc8288
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest


# ── Helper unit tests (Pydantic-free, fast) ──────────────────────────


@pytest.fixture
def tmp_db(monkeypatch, tmp_path):
    """Redirect ``db.DB_PATH`` to a tmp file + initialize tables so
    ``save_agent`` (which the helper calls) doesn't pollute the dev DB
    or fail on a missing table. Each test starts with empty agents."""
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()
    return path


@pytest.fixture
def fake_orchestrator():
    """A MagicMock with the two methods the helper calls. Lets us
    assert what got passed in without booting the real orchestrator
    (which would pull provider configs + tool registries)."""
    orch = MagicMock()
    return orch


def test_helper_known_template_returns_canonical_payload(tmp_db, fake_orchestrator):
    """Happy path: the helper returns the expected response shape and
    invokes orchestrator.add_agent + set_agent_tools + save_agent."""
    from local_ai_platform.system_templates import deploy_template_as_agent

    result = deploy_template_as_agent(
        "general_assistant", None, fake_orchestrator,
    )

    assert result["status"] == "deployed"
    assert result["agent"] == "general_assistant"
    assert result["template"] == "general_assistant"
    # Tools list mirrors the template's tool_ids (general_assistant has 4).
    assert isinstance(result["tools"], list)
    assert len(result["tools"]) > 0

    # Orchestrator received the call.
    fake_orchestrator.add_agent.assert_called_once()
    fake_orchestrator.set_agent_tools.assert_called_once()


def test_helper_unknown_template_raises_keyerror(tmp_db, fake_orchestrator):
    """The helper signals "no such template" via KeyError so the HTTP
    layer can map to 404 without owning template lookup logic."""
    from local_ai_platform.system_templates import deploy_template_as_agent

    with pytest.raises(KeyError) as ei:
        deploy_template_as_agent("does_not_exist", None, fake_orchestrator)
    assert "does_not_exist" in str(ei.value)
    # No partial side effects on missing template.
    fake_orchestrator.add_agent.assert_not_called()


def test_helper_honors_name_override(tmp_db, fake_orchestrator):
    """Body ``{"name": ...}`` overrides the default agent name (=
    template id). Pinned because Flutter relies on it for the
    "deploy second copy" workflow."""
    from local_ai_platform.system_templates import deploy_template_as_agent

    result = deploy_template_as_agent(
        "research_assistant",
        {"name": "my_custom_researcher"},
        fake_orchestrator,
    )
    assert result["agent"] == "my_custom_researcher"
    # The orchestrator got the override too.
    kwargs = fake_orchestrator.add_agent.call_args.kwargs
    assert kwargs["name"] == "my_custom_researcher"


def test_helper_honors_model_override(tmp_db, fake_orchestrator):
    """Body ``{"model_name": ...}`` overrides the template's default
    (first recommended model)."""
    from local_ai_platform.system_templates import deploy_template_as_agent

    deploy_template_as_agent(
        "research_assistant",
        {"model_name": "qwen2.5:14b"},
        fake_orchestrator,
    )
    kwargs = fake_orchestrator.add_agent.call_args.kwargs
    assert kwargs["model_name"] == "qwen2.5:14b"


def test_helper_honors_provider_override(tmp_db, fake_orchestrator):
    """Body ``{"provider": ...}`` overrides the default ``ollama``."""
    from local_ai_platform.system_templates import deploy_template_as_agent

    deploy_template_as_agent(
        "research_assistant",
        {"provider": "openai_compatible"},
        fake_orchestrator,
    )
    kwargs = fake_orchestrator.add_agent.call_args.kwargs
    assert kwargs["provider"] == "openai_compatible"


def test_helper_applies_template_tools_when_present(tmp_db, fake_orchestrator):
    """``set_agent_tools`` is called when the template defines any.
    coding_assistant has 7 tool_ids — pin that they're forwarded."""
    from local_ai_platform.system_templates import (
        deploy_template_as_agent,
        get_template,
    )

    deploy_template_as_agent("coding_assistant", None, fake_orchestrator)

    fake_orchestrator.set_agent_tools.assert_called_once()
    call_args = fake_orchestrator.set_agent_tools.call_args
    assert call_args.args[0] == "coding_assistant"
    expected_tools = get_template("coding_assistant").tool_ids
    assert call_args.args[1] == expected_tools


def test_helper_default_model_when_recommended_list_empty(
    tmp_db, fake_orchestrator, monkeypatch,
):
    """When the template has no recommended_models, the helper falls
    back to ``gemma3:4b`` (matches the pre-IMPROVE-34 inline default
    so old saves don't drift)."""
    from local_ai_platform import system_templates as st_mod
    from local_ai_platform.system_templates import deploy_template_as_agent

    # Build a stub template with empty recommended_models, register it
    # in the lookup dict via monkeypatch so the helper sees it.
    stub = st_mod.SystemTemplate(
        id="stub_no_models",
        name="Stub",
        description="No models",
        icon="x",
        category="general",
        system_prompt="you are stub",
        tool_ids=[],
        recommended_models=[],
        default_settings={},
    )
    monkeypatch.setitem(st_mod.TEMPLATES_BY_ID, "stub_no_models", stub)

    deploy_template_as_agent("stub_no_models", None, fake_orchestrator)

    kwargs = fake_orchestrator.add_agent.call_args.kwargs
    assert kwargs["model_name"] == "gemma3:4b"


def test_helper_skips_set_agent_tools_when_template_has_none(
    tmp_db, fake_orchestrator, monkeypatch,
):
    """Templates with empty tool_ids skip the ``set_agent_tools``
    call. Mirrors the pre-IMPROVE-34 ``if template.tool_ids:`` guard
    — saving a no-tool agent shouldn't leave a stray empty bind-row
    in the agent_tools table."""
    from local_ai_platform import system_templates as st_mod
    from local_ai_platform.system_templates import deploy_template_as_agent

    stub = st_mod.SystemTemplate(
        id="stub_no_tools",
        name="Stub",
        description="No tools",
        icon="x",
        category="general",
        system_prompt="you are stub",
        tool_ids=[],
        recommended_models=["gemma3:1b"],
        default_settings={},
    )
    monkeypatch.setitem(st_mod.TEMPLATES_BY_ID, "stub_no_tools", stub)

    deploy_template_as_agent("stub_no_tools", None, fake_orchestrator)

    fake_orchestrator.add_agent.assert_called_once()
    fake_orchestrator.set_agent_tools.assert_not_called()


def test_helper_persists_agent_with_template_id_field(tmp_db, fake_orchestrator):
    """``template_id`` is written to the saved agent JSON so the UI
    can show a "from template" badge. Pin it because the Flutter agent
    list relies on this field surviving the save."""
    from local_ai_platform.repositories.agents_repo import get_agent_db
    from local_ai_platform.system_templates import deploy_template_as_agent

    deploy_template_as_agent(
        "general_assistant",
        {"name": f"agent_{uuid.uuid4().hex[:8]}"},
        fake_orchestrator,
    )
    # Pull the most-recently-saved agent name from the call. The DB
    # row wraps the actual definition under ``json_definition`` —
    # ``template_id`` lives inside that nested dict, not at the top
    # level of the row.
    saved_name = fake_orchestrator.add_agent.call_args.kwargs["name"]
    row = get_agent_db(saved_name)
    assert row is not None
    assert row["json_definition"].get("template_id") == "general_assistant"


# ── Route integration via TestClient ────────────────────────────────


@pytest.fixture
def client(monkeypatch, tmp_path):
    """In-process TestClient against the real api_server.app with a
    tmp DB so deploys don't pollute the dev DB. Same pattern as
    test_systems_schema_validation.py."""
    from fastapi.testclient import TestClient
    from local_ai_platform import db as db_mod

    path = tmp_path / "app.db"
    monkeypatch.setattr(db_mod, "DB_PATH", path)
    db_mod.init_db()

    import api_server
    with TestClient(api_server.app) as c:
        yield c


def _unique(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def test_new_route_happy_path(client):
    """``POST /agents/from-template/{tid}`` deploys a known template
    and returns the canonical payload."""
    name = _unique("res")
    resp = client.post(
        "/agents/from-template/research_assistant",
        json={"name": name, "model_name": "gemma3:1b"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "deployed"
    assert body["agent"] == name
    assert body["template"] == "research_assistant"
    assert isinstance(body["tools"], list)


def test_new_route_404_on_unknown_template(client):
    """Unknown template id → 404 with template name in the message."""
    resp = client.post("/agents/from-template/no_such_template", json={})
    assert resp.status_code == 404
    body = resp.json()
    detail = body.get("detail") or body
    assert "no_such_template" in str(detail)


def test_new_route_does_not_emit_deprecation_headers(client):
    """The canonical route is brand-new — no Deprecation/Sunset
    headers should appear. Pinned so a future refactor doesn't
    accidentally apply the alias headers to the new route."""
    name = _unique("res")
    resp = client.post(
        "/agents/from-template/research_assistant",
        json={"name": name, "model_name": "gemma3:1b"},
    )
    assert resp.status_code == 200
    assert "deprecation" not in {k.lower() for k in resp.headers.keys()}
    assert "sunset" not in {k.lower() for k in resp.headers.keys()}


def test_new_route_honors_name_override(client):
    """Body name override surfaces in the saved agent + response."""
    custom = _unique("custom")
    resp = client.post(
        "/agents/from-template/general_assistant",
        json={"name": custom, "model_name": "gemma3:1b"},
    )
    assert resp.status_code == 200
    assert resp.json()["agent"] == custom


def test_new_route_honors_model_override(client):
    """Body model_name flows into the saved agent definition."""
    name = _unique("mod")
    resp = client.post(
        "/agents/from-template/general_assistant",
        json={"name": name, "model_name": "qwen2.5:14b"},
    )
    assert resp.status_code == 200

    from local_ai_platform.repositories.agents_repo import get_agent_db
    row = get_agent_db(name)
    assert row is not None
    # ``json_definition`` is the persisted dict; the model_name field
    # lives inside it (not at the row's top level).
    assert row["json_definition"].get("model_name") == "qwen2.5:14b"


def test_old_alias_still_works(client):
    """The deprecated ``POST /systems/deploy/{tid}`` alias must keep
    returning 200 + canonical payload until sunset (2026-10-28).
    Removing it before then breaks the Flutter client."""
    name = _unique("legacy")
    resp = client.post(
        "/systems/deploy/research_assistant",
        json={"name": name, "model_name": "gemma3:1b"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "deployed"
    assert body["agent"] == name
    assert body["template"] == "research_assistant"


def test_old_alias_returns_deprecation_header(client):
    """RFC 9745 ``Deprecation: true`` must be on every alias response
    so HTTP-level audit tools can flag the calls without parsing
    the body."""
    name = _unique("dep")
    resp = client.post(
        "/systems/deploy/general_assistant",
        json={"name": name, "model_name": "gemma3:1b"},
    )
    assert resp.status_code == 200
    # Header lookup is case-insensitive in starlette/httpx but be
    # explicit just in case the contract changes upstream.
    assert resp.headers.get("Deprecation") == "true"


def test_old_alias_returns_sunset_header(client):
    """RFC 8594 ``Sunset`` must be the IMF-fixdate at which the alias
    is removed. Pinned because changing the date silently would let
    clients miss the migration window."""
    name = _unique("sun")
    resp = client.post(
        "/systems/deploy/general_assistant",
        json={"name": name, "model_name": "gemma3:1b"},
    )
    assert resp.status_code == 200
    sunset = resp.headers.get("Sunset")
    assert sunset is not None
    # Spot-check the date format (IMF-fixdate per RFC 7231 §7.1.1.1).
    assert "2026" in sunset
    assert "Oct" in sunset
    assert "GMT" in sunset


def test_old_alias_returns_link_header_with_successor(client):
    """RFC 8288 ``Link`` with ``rel="successor-version"`` is the
    machine-discoverable replacement URL. Clients (or API gateways)
    can follow it without hard-coding the new path."""
    name = _unique("lnk")
    resp = client.post(
        "/systems/deploy/general_assistant",
        json={"name": name, "model_name": "gemma3:1b"},
    )
    assert resp.status_code == 200
    link = resp.headers.get("Link")
    assert link is not None
    assert "/agents/from-template/general_assistant" in link
    assert 'rel="successor-version"' in link


def test_old_alias_404_on_unknown_template(client):
    """Unknown template id still maps to 404 on the alias."""
    resp = client.post("/systems/deploy/ghost_template", json={})
    assert resp.status_code == 404


def test_aliases_produce_equivalent_agent(client):
    """The CRITICAL invariant: deploying the same template via both
    routes yields byte-equivalent saved agents. Without this pin a
    future refactor could accidentally drift the alias behaviour
    while the deprecation window is still open, and Flutter clients
    using the old URL would see different data than callers on the
    new URL."""
    from local_ai_platform.repositories.agents_repo import get_agent_db

    new_name = _unique("via_new")
    old_name = _unique("via_old")

    resp_new = client.post(
        "/agents/from-template/general_assistant",
        json={"name": new_name, "model_name": "gemma3:1b"},
    )
    resp_old = client.post(
        "/systems/deploy/general_assistant",
        json={"name": old_name, "model_name": "gemma3:1b"},
    )

    assert resp_new.status_code == 200
    assert resp_old.status_code == 200

    via_new = get_agent_db(new_name)
    via_old = get_agent_db(old_name)
    assert via_new is not None and via_old is not None

    # The agent definition lives under ``json_definition``; compare
    # the persisted contents (modulo the agent name, which differs by
    # construction) field-by-field.
    defn_new = via_new["json_definition"]
    defn_old = via_old["json_definition"]
    fields_to_compare = (
        "model_name", "system_prompt", "provider",
        "role", "tool_ids", "template_id",
    )
    for f in fields_to_compare:
        assert defn_new.get(f) == defn_old.get(f), (
            f"Drift detected on field '{f}': new={defn_new.get(f)!r} "
            f"vs old={defn_old.get(f)!r}"
        )


def test_canonical_route_is_registered_under_agents_prefix():
    """Static route-table check: the new route lives under /agents/
    so the URL spelling matches what the doc promised. Catches a
    refactor that puts it back under /systems/ by accident."""
    import api_server
    paths = {r.path for r in api_server.app.routes if hasattr(r, "path")}
    assert "/agents/from-template/{template_id}" in paths
    # Old alias is still registered (until sunset).
    assert "/systems/deploy/{template_id}" in paths
