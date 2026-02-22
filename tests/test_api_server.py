import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import api_server


client = TestClient(api_server.app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_model_catalog_provider_unavailable(monkeypatch):
    monkeypatch.setattr(api_server.controller, 'list_local_models_detailed', lambda: (False, [], 'offline'))
    response = client.get('/model-catalog?provider=ollama')
    assert response.status_code == 200
    items = response.json()['items']
    assert items
    assert items[0]['provider_unavailable'] is True


def test_agents_crud_and_validation(monkeypatch):
    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())

    bad = client.post('/agents', json={'name': 'a', 'provider': 'ollama', 'model_id': 'x', 'tool_ids': ['missing']})
    assert bad.status_code == 400

    create = client.post('/agents', json={
        'name': 'agent-a',
        'description': 'test',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are helpful',
        'tool_ids': [],
        'settings': {'temperature': 0.2},
        'resource_limits': {'max_context_messages': 20},
    })
    assert create.status_code == 200

    read = client.get('/agents/agent-a')
    assert read.status_code == 200
    assert read.json()['model_id'] == 'gemma3:1b'

    update = client.put('/agents/agent-a', json={
        'name': 'agent-a',
        'description': 'test2',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are better',
        'tool_ids': [],
        'settings': {'temperature': 0.1},
        'resource_limits': {'max_context_messages': 10},
    })
    assert update.status_code == 200

    eff = client.get('/agents/agent-a/effective-config')
    assert eff.status_code == 200
    assert 'settings' in eff.json()

    delete = client.delete('/agents/agent-a')
    assert delete.status_code == 200


def test_tools_crud_and_tavily_disabled_state():
    create = client.post('/tools', json={
        'name': 'tavily_web_search',
        'type': 'tavily',
        'description': 'search',
        'config_json': {'max_results': 5},
        'is_enabled': True,
    })
    assert create.status_code == 200
    tid = create.json()['tool_id']

    status = client.get('/tools/status')
    assert status.status_code == 200
    reasons = [i['reason'] for i in status.json()['items'] if i['tool_id'] == tid]
    assert reasons

    read = client.get(f'/tools/{tid}')
    assert read.status_code == 200

    upd = client.put(f'/tools/{tid}', json={
        'name': 'tavily_web_search',
        'type': 'tavily',
        'description': 'search-updated',
        'config_json': {'max_results': 3},
        'is_enabled': False,
    })
    assert upd.status_code == 200

    delete = client.delete(f'/tools/{tid}')
    assert delete.status_code == 200


def test_agent_tool_creation(monkeypatch):
    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())
    response = client.post('/tools', json={
        'name': 'call_assistant',
        'type': 'agent_tool',
        'description': 'delegate',
        'config_json': {'target_agent': 'assistant', 'strict_output': True},
        'is_enabled': True,
    })
    assert response.status_code == 200
    assert response.json()['type'] == 'agent_tool'


def test_mcp_server_crud_and_refresh():
    create = client.post('/tools/mcp/servers', json={'name': 'srv1', 'transport': 'http', 'endpoint': 'http://127.0.0.1:8001', 'enabled': True})
    assert create.status_code == 200
    sid = create.json()['id']

    refresh = client.post(f'/tools/mcp/servers/{sid}/refresh', json={})
    assert refresh.status_code == 200
    assert refresh.json()['discovered']

    delete = client.delete(f'/tools/mcp/servers/{sid}')
    assert delete.status_code == 200
