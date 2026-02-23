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

    delete = client.delete(f'/tools/mcp/servers/{sid}')
    assert delete.status_code == 200


def test_agents_list_includes_default_assistant_and_protects_delete():
    response = client.get('/agents')
    assert response.status_code == 200
    names = response.json()['agents']
    assert 'assistant' in names

    protected = client.delete('/agents/assistant')
    assert protected.status_code == 400
    assert protected.json()['detail']['error']['code'] == 'protected_agent'


def test_tools_list_includes_builtin_tools_even_without_db_rows():
    response = client.get('/tools')
    assert response.status_code == 200
    items = response.json()['items']
    assert any(item['name'] == 'tavily_web_search' for item in items)
    assert any(item['name'] == 'mcp_query' for item in items)


def test_model_catalog_exposes_capability_flags():
    response = client.get('/model-catalog')
    assert response.status_code == 200
    items = response.json()['items']
    assert items
    sample = items[0]
    assert 'supports_tools' in sample
    assert 'tool_calling' in sample
    assert 'supports_vision' in sample


def test_chat_multipart_with_attachment(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'mocked reply')
    response = client.post(
        '/chat_with_attachments',
        data={'agent': 'assistant', 'message': 'Summarize attachment'},
        files=[('files', ('note.txt', b'Attachment says hello world', 'text/plain'))],
    )
    assert response.status_code == 200
    body = response.json()
    assert body['conversation_id']
    user_msgs = [m for m in body['messages'] if m['role'] == 'user']
    assert user_msgs
    assert 'note.txt' in user_msgs[-1]['attachments_json']


def test_prompt_draft_minimal_payload_returns_200():
    response = client.post('/agents/prompt-draft', json={'goal': 'Build a bug triage agent'})
    assert response.status_code == 200
    payload = response.json()
    assert 'prompt_text' in payload
    assert 'sections' in payload
    assert 'used_fallback' in payload


def test_prompt_draft_missing_goal_returns_422():
    response = client.post('/agents/prompt-draft', json={})
    assert response.status_code == 422


def test_prompt_draft_fallback_path(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'generate_system_prompt', lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError('offline')))
    response = client.post('/agents/prompt-draft', json={'goal': 'Need deterministic fallback'})
    assert response.status_code == 200
    assert response.json()['used_fallback'] is True


def test_agent_create_with_valid_builtin_tool_id_succeeds(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'ok')
    response = client.post('/agents', json={
        'name': 'tool-agent',
        'description': 'test',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are helpful',
        'tool_ids': ['tavily_web_search'],
        'settings': {'temperature': 0.2},
        'resource_limits': {'max_context_messages': 20},
    })
    assert response.status_code == 200


def test_agent_create_with_unknown_tool_id_fails():
    response = client.post('/agents', json={
        'name': 'bad-tool-agent',
        'description': 'test',
        'provider': 'ollama',
        'model_id': 'gemma3:1b',
        'system_prompt': 'You are helpful',
        'tool_ids': ['not_real_tool'],
        'settings': {'temperature': 0.2},
        'resource_limits': {'max_context_messages': 20},
    })
    assert response.status_code == 400
    assert response.json()['detail']['error']['code'] == 'invalid_tool'


def test_chat_stream_endpoint_emits_events(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'stream_chat_with_agent', lambda *args, **kwargs: iter(['Hi', 'Hi there']))
    response = client.post('/chat/stream', json={'agent': 'assistant', 'message': 'hello'})
    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    body = response.text
    assert 'event: start' in body
    assert 'event: end' in body


def test_tools_help_endpoint():
    response = client.get('/tools/help')
    assert response.status_code == 200
    assert 'TAVILY_API_KEY' in response.json()['tavily']


def test_tools_returns_tavily_with_missing_key_status(monkeypatch):
    monkeypatch.delenv('TAVILY_API_KEY', raising=False)
    response = client.get('/tools')
    assert response.status_code == 200
    items = response.json()['items']
    tavily = next((i for i in items if i['tool_id'] == 'tavily_web_search'), None)
    assert tavily is not None
    assert tavily['status'] in {'missing_key', 'disabled'}


def test_mcp_import_accepts_json_config(monkeypatch):
    monkeypatch.setattr(api_server, '_discover_mcp_tools', lambda server: ([{
        'tool_id': f"mcp:{server['name']}:search",
        'name': f"{server['name']}:search",
        'description': 'Discovered',
        'type': 'mcp',
        'config_json': {'server_id': server['id'], 'tool_name': 'search'},
        'is_enabled': True,
    }], None))
    payload = {
        'description': 'import test',
        'config': {
            'mcpServers': {
                'amap-maps': {
                    'command': 'npx',
                    'args': ['-y', '@amap/amap-maps-mcp-server'],
                    'env': {'AMAP_MAPS_API_KEY': 'api_key'}
                }
            }
        }
    }
    response = client.post('/tools/mcp/import', json=payload)
    assert response.status_code == 200
    assert response.json()['imported_servers']


def test_tool_test_endpoint_returns_output(monkeypatch):
    monkeypatch.setenv('TAVILY_API_KEY', 'dummy')
    response = client.post('/tools/tavily_web_search/test', json={'input': {'query': 'hello'}})
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_mcp_alias_endpoints_work(monkeypatch):
    create = client.post('/mcp/servers', json={'name': 'alias-srv', 'transport': 'http', 'endpoint': 'http://127.0.0.1:9001', 'enabled': True})
    assert create.status_code == 200
    sid = create.json()['id']

    listed = client.get('/mcp/servers')
    assert listed.status_code == 200
    assert any(s['id'] == sid for s in listed.json()['servers'])

    monkeypatch.setattr(api_server, '_discover_mcp_tools', lambda server: ([{
        'tool_id': f"mcp:{server['name']}:x",
        'name': f"{server['name']}:x",
        'description': 'x',
        'type': 'mcp_tool',
        'config_json': {'server_id': server['id'], 'tool_name': 'x'},
        'is_enabled': True,
    }], None))

    discover = client.post(f'/mcp/servers/{sid}/discover', json={})
    assert discover.status_code == 200


def test_mcp_tools_select_persists_tools(monkeypatch):
    create = client.post('/mcp/servers', json={'name': 'select-srv', 'transport': 'http', 'endpoint': 'http://127.0.0.1:9002', 'enabled': True})
    sid = create.json()['id']

    response = client.post('/mcp/tools', json={
        'server_id': sid,
        'selected_tools': [
            {'tool_name': 'search', 'name': 'select-srv:search', 'description': 'search tool', 'schema': {'type': 'object'}, 'enabled': True}
        ]
    })
    assert response.status_code == 200
    assert response.json()['items']
