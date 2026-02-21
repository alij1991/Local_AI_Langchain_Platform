from fastapi.testclient import TestClient

import api_server


client = TestClient(api_server.app)


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_agents_endpoint_shape():
    response = client.get('/agents')
    assert response.status_code == 200
    body = response.json()
    assert 'agents' in body
    assert 'assistant' in body['agents']


def test_chat_endpoint_with_monkeypatched_orchestrator(monkeypatch):
    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda agent, msg: f'echo:{agent}:{msg}')

    response = client.post('/chat', json={'agent': 'assistant', 'message': 'hello'})
    assert response.status_code == 200
    assert response.json()['reply'] == 'echo:assistant:hello'
