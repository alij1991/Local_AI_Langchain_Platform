import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

import api_server


client = TestClient(api_server.app)


@pytest.fixture(autouse=True)
def clean_state():
    # keep tests isolated without touching production-like file paths manually
    yield


def test_health_endpoint():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json()['status'] == 'ok'


def test_conversation_crud_and_messages(monkeypatch):
    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())
    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', lambda *args, **kwargs: 'reply')

    c = client.post('/conversations', json={'title': 'My Chat'})
    assert c.status_code == 200
    cid = c.json()['id']

    send = client.post('/chat', json={'conversation_id': cid, 'agent': 'assistant', 'message': 'hello'})
    assert send.status_code == 200
    assert send.json()['conversation_id'] == cid

    lst = client.get('/conversations')
    assert lst.status_code == 200
    assert any(item['id'] == cid for item in lst.json())

    msgs = client.get(f'/conversations/{cid}/messages')
    assert msgs.status_code == 200
    assert len(msgs.json()) >= 2

    rename = client.patch(f'/conversations/{cid}', json={'title': 'Renamed'})
    assert rename.status_code == 200
    assert rename.json()['title'] == 'Renamed'

    delete = client.delete(f'/conversations/{cid}')
    assert delete.status_code == 200


def test_prompt_draft_fallback_and_shape(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'generate_system_prompt', lambda *_args, **_kwargs: '')

    response = client.post('/agents/prompt-draft', json={'goal': 'Build analyzer'})
    assert response.status_code == 200
    body = response.json()
    assert 'prompt_text' in body
    assert 'sections' in body
    assert 'role' in body['sections']


def test_prompt_draft_llm_refine(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'generate_system_prompt', lambda *_args, **_kwargs: 'LLM prompt')
    response = client.post('/agents/prompt-draft', json={'goal': 'Build analyzer'})
    assert response.status_code == 200
    assert response.json()['prompt_text'] == 'LLM prompt'


def test_system_validation_dag(monkeypatch):
    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())
    valid = {
        'name': 'sys-a',
        'definition': {
            'nodes': [
                {'id': 'n1', 'type': 'agent', 'agent': 'assistant'},
                {'id': 'n2', 'type': 'agent', 'agent': 'assistant'},
            ],
            'edges': [{'source': 'n1', 'target': 'n2'}],
        },
    }
    create = client.post('/systems', json=valid)
    assert create.status_code == 200

    bad = {
        'name': 'sys-cycle',
        'definition': {
            'nodes': [
                {'id': 'a', 'type': 'agent', 'agent': 'assistant'},
                {'id': 'b', 'type': 'agent', 'agent': 'assistant'},
            ],
            'edges': [{'source': 'a', 'target': 'b'}, {'source': 'b', 'target': 'a'}],
        },
    }
    invalid = client.post('/systems', json=bad)
    assert invalid.status_code == 400
    assert 'DAG' in invalid.json()['detail']


def test_prompt_draft_invalid_payload():
    response = client.post('/agents/prompt-draft', json={'goal': 123})
    assert response.status_code in {400, 422}
