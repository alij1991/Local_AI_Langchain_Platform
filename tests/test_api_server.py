import pytest

pytest.importorskip("fastapi")

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


def test_prompt_workflow_tools_systems(monkeypatch):
    monkeypatch.setattr(api_server.orchestrator, 'generate_system_prompt', lambda desc: f'prompt:{desc}')
    monkeypatch.setattr(api_server.orchestrator, 'run_agent_workflow', lambda prompt, seq: {'assistant': f'{prompt}:{seq}'})
    monkeypatch.setattr(api_server.orchestrator, 'get_tool_names', lambda: ['utc_now'])
    monkeypatch.setattr(api_server.orchestrator, 'add_instruction_tool', lambda *_args, **_kwargs: None)

    draft = client.post('/agents/prompt-draft', json={'description': 'email writer'})
    assert draft.status_code == 200
    assert draft.json()['prompt'] == 'prompt:email writer'

    wf = client.post('/workflow/run', json={'prompt': 'hi', 'sequence_csv': 'assistant'})
    assert wf.status_code == 200
    assert 'assistant' in wf.json()['outputs']

    tpl = client.get('/tools/template?mode=instruction')
    assert tpl.status_code == 200
    assert tpl.json()['name'] == 'summarize_for_exec'

    assert client.get('/tools').status_code == 200
    assert client.post('/tools', json={'name': 'my_tool', 'tool_type': 'instruction'}).status_code == 200
    assert client.post('/systems', json={'name': 'demo', 'sequence': 'assistant'}).status_code == 200

    run = client.post('/systems/run', json={'name': 'demo', 'prompt': 'hi'})
    assert run.status_code == 200
    assert 'assistant' in run.json()['outputs']


def test_chat_attachments_endpoint(monkeypatch):
    calls = {}

    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())

    def fake_chat(agent, message, image_paths=None):
        calls['agent'] = agent
        calls['message'] = message
        calls['image_paths'] = image_paths or []
        return 'ok-with-attachments'

    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', fake_chat)

    response = client.post(
        '/chat/attachments',
        data={'agent': 'assistant', 'message': 'read this'},
        files=[('files', ('note.txt', b'hello from attachment', 'text/plain'))],
    )

    assert response.status_code == 200
    assert response.json()['reply'] == 'ok-with-attachments'
    assert calls['agent'] == 'assistant'
    assert 'Attachment context' in calls['message']



def test_chat_attachments_with_image_notice(monkeypatch):
    calls = {}

    monkeypatch.setitem(api_server.orchestrator.definitions, 'assistant', object())

    def fake_chat(agent, message, image_paths=None):
        calls['agent'] = agent
        calls['message'] = message
        calls['image_paths'] = image_paths or []
        return 'ok-image'

    monkeypatch.setattr(api_server.orchestrator, 'chat_with_agent', fake_chat)

    tiny_png = (
        b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde'
        b'\x00\x00\x00\x0bIDATx\x9cc```\x00\x00\x00\x05\x00\x01\x0d\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82'
    )

    response = client.post(
        '/chat/attachments',
        data={'agent': 'assistant', 'message': 'describe this image'},
        files=[('files', ('tiny.png', tiny_png, 'image/png'))],
    )

    assert response.status_code == 200
    assert response.json()['reply'] == 'ok-image'
    assert calls['image_paths']
    assert 'Analyze them directly' in calls['message']
