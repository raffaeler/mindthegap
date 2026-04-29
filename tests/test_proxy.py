import json

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from mindthegap.app import create_app
from mindthegap.config import Settings


@pytest.fixture
def settings():
    return Settings(upstream_base_url="https://upstream.test")


@pytest.fixture
def client(settings):
    app = create_app(settings)
    with TestClient(app) as c:
        yield c


def test_healthz(client):
    assert client.get("/healthz").json() == {"ok": True}


@respx.mock
def test_chat_completions_nonstream_stitches_response(client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hello",
                            "reasoning_content": "thinking",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    respx.post("https://upstream.test/v1/chat/completions").mock(side_effect=handler)

    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-reasoner",
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "<think>\nprev\n</think>\nprior"},
                {"role": "user", "content": "next"},
            ],
        },
        headers={"authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    msg = resp.json()["choices"][0]["message"]
    assert msg["content"] == "<think>\nthinking  \n</think>\n\nHello"
    assert "reasoning_content" not in msg

    # request was unstitched (forward mode for reasoner)
    sent = captured["body"]
    assert sent["messages"][1]["content"] == "prior"
    assert sent["messages"][1]["reasoning_content"] == "prev"


@respx.mock
def test_chat_completions_streaming_rewrites_sse(client):
    sse_body = (
        b'data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"reasoning_content":"plan"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
        b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b"data: [DONE]\n\n"
    )
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )
    )
    resp = client.post(
        "/v1/chat/completions",
        json={
            "model": "deepseek-reasoner",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    body = resp.content.decode()
    payloads = [
        json.loads(line[5:].strip())
        for line in body.splitlines()
        if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]")
    ]
    joined = "".join(
        p["choices"][0]["delta"].get("content") or ""
        for p in payloads
        if isinstance(p["choices"][0]["delta"].get("content"), str)
    )
    assert "<think>\nplan" in joined
    assert "</think>\n\nHi" in joined
    assert "reasoning_content" not in body
    assert "[DONE]" in body


@respx.mock
def test_passthrough_get_models(client):
    respx.get("https://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "deepseek-reasoner"}]})
    )
    resp = client.get("/v1/models", headers={"authorization": "Bearer x"})
    assert resp.status_code == 200
    assert resp.json() == {"data": [{"id": "deepseek-reasoner"}]}


@respx.mock
def test_chat_completions_upstream_error_forwarded(client):
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(400, json={"error": "bad"})
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-reasoner", "messages": []},
        headers={"authorization": "Bearer x"},
    )
    assert resp.status_code == 400
    assert resp.json() == {"error": "bad"}
