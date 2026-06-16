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
                {"role": "assistant", "content": "[[think]]\nprev\n[[/think]]\nprior"},
                {"role": "user", "content": "next"},
            ],
        },
        headers={"authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    msg = resp.json()["choices"][0]["message"]
    assert msg["content"] == "[[think]]  \nthinking  \n[[/think]]\n\nHello"
    assert "reasoning_content" not in msg

    # request was unstitched (forward mode for reasoner)
    sent = captured["body"]
    assert sent["messages"][1]["content"] == "prior"
    assert sent["messages"][1]["reasoning_content"] == "prev"


@respx.mock
def test_chat_completions_does_not_forward_client_accept_encoding(client):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["accept_encoding"] = request.headers.get("accept-encoding")
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    respx.post("https://upstream.test/v1/chat/completions").mock(side_effect=handler)

    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "hi"}]},
        headers={"accept-encoding": "br, zstd", "authorization": "Bearer sk-test"},
    )

    assert resp.status_code == 200
    assert captured["accept_encoding"] != "br, zstd"


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
    assert "[[think]]  \nplan" in joined
    assert "[[/think]]\n\nHi" in joined
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


# ── Multi-upstream tests ──────────────────────────────────────────────────


@respx.mock
def test_multi_upstream_model_based_routing():
    """Requests with different model fields route to different upstreams."""
    settings = Settings(
        upstreams={
            "deepseek": {"base_url": "https://ds.test"},
            "openai": {"base_url": "https://oa.test"},
        },
        upstream_selection={
            "model_mapping": {"deepseek-*": "deepseek", "gpt-*": "openai"},
        },
        default_upstream="deepseek",
    )
    ds_captured = {}
    oa_captured = {}

    def ds_handler(request: httpx.Request) -> httpx.Response:
        ds_captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "from deepseek"}}]},
        )

    def oa_handler(request: httpx.Request) -> httpx.Response:
        oa_captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "from openai"}}]},
        )

    respx.post("https://ds.test/v1/chat/completions").mock(side_effect=ds_handler)
    respx.post("https://oa.test/v1/chat/completions").mock(side_effect=oa_handler)

    with TestClient(create_app(settings)) as c:
        # Model "deepseek-chat" → deepseek upstream
        r = c.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": []},
            headers={"authorization": "Bearer x"},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "from deepseek"

        # Model "gpt-4o" → openai upstream
        r = c.post(
            "/v1/chat/completions",
            json={"model": "gpt-4o", "messages": []},
            headers={"authorization": "Bearer x"},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "from openai"


@respx.mock
def test_multi_upstream_default_when_model_missing():
    """When model is not specified, the default upstream is used."""
    settings = Settings(
        upstreams={
            "deepseek": {"base_url": "https://ds.test"},
            "openai": {"base_url": "https://oa.test"},
        },
        default_upstream="deepseek",
    )
    respx.post("https://ds.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "default"}}]}
        )
    )
    with TestClient(create_app(settings)) as c:
        # No model field → default upstream
        r = c.post(
            "/v1/chat/completions",
            json={"messages": []},
            headers={"authorization": "Bearer x"},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "default"


@respx.mock
def test_multi_upstream_header_selection():
    """The X-Mindthegap-Upstream header selects the upstream."""
    settings = Settings(
        upstreams={
            "deepseek": {"base_url": "https://ds.test"},
            "openai": {"base_url": "https://oa.test"},
        },
    )
    respx.post("https://oa.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "from openai"}}]}
        )
    )
    with TestClient(create_app(settings)) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "anything", "messages": []},
            headers={
                "authorization": "Bearer x",
                "x-mindthegap-upstream": "openai",
            },
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "from openai"


@respx.mock
def test_multi_upstream_path_prefix_routing():
    """A path starting with /<upstream-name>/ routes to that upstream."""
    settings = Settings(
        upstreams={
            "deepseek": {"base_url": "https://ds.test"},
        },
        upstream_selection={"path_prefix_enabled": True},
        default_upstream="deepseek",
    )
    respx.post("https://ds.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "routed"}}]}
        )
    )
    with TestClient(create_app(settings)) as c:
        r = c.post(
            "/deepseek/v1/chat/completions",
            json={"model": "anything", "messages": []},
            headers={"authorization": "Bearer x"},
        )
        assert r.status_code == 200
        assert r.json()["choices"][0]["message"]["content"] == "routed"


@respx.mock
def test_multi_upstream_api_key_injection():
    """Per-upstream API key is injected when the client doesn't provide auth."""
    settings = Settings(
        upstreams={
            "deepseek": {
                "base_url": "https://ds.test",
                "api_key": "sk-upstream-key",
                "api_key_header": "Authorization",
                "api_key_as_bearer": True,
            },
        },
        default_upstream="deepseek",
    )
    captured_auth = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_auth["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    respx.post("https://ds.test/v1/chat/completions").mock(side_effect=handler)

    with TestClient(create_app(settings)) as c:
        # No Authorization header from client → inject upstream key
        r = c.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": []},
        )
        assert r.status_code == 200
        assert captured_auth["auth"] == "Bearer sk-upstream-key"


@respx.mock
def test_multi_upstream_api_key_not_overridden():
    """Client-provided auth is not overwritten by the upstream's API key."""
    settings = Settings(
        upstreams={
            "deepseek": {
                "base_url": "https://ds.test",
                "api_key": "sk-upstream-key",
            },
        },
        default_upstream="deepseek",
    )
    captured_auth = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_auth["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "ok"}}]}
        )

    respx.post("https://ds.test/v1/chat/completions").mock(side_effect=handler)

    with TestClient(create_app(settings)) as c:
        r = c.post(
            "/v1/chat/completions",
            json={"model": "deepseek-chat", "messages": []},
            headers={"authorization": "Bearer sk-client-key"},
        )
        assert r.status_code == 200
        assert captured_auth["auth"] == "Bearer sk-client-key"


# ── Non-prefixed route tests ────────────────────────────────────────────────


@respx.mock
def test_chat_completions_no_v1_prefix(client):
    """POST /chat/completions (no /v1) forwards to upstream /v1/chat/completions."""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "id": "x",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    respx.post("https://upstream.test/v1/chat/completions").mock(side_effect=handler)

    resp = client.post(
        "/chat/completions",
        json={
            "model": "deepseek-reasoner",
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    assert captured["url"] == "https://upstream.test/v1/chat/completions"
    assert resp.json()["choices"][0]["message"]["content"] == "Hello"


@respx.mock
def test_chat_completions_no_v1_prefix_streaming(client):
    """POST /chat/completions (no /v1) with streaming."""
    captured_url = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured_url["url"] = str(request.url)
        sse_body = (
            b'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
            b"data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            content=sse_body,
            headers={"content-type": "text/event-stream"},
        )

    respx.post("https://upstream.test/v1/chat/completions").mock(side_effect=handler)

    resp = client.post(
        "/chat/completions",
        json={
            "model": "deepseek-reasoner",
            "stream": True,
            "messages": [{"role": "user", "content": "hi"}],
        },
        headers={"authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    assert captured_url["url"] == "https://upstream.test/v1/chat/completions"
    assert "[DONE]" in resp.content.decode()


@respx.mock
def test_passthrough_no_v1_prefix(client):
    """GET /models (no /v1) forwards to upstream /v1/models."""
    respx.get("https://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "deepseek-reasoner"}]})
    )
    resp = client.get("/models", headers={"authorization": "Bearer x"})
    assert resp.status_code == 200
    assert resp.json() == {"data": [{"id": "deepseek-reasoner"}]}


@respx.mock
def test_chat_completions_no_prefix_upstream():
    """With upstream_path_prefix="", upstream receives /chat/completions directly."""
    settings = Settings(
        upstream_base_url="https://upstream.test",
        upstream_path_prefix="",
    )
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ok"}}
                ],
            },
        )

    respx.post("https://upstream.test/chat/completions").mock(side_effect=handler)

    with TestClient(create_app(settings)) as c:
        resp = c.post(
            "/chat/completions",
            json={"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert resp.status_code == 200
        assert captured["url"] == "https://upstream.test/chat/completions"


@respx.mock
def test_v1_routes_still_work(client):
    """Existing /v1/chat/completions and /v1/models routes still work."""
    # Chat
    respx.post("https://upstream.test/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "Hi"}}
                ],
            },
        )
    )
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "deepseek-reasoner", "messages": [{"role": "user", "content": "hi"}]},
        headers={"authorization": "Bearer sk-test"},
    )
    assert resp.status_code == 200
    assert resp.json()["choices"][0]["message"]["content"] == "Hi"

    # Passthrough
    respx.get("https://upstream.test/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "m"}]})
    )
    resp = client.get("/v1/models", headers={"authorization": "Bearer x"})
    assert resp.status_code == 200
    assert resp.json() == {"data": [{"id": "m"}]}


@respx.mock
def test_catch_all_no_v1_forwards_to_upstream(client):
    """Non-prefixed path GET /some-endpoint forwards to /v1/some-endpoint."""
    respx.get("https://upstream.test/v1/some-endpoint").mock(
        return_value=httpx.Response(200, json={"result": "ok"})
    )
    resp = client.get("/some-endpoint", headers={"authorization": "Bearer x"})
    assert resp.status_code == 200
    assert resp.json() == {"result": "ok"}
