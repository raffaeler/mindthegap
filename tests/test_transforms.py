from oaipatch.config import Settings
from oaipatch.transforms import (
    stitch_message,
    transform_request_body,
    transform_response_body,
    unstitch_messages,
)

S = Settings()


def test_stitch_message_with_reasoning():
    msg = {"role": "assistant", "content": "Hi", "reasoning_content": "let me think"}
    out = stitch_message(msg, S)
    assert out == {"role": "assistant", "content": "<think>\nlet me think\n</think>\nHi"}
    assert "reasoning_content" not in out


def test_stitch_message_without_reasoning_passes_through():
    msg = {"role": "assistant", "content": "Hi"}
    assert stitch_message(msg, S) == msg


def test_stitch_message_empty_reasoning_strips_field():
    msg = {"role": "assistant", "content": "Hi", "reasoning_content": ""}
    out = stitch_message(msg, S)
    assert out == {"role": "assistant", "content": "Hi"}


def test_stitch_message_null_content_with_tool_calls():
    msg = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "1"}],
        "reasoning_content": "r",
    }
    out = stitch_message(msg, S)
    assert out["tool_calls"] == [{"id": "1"}]
    assert out["content"] == "<think>\nr\n</think>\n"


def test_unstitch_forward_extracts_reasoning():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "<think>\nplan\n</think>\nhello"},
    ]
    out = unstitch_messages(messages, S, "forward")
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1] == {"role": "assistant", "content": "hello", "reasoning_content": "plan"}


def test_unstitch_drop_strips_block():
    messages = [{"role": "assistant", "content": "<think>\nx\n</think>\nhello"}]
    out = unstitch_messages(messages, S, "drop")
    assert out == [{"role": "assistant", "content": "hello"}]


def test_unstitch_keep_passthrough():
    messages = [{"role": "assistant", "content": "<think>\nx\n</think>\nhello"}]
    out = unstitch_messages(messages, S, "keep")
    assert out == messages


def test_unstitch_only_leading_block_treated_as_reasoning():
    messages = [
        {"role": "assistant", "content": "<think>\na\n</think>\nbody <think>nope</think> tail"}
    ]
    out = unstitch_messages(messages, S, "forward")
    assert out[0]["reasoning_content"] == "a"
    assert out[0]["content"] == "body <think>nope</think> tail"


def test_unstitch_no_block_unchanged():
    messages = [{"role": "assistant", "content": "plain"}]
    assert unstitch_messages(messages, S, "forward") == messages


def test_unstitch_skips_non_assistant():
    messages = [{"role": "user", "content": "<think>x</think>hi"}]
    assert unstitch_messages(messages, S, "forward") == messages


def test_unstitch_handles_null_content():
    messages = [{"role": "assistant", "content": None, "tool_calls": []}]
    assert unstitch_messages(messages, S, "forward") == messages


def test_transform_request_body_uses_forward_for_reasoner():
    body = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "assistant", "content": "<think>\nr\n</think>\nA"},
        ],
    }
    out = transform_request_body(body, S)
    assert out["messages"][0]["reasoning_content"] == "r"
    assert out["messages"][0]["content"] == "A"


def test_transform_request_body_uses_drop_for_non_reasoner():
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "assistant", "content": "<think>\nr\n</think>\nA"},
        ],
    }
    out = transform_request_body(body, S)
    assert out["messages"][0] == {"role": "assistant", "content": "A"}


def test_transform_response_body_stitches_choices():
    body = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "hi",
                    "reasoning_content": "thoughts",
                },
            }
        ]
    }
    out = transform_response_body(body, S)
    msg = out["choices"][0]["message"]
    assert msg["content"] == "<think>\nthoughts\n</think>\nhi"
    assert "reasoning_content" not in msg
