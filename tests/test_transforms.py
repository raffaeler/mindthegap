from mindthegap.config import Settings
from mindthegap.transforms import (
    stitch_message,
    transform_request_body,
    transform_response_body,
    unstitch_messages,
)

S = Settings()


def test_stitch_message_with_reasoning():
    msg = {"role": "assistant", "content": "Hi", "reasoning_content": "let me think"}
    out = stitch_message(msg, S)
    assert out == {"role": "assistant", "content": "[[think]]\nlet me think  \n[[/think]]\n\nHi"}
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
    assert out["content"] == "[[think]]\nr  \n[[/think]]\n\n"


def test_unstitch_forward_extracts_reasoning():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "[[think]]\nplan\n[[/think]]\nhello"},
    ]
    out = unstitch_messages(messages, S, "forward")
    assert out[0] == {"role": "user", "content": "hi"}
    assert out[1] == {"role": "assistant", "content": "hello", "reasoning_content": "plan"}


def test_unstitch_drop_strips_block():
    messages = [{"role": "assistant", "content": "[[think]]\nx\n[[/think]]\nhello"}]
    out = unstitch_messages(messages, S, "drop")
    assert out == [{"role": "assistant", "content": "hello"}]


def test_unstitch_keep_passthrough():
    messages = [{"role": "assistant", "content": "[[think]]\nx\n[[/think]]\nhello"}]
    out = unstitch_messages(messages, S, "keep")
    assert out == messages


def test_unstitch_only_leading_block_treated_as_reasoning():
    messages = [
        {
            "role": "assistant",
            "content": "[[think]]\na\n[[/think]]\nbody [[think]]nope[[/think]] tail",
        }
    ]
    out = unstitch_messages(messages, S, "forward")
    assert out[0]["reasoning_content"] == "a"
    assert out[0]["content"] == "body [[think]]nope[[/think]] tail"


def test_unstitch_no_block_unchanged():
    messages = [{"role": "assistant", "content": "plain"}]
    assert unstitch_messages(messages, S, "forward") == messages


def test_stitch_message_normalizes_trailing_newlines_in_reasoning():
    # Reasoning that already ends with newlines must not produce more than
    # one blank line between the body and the closing [[/think]].
    msg = {"role": "assistant", "content": "Hi", "reasoning_content": "thoughts\n"}
    out = stitch_message(msg, S)
    assert out["content"] == "[[think]]\nthoughts  \n[[/think]]\n\nHi"

    msg2 = {"role": "assistant", "content": "Hi", "reasoning_content": "thoughts\n\n\n"}
    out2 = stitch_message(msg2, S)
    assert out2["content"] == "[[think]]\nthoughts  \n[[/think]]\n\nHi"


def test_unstitch_skips_non_assistant():
    messages = [{"role": "user", "content": "[[think]]x[[/think]]hi"}]
    assert unstitch_messages(messages, S, "forward") == messages


def test_unstitch_recovers_unclosed_think_forward():
    # Simulates a previously persisted assistant message whose stream got
    # truncated before [[/think]] was emitted.
    messages = [
        {"role": "assistant", "content": "[[think]]\nhalf a thought"},
    ]
    out = unstitch_messages(messages, S, "forward")
    assert out[0]["reasoning_content"] == "half a thought"
    assert out[0]["content"] == ""


def test_unstitch_recovers_unclosed_think_drop():
    messages = [
        {"role": "assistant", "content": "[[think]]\nhalf a thought"},
    ]
    out = unstitch_messages(messages, S, "drop")
    assert out[0]["content"] == ""
    assert "reasoning_content" not in out[0]


def test_unstitch_unclosed_only_when_no_close_anywhere():
    # If the close tag exists somewhere later, the strict pattern must win
    # (we already test the "leading block only" semantics elsewhere).
    messages = [
        {"role": "assistant", "content": "[[think]]\nplan\n[[/think]]\nhello"},
    ]
    out = unstitch_messages(messages, S, "forward")
    assert out[0]["reasoning_content"] == "plan"
    assert out[0]["content"] == "hello"


def test_unstitch_handles_null_content():
    messages = [{"role": "assistant", "content": None, "tool_calls": []}]
    assert unstitch_messages(messages, S, "forward") == messages


def test_unstitch_normalizes_empty_content_to_none_for_tool_calls():
    # After stripping a leading [[think]] block, an assistant message that
    # carries tool_calls must use content=None (not "") to satisfy DeepSeek
    # / OpenAI strict validation of tool-call messages.
    messages = [
        {
            "role": "assistant",
            "content": "[[think]]\nplan\n[[/think]]\n\n",
            "tool_calls": [{"id": "1", "type": "function"}],
        }
    ]
    out = unstitch_messages(messages, S, "forward")
    assert out[0]["content"] is None
    assert out[0]["tool_calls"] == [{"id": "1", "type": "function"}]
    assert out[0]["reasoning_content"] == "plan"


def test_unstitch_drop_also_normalizes_empty_for_tool_calls():
    messages = [
        {
            "role": "assistant",
            "content": "[[think]]\nplan\n[[/think]]\n\n",
            "tool_calls": [{"id": "1", "type": "function"}],
        }
    ]
    out = unstitch_messages(messages, S, "drop")
    assert out[0]["content"] is None
    assert "reasoning_content" not in out[0]


def test_unstitch_keeps_empty_string_when_no_tool_calls():
    # No tool_calls => stay backward-compatible with the prior "" behavior.
    messages = [{"role": "assistant", "content": "[[think]]\nplan\n[[/think]]\n"}]
    out = unstitch_messages(messages, S, "forward")
    assert out[0]["content"] == ""
    assert out[0]["reasoning_content"] == "plan"


def test_transform_request_body_uses_forward_for_reasoner():
    body = {
        "model": "deepseek-reasoner",
        "messages": [
            {"role": "assistant", "content": "[[think]]\nr\n[[/think]]\nA"},
        ],
    }
    out = transform_request_body(body, S)
    assert out["messages"][0]["reasoning_content"] == "r"
    assert out["messages"][0]["content"] == "A"


def test_transform_request_body_uses_drop_for_non_reasoner():
    body = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "assistant", "content": "[[think]]\nr\n[[/think]]\nA"},
        ],
    }
    out = transform_request_body(body, S)
    assert out["messages"][0] == {"role": "assistant", "content": "A"}


def test_transform_request_body_uses_forward_for_v4_pro():
    body = {
        "model": "deepseek-v4-pro",
        "messages": [
            {"role": "assistant", "content": "[[think]]\nr\n[[/think]]\nA"},
        ],
    }
    out = transform_request_body(body, S)
    assert out["messages"][0]["reasoning_content"] == "r"
    assert out["messages"][0]["content"] == "A"


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
    assert msg["content"] == "[[think]]\nthoughts  \n[[/think]]\n\nhi"
    assert "reasoning_content" not in msg
