"""Unit tests for reasoning text sanitisation."""

from __future__ import annotations

import pytest

from mindthegap.sanitize import has_incomplete_fragment, sanitize_reasoning_text


class TestSanitizeReasoningText:
    def test_empty_and_none(self) -> None:
        assert sanitize_reasoning_text(None) == ""
        assert sanitize_reasoning_text("") == ""
        assert sanitize_reasoning_text("   ") == ""

    def test_passthrough_clean_text(self) -> None:
        text = "This is clean reasoning text."
        assert sanitize_reasoning_text(text) == text

    def test_line_ending_normalisation(self) -> None:
        assert sanitize_reasoning_text("line1\r\nline2\rline3") == "line1\nline2\nline3"

    def test_strip_dsml_tags(self) -> None:
        # DSML tags use U+FF5C fullwidth vertical bar (not U+2016)
        text = "Pre <\uff5cDSML\uff5ctool_calls> mid </\uff5cDSML\uff5cinvoke> post"
        result = sanitize_reasoning_text(text)
        assert "DSML" not in result
        assert result == "Pre mid post"

    def test_strip_dsml_parameter_tag(self) -> None:
        text = 'Hello <\uff5cDSML\uff5cparameter name="x" string="true"> world'
        result = sanitize_reasoning_text(text)
        assert "DSML" not in result
        assert result == "Hello world"

    def test_strip_xml_tags(self) -> None:
        text = "Before </parameter> middle <analysis> after </invoke> end"
        result = sanitize_reasoning_text(text)
        assert "</parameter>" not in result
        assert "<analysis>" not in result
        assert "</invoke>" not in result
        assert result == "Before middle after end"

    def test_strip_nested_tags(self) -> None:
        text = "Start </parameter><analysis> tag soup </invoke> end"
        result = sanitize_reasoning_text(text)
        assert result == "Start tag soup end"

    def test_whitespace_normalisation(self) -> None:
        text = "spaces   \t  \f  \v  collapsed"
        result = sanitize_reasoning_text(text)
        assert result == "spaces collapsed"

    def test_preserve_newlines(self) -> None:
        text = "line1\n\nline2\nline3"
        result = sanitize_reasoning_text(text)
        assert result == text  # newlines should be preserved

    def test_combined_sanitisation(self) -> None:
        text = (
            "Start\r\n</\uff5cDSML\uff5cinvoke> reasoning </parameter> text\n"
            "  \t  more   text"
        )
        result = sanitize_reasoning_text(text)
        assert "DSML" not in result
        assert "</parameter>" not in result
        assert "\r" not in result
        assert result == "Start\n reasoning text\n more text"


class TestHasIncompleteFragment:
    def test_none_and_empty(self) -> None:
        assert has_incomplete_fragment(None) is False
        assert has_incomplete_fragment("") is False
        assert has_incomplete_fragment("   ") is False

    def test_no_tags(self) -> None:
        assert has_incomplete_fragment("plain text") is False

    def test_complete_tag(self) -> None:
        assert has_incomplete_fragment("<tag>text</tag>") is False

    def test_incomplete_opening_tag(self) -> None:
        assert has_incomplete_fragment("</param") is True

    def test_incomplete_tag_with_text(self) -> None:
        assert has_incomplete_fragment("reasoning </parameter") is True

    def test_only_greater_than(self) -> None:
        # Only '>' without '<' is not a fragment
        assert has_incomplete_fragment(">") is False

    def test_both_present(self) -> None:
        assert has_incomplete_fragment("<a>b</c>") is False

    def test_multiple_tags(self) -> None:
        assert has_incomplete_fragment("<a></b>") is False
