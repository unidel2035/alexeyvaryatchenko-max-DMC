"""Tests for the Claude AI client module."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from unittest.mock import MagicMock, patch

from claude_client import get_client, ask_claude, analyze_pattern_with_claude, suggest_pattern_improvements


class TestGetClient:
    def test_raises_without_api_key(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            get_client()

    def test_returns_client_with_api_key(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key-123")
        import anthropic
        client = get_client()
        assert isinstance(client, anthropic.Anthropic)


class TestAskClaude:
    def _make_mock_response(self, text: str):
        mock_resp = MagicMock()
        mock_resp.content = [MagicMock(text=text)]
        return mock_resp

    def test_ask_claude_returns_response_and_history(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response("Привет!")

        with patch("claude_client.get_client", return_value=mock_client):
            response, history = ask_claude("Hello")

        assert response == "Привет!"
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"
        assert history[1]["content"] == "Привет!"

    def test_ask_claude_preserves_history(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response("Ответ 2")

        prior_history = [
            {"role": "user", "content": "Вопрос 1"},
            {"role": "assistant", "content": "Ответ 1"},
        ]

        with patch("claude_client.get_client", return_value=mock_client):
            response, history = ask_claude("Вопрос 2", prior_history)

        assert response == "Ответ 2"
        assert len(history) == 4
        # Verify messages were sent with full history
        call_args = mock_client.messages.create.call_args
        messages_sent = call_args.kwargs["messages"]
        assert len(messages_sent) == 3  # 2 prior + 1 new user

    def test_ask_claude_uses_system_prompt(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response("OK")

        with patch("claude_client.get_client", return_value=mock_client):
            ask_claude("Test")

        call_args = mock_client.messages.create.call_args
        assert "system" in call_args.kwargs
        assert "DMC" in call_args.kwargs["system"] or "вышива" in call_args.kwargs["system"]

    def test_ask_claude_default_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._make_mock_response("OK")

        with patch("claude_client.get_client", return_value=mock_client):
            ask_claude("Test")

        call_args = mock_client.messages.create.call_args
        assert call_args.kwargs["model"] == "claude-sonnet-4-6"


class TestAnalyzePatternWithClaude:
    def _sample_pattern_json(self) -> str:
        import json
        return json.dumps({
            "width": 30,
            "height": 20,
            "grid": [],
            "color_legend": {
                "310": {"name": "Black", "rgb": [0, 0, 0], "symbol": "■", "count": 100},
                "321": {"name": "Red", "rgb": [196, 29, 35], "symbol": "●", "count": 50},
            }
        })

    def test_analyze_calls_ask_claude(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Отличная схема!")]
        )

        with patch("claude_client.get_client", return_value=mock_client):
            result = analyze_pattern_with_claude(self._sample_pattern_json())

        assert result == "Отличная схема!"

    def test_analyze_includes_pattern_info(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="OK")]
        )

        with patch("claude_client.get_client", return_value=mock_client):
            analyze_pattern_with_claude(self._sample_pattern_json())

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[0]["content"]
        assert "30" in user_msg  # width
        assert "20" in user_msg  # height
        assert "2" in user_msg   # num colors


class TestSuggestPatternImprovements:
    def test_suggest_calls_ask_claude(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="Рекомендация")]
        )

        with patch("claude_client.get_client", return_value=mock_client):
            result = suggest_pattern_improvements("Хочу вышить кота")

        assert result == "Рекомендация"

    def test_suggest_includes_description(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_client = MagicMock()
        mock_client.messages.create.return_value = MagicMock(
            content=[MagicMock(text="OK")]
        )

        with patch("claude_client.get_client", return_value=mock_client):
            suggest_pattern_improvements("вышить пейзаж с горами")

        call_args = mock_client.messages.create.call_args
        messages = call_args.kwargs["messages"]
        user_msg = messages[0]["content"]
        assert "пейзаж с горами" in user_msg
