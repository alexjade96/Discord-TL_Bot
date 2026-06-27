"""Tests for Prompt/prompt.py — all HF/Ollama API calls are mocked."""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent.parent / ".env")

from prompt import ask, _OLLAMA_MODEL, _HF_MODEL, _hf_call, _RETRY_DELAYS

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore[no-redef]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_client(content="Mock reply."):
    """Return a mock InferenceClient whose chat_completion returns `content`."""
    mock = MagicMock()
    mock.chat_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content=content))]
    )
    return mock


def _http_error(status_code: int, retry_after: str | None = None) -> HfHubHTTPError:
    """Build a minimal HfHubHTTPError with the given status code."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"Retry-After": retry_after} if retry_after else {}
    return HfHubHTTPError(f"HTTP {status_code}", response=response)


_SINGLE_TURN = [{"role": "user", "content": "Hello"}]
_MULTI_TURN = [
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."},
    {"role": "user",      "content": "What is its population?"},
]

_HAS_TOKEN = bool(os.environ.get("HF_TOKEN"))
needs_token = pytest.mark.skipif(not _HAS_TOKEN, reason="HF_TOKEN not set")


# ---------------------------------------------------------------------------
# Unit tests — Ollama path (primary)
# ---------------------------------------------------------------------------

class TestAskOllama:
    @patch("prompt.InferenceClient")
    def test_returns_string(self, mock_cls):
        mock_cls.return_value = _mock_client("Hello there!")
        result = ask(_SINGLE_TURN)
        assert isinstance(result, str)

    @patch("prompt.InferenceClient")
    def test_reply_content_matches_mock(self, mock_cls):
        mock_cls.return_value = _mock_client("Expected reply")
        assert ask(_SINGLE_TURN) == "Expected reply"

    @patch("prompt.InferenceClient")
    def test_whitespace_stripped(self, mock_cls):
        mock_cls.return_value = _mock_client("  padded  ")
        assert ask(_SINGLE_TURN) == "padded"

    @patch("prompt.InferenceClient")
    def test_uses_ollama_model(self, mock_cls):
        client = _mock_client()
        mock_cls.return_value = client
        ask(_SINGLE_TURN)
        _, kwargs = client.chat_completion.call_args
        assert kwargs.get("model") == _OLLAMA_MODEL

    @patch("prompt.InferenceClient")
    def test_passes_messages_intact(self, mock_cls):
        client = _mock_client()
        mock_cls.return_value = client
        ask(_SINGLE_TURN)
        assert client.chat_completion.call_args[0][0] == _SINGLE_TURN

    @patch("prompt.InferenceClient")
    def test_multi_turn_messages_passed_intact(self, mock_cls):
        client = _mock_client("About 2 million.")
        mock_cls.return_value = client
        ask(_MULTI_TURN)
        called = client.chat_completion.call_args[0][0]
        assert called == _MULTI_TURN and len(called) == 3

    @patch("prompt.InferenceClient")
    def test_empty_messages_calls_api(self, mock_cls):
        mock_cls.return_value = _mock_client("ok")
        assert isinstance(ask([]), str)


# ---------------------------------------------------------------------------
# Unit tests — HF fallback path (Ollama unavailable)
# ---------------------------------------------------------------------------

class TestAskHFFallback:
    @patch("prompt.InferenceClient")
    def test_falls_back_when_ollama_connection_error(self, mock_cls):
        hf_client = _mock_client("HF reply")
        # First call (Ollama) raises ConnectionError; second call (HF) succeeds.
        mock_cls.side_effect = [
            _make_failing_client(ConnectionError("refused")),
            hf_client,
        ]
        result = ask(_SINGLE_TURN)
        assert result == "HF reply"
        assert mock_cls.call_count == 2

    @patch("prompt.InferenceClient")
    def test_hf_fallback_uses_hf_model(self, mock_cls):
        hf_client = _mock_client("HF reply")
        mock_cls.side_effect = [
            _make_failing_client(ConnectionError("refused")),
            hf_client,
        ]
        ask(_SINGLE_TURN)
        _, kwargs = hf_client.chat_completion.call_args
        assert kwargs.get("model") == _HF_MODEL

    @patch("prompt.InferenceClient")
    def test_ollama_http_error_propagates_not_fallback(self, mock_cls):
        mock_cls.return_value = _make_failing_client(_http_error(500))
        with pytest.raises(HfHubHTTPError):
            ask(_SINGLE_TURN)
        # Should not have tried HF (only one InferenceClient constructed)
        assert mock_cls.call_count == 1

    def test_hf_fallback_raises_without_token(self):
        with patch("prompt.InferenceClient") as mock_cls:
            mock_cls.side_effect = [
                _make_failing_client(ConnectionError("refused")),
            ]
            with patch.dict(os.environ, {}, clear=True):
                with pytest.raises(EnvironmentError, match="HF_TOKEN"):
                    ask(_SINGLE_TURN)


def _make_failing_client(exc):
    """Return a mock InferenceClient whose chat_completion raises exc."""
    mock = MagicMock()
    mock.chat_completion.side_effect = exc
    return mock


# ---------------------------------------------------------------------------
# _hf_call retry logic — no network
# ---------------------------------------------------------------------------

class TestHfCallRetry:
    @patch("prompt.time.sleep")
    def test_succeeds_on_first_try(self, mock_sleep):
        fn = MagicMock(return_value="ok")
        assert _hf_call(fn, "arg") == "ok"
        mock_sleep.assert_not_called()

    @patch("prompt.time.sleep")
    def test_retries_on_429_and_succeeds(self, mock_sleep):
        fn = MagicMock(side_effect=[_http_error(429), "ok"])
        assert _hf_call(fn) == "ok"
        assert fn.call_count == 2
        mock_sleep.assert_called_once_with(_RETRY_DELAYS[0])

    @patch("prompt.time.sleep")
    def test_retries_on_503_and_succeeds(self, mock_sleep):
        fn = MagicMock(side_effect=[_http_error(503), "ok"])
        assert _hf_call(fn) == "ok"
        assert fn.call_count == 2

    @patch("prompt.time.sleep")
    def test_respects_retry_after_header(self, mock_sleep):
        fn = MagicMock(side_effect=[_http_error(429, retry_after="7"), "ok"])
        _hf_call(fn)
        mock_sleep.assert_called_once_with(7.0)

    @patch("prompt.time.sleep")
    def test_raises_after_all_retries_exhausted(self, mock_sleep):
        fn = MagicMock(side_effect=[_http_error(429)] * (len(_RETRY_DELAYS) + 1))
        with pytest.raises(HfHubHTTPError):
            _hf_call(fn)
        assert fn.call_count == len(_RETRY_DELAYS) + 1

    @patch("prompt.time.sleep")
    def test_non_retriable_error_raises_immediately(self, mock_sleep):
        fn = MagicMock(side_effect=_http_error(400))
        with pytest.raises(HfHubHTTPError):
            _hf_call(fn)
        assert fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("prompt.time.sleep")
    def test_uses_increasing_backoff_on_consecutive_failures(self, mock_sleep):
        fn = MagicMock(side_effect=[_http_error(429), _http_error(429), "ok"])
        _hf_call(fn)
        assert mock_sleep.call_args_list == [call(_RETRY_DELAYS[0]), call(_RETRY_DELAYS[1])]


# ---------------------------------------------------------------------------
# Integration tests — real Ollama call, requires running server
# ---------------------------------------------------------------------------

_OLLAMA_RUNNING = False
try:
    import urllib.request
    urllib.request.urlopen("http://localhost:11434", timeout=1)
    _OLLAMA_RUNNING = True
except Exception:
    pass

needs_ollama = pytest.mark.skipif(not _OLLAMA_RUNNING, reason="Ollama not running")


@needs_ollama
class TestAskOllamaE2E:
    """Real Ollama calls — require local Ollama server with llama3.1."""

    def test_single_turn_returns_nonempty_string(self):
        result = ask([{"role": "user", "content": "Say the word 'hello'."}])
        assert isinstance(result, str) and len(result) > 0

    def test_multi_turn_retains_context(self):
        messages = [
            {"role": "user",      "content": "My favourite colour is blue. Remember that."},
            {"role": "assistant", "content": "Got it, your favourite colour is blue."},
            {"role": "user",      "content": "What is my favourite colour?"},
        ]
        result = ask(messages)
        assert "blue" in result.lower()

    def test_different_prompts_give_different_replies(self):
        a = ask([{"role": "user", "content": "What is 2 + 2?"}])
        b = ask([{"role": "user", "content": "Name a random animal."}])
        assert a != b

    def test_reply_is_not_empty(self):
        result = ask([{"role": "user", "content": "Ping"}])
        assert result.strip()


# ---------------------------------------------------------------------------
# Integration tests — real HF API, require HF_TOKEN + enabled provider
# ---------------------------------------------------------------------------

@needs_token
class TestAskHFE2E:
    """Real HF API calls — require HF_TOKEN and an enabled inference provider."""

    def test_single_turn_returns_nonempty_string(self):
        # Force HF path by pointing Ollama at an invalid port
        with patch("prompt._OLLAMA_URL", "http://localhost:19999"):
            result = ask([{"role": "user", "content": "Say the word 'hello'."}])
        assert isinstance(result, str) and len(result) > 0
