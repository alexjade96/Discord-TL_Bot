"""Chat completion with Ollama-first routing and HF Inference API fallback.

Tries the local Ollama server first (no token required, no rate limits).
Falls back to the HF auto-router if Ollama is not running.

Public API:
    ask(messages) -> str

Environment:
    HF_TOKEN — HuggingFace Inference API token (required for HF fallback)
"""

from __future__ import annotations

import os
import time

from huggingface_hub import InferenceClient

try:
    from huggingface_hub.errors import HfHubHTTPError
except ImportError:
    from huggingface_hub.utils import HfHubHTTPError  # type: ignore[no-redef]

_OLLAMA_URL   = "http://localhost:11434"
_OLLAMA_MODEL = "llama3.1"

# HF auto-router fallback — works once a provider (Together AI, Fireworks, etc.)
# is enabled at huggingface.co → Settings → Inference Providers.
_HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

_MAX_NEW_TOKENS = 512
_RETRY_DELAYS = (1, 4)  # seconds between attempts; len+1 == total attempts


def _hf_call(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying on 429/503 with backoff.

    Respects the Retry-After response header when present.
    Raises immediately for any other HTTP error or after all retries are spent.
    """
    for attempt, delay in enumerate(_RETRY_DELAYS + (None,)):
        try:
            return fn(*args, **kwargs)
        except HfHubHTTPError as exc:
            if exc.response.status_code not in (429, 503) or delay is None:
                raise
            wait = float(exc.response.headers.get("Retry-After", delay))
            time.sleep(wait)


def ask(messages: list[dict]) -> str:
    """Send a conversation and return the assistant reply.

    Tries Ollama (local) first, then falls back to HF Inference API.

    Args:
        messages: List of {"role": "user"/"assistant", "content": "..."} dicts.

    Returns:
        The model's reply as a plain string.
    """
    # --- Ollama (local, primary) ---
    try:
        client = InferenceClient(base_url=_OLLAMA_URL, api_key="ollama")
        response = _hf_call(
            client.chat_completion, messages,
            model=_OLLAMA_MODEL, max_tokens=_MAX_NEW_TOKENS,
        )
        return response.choices[0].message.content.strip()
    except HfHubHTTPError:
        raise  # real API error from Ollama — propagate, don't fall through
    except Exception:
        pass  # Ollama not running — fall through to HF

    # --- HF Inference API (fallback) ---
    token = os.getenv("HF_TOKEN")
    if not token:
        raise EnvironmentError("HF_TOKEN environment variable is not set.")
    client = InferenceClient(api_key=token)
    response = _hf_call(
        client.chat_completion, messages,
        model=_HF_MODEL, max_tokens=_MAX_NEW_TOKENS,
    )
    return response.choices[0].message.content.strip()
