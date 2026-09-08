"""
Thin wrapper around an OpenAI-compatible chat completion client.

Per the project spec, Nebius AI Studio (used in Sprint 16) is the primary
LLM provider. Nebius exposes an OpenAI-compatible REST API, so the
official `openai` Python SDK works against it by overriding `base_url`.
Plain OpenAI is supported too, as a convenience for local development.

If no API key is configured at all, `get_llm_client()` returns
`(None, None)` and callers fall back to the deterministic, rule-based
parser/response templates in src/interface.py. This keeps the project
runnable end-to-end (including the automated test suite) without
requiring anyone to hold a paid API key.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

DEFAULT_NEBIUS_BASE_URL = "https://api.studio.nebius.ai/v1"
DEFAULT_NEBIUS_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_llm_client() -> Tuple[Optional[object], Optional[str]]:
    """Build an OpenAI-SDK client pointed at whichever provider is
    configured via environment variables.

    Priority:
      1. NEBIUS_API_KEY  -> Nebius AI Studio (NEBIUS_BASE_URL / NEBIUS_MODEL
         optionally override the defaults above)
      2. OPENAI_API_KEY  -> OpenAI (or an OPENAI_BASE_URL-compatible proxy)
      3. neither set      -> (None, None), triggering rule-based fallback

    Returns:
        (client, model_name) or (None, None)
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None, None

    nebius_key = os.environ.get("NEBIUS_API_KEY")
    if nebius_key:
        base_url = os.environ.get("NEBIUS_BASE_URL", DEFAULT_NEBIUS_BASE_URL)
        model = os.environ.get("NEBIUS_MODEL", DEFAULT_NEBIUS_MODEL)
        return OpenAI(api_key=nebius_key, base_url=base_url), model

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        base_url = os.environ.get("OPENAI_BASE_URL")  # None -> official endpoint
        model = os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
        return OpenAI(api_key=openai_key, base_url=base_url), model

    return None, None


def chat_json(client, model: str, system_prompt: str, user_prompt: str) -> str:
    """Send one chat completion request asking for a JSON object back.
    Returns the raw JSON text from the model."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content


def chat_text(client, model: str, system_prompt: str, user_prompt: str) -> str:
    """Send one chat completion request and return plain text."""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content
