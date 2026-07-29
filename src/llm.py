"""Shared single-prompt text-completion helper with provider fallback.

A generic, free-form counterpart to `src/rank.py`'s scorer: given one prompt, return the
model's text. Providers run in order — Anthropic (Claude) primary, OpenAI (GPT) fallback.
An *unavailable* condition (billing/auth/quota) drops that provider and the next takes over;
a transient error just tries the next provider. With no keys / no packages, `complete`
returns None quietly. It NEVER raises to the caller — so an assist step that wants a
completion can treat None as "no LLM available" and degrade gracefully.
"""
from __future__ import annotations

import logging
import os
from typing import Callable, NamedTuple

log = logging.getLogger("llm")


# --- providers ---------------------------------------------------------------

class _Provider(NamedTuple):
    name: str
    client: object
    # (client, prompt, model, max_tokens) -> raw model text
    call: Callable[[object, str, str, int], str]


def _anthropic_client():
    key = os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        return None
    try:
        import anthropic
    except ImportError:
        log.warning("anthropic package not installed — Claude completion disabled")
        return None
    return anthropic.Anthropic(api_key=key)


def _openai_client():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        return None
    try:
        import openai
    except ImportError:
        log.warning("openai package not installed — GPT fallback disabled")
        return None
    return openai.OpenAI(api_key=key)


def _call_anthropic(client, prompt: str, model: str, max_tokens: int) -> str:
    resp = client.messages.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _call_openai(client, prompt: str, model: str, max_tokens: int) -> str:
    # gpt-5 and the o-series are reasoning models: they reject the legacy `max_tokens`
    # param (need `max_completion_tokens`) AND spend part of that budget on hidden
    # reasoning tokens — too small a budget leaves nothing for the answer (empty text).
    # So give reasoning models plenty of headroom; older chat models use `max_tokens`.
    if model.startswith("gpt-5"):
        # reasoning_effort='minimal' keeps reasoning tokens near zero so the answer
        # actually fits the budget (default effort can consume the whole thing -> empty).
        kwargs = {"max_completion_tokens": max(max_tokens, 8192), "reasoning_effort": "minimal"}
    elif model.startswith(("o1", "o3", "o4")):
        kwargs = {"max_completion_tokens": max(max_tokens, 8192)}
    else:
        kwargs = {"max_tokens": max_tokens}
    resp = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}], **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def _providers() -> list[_Provider]:
    """Available providers in priority order: Anthropic primary, OpenAI fallback."""
    out: list[_Provider] = []
    a = _anthropic_client()
    if a is not None:
        out.append(_Provider("anthropic", a, _call_anthropic))
    o = _openai_client()
    if o is not None:
        out.append(_Provider("openai", o, _call_openai))
    return out


def _is_unavailable(exc: Exception) -> bool:
    """True when an error means the provider is *out of service* for this run (billing,
    auth, quota) — as opposed to a transient blip. Drives dropping vs. retrying a provider.
    """
    status = getattr(exc, "status_code", None) or getattr(
        getattr(exc, "response", None), "status_code", None)
    if status in (401, 403):
        return True
    msg = str(exc).lower()
    return any(s in msg for s in (
        "credit balance is too low", "insufficient_quota", "billing",
        "exceeded your current quota", "invalid_api_key", "invalid api key",
        "authentication", "account is not active",
    ))


def _short(exc: Exception) -> str:
    return str(exc).splitlines()[0][:120] if str(exc) else exc.__class__.__name__


def complete(
    prompt: str,
    *,
    max_tokens: int = 2048,
    anthropic_model: str = os.environ.get("LLM_ANTHROPIC_MODEL", "claude-sonnet-4-6"),
    openai_model: str = os.environ.get("LLM_OPENAI_MODEL", "gpt-4o"),
) -> str | None:
    """Return the first provider's non-empty completion text, or None.

    Iterates providers in priority order. An *unavailable* error drops that provider and
    tries the next; a transient error just tries the next. Returns the first successful
    non-empty stripped text. Returns None if no provider is configured or every provider
    failed / was unavailable. NEVER raises to the caller.
    """
    for prov in _providers():
        model = anthropic_model if prov.name == "anthropic" else openai_model
        try:
            text = (prov.call(prov.client, prompt, model, max_tokens) or "").strip()
        except Exception as exc:  # noqa: BLE001 — never let a completion crash the caller
            if _is_unavailable(exc):
                log.error("llm %s unavailable — %s", prov.name, _short(exc))
            else:
                log.error("llm %s call failed: %s", prov.name, exc)
            continue
        if text:
            log.info("llm completion served by %s", prov.name)
            return text
        log.warning("llm %s returned empty text — trying next provider", prov.name)
    return None
