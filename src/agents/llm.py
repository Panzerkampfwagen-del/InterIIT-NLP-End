"""LLM client (Phase 5, ADR-05) - Groq via the OpenAI-compatible SDK.

Production-hardened while keeping the strict fallback contract (per the PS and
prompts): every LLM-dependent step returns ``None`` on failure and the calling
agent then flows its **deterministic statistical findings** onward - the
pipeline must never require a live LLM to produce a valid finding.

Hardening:
- bounded retry with exponential backoff + jitter on TRANSIENT failures
  (429 rate limits, timeouts, 5xx); PERMANENT failures (auth/invalid) fail fast;
- context-window overflow: oversized prompts are truncated once and retried,
  so a big evidence bundle degrades instead of losing the signal entirely;
- availability is re-probed on a TTL: one transient blip (at startup or mid-run)
  no longer disables the LLM for the process lifetime;
- PII is sanitized at the ``complete()`` choke point before any prompt leaves
  the process (``redact_value``);
- structured logging only - prompts and payload content are never logged.
"""
from __future__ import annotations

import json
import random
import time
from typing import Any, Protocol

from src.config import get_settings
from src.logging_setup import get_logger
from src.observability.pii_scan import redact_value

log = get_logger("llm")

# Preference-ordered model fallback (confirmed at runtime via /models).
PREFERRED_MODELS = (
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
)

# --- retry policy (bounded, exponential backoff with jitter) ---
MAX_RETRIES = 2
BACKOFF_BASE_S = 0.5
BACKOFF_CAP_S = 4.0

# --- context-window overflow guard ---
TRUNCATED_USER_CHARS = 60_000

# --- availability probe caching ---
AVAILABLE_TTL_S = 300.0
PROBE_RETRY_TTL_S = 60.0

# response size bound (inference JSON is small)
MAX_OUTPUT_TOKENS = 2048

# retryable HTTP status codes (transient); everything else 4xx is permanent
_TRANSIENT_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _classify(exc: Exception) -> str:
    """Classify an LLM failure: overflow | transient | permanent."""
    text = str(exc).lower()
    name = type(exc).__name__.lower()
    if "context_length" in text or "context length" in text or "too many tokens" in text:
        return "overflow"
    status = getattr(exc, "status_code", None)
    if status is not None:
        return "transient" if status in _TRANSIENT_STATUS else "permanent"
    if "timeout" in name or "connection" in name or "ratelimit" in name.replace("_", ""):
        return "transient"
    if "auth" in text or "permission" in text or "invalid_api_key" in text or "not found" in text:
        return "permanent"
    return "transient"  # unknown: safest to retry a bounded number of times


class LLMClient(Protocol):
    """Minimal LLM interface: structured completion with fallback."""

    def complete(self, system: str, user: str) -> dict[str, Any] | None:
        """Return parsed JSON or None on any failure (fallback contract)."""
        ...

    def available(self) -> bool:
        ...


class GroqLLMClient:
    """Groq (OpenAI-compatible) client; model confirmed at runtime."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        s = get_settings()
        self.api_key = api_key if api_key is not None else s.llm_api_key
        self.base_url = base_url or s.llm_base_url
        self.model = model or s.llm_model
        self.timeout_s = timeout_s if timeout_s is not None else s.llm_timeout_s
        self._available: bool | None = None
        self._probed_at: float | None = None
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    def available(self) -> bool:
        """Confirm the key works and the model exists; verdict cached on a TTL.

        A failed probe is retried after PROBE_RETRY_TTL_S (no longer sticky),
        so one transient network blip cannot disable the LLM for the run.
        """
        now = time.monotonic()
        if self._available is True and self._probed_at is not None and (now - self._probed_at) < AVAILABLE_TTL_S:
            return True
        if self._available is False and self._probed_at is not None and (now - self._probed_at) < PROBE_RETRY_TTL_S:
            return False
        if not self.api_key:
            self._available, self._probed_at = False, now
            return False
        try:
            models = self._get_client().models.list()
            ids = {m.id for m in models.data}
            if self.model in ids:
                self._available = True
            else:
                for pref in PREFERRED_MODELS:
                    if pref in ids:
                        self.model = pref
                        self._available = True
                        break
                else:
                    self._available = False
        except Exception as exc:
            if self._available is not True:
                log.warning("llm_probe_failed", extra={"error": type(exc).__name__})
            # keep a previously-successful verdict on a transient re-probe failure
            if self._available is None:
                self._available = False
        finally:
            self._probed_at = now
        return self._available is True

    def complete(self, system: str, user: str) -> dict[str, Any] | None:
        """Structured JSON completion; None on any failure (fallback contract).

        Sanitization choke point: PII is redacted before the prompt leaves the
        process. Transient failures retry with backoff; context overflow
        truncates the prompt once and retries; permanent failures fail fast.
        """
        if not self.available():
            return None
        prompt = redact_value(user)
        for attempt in range(MAX_RETRIES + 1):
            try:
                response = self._get_client().chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.1,
                    max_tokens=MAX_OUTPUT_TOKENS,
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as exc:
                kind = _classify(exc)
                log.warning(
                    "llm_complete_failed",
                    extra={"attempt": attempt + 1, "class": kind, "error": type(exc).__name__, "user_chars": len(prompt)},
                )
                if kind == "overflow" and len(prompt) > TRUNCATED_USER_CHARS:
                    prompt = prompt[:TRUNCATED_USER_CHARS]  # degrade, retry
                    continue
                if kind == "transient" and attempt < MAX_RETRIES:
                    delay = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (2**attempt)) * (1.0 + random.random() * 0.1)
                    time.sleep(delay)
                    continue
                return None  # permanent, retries exhausted, or non-truncatable
        return None


class FakeLLMClient:
    """Deterministic LLM double for tests (never touches the network)."""

    def __init__(self, responses: list[dict[str, Any]] | None = None, fail: bool = False) -> None:
        self._responses = list(responses or [])
        self._fail = fail
        self.calls: list[tuple[str, str]] = []

    def available(self) -> bool:
        return not self._fail

    def complete(self, system: str, user: str) -> dict[str, Any] | None:
        self.calls.append((system, user))
        if self._fail:
            return None
        if self._responses:
            return self._responses.pop(0)
        return {}


def get_llm_client() -> LLMClient:
    """Build the configured client (Groq per ADR-05)."""
    s = get_settings()
    if not s.llm_enabled:
        return FakeLLMClient(fail=True)
    return GroqLLMClient()
