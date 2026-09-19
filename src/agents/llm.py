"""LLM client (Phase 5, ADR-05) - Groq via the OpenAI-compatible SDK.

Strict fallback contract (per the PS and prompts): every LLM-dependent step
returns ``None`` on failure (missing key, network error, invalid JSON) and the
calling agent then flows its **deterministic statistical findings** onward -
the pipeline must never require a live LLM to produce a valid finding.

Model availability is confirmed at runtime against the provider's model list
(web-research policy: confirm model/library versions); a preference-ordered
fallback list is used.
"""
from __future__ import annotations

import json
from typing import Any, Protocol

from src.config import get_settings

# Preference-ordered model fallback (confirmed at runtime via /models).
PREFERRED_MODELS = (
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
)


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
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)
        return self._client

    def available(self) -> bool:
        """Confirm the API key works and the model exists (runtime check)."""
        if self._available is not None:
            return self._available
        if not self.api_key:
            self._available = False
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
        except Exception:
            self._available = False
        return self._available

    def complete(self, system: str, user: str) -> dict[str, Any] | None:
        """Structured JSON completion; None on any failure (fallback contract)."""
        if not self.available():
            return None
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception:
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
