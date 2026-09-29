"""Structured logging setup (production cleanliness).

One factory for every module: JSON-structured records (machine-parseable) with an
env-configurable level and format. Prompts, payloads, and PII are NEVER logged -
only lengths, ids, and failure classes.

Env knobs:
- C360_LOG_LEVEL: DEBUG|INFO|WARNING|ERROR (default INFO)
- C360_LOG_FORMAT: json|console (default json in production, console when a TTY)
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime

_RESERVED = logging.LogRecord("r", 0, "p", 0, "m", None, None).__dict__.keys()


class JsonFormatter(logging.Formatter):
    """JSON lines formatter: one machine-parseable object per record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure(force_console: bool | None = None) -> None:
    """Idempotently configure the root 'c360' logger from the environment."""
    root = logging.getLogger("c360")
    if getattr(root, "_c360_configured", False):
        return
    level = os.environ.get("C360_LOG_LEVEL", "INFO").upper()
    fmt = os.environ.get("C360_LOG_FORMAT") or ("console" if (force_console if force_console is not None else sys.stderr.isatty()) else "json")
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
    root.addHandler(handler)
    root.setLevel(getattr(logging, level, logging.INFO))
    root._c360_configured = True  # type: ignore[attr-defined]
    root.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Module logger under the 'c360' namespace (configured on first use)."""
    configure()
    return logging.getLogger(f"c360.{name}")
