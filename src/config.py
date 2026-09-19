"""Central configuration for the Agentic Customer 360 system.

All configuration is environment-driven with C360_* variables so the same code
runs against local docker-compose services, CI, or a demo deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def load_groq_env(path: str = os.path.expanduser("~/.groq_env")) -> dict[str, str]:
    """Load KEY=VALUE pairs from a dotenv-style file if present (never logged)."""
    values: dict[str, str] = {}
    p = Path(path)
    if p.is_file():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                values[k.strip()] = v.strip()
    return values


@dataclass(frozen=True)
class Settings:
    """Frozen settings object; construct once per process via ``get_settings``."""

    # Event bus (Redpanda / Kafka API)
    kafka_bootstrap: str = field(default_factory=lambda: _env("C360_KAFKA_BOOTSTRAP", "localhost:9092"))
    kafka_topic_prefix: str = field(default_factory=lambda: _env("C360_KAFKA_TOPIC_PREFIX", "c360"))

    # Postgres (state board, episodic store, identity graph, retrieval)
    pg_dsn: str = field(default_factory=lambda: _env("C360_PG_DSN", "postgresql://c360:c360@localhost:5433/c360"))

    # Redis (hot current-state read cache)
    redis_url: str = field(default_factory=lambda: _env("C360_REDIS_URL", "redis://localhost:6380/0"))

    # LLM (Groq, OpenAI-compatible endpoint)
    llm_base_url: str = field(default_factory=lambda: _env("C360_LLM_BASE_URL", "https://api.groq.com/openai/v1"))
    llm_model: str = field(default_factory=lambda: _env("C360_LLM_MODEL", "llama-3.3-70b-versatile"))
    llm_api_key: str = field(
        default_factory=lambda: _env("C360_LLM_API_KEY", "") or load_groq_env().get("GROQ_API_KEY", "")
    )
    llm_enabled: bool = field(default_factory=lambda: _env("C360_LLM_ENABLED", "1") == "1")
    llm_timeout_s: float = field(default_factory=lambda: _env_float("C360_LLM_TIMEOUT_S", 30.0))

    # Observability (OTLP HTTP endpoint of the local collector)

    # Streaming / watermark semantics (ADR-02)
    watermark_out_of_order_max_s: float = field(default_factory=lambda: _env_float("C360_WATERMARK_OOO_MAX_S", 30.0))
    watermark_idle_source_s: float = field(default_factory=lambda: _env_float("C360_WATERMARK_IDLE_S", 5.0))
    allowed_lateness_s: float = field(default_factory=lambda: _env_float("C360_ALLOWED_LATENESS_S", 900.0))

    # Memory / decay (RQ-M2)
    default_half_life_days: float = field(default_factory=lambda: _env_float("C360_HALF_LIFE_DAYS", 30.0))
    decay_min_confidence: float = field(default_factory=lambda: _env_float("C360_DECAY_MIN_CONF", 0.10))

    # Retrieval (RQ-R2: score = similarity - lambda * age_days)
    retrieval_recency_lambda: float = field(default_factory=lambda: _env_float("C360_RECENCY_LAMBDA", 0.01))
    retrieval_top_k: int = field(default_factory=lambda: _env_int("C360_RETRIEVAL_TOP_K", 8))

    # Working memory bounds
    recent_events_ring_size: int = field(default_factory=lambda: _env_int("C360_RING_SIZE", 50))

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def set_settings(settings: Settings) -> None:
    """Inject settings explicitly (used by tests and scripts)."""
    global _settings
    _settings = settings
