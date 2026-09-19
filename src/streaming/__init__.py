"""Streaming: watermark-aware consumer, windowed features (Phase 4)."""
from src.streaming.consumer import EventTimeConsumer, Outcome, ProcessingResult
from src.streaming.features import WindowSpec, compute_window_features
from src.streaming.watermarks import WatermarkTracker

__all__ = ["EventTimeConsumer", "Outcome", "ProcessingResult", "WindowSpec", "compute_window_features", "WatermarkTracker"]
