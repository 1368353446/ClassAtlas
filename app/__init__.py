"""Top-level package exports."""

from .logging_config import configure_logging

configure_logging()

from .knowledge_base import build_assets  # noqa: E402
from .pipeline.runner import run_pipeline  # noqa: E402

__all__ = ["build_assets", "run_pipeline"]
