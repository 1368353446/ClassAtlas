from __future__ import annotations

import logging
import os
from pathlib import Path

_CONFIGURED = False


def configure_logging() -> None:
    """
    Configure application-wide logging once.

    Respects APP_LOG_LEVEL env var (defaults to INFO) and uses a concise format
    listing timestamp, severity, logger name, and message.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return
    level_name = os.getenv("APP_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_dir = Path(os.getenv("APP_LOG_DIR", "")).expanduser()
    handlers = None
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
        handlers = [file_handler]
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    # Reduce noisy third-party loggers.
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _CONFIGURED = True
