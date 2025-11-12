from __future__ import annotations

from pathlib import Path

from .settings_manager import get_settings

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_settings(force: bool = False) -> dict:
    settings = get_settings(force_reload=force)
    lecture_root_raw = settings.get("lecture_root") or "data/lectures"
    lecture_root = Path(lecture_root_raw)
    if not lecture_root.is_absolute():
        lecture_root = (PROJECT_ROOT / lecture_root).resolve()
    return {
        "MODELSCOPE_API_KEY": settings.get("llm_api_key") or "",
        "LLM_MODEL": settings.get("llm_model") or "Qwen/Qwen3-8B",
        "LLM_BASE_URL": settings.get("llm_base_url") or "https://api-inference.modelscope.cn/v1/",
        "DEFAULT_WHISPER_MODEL": settings.get("whisper_model") or "medium",
        "LECTURE_ROOT": lecture_root,
    }


_settings_cache = _resolve_settings()

MODELSCOPE_API_KEY = _settings_cache["MODELSCOPE_API_KEY"]
LLM_MODEL = _settings_cache["LLM_MODEL"]
LLM_BASE_URL = _settings_cache["LLM_BASE_URL"]
DEFAULT_WHISPER_MODEL = _settings_cache["DEFAULT_WHISPER_MODEL"]
LECTURE_ROOT = _settings_cache["LECTURE_ROOT"]


def refresh_runtime_settings():
    global MODELSCOPE_API_KEY, LLM_MODEL, LLM_BASE_URL, DEFAULT_WHISPER_MODEL, LECTURE_ROOT
    updated = _resolve_settings(force=True)
    MODELSCOPE_API_KEY = updated["MODELSCOPE_API_KEY"]
    LLM_MODEL = updated["LLM_MODEL"]
    LLM_BASE_URL = updated["LLM_BASE_URL"]
    DEFAULT_WHISPER_MODEL = updated["DEFAULT_WHISPER_MODEL"]
    LECTURE_ROOT = updated["LECTURE_ROOT"]


class UIConfig:
    PAGE_TITLE = "Lecture Knowledge Companion"
    PAGE_ICON = "🎓"
