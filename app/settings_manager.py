from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SETTINGS_PATH = Path(os.getenv("CLASS_EXTRACT_SETTINGS", PROJECT_ROOT / "settings.json"))

DEFAULT_SETTINGS: Dict[str, Any] = {
    "llm_model": "Qwen/Qwen3-8B",
    "llm_base_url": "https://api-inference.modelscope.cn/v1/",
    "llm_api_key": "",
    "whisper_model": "medium",
    "lecture_root": "data/lectures",
}

_SETTINGS_CACHE: Dict[str, Any] | None = None


def _load_from_disk() -> Dict[str, Any]:
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(DEFAULT_SETTINGS, ensure_ascii=False, indent=2), encoding="utf-8")
        return deepcopy(DEFAULT_SETTINGS)
    try:
        payload = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        merged = deepcopy(DEFAULT_SETTINGS)
        merged.update(payload)
        return merged
    except Exception:
        return deepcopy(DEFAULT_SETTINGS)


def get_settings(*, force_reload: bool = False) -> Dict[str, Any]:
    global _SETTINGS_CACHE
    if _SETTINGS_CACHE is None or force_reload:
        _SETTINGS_CACHE = _load_from_disk()
    return deepcopy(_SETTINGS_CACHE)


def save_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    data = deepcopy(DEFAULT_SETTINGS)
    data.update(settings)
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return get_settings(force_reload=True)


def update_settings(changes: Dict[str, Any]) -> Dict[str, Any]:
    current = get_settings()
    current.update(changes)
    return save_settings(current)
