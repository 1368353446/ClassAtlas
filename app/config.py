from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LECTURE_ROOT = Path(os.getenv("LECTURE_ROOT", PROJECT_ROOT / "data" / "lectures"))

MODELSCOPE_API_KEY = os.getenv("MODELSCOPE_API_KEY")
LLM_MODEL = os.getenv("LLM_MODEL_NAME", "Qwen/Qwen3-8B")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api-inference.modelscope.cn/v1/")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class UIConfig:
    PAGE_TITLE = "Lecture Knowledge Companion"
    PAGE_ICON = "🎓"
