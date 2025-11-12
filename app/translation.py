from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

from .llm_utils import AgentNotAvailableError, invoke_text_prompt

logger = logging.getLogger(__name__)

LANGUAGE_CONFIG = {
    "zh": {"label": "中文", "name": "Chinese"},
    "en": {"label": "English", "name": "English"},
    "ja": {"label": "日本語", "name": "Japanese"},
    "ko": {"label": "한국어", "name": "Korean"},
    "fr": {"label": "Français", "name": "French"},
    "de": {"label": "Deutsch", "name": "German"},
    "es": {"label": "Español", "name": "Spanish"},
    "pt": {"label": "Português", "name": "Portuguese"},
    "ru": {"label": "Русский", "name": "Russian"},
    "hi": {"label": "हिंदी", "name": "Hindi"},
}


class Translator:
    def __init__(self, target_language: str):
        if target_language not in LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {target_language}")
        self.target_language = target_language
        self.language_name = LANGUAGE_CONFIG[target_language]["name"]
        self.system_template = (
            "You are a professional translator. Convert any provided text into {language_name} while keeping meaning and tone. "
            "Do not add explanations."
        )
        self.human_template = (
            "Text:\n{text}\n\n"
            "{input}"
        )

    def translate_text(self, text: str) -> str:
        if not text:
            return text
        try:
            translated = invoke_text_prompt(
                system_template=self.system_template,
                human_template=self.human_template,
                variables={
                    "language_name": self.language_name,
                    "text": text,
                    "input": "Return only the translated text.",
                },
            )
            translated = translated.strip()
            logger.info("Translated text to %s (%d chars).", self.target_language, len(translated))
            return translated
        except AgentNotAvailableError as exc:
            raise RuntimeError("LLM is not configured; translation is unavailable.") from exc


def translate_segments_file(source_path: Path, target_dir: Path, translator: Translator):
    segments = json.loads(source_path.read_text(encoding="utf-8"))
    for segment in segments:
        segment["text"] = translator.translate_text(segment.get("text", ""))
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "transcript_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Translated transcript_segments.json into %s.", translator.target_language)


def translate_outline_file(source_path: Path, target_dir: Path, translator: Translator):
    outline = json.loads(source_path.read_text(encoding="utf-8"))
    outline["session_topic"] = translator.translate_text(outline.get("session_topic", ""))
    outline["overall_summary"] = translator.translate_text(outline.get("overall_summary", ""))
    for kp in outline.get("knowledge_points", []):
        for field in ["title", "summary", "content", "teaching_method", "emphasis"]:
            kp[field] = translator.translate_text(kp.get(field, ""))
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "knowledge_outline_enriched.json").write_text(
        json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Translated knowledge_outline_enriched.json into %s.", translator.target_language)


def translate_outputs(lecture_dir: Path, languages: List[str]):
    for lang in languages:
        translator = Translator(lang)
        translation_dir = lecture_dir / "translations" / lang
        translate_segments_file(lecture_dir / "transcript_segments.json", translation_dir, translator)
        translate_outline_file(lecture_dir / "knowledge_outline_enriched.json", translation_dir, translator)
        logger.info("Completed translation for %s in %s.", lang, lecture_dir)
