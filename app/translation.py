from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .qa import build_default_llm

LANGUAGE_CONFIG = {
    "zh": {"label": "中文", "name": "Chinese"},
    "en": {"label": "English", "name": "English"},
}


class Translator:
    def __init__(self, target_language: str):
        if target_language not in LANGUAGE_CONFIG:
            raise ValueError(f"Unsupported language: {target_language}")
        self.target_language = target_language
        self.language_name = LANGUAGE_CONFIG[target_language]["name"]
        self.llm = build_default_llm()
        if self.llm is None:
            raise RuntimeError("LLM is not configured; translation is unavailable.")

    def translate_text(self, text: str) -> str:
        if not text:
            return text
        prompt = (
            f"Translate the following text into {self.language_name}. "
            "Preserve meaning and keep the tone natural.\n\n"
            f"Text:\n{text}"
        )
        response = self.llm.invoke(prompt)
        content = getattr(response, "content", None)
        if isinstance(content, list):
            return "".join(part.get("text", "") for part in content)
        return str(content or response)


def translate_segments_file(source_path: Path, target_dir: Path, translator: Translator):
    segments = json.loads(source_path.read_text(encoding="utf-8"))
    for segment in segments:
        segment["text"] = translator.translate_text(segment.get("text", ""))
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "transcript_segments.json").write_text(
        json.dumps(segments, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def translate_outline_file(source_path: Path, target_dir: Path, translator: Translator):
    outline = json.loads(source_path.read_text(encoding="utf-8"))
    outline["session_topic"] = translator.translate_text(outline.get("session_topic", ""))
    outline["overall_summary"] = translator.translate_text(outline.get("overall_summary", ""))
    for kp in outline.get("knowledge_points", []):
        for field in ["title", "summary", "teaching_method", "emphasis"]:
            kp[field] = translator.translate_text(kp.get(field, ""))
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "knowledge_outline_enriched.json").write_text(
        json.dumps(outline, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def translate_outputs(lecture_dir: Path, languages: List[str]):
    for lang in languages:
        translator = Translator(lang)
        translation_dir = lecture_dir / "translations" / lang
        translate_segments_file(lecture_dir / "transcript_segments.json", translation_dir, translator)
        translate_outline_file(lecture_dir / "knowledge_outline_enriched.json", translation_dir, translator)
