from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .loaders import load_knowledge_outline, load_slide_manifest, load_transcript_segments
from .models import KnowledgeOutline, SlideManifestEntry, TranscriptSegment


@dataclass
class KnowledgeAssets:
    segments: List[TranscriptSegment]
    outline: KnowledgeOutline
    slides: List[SlideManifestEntry]


def build_assets(lecture_dir: Path, language: str | None = None) -> KnowledgeAssets:
    base_dir = lecture_dir
    if language and language != "base":
        base_dir = lecture_dir / "translations" / language
    segments = load_transcript_segments(base_dir)
    outline = load_knowledge_outline(base_dir)
    slides = load_slide_manifest(lecture_dir)
    return KnowledgeAssets(
        segments=segments,
        outline=outline,
        slides=slides,
    )


def slides_by_index(slides: List[SlideManifestEntry]) -> dict[int, SlideManifestEntry]:
    return {slide.slide_index: slide for slide in slides}
