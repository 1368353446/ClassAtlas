from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List

from .models import KnowledgeOutline, SlideManifestEntry, TranscriptSegment


def _read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Expected file at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _segment_cache(path: str) -> List[TranscriptSegment]:
    payload = _read_json(Path(path))
    return [TranscriptSegment.from_dict(item) for item in payload]


@lru_cache(maxsize=None)
def _outline_cache(path: str) -> KnowledgeOutline:
    payload = _read_json(Path(path))
    return KnowledgeOutline.from_dict(payload)


@lru_cache(maxsize=None)
def _slide_cache(path: str) -> List[SlideManifestEntry]:
    payload = _read_json(Path(path))
    return [SlideManifestEntry.from_dict(item) for item in payload]


def load_transcript_segments(base_dir: Path) -> List[TranscriptSegment]:
    return _segment_cache(str((base_dir / "transcript_segments.json").resolve()))


def load_knowledge_outline(base_dir: Path) -> KnowledgeOutline:
    return _outline_cache(str((base_dir / "knowledge_outline_enriched.json").resolve()))


def load_slide_manifest(base_dir: Path) -> List[SlideManifestEntry]:
    return _slide_cache(str((base_dir / "slides_manifest.json").resolve()))
