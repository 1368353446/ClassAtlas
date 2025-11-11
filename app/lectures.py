from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List

from .config import LECTURE_ROOT

METADATA_FILENAME = "metadata.json"


@dataclass
class LectureRecord:
    lecture_id: str
    title: str
    path: Path
    created_at: datetime
    updated_at: datetime
    video_path: Path | None
    source_language: str = "unknown"
    translations: List[str] = field(default_factory=list)


def _now() -> datetime:
    return datetime.utcnow()


def ensure_root() -> Path:
    LECTURE_ROOT.mkdir(parents=True, exist_ok=True)
    return LECTURE_ROOT


def slugify(text: str) -> str:
    text = re.sub(r"\s+", "-", text.strip().lower())
    text = re.sub(r"[^a-z0-9\-]", "", text)
    text = text.strip("-")
    return text or f"lecture-{_now().strftime('%Y%m%d%H%M%S')}"


def load_metadata(lecture_dir: Path) -> LectureRecord | None:
    metadata_path = lecture_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    return LectureRecord(
        lecture_id=payload.get("lecture_id", lecture_dir.name),
        title=payload.get("title", lecture_dir.name),
        path=lecture_dir,
        created_at=datetime.fromisoformat(payload.get("created_at")),
        updated_at=datetime.fromisoformat(payload.get("updated_at")),
        video_path=Path(payload["video_path"]) if payload.get("video_path") else None,
        source_language=payload.get("source_language", "unknown"),
        translations=list(payload.get("translations", [])),
    )


def save_metadata(
    lecture_dir: Path,
    *,
    lecture_id: str,
    title: str,
    video_path: Path | None,
    source_language: str = "unknown",
    translations: List[str] | None = None,
) -> LectureRecord:
    lecture_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = lecture_dir / METADATA_FILENAME
    existing = load_metadata(lecture_dir)
    created_at = existing.created_at if existing else _now()
    record = LectureRecord(
        lecture_id=lecture_id,
        title=title,
        path=lecture_dir,
        created_at=created_at,
        updated_at=_now(),
        video_path=video_path.resolve() if video_path else None,
        source_language=source_language,
        translations=list(translations or existing.translations if existing else []),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "lecture_id": record.lecture_id,
                "title": record.title,
                "video_path": str(record.video_path) if record.video_path else "",
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "source_language": record.source_language,
                "translations": record.translations,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return record


def list_lectures() -> List[LectureRecord]:
    ensure_root()
    records: List[LectureRecord] = []
    for child in LECTURE_ROOT.iterdir():
        if not child.is_dir():
            continue
        metadata = load_metadata(child)
        if metadata:
            records.append(metadata)
    records.sort(key=lambda rec: rec.updated_at, reverse=True)
    return records


def get_lecture_dir(lecture_id: str) -> Path:
    return ensure_root() / lecture_id


def update_translations(lecture_dir: Path, translations: List[str]):
    metadata = load_metadata(lecture_dir)
    if not metadata:
        return
    save_metadata(
        lecture_dir,
        lecture_id=metadata.lecture_id,
        title=metadata.title,
        video_path=metadata.video_path,
        source_language=metadata.source_language,
        translations=translations,
    )
