from __future__ import annotations

import hashlib
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
    pdf_path: Path | None
    source_language: str = "unknown"
    translations: List[str] = field(default_factory=list)
    video_hash: str | None = None
    pdf_hash: str | None = None


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
        pdf_path=Path(payload["pdf_path"]) if payload.get("pdf_path") else None,
        source_language=payload.get("source_language", "unknown"),
        translations=list(payload.get("translations", [])),
        video_hash=payload.get("video_hash"),
        pdf_hash=payload.get("pdf_hash"),
    )


def save_metadata(
    lecture_dir: Path,
    *,
    lecture_id: str,
    title: str,
    video_path: Path | None,
    source_language: str = "unknown",
    translations: List[str] | None = None,
    video_hash: str | None = None,
    pdf_path: Path | None = None,
    pdf_hash: str | None = None,
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
        translations=list(translations or (existing.translations if existing else [])),
        video_hash=video_hash or (existing.video_hash if existing else None),
        pdf_path=pdf_path.resolve() if pdf_path else (existing.pdf_path if existing else None),
        pdf_hash=pdf_hash or (existing.pdf_hash if existing else None),
    )
    metadata_path.write_text(
        json.dumps(
            {
                "lecture_id": record.lecture_id,
                "title": record.title,
                "video_path": str(record.video_path) if record.video_path else "",
                "pdf_path": str(record.pdf_path) if record.pdf_path else "",
                "created_at": record.created_at.isoformat(),
                "updated_at": record.updated_at.isoformat(),
                "source_language": record.source_language,
                "translations": record.translations,
                "video_hash": record.video_hash,
                "pdf_hash": record.pdf_hash,
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
        video_hash=metadata.video_hash,
        pdf_path=metadata.pdf_path,
        pdf_hash=metadata.pdf_hash,
    )


def rename_lecture(old_id: str, new_title: str) -> LectureRecord:
    old_dir = get_lecture_dir(old_id)
    if not old_dir.exists():
        raise FileNotFoundError(f"Lecture {old_id} does not exist.")
    metadata = load_metadata(old_dir)
    if not metadata:
        raise FileNotFoundError(f"Metadata for {old_id} not found.")
    new_id = slugify(new_title)
    if new_id == old_id:
        return save_metadata(
            old_dir,
            lecture_id=old_id,
            title=new_title,
            video_path=metadata.video_path,
            source_language=metadata.source_language,
            translations=metadata.translations,
            video_hash=metadata.video_hash,
            pdf_path=metadata.pdf_path,
            pdf_hash=metadata.pdf_hash,
        )
    new_dir = get_lecture_dir(new_id)
    if new_dir.exists():
        raise FileExistsError(f"Lecture ID {new_id} already exists.")
    old_dir.rename(new_dir)
    video_path = metadata.video_path
    if video_path:
        try:
            rel_path = video_path.relative_to(old_dir)
            video_path = new_dir / rel_path
        except ValueError:
            pass
    return save_metadata(
        new_dir,
        lecture_id=new_id,
        title=new_title,
        video_path=video_path,
        source_language=metadata.source_language,
        translations=metadata.translations,
        video_hash=metadata.video_hash,
        pdf_path=metadata.pdf_path,
        pdf_hash=metadata.pdf_hash,
    )


def compute_file_hash(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()
