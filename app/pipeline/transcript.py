from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import torch
from faster_whisper import WhisperModel


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def transcribe_video_audio(
    video_path: Path | str,
    *,
    model_size: str = "medium",
    compute_type: str = "float16",
) -> List:
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    model = WhisperModel(model_size, device=_device(), compute_type=compute_type)
    segments, _ = model.transcribe(str(video_path))
    return list(segments)


def serialize_transcript_segments(segments: Iterable) -> List[dict]:
    payload = []
    for idx, segment in enumerate(segments, start=1):
        payload.append(
            {
                "segment_index": idx,
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
        )
    return payload


def build_transcript_outline_text(segments: Iterable) -> str:
    lines = []
    for segment in segments:
        text = _get_text(segment)
        if not text:
            continue
        start = float(_get_attr(segment, "start", 0.0))
        end = float(_get_attr(segment, "end", start))
        lines.append(f"[{start:.2f}s - {end:.2f}s] {text}")
    return "\n".join(lines)


def infer_language_from_segments(segments: Iterable[dict] | Iterable) -> str:
    chinese = 0
    latin = 0
    for segment in segments:
        text = _get_text(segment)
        for char in text:
            if "\u4e00" <= char <= "\u9fff":
                chinese += 1
            elif char.isalpha():
                latin += 1
    if chinese == 0 and latin == 0:
        return "unknown"
    return "zh" if chinese >= latin else "en"


def _get_attr(segment, name: str, default):
    if isinstance(segment, dict):
        return segment.get(name, default)
    return getattr(segment, name, default)


def _get_text(segment) -> str:
    value = _get_attr(segment, "text", "")
    return value.strip() if isinstance(value, str) else ""
