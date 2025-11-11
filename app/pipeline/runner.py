from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

from ..config import LECTURE_ROOT
from ..models import KnowledgeOutline
from .outline import generate_knowledge_outline, outline_asdict
from .slides import extract_slide_keyframes, persist_slides_and_build_pdf
from .transcript import (
    build_transcript_outline_text,
    infer_language_from_segments,
    serialize_transcript_segments,
    transcribe_video_audio,
)


@dataclass
class PipelineResult:
    outline: KnowledgeOutline
    output_dir: Path
    transcript_segments_path: Path
    transcript_outline_path: Path
    knowledge_outline_path: Path
    slide_manifest_path: Path
    slides_pdf_path: Path | None
    segment_count: int
    slide_count: int
    timings: dict[str, float]
    language: str


def _write_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_pipeline(
    *,
    video_path: Path,
    output_dir: Path | None = None,
    whisper_model: str = "medium",
    lecture_id: str | None = None,
    progress_callback: Callable[[str], None] | None = None,
) -> PipelineResult:
    video_path = Path(video_path)
    if output_dir is None:
        slug = lecture_id or f"lecture-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        output_dir = LECTURE_ROOT / slug
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transcript_segments_path = output_dir / "transcript_segments.json"
    transcript_outline_path = output_dir / "transcript_outline.txt"
    knowledge_outline_raw_path = output_dir / "knowledge_outline_raw.json"
    knowledge_outline_enriched_path = output_dir / "knowledge_outline_enriched.json"
    slide_manifest_path = output_dir / "slides_manifest.json"
    slides_dir = output_dir / "slides"
    slides_pdf_path = output_dir / "slides.pdf"
    timings: dict[str, float] = {}

    def notify(message: str):
        if progress_callback:
            progress_callback(message)

    notify("Running ASR transcription...")
    start = perf_counter()
    segments = transcribe_video_audio(video_path, model_size=whisper_model)
    serialized_segments = serialize_transcript_segments(segments)
    language_code = infer_language_from_segments(serialized_segments)
    _write_json(transcript_segments_path, serialized_segments)
    timings["transcription"] = perf_counter() - start
    notify(f"Transcription done: {len(serialized_segments)} segments.")

    notify("Generating knowledge outline...")
    start = perf_counter()
    transcript_outline_text = build_transcript_outline_text(segments)
    transcript_outline_path.write_text(transcript_outline_text, encoding="utf-8")
    knowledge_outline = generate_knowledge_outline(transcript_outline_text, language_hint=language_code)
    _write_json(knowledge_outline_raw_path, outline_asdict(knowledge_outline))
    timings["outline"] = perf_counter() - start
    notify(f"Outline ready: {len(knowledge_outline.knowledge_points)} knowledge points.")

    notify("Extracting slides and building PDF...")
    start = perf_counter()
    slide_segments = extract_slide_keyframes(video_path)
    slide_manifest, pdf_file = persist_slides_and_build_pdf(slide_segments, slides_dir, slides_pdf_path)
    _write_json(slide_manifest_path, slide_manifest)
    timings["slides"] = perf_counter() - start
    notify(f"Slides ready: {len(slide_manifest)} frames.")

    start = perf_counter()
    attach_slide_references(knowledge_outline, slide_manifest)
    _write_json(knowledge_outline_enriched_path, asdict(knowledge_outline))
    timings["enrichment"] = perf_counter() - start
    notify("Pipeline complete. Knowledge base updated.")

    return PipelineResult(
        outline=knowledge_outline,
        output_dir=output_dir,
        transcript_segments_path=transcript_segments_path,
        transcript_outline_path=transcript_outline_path,
        knowledge_outline_path=knowledge_outline_enriched_path,
        slide_manifest_path=slide_manifest_path,
        slides_pdf_path=Path(pdf_file) if pdf_file else None,
        segment_count=len(serialized_segments),
        slide_count=len(slide_manifest),
        timings=timings,
        language=language_code,
    )


def attach_slide_references(
    outline: KnowledgeOutline,
    slide_segments: list[dict],
    *,
    min_overlap_seconds: float = 0.5,
):
    for kp in outline.knowledge_points:
        kp_start = float(kp.start_time)
        kp_end = float(kp.end_time)
        matched_slides: list[int] = []
        for slide in slide_segments:
            slide_start = float(slide["start_time"])
            slide_end = float(slide["end_time"])
            overlap = min(kp_end, slide_end) - max(kp_start, slide_start)
            if overlap >= min_overlap_seconds:
                matched_slides.append(int(slide["slide_index"]))
        kp.slides = matched_slides
