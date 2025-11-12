from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Callable

from ..config import DEFAULT_WHISPER_MODEL, LECTURE_ROOT
from ..models import KnowledgeOutline, TranscriptSegment
from .outline import generate_knowledge_outline, outline_asdict
from .slides import build_slide_manifest
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
    pdf_path: Path | None = None,
    output_dir: Path | None = None,
    whisper_model: str | None = None,
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

    def emit(stage: str, action: str, detail: str):
        if not progress_callback:
            return
        payload = {"stage": stage, "action": action, "detail": detail}
        try:
            progress_callback(payload)
        except TypeError:
            progress_callback(f"[{stage.upper()}] {detail}")

    selected_model = whisper_model or DEFAULT_WHISPER_MODEL
    pdf_source = Path(pdf_path) if pdf_path is not None else None
    pdf_available = pdf_source is not None and pdf_source.exists()
    def load_cached_outline() -> KnowledgeOutline | None:
        if knowledge_outline_enriched_path.exists():
            return KnowledgeOutline.from_dict(json.loads(knowledge_outline_enriched_path.read_text(encoding="utf-8")))
        if knowledge_outline_raw_path.exists():
            return KnowledgeOutline.from_dict(json.loads(knowledge_outline_raw_path.read_text(encoding="utf-8")))
        return None

    if transcript_segments_path.exists():
        emit(
            "transcription",
            "skip",
            f"Found existing transcript at {transcript_segments_path.name}; skipping transcription.",
        )
        serialized_segments = json.loads(transcript_segments_path.read_text(encoding="utf-8"))
        segments = [TranscriptSegment.from_dict(item) for item in serialized_segments]
        language_code = infer_language_from_segments(serialized_segments)
        timings["transcription"] = 0.0
    else:
        emit("transcription", "start", f"Running Whisper ({selected_model}) ...")
        start = perf_counter()
        raw_segments = transcribe_video_audio(video_path, model_size=selected_model)
        serialized_segments = serialize_transcript_segments(raw_segments)
        segments = [TranscriptSegment.from_dict(item) for item in serialized_segments]
        language_code = infer_language_from_segments(serialized_segments)
        _write_json(transcript_segments_path, serialized_segments)
        timings["transcription"] = perf_counter() - start
        emit(
            "transcription",
            "complete",
            f"Captured {len(serialized_segments)} segments in {timings['transcription']:.2f}s.",
        )

    outline_ran = False
    outline_artifacts_exist = transcript_outline_path.exists() and (
        knowledge_outline_raw_path.exists() or knowledge_outline_enriched_path.exists()
    )

    if outline_artifacts_exist:
        emit(
            "outline",
            "skip",
            f"Detected {transcript_outline_path.name}; reusing existing outline artifacts.",
        )
        transcript_outline_text = transcript_outline_path.read_text(encoding="utf-8")
        knowledge_outline = load_cached_outline()
        if knowledge_outline is None:
            raise RuntimeError("Expected knowledge outline artifacts but none were found.")
        timings["outline"] = 0.0
    else:
        emit("outline", "start", "Generating turning points and detailed summaries ...")
        start = perf_counter()
        transcript_outline_text = build_transcript_outline_text(segments)
        transcript_outline_path.write_text(transcript_outline_text, encoding="utf-8")
        knowledge_outline = generate_knowledge_outline(segments, language_hint=language_code)
        _write_json(knowledge_outline_raw_path, outline_asdict(knowledge_outline))
        timings["outline"] = perf_counter() - start
        emit(
            "outline",
            "complete",
            f"Created {len(knowledge_outline.knowledge_points)} sections in {timings['outline']:.2f}s.",
        )
        outline_ran = True

    local_pdf_path = None
    if pdf_available and pdf_source is not None:
        slides_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if pdf_source.resolve() != slides_pdf_path.resolve():
            shutil.copy2(pdf_source, slides_pdf_path)
        local_pdf_path = slides_pdf_path

    slides_updated = False
    if not pdf_available:
        slide_manifest = []
        _write_json(slide_manifest_path, slide_manifest)
        timings["slides"] = 0.0
        emit("slides", "skip", "No slides PDF provided; skipping alignment and clearing cached slides.")
        pdf_file = None
        slides_updated = True
    elif not slide_manifest_path.exists() and local_pdf_path is not None:
        emit("slides", "start", "Aligning PDF pages with video frames ...")
        start = perf_counter()
        slide_manifest = build_slide_manifest(local_pdf_path, video_path, slides_dir)
        _write_json(slide_manifest_path, slide_manifest)
        timings["slides"] = perf_counter() - start
        emit(
            "slides",
            "complete",
            f"Aligned {len(slide_manifest)} pages in {timings['slides']:.2f}s.",
        )
        pdf_file = str(local_pdf_path)
        slides_updated = True
    else:
        if slide_manifest_path.exists():
            slide_manifest = json.loads(slide_manifest_path.read_text(encoding="utf-8"))
        else:
            slide_manifest = []
            _write_json(slide_manifest_path, slide_manifest)
        timings.setdefault("slides", 0.0)
        emit("slides", "skip", f"Reusing cached slide alignment from {slide_manifest_path.name}.")
        pdf_file = str(local_pdf_path) if local_pdf_path and local_pdf_path.exists() else None
        slides_updated = False

    enrichment_artifacts_exist = knowledge_outline_enriched_path.exists()
    should_refresh_enrichment = outline_ran or slides_updated or not enrichment_artifacts_exist

    if should_refresh_enrichment:
        emit("enrichment", "start", "Linking slides to knowledge segments ...")
        start = perf_counter()
        attach_slide_references(knowledge_outline, slide_manifest)
        _write_json(knowledge_outline_enriched_path, asdict(knowledge_outline))
        timings["enrichment"] = perf_counter() - start
        emit("enrichment", "complete", f"Updated references in {timings['enrichment']:.2f}s.")
    else:
        emit(
            "enrichment",
            "skip",
            f"Knowledge outline already synchronized (found {knowledge_outline_enriched_path.name}).",
        )
        timings.setdefault("enrichment", 0.0)
        knowledge_outline = KnowledgeOutline.from_dict(
            json.loads(knowledge_outline_enriched_path.read_text(encoding="utf-8"))
        )
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
