from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline.runner import run_pipeline


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _prepare_cached_artifacts(output_dir: Path):
    segments_path = output_dir / "transcript_segments.json"
    outline_txt_path = output_dir / "transcript_outline.txt"
    outline_raw_path = output_dir / "knowledge_outline_raw.json"
    outline_enriched_path = output_dir / "knowledge_outline_enriched.json"
    slide_manifest_path = output_dir / "slides_manifest.json"

    segments_payload = [
        {"segment_index": 0, "start": 0.0, "end": 5.0, "text": "Intro line one."},
        {"segment_index": 1, "start": 5.0, "end": 10.0, "text": "Second line of the demo."},
    ]
    segments_path.write_text(json.dumps(segments_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    outline_txt_path.write_text("00:00 Intro\n00:05 Demo point\n", encoding="utf-8")

    outline_payload = {
        "session_topic": "Demo Topic",
        "overall_summary": "Demo summary.",
        "knowledge_points": [
            {
                "title": "Point 1",
                "start_time": 0.0,
                "end_time": 10.0,
                "summary": "Short summary",
                "content": "Demo content",
                "teaching_method": "Lecture",
                "emphasis": "Core",
                "slides": [1],
            }
        ],
    }
    _write_json(outline_raw_path, outline_payload)
    _write_json(outline_enriched_path, outline_payload)

    slide_manifest_payload = [
        {"slide_index": 1, "start_time": 0.0, "end_time": 10.0, "image_path": "slides/slide_001.png"}
    ]
    _write_json(slide_manifest_path, slide_manifest_payload)


def main():
    with TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "demo_lecture"
        output_dir.mkdir(parents=True, exist_ok=True)
        _prepare_cached_artifacts(output_dir)

        dummy_video = Path(tmpdir) / "demo.mp4"
        dummy_video.write_bytes(b"fake video content")

        events = []

        run_pipeline(
            video_path=dummy_video,
            output_dir=output_dir,
            progress_callback=events.append,
        )

        print("Progress log:")
        for entry in events:
            if isinstance(entry, dict):
                print(f" - {entry['stage']}:{entry['action']} -> {entry['detail']}")
            else:
                print(f" - {entry}")


if __name__ == "__main__":
    main()
