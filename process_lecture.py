from __future__ import annotations

import argparse
from pathlib import Path

from app.lectures import compute_file_hash, save_metadata
from app.pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Process a lecture video into structured artifacts.")
    parser.add_argument("--video", type=Path, required=True, help="Path to lecture video file")
    parser.add_argument("--slides-pdf", type=Path, help="Optional slides PDF to align with the lecture")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store generated artifacts")
    parser.add_argument(
        "--whisper-model",
        type=str,
        default=None,
        help="Whisper model size (tiny/base/small/medium/large). Defaults to the value configured in settings.",
    )
    parser.add_argument("--lecture-id", type=str, help="Lecture identifier (defaults to output directory name)")
    parser.add_argument("--title", type=str, help="Human-readable lecture title")
    args = parser.parse_args()

    def show_progress(event):
        if isinstance(event, dict):
            stage = event.get("stage", "stage").upper()
            action = event.get("action", "info")
            detail = event.get("detail", "")
            print(f"[{stage}][{action}] {detail}")
        else:
            print(event)

    result = run_pipeline(
        video_path=args.video,
        pdf_path=args.slides_pdf,
        output_dir=args.output,
        whisper_model=args.whisper_model,
        lecture_id=args.lecture_id,
        progress_callback=show_progress,
    )
    lecture_id = args.lecture_id or result.output_dir.name
    title = args.title or lecture_id
    save_metadata(
        result.output_dir,
        lecture_id=lecture_id,
        title=title,
        video_path=args.video,
        source_language=result.language,
        video_hash=compute_file_hash(args.video),
        pdf_path=Path(result.slides_pdf_path) if result.slides_pdf_path else None,
        pdf_hash=compute_file_hash(args.slides_pdf) if args.slides_pdf else None,
    )
    print(f"Lecture '{title}' processed at {result.output_dir}")


if __name__ == "__main__":
    main()
