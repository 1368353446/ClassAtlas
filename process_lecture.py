from __future__ import annotations

import argparse
from pathlib import Path

from app.lectures import save_metadata
from app.pipeline.runner import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Process a lecture video into structured artifacts.")
    parser.add_argument("--video", type=Path, required=True, help="Path to lecture video file")
    parser.add_argument("--output", type=Path, required=True, help="Directory to store generated artifacts")
    parser.add_argument("--whisper-model", type=str, default="medium", help="Whisper model size (tiny/base/small/medium/large)")
    parser.add_argument("--lecture-id", type=str, help="Lecture identifier (defaults to output directory name)")
    parser.add_argument("--title", type=str, help="Human-readable lecture title")
    args = parser.parse_args()

    result = run_pipeline(
        video_path=args.video,
        output_dir=args.output,
        whisper_model=args.whisper_model,
        lecture_id=args.lecture_id,
    )
    lecture_id = args.lecture_id or result.output_dir.name
    title = args.title or lecture_id
    save_metadata(
        result.output_dir,
        lecture_id=lecture_id,
        title=title,
        video_path=args.video,
        source_language=result.language,
    )
    print(f"Lecture '{title}' processed at {result.output_dir}")


if __name__ == "__main__":
    main()
