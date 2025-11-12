from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from typing import Iterable, List, Literal

if __name__ == "__main__" and __package__ is None:  # pragma: no cover - script execution helper
    from pathlib import Path
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    __package__ = "app.pipeline"

from pydantic import BaseModel, Field, ValidationError

from .. import models
from ..models import TranscriptSegment
from ..llm_utils import AgentNotAvailableError, invoke_structured_prompt, invoke_text_prompt
from .transcript import build_transcript_outline_text

logger = logging.getLogger(__name__)


LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
}

TURNING_POINT_SYSTEM = (
    "You are a lecture analysis agent. Identify major turning points, demos, Q&A moments, or digressions. "
    "Return structured data that follows the JSON schema exactly. {language_instruction}"
)
TURNING_POINT_HUMAN = (
    "Here is the transcript outline with timestamps:\n{transcript}\n\n"
    "Instruction:\n{input}"
)

SECTION_SYSTEM = (
    "You generate exhaustive study notes from transcript snippets. "
    "Summaries must follow the JSON schema and stay faithful to the lecture. {language_instruction}"
)
SECTION_HUMAN = (
    "Section category: {category}\n"
    "Title hint: {title_hint}\n"
    "Description: {description}\n"
    "Transcript snippets:\n{transcript}\n\n"
    "{input}"
)

OVERVIEW_SYSTEM = (
    "You craft high-level overviews of entire lectures. "
    "Produce a concise topic and a detailed overview that matches the JSON schema. {language_instruction}"
)
OVERVIEW_HUMAN = "Transcript outline:\n{transcript}\n\n{input}"


class TurningPointModel(BaseModel):
    time_seconds: float = Field(..., ge=0.0)
    title: str
    description: str
    category: Literal["instruction", "aside", "qa", "other"]


class TurningPointResponse(BaseModel):
    points: List[TurningPointModel]


class SectionSummaryModel(BaseModel):
    title: str
    summary: str
    content: str
    teaching_method: str
    emphasis: str
    category: Literal["instruction", "aside", "qa", "other"]


class OverviewModel(BaseModel):
    topic: str
    overview: str


@dataclass
class SegmentGroup:
    start_time: float
    end_time: float
    title_hint: str
    description: str
    category: str
    segments: List[TranscriptSegment]


def _language_instruction(language_hint: str | None) -> str:
    language_name = LANGUAGE_NAMES.get(language_hint, "the transcript language")
    return f"Respond entirely in {language_name}."


def _clip_time(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _format_segments_for_prompt(segments: Iterable[TranscriptSegment]) -> str:
    lines = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        lines.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")
    return "\n".join(lines[:500])


def _parse_time_token(token: str) -> float | None:
    cleaned = token.strip().lower().replace("s", "").replace("秒", "")
    if not cleaned:
        return None
    try:
        if ":" in cleaned:
            parts = cleaned.split(":")
            total = 0.0
            for part in parts:
                total = total * 60 + float(part)
            return total
        return float(cleaned)
    except ValueError:
        return None


def _fallback_turning_points(raw_text: str) -> List[TurningPointModel]:
    lines = [line.strip(" -*\t") for line in raw_text.splitlines() if line.strip()]
    category_pattern = re.compile(r"\b(instruction|aside|qa|other)\b", re.IGNORECASE)
    time_pattern = re.compile(r"(\d+(?::\d+)*(?:\.\d+)?)")

    candidates: list[TurningPointModel] = []
    for line in lines:
        time_match = time_pattern.search(line)
        if not time_match:
            continue
        time_value = _parse_time_token(time_match.group(1))
        if time_value is None:
            continue
        remainder = line[time_match.end():].strip(" -:：,，")
        if not remainder:
            remainder = line[: time_match.start()].strip(" -:：,，")
        category_match = category_pattern.search(line)
        category = category_match.group(1).lower() if category_match else "instruction"
        if category_match:
            start, end = category_match.span()
            trimmed = (line[:start] + line[end:]).strip()
        else:
            trimmed = remainder
        parts = [part.strip() for part in re.split(r"[，,；;]", trimmed) if part.strip()]
        title = parts[0] if parts else f"Section @ {time_value:.0f}s"
        description = parts[1] if len(parts) > 1 else title
        candidates.append(
            TurningPointModel(
                time_seconds=time_value,
                title=title,
                description=description,
                category=category,
            )
        )
    deduped: dict[float, TurningPointModel] = {}
    for item in candidates:
        key = round(item.time_seconds, 2)
        if key not in deduped:
            deduped[key] = item
    return sorted(deduped.values(), key=lambda item: item.time_seconds)


def detect_turning_points(transcript_outline: str, language_hint: str | None) -> List[TurningPointModel]:
    variables = {
        "language_instruction": _language_instruction(language_hint),
        "transcript": transcript_outline,
        "input": (
            "Return 5-12 ordered turning points that cover the lecture from beginning to end. "
            "Each point must include the starting second (float), a short title, description, and the category (instruction/aside/qa/other). "
            "Only output the JSON object."
        ),
    }
    try:
        response = invoke_structured_prompt(
            system_template=TURNING_POINT_SYSTEM,
            human_template=TURNING_POINT_HUMAN,
            variables=variables,
            response_model=TurningPointResponse,
        )
        parsed_points = response.points
    except AgentNotAvailableError:
        raise
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(
            "Failed to parse turning points via response_format. Falling back to heuristic parser. Error: %s", exc
        )
        raw_text = invoke_text_prompt(
            system_template=TURNING_POINT_SYSTEM,
            human_template=TURNING_POINT_HUMAN,
            variables=variables,
        )
        parsed_points = _fallback_turning_points(raw_text)
        if not parsed_points:
            raise RuntimeError("Unable to parse turning points from LLM response.") from exc
    return sorted(parsed_points, key=lambda item: item.time_seconds)


def _ensure_sentinels(points: List[TurningPointModel], lecture_end: float) -> List[TurningPointModel]:
    normalized: List[TurningPointModel] = []
    last_time = -1.0
    for point in points:
        t = _clip_time(point.time_seconds, 0.0, lecture_end)
        if not normalized or t - last_time >= 1.0:
            normalized.append(
                TurningPointModel(
                    time_seconds=t,
                    title=point.title.strip() or "Section",
                    description=point.description,
                    category=point.category,
                )
            )
            last_time = t
    if not normalized or normalized[0].time_seconds > 0.5:
        normalized.insert(
            0,
            TurningPointModel(
                time_seconds=0.0,
                title="Introduction",
                description="Opening remarks",
                category="instruction",
            ),
        )
    if normalized[-1].time_seconds < lecture_end - 0.5:
        normalized.append(
            TurningPointModel(
                time_seconds=lecture_end,
                title="Wrap-up",
                description="Session ending",
                category="other",
            )
        )
    return normalized


def build_segment_groups(segments: List[TranscriptSegment], points: List[TurningPointModel]) -> List[SegmentGroup]:
    if not segments:
        return []
    lecture_end = float(max(segment.end for segment in segments))
    normalized_points = _ensure_sentinels(points, lecture_end)
    groups: List[SegmentGroup] = []
    for idx in range(len(normalized_points) - 1):
        current = normalized_points[idx]
        next_point = normalized_points[idx + 1]
        bucket = [
            segment
            for segment in segments
            if current.time_seconds - 0.05 <= segment.start < next_point.time_seconds + 0.05
        ]
        if not bucket:
            continue
        groups.append(
            SegmentGroup(
                start_time=bucket[0].start,
                end_time=bucket[-1].end,
                title_hint=current.title,
                description=current.description,
                category=current.category,
                segments=bucket,
            )
        )
    return groups


def summarize_group(group: SegmentGroup, language_hint: str | None) -> SectionSummaryModel:
    segment_text = _format_segments_for_prompt(group.segments)
    variables = {
        "language_instruction": _language_instruction(language_hint),
        "category": group.category,
        "title_hint": group.title_hint,
        "description": group.description,
        "transcript": segment_text,
        "input": (
            "Produce the JSON object with title, summary, content, teaching_method, emphasis, and category. "
            "Be faithful to the transcript."
        ),
    }
    return invoke_structured_prompt(
        system_template=SECTION_SYSTEM,
        human_template=SECTION_HUMAN,
        variables=variables,
        response_model=SectionSummaryModel,
    )


def summarize_overview(transcript_outline: str, language_hint: str | None) -> OverviewModel:
    variables = {
        "language_instruction": _language_instruction(language_hint),
        "transcript": transcript_outline,
        "input": "Provide the JSON object with 'topic' and 'overview' fields only.",
    }
    return invoke_structured_prompt(
        system_template=OVERVIEW_SYSTEM,
        human_template=OVERVIEW_HUMAN,
        variables=variables,
        response_model=OverviewModel,
    )


def generate_knowledge_outline(
    segments: List[TranscriptSegment], language_hint: str | None = None
) -> models.KnowledgeOutline:
    transcript_outline_text = build_transcript_outline_text(segments)
    turning_points = detect_turning_points(transcript_outline_text, language_hint)
    groups = build_segment_groups(segments, turning_points)
    if not groups and segments:
        groups = [
            SegmentGroup(
                start_time=segments[0].start,
                end_time=segments[-1].end,
                title_hint="Session Overview",
                description="Full lecture",
                category="instruction",
                segments=segments,
            )
        ]
    knowledge_points: List[models.KnowledgePoint] = []
    for group in groups:
        section = summarize_group(group, language_hint)
        knowledge_points.append(
            models.KnowledgePoint(
                title=section.title or group.title_hint,
                start_time=group.start_time,
                end_time=group.end_time,
                summary=section.summary,
                content=section.content,
                teaching_method=section.teaching_method,
                emphasis=section.emphasis,
                slides=[],
            )
        )
    overview = summarize_overview(transcript_outline_text, language_hint)
    return models.KnowledgeOutline(
        session_topic=overview.topic,
        overall_summary=overview.overview,
        knowledge_points=knowledge_points,
    )


if __name__ == "__main__":
    # Lightweight self-test that avoids real LLM calls by mocking prompt invocations.
    from copy import deepcopy

    sample_segments = [
        TranscriptSegment(segment_index=0, start=0.0, end=30.0, text="课程介绍以及今天要讲的主要内容。"),
        TranscriptSegment(segment_index=1, start=30.0, end=90.0, text="第一部分：讲解基本概念和案例。"),
        TranscriptSegment(segment_index=2, start=90.0, end=150.0, text="第二部分：深入讨论和问答互动。"),
    ]

    original_structured = invoke_structured_prompt
    original_text = invoke_text_prompt

    def _fake_structured(*, response_model=None, **_kwargs):
        if response_model is TurningPointResponse:
            return TurningPointResponse(
                points=[
                    TurningPointModel(time_seconds=0.0, title="开场", description="老师介绍课程", category="instruction"),
                    TurningPointModel(time_seconds=30.0, title="基础概念", description="进入核心内容", category="instruction"),
                    TurningPointModel(time_seconds=90.0, title="讨论与问答", description="互动环节", category="qa"),
                ]
            )
        if response_model is SectionSummaryModel:
            return SectionSummaryModel(
                title="示例段落",
                summary="示例摘要",
                content="详述示例内容。",
                teaching_method="讲授+示例",
                emphasis="牢记关键定义和应用",
                category="instruction",
            )
        if response_model is OverviewModel:
            return OverviewModel(
                topic="示例课程主题",
                overview="简短概括：从介绍到问答的整体流程。",
            )
        raise ValueError("Unexpected response model")

    def _fake_text(**_kwargs):
        return (
            "1. 0.0 开场 - instruction\n"
            "2. 30.0 基础概念 - instruction\n"
            "3. 90.0 讨论与问答 - qa"
        )

    try:
        globals()["invoke_structured_prompt"] = _fake_structured
        globals()["invoke_text_prompt"] = _fake_text
        outline = generate_knowledge_outline(deepcopy(sample_segments), language_hint="zh")
        print("Self-test generated outline:")
        print(f"  Topic: {outline.session_topic}")
        print(f"  Summary: {outline.overall_summary}")
        print(f"  Knowledge points: {len(outline.knowledge_points)}")
        for idx, kp in enumerate(outline.knowledge_points, start=1):
            print(f"    {idx}. {kp.title} ({kp.start_time:.0f}s - {kp.end_time:.0f}s)")
    finally:
        globals()["invoke_structured_prompt"] = original_structured
        globals()["invoke_text_prompt"] = original_text


def ensure_outline_dataclass(response) -> models.KnowledgeOutline:
    if isinstance(response, models.KnowledgeOutline):
        return response
    if isinstance(response, dict) and "structured_response" in response:
        response = response["structured_response"]
    if isinstance(response, dict):
        return models.KnowledgeOutline.from_dict(response)
    raise TypeError(f"Unexpected response type: {type(response)}")


def outline_asdict(outline: models.KnowledgeOutline) -> dict:
    return asdict(outline)
