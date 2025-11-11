from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document


@dataclass
class TranscriptSegment:
    segment_index: int
    start: float
    end: float
    text: str

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptSegment":
        return cls(
            segment_index=int(data.get("segment_index", 0)),
            start=float(data.get("start", 0.0)),
            end=float(data.get("end", 0.0)),
            text=data.get("text", "").strip(),
        )


@dataclass
class KnowledgePoint:
    title: str
    start_time: float
    end_time: float
    summary: str
    teaching_method: str
    emphasis: str
    slides: List[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgePoint":
        return cls(
            title=data.get("title", ""),
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            summary=data.get("summary", ""),
            teaching_method=data.get("teaching_method", ""),
            emphasis=data.get("emphasis", ""),
            slides=list(data.get("slides", [])),
        )


@dataclass
class KnowledgeOutline:
    session_topic: str
    overall_summary: str
    knowledge_points: List[KnowledgePoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeOutline":
        points = [KnowledgePoint.from_dict(item) for item in data.get("knowledge_points", [])]
        return cls(
            session_topic=data.get("session_topic", ""),
            overall_summary=data.get("overall_summary", ""),
            knowledge_points=points,
        )


@dataclass
class SlideManifestEntry:
    slide_index: int
    start_time: float
    end_time: float
    image_path: str

    @classmethod
    def from_dict(cls, data: dict) -> "SlideManifestEntry":
        return cls(
            slide_index=int(data.get("slide_index", 0)),
            start_time=float(data.get("start_time", 0.0)),
            end_time=float(data.get("end_time", 0.0)),
            image_path=data.get("image_path", ""),
        )


def segments_to_documents(segments: List[TranscriptSegment]) -> List[Document]:
    docs: List[Document] = []
    for segment in segments:
        if not segment.text:
            continue
        metadata = {
            "segment_index": segment.segment_index,
            "start_time": segment.start,
            "end_time": segment.end,
        }
        docs.append(Document(page_content=segment.text, metadata=metadata))
    return docs
