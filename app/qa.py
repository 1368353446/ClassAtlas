from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Literal, Sequence

from pydantic import BaseModel, Field, ValidationError

from .llm_utils import (
    AgentNotAvailableError,
    invoke_structured_prompt,
    invoke_text_prompt,
    prepare_history,
    blank_history,
    build_chat_model,
)
from .models import TranscriptSegment

def build_default_llm(streaming: bool = True):
    return build_chat_model(streaming=streaming)


def _detect_language(text: str) -> str:
    chinese = 0
    latin = 0
    for ch in text:
        if "\u4e00" <= ch <= "\u9fff":
            chinese += 1
        elif ch.isalpha():
            latin += 1
    if chinese == 0 and latin == 0:
        return "unknown"
    return "zh" if chinese >= latin else "en"


def _language_instruction(lang_code: str) -> str:
    if lang_code == "zh":
        return "Answer entirely in Chinese."
    if lang_code == "en":
        return "Answer entirely in English."
    return "Respond in the same language used in the latest question."


def _llm_missing_message(lang_code: str) -> str:
    if lang_code == "zh":
        return "尚未配置大模型，暂时无法生成回答。"
    if lang_code == "en":
        return "The language model is unavailable, so I cannot answer this yet."
    return "The language model is unavailable, so I cannot answer this yet."


def _not_found_message(lang_code: str) -> str:
    if lang_code == "zh":
        return "在课堂转录中没有找到相关内容。"
    if lang_code == "en":
        return "No matching content was found in the lecture transcript."
    return "No matching content was found in the lecture transcript."


class TranscriptMatchModel(BaseModel):
    start_time: float = Field(..., description="Start time in seconds of the relevant segment.")
    end_time: float = Field(..., description="End time in seconds of the relevant segment.")
    excerpt: str = Field(..., description="Short quote or paraphrase from the transcript.")
    summary: str = Field(..., description="Brief summary showing how this segment answers the question.")


class TranscriptQAResponseModel(BaseModel):
    status: Literal["found", "not_found"] = Field(
        ...,
        description="Use 'found' when the transcript covers the topic, otherwise 'not_found'.",
    )
    language: str = Field(
        ...,
        description="Language code (e.g., zh or en) that matches the user's latest question.",
    )
    matches: List[TranscriptMatchModel] = Field(
        default_factory=list,
        description="Chronological list of up to four relevant transcript segments.",
    )
    answer: str = Field(
        ...,
        description="User-facing summary in the user's language. For 'not_found' explain that nothing matched.",
    )


@dataclass
class TranscriptHit:
    start_time: float
    end_time: float
    excerpt: str
    summary: str


@dataclass
class TranscriptQAResult:
    status: str
    language: str
    matches: List[TranscriptHit]
    answer: str


@dataclass
class GeneralQAResponse:
    answer: str
    language: str


TRANSCRIPT_QA_SYSTEM = (
    "You analyze full lecture transcripts with timestamps. "
    "Find the segments that answer the learner's question, summarize them, and respond strictly using the JSON schema. "
    "{language_instruction}"
)
TRANSCRIPT_QA_HUMAN = (
    "Question:\n{question}\n\nTranscript:\n{transcript}\n\n"
    "{input}"
)

GENERAL_CHAT_SYSTEM = (
    "You are a thoughtful teaching assistant who reasons carefully and never invents transcript details. {language_instruction}"
)

class TranscriptQASystem:
    def __init__(self, segments: Sequence[TranscriptSegment], *, llm=None):
        self.segments = [segment for segment in segments if segment.text.strip()]
        self.transcript_text = self._format_transcript(self.segments)

    def answer(self, question: str) -> TranscriptQAResult:
        lang_code = _detect_language(question)
        if not self.segments:
            return TranscriptQAResult(
                status="not_found",
                language=lang_code,
                matches=[],
                answer=_not_found_message(lang_code),
            )
        variables = {
            "question": question,
            "transcript": self.transcript_text,
            "language_instruction": _language_instruction(lang_code),
            "input": (
                "Return the JSON response with status, language, matches, and answer fields. "
                "Matches must be chronological."
            ),
        }
        try:
            response = invoke_structured_prompt(
                system_template=TRANSCRIPT_QA_SYSTEM,
                human_template=TRANSCRIPT_QA_HUMAN,
                variables=variables,
                response_model=TranscriptQAResponseModel,
            )
        except AgentNotAvailableError:
            return TranscriptQAResult(
                status="not_found",
                language=lang_code,
                matches=[],
                answer=_llm_missing_message(lang_code),
            )
        except (json.JSONDecodeError, ValidationError):
            try:
                raw_text = invoke_text_prompt(
                    system_template=TRANSCRIPT_QA_SYSTEM,
                    human_template=TRANSCRIPT_QA_HUMAN,
                    variables=variables,
                )
                response = TranscriptQAResponseModel.model_validate_json(raw_text)
            except Exception:
                return TranscriptQAResult(
                    status="not_found",
                    language=lang_code,
                    matches=[],
                    answer=_llm_missing_message(lang_code),
                )
        matches = [
            TranscriptHit(
                start_time=item.start_time,
                end_time=item.end_time,
                excerpt=item.excerpt,
                summary=item.summary,
            )
            for item in response.matches
        ]
        return TranscriptQAResult(
            status=response.status,
            language=response.language or lang_code,
            matches=matches,
            answer=response.answer,
        )

    @staticmethod
    def _format_transcript(segments: Sequence[TranscriptSegment]) -> str:
        lines: List[str] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            lines.append(f"[{segment.start:.2f}s - {segment.end:.2f}s] {text}")
        return "\n".join(lines)


class GeneralChatAgent:
    def __init__(self):
        self.system_template = GENERAL_CHAT_SYSTEM

    def answer(self, question: str, history: List[dict[str, str]] | None = None) -> GeneralQAResponse:
        lang_code = _detect_language(question)
        chat_history = prepare_history(history or []) if history else blank_history()
        try:
            text = invoke_text_prompt(
                system_template=self.system_template,
                human_template="{input}",
                variables={
                    "language_instruction": _language_instruction(lang_code),
                    "input": question,
                },
                history=chat_history,
            )
        except AgentNotAvailableError:
            return GeneralQAResponse(answer=_llm_missing_message(lang_code), language=lang_code)
        return GeneralQAResponse(answer=text, language=lang_code)
