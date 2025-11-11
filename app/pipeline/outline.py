from __future__ import annotations

from dataclasses import asdict

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from .. import models
from ..config import LLM_BASE_URL, LLM_MODEL, MODELSCOPE_API_KEY

LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
}

BASE_PROMPT = """
You are an expert note-taker who restructures timestamped classroom transcripts into a coherent knowledge graph.
Please complete the following:
1) `session_topic`: one concise sentence capturing the core theme/question of this lecture.
2) `overall_summary`: 2-4 sentences summarizing the lecture flow (introduction → development → wrap-up).
3) `knowledge_points`: list each key point with title, start/end time in seconds, `summary`, `teaching_method`, and `emphasis`.
Keep the narrative natural yet structured, and make sure every time range matches the described content.
"""


def _build_model():
    if not MODELSCOPE_API_KEY:
        raise ValueError("MODELSCOPE_API_KEY is not set. Configure it in .env.")
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=MODELSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        streaming=True,
    )


def _system_prompt(language_hint: str | None) -> str:
    if not language_hint or language_hint == "unknown":
        return BASE_PROMPT
    language_name = LANGUAGE_NAMES.get(language_hint, "the transcript language")
    return BASE_PROMPT + f"\nAlways respond entirely in {language_name}."


def generate_knowledge_outline(transcript_outline: str, language_hint: str | None = None) -> models.KnowledgeOutline:
    model = _build_model()
    agent = create_agent(
        model=model,
        system_prompt=_system_prompt(language_hint),
        response_format=models.KnowledgeOutline,
    )
    response = agent.invoke({"messages": [{"role": "user", "content": transcript_outline}]})
    return ensure_outline_dataclass(response)


def ensure_outline_dataclass(response) -> models.KnowledgeOutline:
    if isinstance(response, dict) and "structured_response" in response:
        response = response["structured_response"]
    if isinstance(response, models.KnowledgeOutline):
        return response
    if isinstance(response, dict):
        return models.KnowledgeOutline.from_dict(response)
    raise TypeError(f"Unexpected response type: {type(response)}")


def outline_asdict(outline: models.KnowledgeOutline) -> dict:
    return asdict(outline)
