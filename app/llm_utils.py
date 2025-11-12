from __future__ import annotations

import json
import logging
from typing import Iterable, Sequence, Type

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ValidationError

from .config import LLM_BASE_URL, LLM_MODEL, MODELSCOPE_API_KEY

logger = logging.getLogger("app.llm")


class AgentNotAvailableError(RuntimeError):
    """Raised when the LLM agent cannot be constructed."""


def build_chat_model(*, streaming: bool = False) -> ChatOpenAI | None:
    if not MODELSCOPE_API_KEY:
        return None
    return ChatOpenAI(
        model=LLM_MODEL,
        api_key=MODELSCOPE_API_KEY,
        base_url=LLM_BASE_URL,
        streaming=streaming,
    )


def _safe_format(template: str, variables: dict) -> str:
    try:
        return template.format(**variables)
    except KeyError:
        return template


def _summarize_variables(variables: dict) -> dict:
    summary = {}
    for key, value in variables.items():
        if isinstance(value, str):
            summary[key] = f"{len(value)} chars"
        elif isinstance(value, (list, tuple, set)):
            summary[key] = f"{len(value)} items"
        elif isinstance(value, dict):
            summary[key] = f"{len(value)} keys"
        else:
            summary[key] = value
    return summary


def _invoke_agent(
    *,
    system_template: str,
    human_template: str,
    variables: dict,
    response_model: Type[BaseModel] | None,
    streaming: bool = False,
    history: Sequence[BaseMessage] | None = None,
):
    llm = build_chat_model(streaming=streaming)
    if llm is None:
        return None
    system_prompt = _safe_format(system_template, variables)
    human_prompt = _safe_format(human_template, variables)
    model_name = response_model.__name__ if response_model else "text"
    logger.info(
        "LLM request start | model=%s | response=%s | vars=%s",
        LLM_MODEL,
        model_name,
        _summarize_variables(variables),
    )
    agent = create_agent(
        model=llm,
        tools=None,
        system_prompt=system_prompt,
        response_format=response_model,
    )
    messages: list[BaseMessage] = list(history or [])
    messages.append(HumanMessage(content=human_prompt))
    try:
        state = agent.invoke({"messages": messages})
        structured = state.get("structured_response")
        snippet = ""
        if isinstance(structured, BaseModel):
            snippet = structured.__class__.__name__
        elif isinstance(structured, str):
            snippet = structured[:160]
        elif structured is None and state.get("messages"):
            last = state["messages"][-1]
            content = getattr(last, "content", "")
            snippet = content[:160] if isinstance(content, str) else str(content)[:160]
        logger.info(
            "LLM request complete | response=%s | snippet=%s",
            model_name,
            snippet,
        )
        return state
    except Exception as exc:
        logger.exception("LLM request failed | response=%s | error=%s", model_name, exc)
        raise


def _coerce_structured(model: Type[BaseModel], payload) -> BaseModel:
    if isinstance(payload, model):
        return payload
    if isinstance(payload, BaseModel):
        return model.model_validate(payload.model_dump())
    if isinstance(payload, str):
        data = json.loads(payload)
    else:
        data = payload
    return model.model_validate(data)


def invoke_structured_prompt(
    *,
    system_template: str,
    human_template: str,
    variables: dict,
    response_model: Type[BaseModel],
    history: Sequence[BaseMessage] | None = None,
) -> BaseModel:
    state = _invoke_agent(
        system_template=system_template,
        human_template=human_template,
        variables=variables,
        response_model=response_model,
        history=history,
    )
    if state is None:
        raise AgentNotAvailableError("LLM is not configured.")
    structured = state.get("structured_response")
    if structured is None:
        raise ValidationError(
            f"Agent did not return structured output for {response_model.__name__}", response_model
        )
    return _coerce_structured(response_model, structured)


def invoke_text_prompt(
    *,
    system_template: str,
    human_template: str,
    variables: dict,
    history: Sequence[BaseMessage] | None = None,
) -> str:
    state = _invoke_agent(
        system_template=system_template,
        human_template=human_template,
        variables=variables,
        response_model=None,
        history=history,
    )
    if state is None:
        raise AgentNotAvailableError("LLM is not configured.")
    structured = state.get("structured_response")
    if structured is not None:
        if isinstance(structured, BaseModel):
            return structured.json()
        if isinstance(structured, str):
            return structured
        return json.dumps(structured, ensure_ascii=False)
    messages: list[BaseMessage] = state.get("messages", [])
    if not messages:
        return ""
    last = messages[-1]
    content = getattr(last, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for chunk in content:
            if isinstance(chunk, dict):
                parts.append(chunk.get("text", ""))
            else:
                parts.append(str(chunk))
        return "".join(parts)
    return str(content)


def blank_history() -> list[BaseMessage]:
    return []


def prepare_history(messages: Iterable[dict[str, str]]) -> list[BaseMessage]:
    prepared: list[BaseMessage] = []
    for item in messages:
        role = item.get("role")
        content = item.get("content", "")
        if not content:
            continue
        if role == "assistant":
            prepared.append(AIMessage(content=content))
        else:
            prepared.append(HumanMessage(content=content))
    return prepared
