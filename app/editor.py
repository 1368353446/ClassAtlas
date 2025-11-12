from __future__ import annotations

from dataclasses import replace

from pydantic import BaseModel

from .llm_utils import AgentNotAvailableError, invoke_structured_prompt
from .models import KnowledgePoint


class KnowledgePointEditModel(BaseModel):
    summary: str
    teaching_method: str
    emphasis: str


class KnowledgePointEditor:
    def __init__(self, llm=None):
        self.system_template = (
            "You refine structured lecture notes. Update the summary, teaching method, and emphasis fields "
            "based on the editor's instruction."
        )
        self.human_template = (
            "Title: {title}\n"
            "Summary: {summary}\n"
            "Teaching method: {teaching_method}\n"
            "Emphasis: {emphasis}\n"
            "Instruction: {instruction}\n\n"
            "{input}"
        )

    def apply(self, kp: KnowledgePoint, instruction: str) -> KnowledgePoint:
        instruction = instruction.strip()
        if not instruction:
            return kp
        variables = {
            "title": kp.title,
            "summary": kp.summary,
            "teaching_method": kp.teaching_method,
            "emphasis": kp.emphasis,
            "instruction": instruction,
            "input": "Return the updated JSON object for summary, teaching_method, and emphasis.",
        }
        try:
            response = invoke_structured_prompt(
                system_template=self.system_template,
                human_template=self.human_template,
                variables=variables,
                response_model=KnowledgePointEditModel,
            )
            payload = response.model_dump()
        except AgentNotAvailableError:
            updated_summary = f"{kp.summary}\n\n[Pending instruction] {instruction}"
            payload = {
                "summary": updated_summary,
                "teaching_method": kp.teaching_method,
                "emphasis": kp.emphasis,
            }
        return replace(
            kp,
            summary=payload.get("summary", kp.summary),
            teaching_method=payload.get("teaching_method", kp.teaching_method),
            emphasis=payload.get("emphasis", kp.emphasis),
        )
