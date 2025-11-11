from __future__ import annotations

import json
from dataclasses import replace

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

from .models import KnowledgePoint
from .qa import build_default_llm


class KnowledgePointEditor:
    def __init__(self, llm=None):
        self.llm = llm or build_default_llm()
        if self.llm is None:
            self.chain = None
        else:
            prompt = PromptTemplate(
                template=(
                    "You are updating a lecture knowledge point.\n"
                    "Title: {title}\nSummary: {summary}\nTeaching method: {teaching_method}\nEmphasis: {emphasis}\n\n"
                    "Instruction: {instruction}\n"
                    "Return a compact JSON object with keys summary/teaching_method/emphasis containing the updated text."
                ),
                input_variables=["title", "summary", "teaching_method", "emphasis", "instruction"],
            )
            self.chain = prompt | self.llm | StrOutputParser()

    def apply(self, kp: KnowledgePoint, instruction: str) -> KnowledgePoint:
        instruction = instruction.strip()
        if not instruction:
            return kp
        if self.chain is None:
            updated_summary = f"{kp.summary}\n\n[Pending instruction] {instruction}"
            return replace(kp, summary=updated_summary)
        raw = self.chain.invoke(
            {
                "title": kp.title,
                "summary": kp.summary,
                "teaching_method": kp.teaching_method,
                "emphasis": kp.emphasis,
                "instruction": instruction,
            }
        )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"summary": raw.strip()}
        return replace(
            kp,
            summary=payload.get("summary", kp.summary),
            teaching_method=payload.get("teaching_method", kp.teaching_method),
            emphasis=payload.get("emphasis", kp.emphasis),
        )
