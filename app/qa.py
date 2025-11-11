from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from . import config
from .retrievers import BM25Retriever, RetrievalHit
from .vector_store import FaissRetriever


def build_default_llm():
    if not config.MODELSCOPE_API_KEY:
        return None
    return ChatOpenAI(
        model=config.LLM_MODEL,
        api_key=config.MODELSCOPE_API_KEY,
        base_url=config.LLM_BASE_URL,
        streaming=True,
    )


@dataclass
class QAResponse:
    answer: str
    hits: List[RetrievalHit]
    source_mode: str


class KnowledgeQASystem:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        llm=None,
        bm25_retriever: BM25Retriever | None = None,
        faiss_retriever: FaissRetriever | None = None,
        top_k: int = 5,
    ):
        self.documents = list(documents)
        self.top_k = top_k
        self.bm25 = bm25_retriever or BM25Retriever(self.documents)
        self.faiss = faiss_retriever
        self.llm = llm or build_default_llm()
        self.prompt = PromptTemplate(
            template=(
                "You are a helpful teaching assistant. Answer the user's question using only the given context.\n"
                "If the context does not contain the answer, reply that it is unknown.\n\n"
                "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
            ),
            input_variables=["question", "context"],
        )
        if self.llm:
            self.chain = self.prompt | self.llm | StrOutputParser()
        else:
            self.chain = None

    def _retrieve(self, question: str, mode: str) -> List[RetrievalHit]:
        if mode == "faiss" and self.faiss is not None:
            return self.faiss.search(question, top_k=self.top_k)
        return self.bm25.search(question, top_k=self.top_k)

    @staticmethod
    def _format_context(hits: List[RetrievalHit]) -> str:
        lines = []
        for hit in hits:
            meta = hit.document.metadata or {}
            start = meta.get("start_time")
            end = meta.get("end_time")
            prefix = ""
            if start is not None and end is not None:
                prefix = f"[{start:.2f}-{end:.2f}s] "
            lines.append(f"{prefix}{hit.document.page_content}")
        return "\n".join(lines[:5])

    def _fallback_answer(self, hits: List[RetrievalHit]) -> str:
        if not hits:
            return "No relevant segments were found in this lecture."
        best = hits[0]
        meta = best.document.metadata or {}
        start = meta.get("start_time")
        end = meta.get("end_time")
        if start is not None and end is not None:
            return f"{best.document.page_content} ({start:.2f}-{end:.2f}s)"
        return best.document.page_content

    def answer(self, question: str, *, mode: str = "bm25", use_llm: bool = True) -> QAResponse:
        hits = self._retrieve(question, mode=mode)
        if not self.chain or not use_llm:
            answer = self._fallback_answer(hits)
            return QAResponse(answer=answer, hits=hits, source_mode=mode)
        context = self._format_context(hits)
        response = self.chain.invoke({"question": question, "context": context})
        return QAResponse(answer=response, hits=hits, source_mode=mode)
