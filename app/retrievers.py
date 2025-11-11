from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

from langchain_core.documents import Document

Tokenizer = Callable[[str], List[str]]


@dataclass
class RetrievalHit:
    document: Document
    score: float


class BM25Retriever:
    def __init__(
        self,
        documents: Sequence[Document],
        *,
        tokenizer: Tokenizer | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.documents = list(documents)
        self.tokenizer = tokenizer or self._default_tokenizer
        self.k1 = k1
        self.b = b
        self.doc_tokens: List[List[str]] = []
        self.doc_lengths: List[int] = []
        self.doc_freq = Counter()
        for doc in self.documents:
            tokens = self.tokenizer(doc.page_content)
            self.doc_tokens.append(tokens)
            self.doc_lengths.append(len(tokens))
            self.doc_freq.update(set(tokens))
        self.corpus_size = len(self.documents)
        self.avg_doc_length = (
            sum(self.doc_lengths) / self.corpus_size if self.corpus_size else 0.0
        )

    @staticmethod
    def _default_tokenizer(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _idf(self, term: str) -> float:
        freq = self.doc_freq.get(term, 0)
        if freq == 0:
            return 0.0
        return math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def _score(self, query_tokens: Iterable[str], doc_index: int) -> float:
        doc_tokens = self.doc_tokens[doc_index]
        if not doc_tokens:
            return 0.0
        tf = defaultdict(int)
        for term in doc_tokens:
            tf[term] += 1
        doc_len = self.doc_lengths[doc_index]
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            idf = self._idf(term)
            numerator = tf[term] * (self.k1 + 1)
            denominator = tf[term] + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_length)
            score += idf * numerator / (denominator + 1e-8)
        return score

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        if not query.strip():
            return []
        query_tokens = self.tokenizer(query)
        hits = []
        for idx in range(self.corpus_size):
            score = self._score(query_tokens, idx)
            if score > 0:
                hits.append(RetrievalHit(self.documents[idx], score))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
