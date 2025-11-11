from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from langchain_core.documents import Document

from . import config
from .retrievers import RetrievalHit

try:
    from langchain_community.vectorstores import FAISS as LCFAISS
except ImportError:
    LCFAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
    except ImportError:
        HuggingFaceEmbeddings = None


@dataclass
class VectorStoreStatus:
    ready: bool
    message: str


class FaissRetriever:
    def __init__(self, store):
        self.store = store

    def search(self, query: str, top_k: int = 5) -> List[RetrievalHit]:
        if not query.strip():
            return []
        results = self.store.similarity_search_with_score(query, k=top_k)
        hits = [RetrievalHit(doc, float(score)) for doc, score in results]
        hits.sort(key=lambda hit: hit.score)
        return hits


def build_faiss_store(
    documents: Sequence[Document],
    *,
    persist_path: Path,
    embedding_model_name: str | None = None,
) -> tuple[FaissRetriever | None, VectorStoreStatus]:
    if LCFAISS is None or HuggingFaceEmbeddings is None:
        return None, VectorStoreStatus(
            False,
            "FAISS unavailable; install `langchain-community[faiss]` and compatible embeddings.",
        )

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name or config.EMBEDDING_MODEL)
    persist_path = Path(persist_path)
    try:
        if persist_path.exists():
            store = LCFAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)
            return FaissRetriever(store), VectorStoreStatus(True, f"Loaded FAISS index from {persist_path}")

        store = LCFAISS.from_documents(list(documents), embeddings)
        persist_path.parent.mkdir(parents=True, exist_ok=True)
        store.save_local(persist_path)
        return FaissRetriever(store), VectorStoreStatus(True, f"Built FAISS index at {persist_path}")
    except ImportError:
        return None, VectorStoreStatus(
            False,
            "FAISS import failed. Install a NumPy-compatible build (e.g., `pip install 'numpy<2' faiss-cpu`).",
        )
    except Exception as exc:  # pragma: no cover
        return None, VectorStoreStatus(False, f"Unable to initialize FAISS: {exc}")
