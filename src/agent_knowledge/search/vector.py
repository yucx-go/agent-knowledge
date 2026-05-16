"""Vector search index: embed documents, search by cosine similarity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .embedder import Embedder, cosine_similarity
from .engine import SearchResult


class VectorIndex:
    """In-memory vector index over vault pages."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.docs: dict[str, dict] = {}  # doc_id -> {title, text, type, vector}

    def add(self, doc_id: str, title: str, text: str, page_type: str) -> None:
        """Add a document (embedding computed lazily in build())."""
        self.docs[doc_id] = {
            "title": title,
            "text": text,
            "type": page_type,
            "vector": None,
        }

    def build(self) -> int:
        """Compute embeddings for all documents. Returns count."""
        if not self.docs:
            return 0

        doc_ids = list(self.docs.keys())
        texts = [self.docs[did]["text"][:2000] for did in doc_ids]  # truncate for embedding

        vectors = self.embedder.embed(texts)
        for did, vec in zip(doc_ids, vectors):
            self.docs[did]["vector"] = vec

        return len(doc_ids)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search by cosine similarity."""
        if not self.docs:
            return []

        query_vec = self.embedder.embed([query])[0]
        if all(v == 0.0 for v in query_vec):
            return []  # embedding failed

        scored = []
        for doc_id, doc in self.docs.items():
            if doc["vector"] is None:
                continue
            sim = cosine_similarity(query_vec, doc["vector"])
            if sim > 0.0:
                scored.append((doc_id, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for doc_id, score in scored[:top_k]:
            doc = self.docs[doc_id]
            # Simple snippet: first 200 chars
            snippet = doc["text"][:200].replace("\n", " ").strip()
            results.append(SearchResult(
                id=doc_id,
                page_type=doc["type"],
                title=doc["title"],
                snippet=snippet,
                score=round(score, 4),
            ))
        return results
