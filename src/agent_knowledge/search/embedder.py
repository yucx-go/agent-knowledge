"""Embedding backends: API-based (OpenAI-compatible) or local (sentence-transformers).

Default: API-based, zero local deps. Just needs an OpenAI-compatible endpoint.
Optional: local sentence-transformers for fully offline use.
"""

from __future__ import annotations

import json
import math
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from typing import Optional


class Embedder(ABC):
    """Abstract embedding interface."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns list of vectors."""
        ...

    @abstractmethod
    def dim(self) -> int:
        """Embedding dimension."""
        ...


class APIEmbedder(Embedder):
    """OpenAI-compatible embedding API client. Zero local deps."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "text-embedding-ada-002",
        timeout: float = 30.0,
        batch_size: int = 32,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.batch_size = batch_size
        self._dim: Optional[int] = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            embeddings = self._call_api(batch)
            all_embeddings.extend(embeddings)
        return all_embeddings

    def dim(self) -> int:
        if self._dim is None:
            # Probe with a short text
            vecs = self.embed(["test"])
            self._dim = len(vecs[0]) if vecs else 0
        return self._dim

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        url = f"{self.base_url}/embeddings"
        payload = json.dumps({
            "input": texts,
            "model": self.model,
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(url, data=payload, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                # OpenAI format: {"data": [{"embedding": [...], "index": 0}, ...]}
                items = sorted(data["data"], key=lambda x: x["index"])
                embeddings = [item["embedding"] for item in items]
                if self._dim is None and embeddings:
                    self._dim = len(embeddings[0])
                return embeddings
        except (urllib.error.URLError, json.JSONDecodeError, KeyError) as e:
            # Return zero vectors on failure (graceful degradation)
            dim = self._dim or 384
            return [[0.0] * dim] * len(texts)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)
