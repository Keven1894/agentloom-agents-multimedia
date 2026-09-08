"""Embeddings for window vectors.

v1 uses the OpenAI embedder, consistent with the API-first decision: no model download, and
the trade-off being accepted knowingly is that **transcript text leaves the machine**. That is
fine for public YouTube content and would need revisiting before private recordings or paid
course material.

The model id is stored per row by the index, because an index built with two different
embedders is silently corrupt: the vectors live in the same space only if they came from the
same model.

See plan §3.4.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

# Cheapest current OpenAI embedder. Quality is ample for 45-second windows, and per-video cost
# is negligible either way (a 16-minute video is roughly 2k tokens).
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

KNOWN_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

# The API accepts batches; keep them modest so a failure retries cheaply.
BATCH_SIZE = 64


class EmbeddingUnavailable(RuntimeError):
    """Raised when embeddings cannot be produced, so callers can degrade to lexical search."""


def resolve_embedding_model(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    return (os.environ.get("EMBEDDING_MODEL") or "").strip() or DEFAULT_EMBEDDING_MODEL


def embeddings_enabled() -> bool:
    """Whether vector search can be built at all."""
    if (os.environ.get("SEARCH_DISABLE_EMBEDDINGS") or "").strip().lower() in {
        "1",
        "true",
        "on",
        "yes",
    }:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))


class Embedder:
    """Batched embedder that reports which model produced its vectors."""

    def __init__(self, model: Optional[str] = None, api_key: Optional[str] = None):
        self.model = resolve_embedding_model(model)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        self._dimensions: Optional[int] = KNOWN_DIMENSIONS.get(self.model)

    @property
    def dimensions(self) -> Optional[int]:
        """Vector width, known up front for OpenAI models and otherwise after first call."""
        return self._dimensions

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        if not self._api_key:
            raise EmbeddingUnavailable(
                "OPENAI_API_KEY is not set, so window vectors cannot be produced. "
                "Search will fall back to lexical (FTS5) ranking only."
            )
        import openai

        self._client = openai.OpenAI(api_key=self._api_key)
        return self._client

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts, preserving input order."""
        if not texts:
            return []

        client = self._ensure_client()
        vectors: List[List[float]] = []

        for start in range(0, len(texts), BATCH_SIZE):
            batch = [t if t.strip() else " " for t in texts[start : start + BATCH_SIZE]]
            response = client.embeddings.create(model=self.model, input=batch)
            # The API may return items out of order; `index` is authoritative.
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(list(item.embedding) for item in ordered)

        if vectors:
            self._dimensions = len(vectors[0])
        return vectors

    def embed_one(self, text: str) -> List[float]:
        result = self.embed([text])
        if not result:
            raise EmbeddingUnavailable("Embedding request returned no vectors.")
        return result[0]
