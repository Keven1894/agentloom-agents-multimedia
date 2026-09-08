"""Hybrid time-range search over transcripts.

Turns an opaque video library into a searchable corpus: a query returns time ranges in
specific videos, not documents. See plan §3.
"""

from agentloom_media.search.embed import Embedder, resolve_embedding_model
from agentloom_media.search.index import SearchHit, SearchIndex, default_db_path
from agentloom_media.search.windows import Window, build_windows

__all__ = [
    "Embedder",
    "SearchHit",
    "SearchIndex",
    "Window",
    "build_windows",
    "default_db_path",
    "resolve_embedding_model",
]
