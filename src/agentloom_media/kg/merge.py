"""Merge *suggestion* for near-duplicate surfaces.

ATOM merges entities automatically above a cosine threshold. We do not. A similarity score
measures how alike two strings look in embedding space, which is not evidence that they denote
the same thing — and an automatic merge is unreviewable after the fact, because the losing
surface is gone.

So similarity produces candidates a human decides on. The graph keeps both surfaces until
then, which is the recoverable failure mode: an unmerged duplicate is visible and fixable, a
wrong merge is silent.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from agentloom_media.distillation.segmentation import cosine_distance
from agentloom_media.kg.model import GraphProposal

DEFAULT_MERGE_THRESHOLD = 0.88

# Comparing every pair is quadratic; with a few hundred surfaces per video that is fine, but
# refuse rather than stall if a graph is unexpectedly large.
MAX_SURFACES_FOR_PAIRWISE = 400


def merge_threshold(explicit: Optional[float] = None) -> float:
    if explicit is not None:
        return explicit
    raw = (os.environ.get("KG_MERGE_THRESHOLD") or "").strip()
    try:
        return float(raw) if raw else DEFAULT_MERGE_THRESHOLD
    except ValueError:
        return DEFAULT_MERGE_THRESHOLD


def suggest_merges(
    graph: GraphProposal,
    embedder: Optional[Any] = None,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Propose surface pairs a reviewer might merge. Never mutates the graph."""
    limit = merge_threshold(threshold)
    surfaces = list(graph.nodes.values())

    if embedder is None or len(surfaces) < 2:
        return []
    if len(surfaces) > MAX_SURFACES_FOR_PAIRWISE:
        graph.stats["merge_suggestion_skipped"] = (
            f"{len(surfaces)} surfaces exceeds the pairwise limit "
            f"of {MAX_SURFACES_FOR_PAIRWISE}"
        )
        return []

    texts = [node.surface_forms[0] if node.surface_forms else node.normalized_surface for node in surfaces]
    vectors = embedder.embed(texts)

    candidates: List[Dict[str, Any]] = []
    for i in range(len(surfaces)):
        for j in range(i + 1, len(surfaces)):
            similarity = 1.0 - cosine_distance(vectors[i], vectors[j])
            if similarity < limit:
                continue
            candidates.append(
                {
                    "a": surfaces[i].id,
                    "b": surfaces[j].id,
                    "a_surface": texts[i],
                    "b_surface": texts[j],
                    "similarity": round(similarity, 4),
                    "decision": "pending_human_review",
                    "note": (
                        "Embedding similarity only. This is not evidence that the two "
                        "surfaces denote the same thing; both remain in the graph until a "
                        "reviewer decides."
                    ),
                }
            )

    candidates.sort(key=lambda c: c["similarity"], reverse=True)
    return candidates
