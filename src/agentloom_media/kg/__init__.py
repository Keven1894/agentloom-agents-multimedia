"""Knowledge graph expansion: typed, grounded, bi-temporal.

Extraction and merging are ours rather than ATOM's; see
`docs/decisions/2026-09-08-atom-rejected-for-kg-extraction.md` for the measured reasons.
"""

from agentloom_media.kg.extract import extract_graph
from agentloom_media.kg.merge import suggest_merges
from agentloom_media.kg.model import (
    EDGE_FAMILIES,
    Edge,
    Evidence,
    GraphProposal,
    Node,
    normalize_surface,
)

__all__ = [
    "EDGE_FAMILIES",
    "Edge",
    "Evidence",
    "GraphProposal",
    "Node",
    "extract_graph",
    "normalize_surface",
    "suggest_merges",
]
