"""Knowledge graph node and edge model.

Two commitments shape this schema.

**Nodes are normalized typed surfaces, not resolved identities.** A node is a case-folded,
punctuation-stripped, whitespace-collapsed *surface form* with the type it was first seen
carrying. It is not a claim that two occurrences of that surface denote the same thing.
Aggregating by surface merges homonyms, and neither lexical nor embedding similarity settles
identity. Field names say `surface` for this reason, and nothing downstream may describe these
as entities resolved by coreference — see plan §5.3, a lesson already paid for in the
SESAME/JCDL work.

**Edges carry three independent time axes.** A claim extracted from a recording needs all
three, and collapsing any two of them loses information:

- *media time* — `(t0, t1)`, where in the recording it is said; makes the claim checkable
- *observation time* — when the claim was made, i.e. the publish date; a platform-behaviour
  claim from 2026-09 is not a timeless fact
- *validity time* — `valid_from` / `valid_until`, the period the claim is asserted to hold

See plan §5.2 and `docs/decisions/2026-09-08-atom-rejected-for-kg-extraction.md`.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

# The six edge families of plan §5.2. Predicates are closed: an unrecognized predicate is a
# sign the model invented a relation type, and inventing relation types makes the families
# unreviewable.
EDGE_FAMILIES: Dict[str, List[str]] = {
    "causal": ["enables", "causes", "reduces_need_for", "prevents"],
    "structural": ["part_of", "alternative_to", "instance_of"],
    "procedural": ["step_of", "precondition_of"],
    "definitional": ["defined_as", "measured_by"],
    "temporal": ["superseded_by", "valid_during"],
    # Evidential edges are not model output. Every node and edge gets one by construction,
    # from the grounding gate; the family is named here so the digest can enumerate all six.
    "evidential": ["supported_by_span"],
}

MODEL_EMITTED_FAMILIES = [f for f in EDGE_FAMILIES if f != "evidential"]

PREDICATE_TO_FAMILY: Dict[str, str] = {
    predicate: family
    for family, predicates in EDGE_FAMILIES.items()
    for predicate in predicates
}

_PUNCT_CATEGORIES = {"Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Sm", "Sk"}
_WHITESPACE = re.compile(r"\s+")

# A node has to be an entity surface to be useful in a graph. Models drift toward clauses
# ("Meta Andromeda以后85%以上的Ads Account"), which are unmergeable across videos and
# unreadable as graph labels. These bounds reject clauses without over-fitting to a language.
MAX_SURFACE_CHARS = 32
_CLAUSE_PUNCTUATION = re.compile(r"[，,。、；;？?！!]")


def is_atomic_surface(text: str, max_chars: int = MAX_SURFACE_CHARS) -> bool:
    """Whether a surface is short and clause-free enough to serve as a node."""
    normalized = normalize_surface(text)
    if not normalized:
        return False
    if len(normalized) > max_chars:
        return False
    # Punctuation is checked pre-normalization, since normalization strips it.
    return not _CLAUSE_PUNCTUATION.search(str(text))


def normalize_surface(text: str) -> str:
    """Case-fold, strip punctuation, collapse whitespace — nothing more.

    Deliberately conservative: no stemming, no synonym mapping, no transliteration. Each of
    those would assert an identity the surface does not establish. NFKC first so full-width
    and half-width Latin, common in mixed Chinese/English speech, normalize together.
    """
    if not text:
        return ""
    normalized = unicodedata.normalize("NFKC", str(text))
    kept = [
        char
        for char in normalized
        if unicodedata.category(char) not in _PUNCT_CATEGORIES
    ]
    collapsed = _WHITESPACE.sub(" ", "".join(kept)).strip()
    return collapsed.casefold()


@dataclass
class Evidence:
    """Where a node or edge is grounded in a recording."""

    media_id: str
    t0: float
    t1: float
    quote: str
    anchor_url: Optional[str] = None
    segment_index: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "media_id": self.media_id,
            "t0": self.t0,
            "t1": self.t1,
            "quote": self.quote,
            "anchor_url": self.anchor_url,
            "segment_index": self.segment_index,
        }


@dataclass
class Node:
    """A normalized typed surface. NOT a resolved entity — see the module docstring."""

    normalized_surface: str
    first_seen_type: str
    # Every distinct spelling seen for this normalized surface, in first-seen order. Kept so a
    # reviewer can judge whether the normalization merged things it should not have.
    surface_forms: List[str] = field(default_factory=list)
    evidence: List[Evidence] = field(default_factory=list)
    description: Optional[str] = None

    @property
    def id(self) -> str:
        return f"surface:{self.normalized_surface}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "normalized_surface": self.normalized_surface,
            "surface_forms": self.surface_forms,
            "first_seen_type": self.first_seen_type,
            "description": self.description,
            "evidence": [e.to_dict() for e in self.evidence],
            "identity_claim": "none",
            "identity_note": (
                "Normalized typed surface, aggregated by exact normalized match. This is not "
                "a resolved identity: homonyms may be merged and no coreference was performed."
            ),
        }


@dataclass
class Edge:
    """A typed relation between two surfaces, with evidence and three time axes."""

    source: str
    target: str
    predicate: str
    family: str
    evidence: Evidence
    # Observation time: when the claim was made. Wall-clock, from the media publish date.
    observed_at: Optional[str] = None
    # Validity: the period the claim is asserted to hold. Usually unknown, and unknown is
    # recorded as null rather than guessed from the observation date.
    valid_from: Optional[str] = None
    valid_until: Optional[str] = None
    # A claim the speaker asserts is no longer true, e.g. "Winning Ads 已死".
    supersedes: Optional[str] = None
    claim: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "predicate": self.predicate,
            "family": self.family,
            "claim": self.claim,
            "evidence": self.evidence.to_dict(),
            "observed_at": self.observed_at,
            "valid_from": self.valid_from,
            "valid_until": self.valid_until,
            "supersedes": self.supersedes,
        }


@dataclass
class GraphProposal:
    """A working graph for one media item, ready to be normalized into a proposal."""

    media_id: str
    nodes: Dict[str, Node] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    merge_candidates: List[Dict[str, Any]] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def add_node(
        self,
        surface: str,
        node_type: str,
        evidence: Optional[Evidence] = None,
        description: Optional[str] = None,
    ) -> Optional[Node]:
        """Add or extend a surface. The first type seen wins; later types are not overridden."""
        normalized = normalize_surface(surface)
        if not normalized:
            return None

        node = self.nodes.get(normalized)
        if node is None:
            node = Node(
                normalized_surface=normalized,
                first_seen_type=node_type or "unknown",
                surface_forms=[surface],
                description=description,
            )
            self.nodes[normalized] = node
        elif surface not in node.surface_forms:
            node.surface_forms.append(surface)

        if description and not node.description:
            node.description = description
        if evidence:
            node.evidence.append(evidence)
        return node

    def to_dict(self) -> Dict[str, Any]:
        by_family: Dict[str, int] = {}
        for edge in self.edges:
            by_family[edge.family] = by_family.get(edge.family, 0) + 1

        return {
            "schema": "medialoom/kg-proposal/v1",
            "schema_version": SCHEMA_VERSION,
            "media_id": self.media_id,
            "node_semantics": (
                "Nodes are normalized typed entity surfaces, not resolved identities. "
                "Aggregation is by exact normalized match only; no coreference was performed."
            ),
            "edge_families": {
                family: EDGE_FAMILIES[family] for family in EDGE_FAMILIES
            },
            "time_axes": {
                "media_time": "evidence.t0/t1 — where in the recording the claim is made",
                "observation_time": "observed_at — when the claim was made (publish date)",
                "validity_time": "valid_from/valid_until — when the claim is asserted to hold",
            },
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "edges": [edge.to_dict() for edge in self.edges],
            "edges_by_family": by_family,
            "merge_candidates": self.merge_candidates,
            "stats": self.stats,
        }
