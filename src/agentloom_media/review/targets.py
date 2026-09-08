"""Flattens a proposal into reviewable targets with stable ids.

The ids produced here are the contract between the review UI and the decision log: they are
derived from content rather than list position, so re-running a distillation and re-reviewing
does not silently reattach an old verdict to a different claim.

Every target declares its evidence. A target with an empty evidence list is not hidden — it is
shown as ungrounded, because "this claim has no span" is precisely what a reviewer needs to
know, and quietly omitting it would misrepresent the proposal's coverage.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional

from agentloom_media.distillation.anchors import parse_anchor_seconds


def _short_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]


def _evidence(
    t0: Optional[float],
    t1: Optional[float] = None,
    quote: Optional[str] = None,
    anchor_url: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if t0 is None:
        return None
    return {
        "t0": float(t0),
        "t1": float(t1) if t1 is not None else None,
        "quote": quote,
        "anchor_url": anchor_url,
    }


def build_review_items(proposal: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return every reviewable target in a proposal, grouped by kind."""
    items: List[Dict[str, Any]] = []

    # --- key points, in playback order -------------------------------------------
    for segment in proposal.get("segments") or []:
        seg_title = segment.get("title") or f"Segment {segment.get('index')}"
        for position, point in enumerate(segment.get("key_points") or []):
            if not isinstance(point, dict) or not point.get("point"):
                continue
            text = str(point["point"])
            seconds = parse_anchor_seconds(point.get("anchor"))
            evidence = _evidence(seconds)
            items.append(
                {
                    "kind": "key_point",
                    "id": f"kp:{segment.get('index')}:{_short_hash(text)}",
                    "label": text,
                    "context": seg_title,
                    "evidence": [evidence] if evidence else [],
                }
            )

    # --- grounded terms -----------------------------------------------------------
    for term in proposal.get("terms") or []:
        if not isinstance(term, dict) or not term.get("term"):
            continue
        evidence = _evidence(
            term.get("t0"), term.get("t1"), term.get("quote"), term.get("anchor_url")
        )
        items.append(
            {
                "kind": "term",
                "id": f"term:{_short_hash(str(term['term']))}",
                "label": str(term["term"]),
                "context": term.get("type") or "keyword",
                "detail": term.get("description"),
                "evidence": [evidence] if evidence else [],
            }
        )

    graph = proposal.get("knowledge_graph") or {}
    node_label = {}
    for node in graph.get("nodes") or []:
        surfaces = node.get("surface_forms") or []
        node_label[node.get("id")] = (
            surfaces[0] if surfaces else node.get("normalized_surface", "")
        )

    # --- graph surfaces -----------------------------------------------------------
    for node in graph.get("nodes") or []:
        node_id = node.get("id")
        if not node_id:
            continue
        evidence = [
            e
            for e in (
                _evidence(
                    item.get("t0"), item.get("t1"), item.get("quote"), item.get("anchor_url")
                )
                for item in (node.get("evidence") or [])
            )
            if e
        ]
        items.append(
            {
                "kind": "node",
                "id": node_id,
                "label": node_label.get(node_id) or node_id,
                "context": node.get("first_seen_type") or "surface",
                "detail": node.get("description"),
                "evidence": evidence,
                "meta": {
                    "surface_forms": node.get("surface_forms") or [],
                    "identity_note": node.get("identity_note"),
                },
            }
        )

    # --- graph edges --------------------------------------------------------------
    for edge in graph.get("edges") or []:
        source = edge.get("source")
        target = edge.get("target")
        predicate = edge.get("predicate")
        if not (source and target and predicate):
            continue
        raw = edge.get("evidence") or {}
        evidence = _evidence(
            raw.get("t0"), raw.get("t1"), raw.get("quote"), raw.get("anchor_url")
        )
        items.append(
            {
                "kind": "edge",
                "id": f"edge:{_short_hash(f'{source}|{predicate}|{target}')}",
                "label": (
                    f"{node_label.get(source, source)} → {predicate} → "
                    f"{node_label.get(target, target)}"
                ),
                "context": edge.get("family") or "unknown",
                "detail": edge.get("claim"),
                "evidence": [evidence] if evidence else [],
                "meta": {
                    "observed_at": edge.get("observed_at"),
                    "valid_from": edge.get("valid_from"),
                    "valid_until": edge.get("valid_until"),
                    "supersedes": edge.get("supersedes"),
                },
            }
        )

    # --- skill steps --------------------------------------------------------------
    skill = proposal.get("skill") or {}
    for position, step in enumerate(skill.get("steps") or [], start=1):
        if not isinstance(step, dict):
            continue
        title = str(step.get("title") or f"Step {position}")
        seconds = parse_anchor_seconds(step.get("anchor"))
        evidence = _evidence(seconds)
        items.append(
            {
                "kind": "skill_step",
                "id": f"step:{_short_hash(title)}",
                "label": f"{position}. {title}",
                "context": skill.get("name") or "candidate skill",
                "detail": step.get("detail"),
                "evidence": [evidence] if evidence else [],
            }
        )

    # --- document takeaways -------------------------------------------------------
    # Raw takeaways carry no span of their own. P6 decomposes them into typed atomic claims
    # and retrieves a span per claim, so when a takeaway is present they are reviewed as
    # claims instead — reviewing the paraphrase would give a reviewer nothing to check
    # against. Without a takeaway pass they are still listed, ungrounded, rather than hidden.
    takeaway = proposal.get("takeaway") or {}
    claims = takeaway.get("claims") or []

    if claims:
        for claim in claims:
            text = str(claim.get("claim") or "")
            if not text:
                continue
            ground = claim.get("groundedness") or {}
            span = ground.get("evidence") or {}
            evidence: List[Dict[str, Any]] = []
            # Only a grounded claim offers its span as evidence. An ungrounded claim's best
            # candidate was rejected, and presenting it for review would invite a reviewer to
            # accept a citation the pipeline already refused.
            if ground.get("verdict") == "grounded" and span.get("t0") is not None:
                evidence.append(
                    {
                        "t0": span.get("t0"),
                        "t1": span.get("t1"),
                        "quote": span.get("text"),
                        "how": f"groundedness tier: {ground.get('tier')}",
                    }
                )
            veracity = claim.get("veracity") or {}
            items.append(
                {
                    "kind": "claim",
                    "id": f"claim:{_short_hash(text)}",
                    "label": text,
                    "context": (
                        f"{claim.get('type')} · groundedness "
                        f"{ground.get('verdict')} · veracity {veracity.get('verdict')}"
                    ),
                    "claim_type": claim.get("type"),
                    "groundedness": ground.get("verdict"),
                    "veracity": veracity.get("verdict"),
                    "evidence": evidence,
                }
            )
    else:
        for raw in (proposal.get("document") or {}).get("takeaways") or []:
            text = str(raw)
            items.append(
                {
                    "kind": "takeaway",
                    "id": f"takeaway:{_short_hash(text)}",
                    "label": text,
                    "context": "document synthesis",
                    "evidence": [],
                }
            )

    return items


def coverage(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """How much of the proposal is actually citable, by kind."""
    by_kind: Dict[str, Dict[str, int]] = {}
    for item in items:
        bucket = by_kind.setdefault(
            item["kind"], {"total": 0, "grounded": 0, "ungrounded": 0}
        )
        bucket["total"] += 1
        if item["evidence"]:
            bucket["grounded"] += 1
        else:
            bucket["ungrounded"] += 1

    return {
        "by_kind": by_kind,
        "total": len(items),
        "grounded": sum(b["grounded"] for b in by_kind.values()),
        "ungrounded": sum(b["ungrounded"] for b in by_kind.values()),
    }
