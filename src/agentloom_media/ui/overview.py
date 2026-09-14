"""Client-facing overview: whole-video + per-segment summaries from a typed proposal.

P3 already writes `document` and per-segment `analysis`. This module only reshapes those
fields for the Overview tab. It does not call an LLM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

from agentloom_media.distillation.anchors import (
    anchor_url,
    format_timestamp,
    parse_anchor_seconds,
)
from agentloom_media.review.decisions import current_decisions, read_review, summarize
from agentloom_media.review.targets import (
    build_review_items,
    digest_path_for_proposal,
    media_id_from_source,
)


def resolve_proposal_filename(proposal_file: str) -> str:
    """Basename only, after peeling leftover percent-encoding.

    The Overview tab used to encode the Chinese filename in the hash *and* in
    `encodeURIComponent` before fetch, so the server saw `meta%E5%B9%BF....json`
    (which is also over the macOS name-length limit).
    """
    name = Path(str(proposal_file or "")).name
    for _ in range(3):
        decoded = unquote(name)
        next_name = Path(decoded).name
        if next_name == name:
            break
        name = next_name
    return name


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def duration_seconds(proposal: Dict[str, Any]) -> Optional[float]:
    source = proposal.get("source") or {}
    for key in ("duration", "duration_seconds", "length"):
        parsed = _as_float(source.get(key))
        if parsed is not None and parsed > 0:
            return parsed
    ends = [
        _as_float(segment.get("t1"))
        for segment in proposal.get("segments") or []
        if isinstance(segment, dict)
    ]
    ends = [t for t in ends if t is not None]
    return max(ends) if ends else None


def _seek_url(video_url: str, seconds: Optional[float]) -> Optional[str]:
    if not video_url or seconds is None:
        return None
    return anchor_url(video_url, seconds)


def _key_points(raw: Any, video_url: str) -> List[Dict[str, Any]]:
    points: List[Dict[str, Any]] = []
    for item in raw or []:
        if isinstance(item, str):
            text = item.strip()
            if text:
                points.append(
                    {"point": text, "t0": None, "label": None, "url": None}
                )
            continue
        if not isinstance(item, dict):
            continue
        text = str(item.get("point") or "").strip()
        if not text:
            continue
        t0 = parse_anchor_seconds(item.get("anchor"))
        points.append(
            {
                "point": text,
                "t0": t0,
                "label": format_timestamp(t0) if t0 is not None else None,
                "url": _seek_url(video_url, t0),
            }
        )
    return points


def _segments(proposal: Dict[str, Any], video_url: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in proposal.get("segments") or []:
        if not isinstance(raw, dict):
            continue
        index = raw.get("index")
        t0 = _as_float(raw.get("t0"))
        t1 = _as_float(raw.get("t1"))
        title = str(raw.get("title") or "").strip() or (
            f"Segment {int(index) + 1}" if index is not None else "Segment"
        )
        span = None
        if t0 is not None and t1 is not None:
            span = f"{format_timestamp(t0)}–{format_timestamp(t1)}"
        elif t0 is not None:
            span = format_timestamp(t0)
        watch_spans = []
        for item in raw.get("watch_spans") or []:
            if not isinstance(item, dict):
                continue
            span_t0 = _as_float(item.get("t0"))
            if span_t0 is None:
                continue
            span_t1 = _as_float(item.get("t1"))
            watch_spans.append(
                {
                    "t0": span_t0,
                    "t1": span_t1,
                    "label": item.get("label")
                    or (
                        f"{format_timestamp(span_t0)}–{format_timestamp(span_t1)}"
                        if span_t1 is not None
                        else format_timestamp(span_t0)
                    ),
                    "reason": item.get("reason"),
                    "url": _seek_url(video_url, span_t0),
                }
            )
        rows.append(
            {
                "index": index,
                "title": title,
                "t0": t0,
                "t1": t1,
                "span_label": span,
                "url": _seek_url(video_url, t0),
                "analysis": (str(raw.get("analysis")).strip() if raw.get("analysis") else None),
                "speech_mode": raw.get("speech_mode"),
                "watch": bool(raw.get("watch") or watch_spans),
                "watch_reason": raw.get("watch_reason"),
                "watch_spans": watch_spans,
                "key_points": _key_points(raw.get("key_points"), video_url),
            }
        )
    return rows


def evidence_progress(
    proposal: Dict[str, Any], review: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """How many reviewable claims still need an item-level verdict."""
    items = build_review_items(proposal)
    latest = current_decisions(review or {})
    decided = 0
    for item in items:
        key = f"{item['kind']}|{item['id']}|-1"
        if key in latest:
            decided += 1
    total = len(items)
    return {
        "total": total,
        "decided": decided,
        "pending": max(0, total - decided),
        "legacy": total == 0,
        "summary": summarize(review or {}),
    }


def client_overview(
    proposal: Dict[str, Any],
    proposal_file: str,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Structured briefing for the Overview tab. Operator dump stays out."""
    source = dict(proposal.get("source") or {})
    document = proposal.get("document") or {}
    if not isinstance(document, dict):
        document = {}
    video_url = str(source.get("url") or "")
    media_id = media_id_from_source(source)
    if media_id and not source.get("media_id"):
        source = {**source, "media_id": media_id}

    seconds = duration_seconds(proposal)
    segments = _segments(proposal, video_url)
    watch_sections = [s for s in segments if s.get("watch")]
    takeaways = [
        str(item).strip()
        for item in (document.get("takeaways") or [])
        if str(item).strip()
    ]
    has_document = bool(
        document.get("executive_summary")
        or document.get("thesis")
        or document.get("structure")
        or takeaways
    )

    digest_name = None
    if repo_root is not None:
        digest_path = digest_path_for_proposal(repo_root, proposal_file)
        if digest_path is not None:
            digest_name = digest_path.name

    return {
        "proposal_file": Path(proposal_file).name,
        "digest_filename": digest_name,
        "source": {
            "title": source.get("title") or Path(proposal_file).stem,
            "channel": source.get("channel"),
            "url": video_url or None,
            "media_id": media_id,
            "duration_seconds": seconds,
            "duration_label": format_timestamp(seconds) if seconds is not None else None,
        },
        "document": {
            "executive_summary": document.get("executive_summary") or None,
            "thesis": document.get("thesis") or None,
            "structure": document.get("structure") or None,
            "takeaways": takeaways,
        },
        "segments": segments,
        "watch_count": len(watch_sections),
        "has_structured_overview": has_document or bool(segments),
    }


def overview_card(
    proposal: Dict[str, Any],
    proposal_file: str,
    review: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """One row for the Overview list and the review queue."""
    briefing = client_overview(proposal, proposal_file, repo_root=repo_root)
    source = briefing["source"]
    return {
        "proposal_file": briefing["proposal_file"],
        "digest_filename": briefing["digest_filename"],
        "title": source["title"],
        "channel": source["channel"],
        "url": source["url"],
        "media_id": source["media_id"],
        "duration_label": source["duration_label"],
        "segment_count": len(briefing["segments"]),
        "watch_count": briefing.get("watch_count") or 0,
        "has_structured_overview": briefing["has_structured_overview"],
        "review": evidence_progress(proposal, review),
    }


def list_overview_cards(repo_root: Path) -> List[Dict[str, Any]]:
    proposals_dir = Path(repo_root) / "proposals"
    if not proposals_dir.exists():
        return []
    cards: List[Dict[str, Any]] = []
    for path in sorted(proposals_dir.glob("*.json"), reverse=True):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                proposal = json.load(handle)
        except (OSError, ValueError):
            continue
        if not isinstance(proposal, dict):
            continue
        review = read_review(repo_root, path.name)
        cards.append(overview_card(proposal, path.name, review=review, repo_root=repo_root))
    return cards
