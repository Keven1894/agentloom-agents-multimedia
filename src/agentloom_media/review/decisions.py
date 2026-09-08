"""Per-item and per-evidence review decisions.

The governing idea from plan §6.1: **label the decision, not the item.** A node can be correct
while one of its citations points at the wrong moment, and that is still a governance failure
— so an evidence link is a reviewable target in its own right, independent of the claim it
supports. Rejecting a citation must not reject the claim, and accepting a claim must not
launder its citations.

Decisions are keyed by `(target_kind, target_id, evidence_index)`. A decision with
`evidence_index = None` is about the item; with an index, it is about that one evidence link.
Both can coexist and disagree, which is the point.

Decisions are stored append-only per proposal, so a changed verdict keeps its history rather
than overwriting it. `current_decisions` collapses the log to the latest verdict per target.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = "medialoom/review-decisions/v1"

TARGET_KINDS = {
    "node",
    "edge",
    "term",
    "key_point",
    "skill_step",
    "takeaway",
    # P6 atomic claims. Kept alongside "takeaway" rather than replacing it, so decisions
    # recorded against pre-P6 proposals stay readable.
    "claim",
}
VERDICTS = {"accept", "reject", "unsure"}

_SAFE = re.compile(r"[^A-Za-z0-9._\u4e00-\u9fff-]+")


def reviews_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "reviews"


def review_path(repo_root: Path, proposal_file: str) -> Path:
    stem = Path(proposal_file).stem
    return reviews_dir(repo_root) / f"review-{stem}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def target_key(
    kind: str, target_id: str, evidence_index: Optional[int]
) -> Tuple[str, str, int]:
    """A hashable key. -1 stands for "the item itself" so keys sort and compare cleanly."""
    return (kind, target_id, -1 if evidence_index is None else int(evidence_index))


def read_review(repo_root: Path, proposal_file: str) -> Dict[str, Any]:
    """Load the decision log, returning an empty one when nothing is recorded yet."""
    path = review_path(repo_root, proposal_file)
    empty = {
        "schema": SCHEMA,
        "proposal_file": Path(proposal_file).name,
        "decisions": [],
    }
    if not path.exists():
        return empty
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return empty
    if not isinstance(data, dict) or not isinstance(data.get("decisions"), list):
        return empty
    return data


def record_decision(
    repo_root: Path,
    proposal_file: str,
    kind: str,
    target_id: str,
    verdict: str,
    evidence_index: Optional[int] = None,
    note: Optional[str] = None,
    reviewer: str = "unknown",
) -> Dict[str, Any]:
    """Append one decision. Raises ValueError on an unknown kind or verdict.

    Rejecting unknown vocabulary matters: a typo'd verdict that silently persisted would make
    the review log unreadable, and the log is the audit trail.
    """
    if kind not in TARGET_KINDS:
        raise ValueError(
            f"Unknown target kind {kind!r}; expected one of {sorted(TARGET_KINDS)}"
        )
    if verdict not in VERDICTS:
        raise ValueError(
            f"Unknown verdict {verdict!r}; expected one of {sorted(VERDICTS)}"
        )
    if not str(target_id or "").strip():
        raise ValueError("target_id is required")

    review = read_review(repo_root, proposal_file)
    entry = {
        "kind": kind,
        "target_id": str(target_id),
        "evidence_index": None if evidence_index is None else int(evidence_index),
        "scope": "item" if evidence_index is None else "evidence",
        "verdict": verdict,
        "note": (note or "").strip() or None,
        "reviewer": reviewer or "unknown",
        "timestamp": _now(),
    }
    review["decisions"].append(entry)
    review["updated_at"] = entry["timestamp"]
    review["schema"] = SCHEMA
    review["proposal_file"] = Path(proposal_file).name

    path = review_path(repo_root, proposal_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(review, f, indent=2, ensure_ascii=False)

    return entry


def current_decisions(review: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Collapse the append-only log to the latest verdict per target.

    Keys are `"<kind>|<target_id>|<evidence_index>"` with `-1` for item-level, so the result
    is JSON-serializable for the front end.
    """
    latest: Dict[str, Dict[str, Any]] = {}
    for entry in review.get("decisions", []):
        if not isinstance(entry, dict):
            continue
        key = target_key(
            entry.get("kind", ""),
            entry.get("target_id", ""),
            entry.get("evidence_index"),
        )
        flat = f"{key[0]}|{key[1]}|{key[2]}"
        previous = latest.get(flat)
        # The log is append-only and written in order, but compare timestamps anyway so an
        # out-of-order or merged log still resolves to the newest verdict.
        if previous is None or str(entry.get("timestamp", "")) >= str(
            previous.get("timestamp", "")
        ):
            latest[flat] = entry
    return latest


def summarize(review: Dict[str, Any]) -> Dict[str, Any]:
    """Counts by scope and verdict, plus the disagreements worth a reviewer's attention."""
    latest = current_decisions(review)
    counts: Dict[str, Dict[str, int]] = {
        "item": {v: 0 for v in sorted(VERDICTS)},
        "evidence": {v: 0 for v in sorted(VERDICTS)},
    }
    for entry in latest.values():
        scope = entry.get("scope") or (
            "item" if entry.get("evidence_index") is None else "evidence"
        )
        verdict = entry.get("verdict")
        if scope in counts and verdict in counts[scope]:
            counts[scope][verdict] += 1

    # An item accepted while one of its citations is rejected. Not an error — it is exactly
    # the distinction §6.1 asks for — but it is the case a governance reader should see.
    accepted_items = {
        (e["kind"], e["target_id"])
        for e in latest.values()
        if e.get("evidence_index") is None and e.get("verdict") == "accept"
    }
    split = sorted(
        {
            f"{e['kind']}|{e['target_id']}"
            for e in latest.values()
            if e.get("evidence_index") is not None
            and e.get("verdict") == "reject"
            and (e["kind"], e["target_id"]) in accepted_items
        }
    )

    return {
        "decided_targets": len(latest),
        "counts": counts,
        "accepted_with_rejected_evidence": split,
    }
