"""Assembles the verified takeaway.

Four sections, per plan §7.4:

1. **What the video claims** — anchored, grounded, per-claim spans.
2. **What holds up** — claims with external support.
3. **What to verify** — contradicted, unsupported, and time-sensitive claims, each with why it
   was flagged and what would settle it.
4. **What is unfalsifiable** — personal results and recommendations, as the author's position
   rather than as knowledge.

Section 4 is what makes the output trustworthy rather than sycophantic. An honest takeaway on
the DAOJIE video says the mechanism argument is consistent with documented platform behaviour,
*and* that the ROAS figure is one unaudited operator's claim that must not enter the knowledge
graph as fact.

The takeaway is itself a proposal. It is never auto-accepted.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence

from agentloom_media.verification.claims import (
    UNFALSIFIABLE_TYPES,
    decompose_and_type,
    type_counts,
)
from agentloom_media.verification.groundedness import assess_claims
from agentloom_media.verification.veracity import check_claims

# Verdicts that mean "a human should look at this before relying on it".
NEEDS_VERIFICATION = {"contradicted", "unsupported", "time-sensitive", "unverifiable"}


def collect_statements(
    document: Dict[str, Any], segments: Sequence[Any]
) -> List[Dict[str, Any]]:
    """Gather the statements worth verifying, keeping any span they already carry.

    Key points arrive with a verified anchor from P3, so they can skip retrieval. Takeaways
    and the thesis carry nothing — they are document-level synthesis, and closing that gap is
    the point of this stage.
    """
    statements: List[Dict[str, Any]] = []

    for takeaway in document.get("takeaways") or []:
        text = str(takeaway).strip()
        if text:
            statements.append({"text": text, "origin": "takeaway", "evidence": []})

    thesis = str(document.get("thesis") or "").strip()
    if thesis:
        statements.append({"text": thesis, "origin": "thesis", "evidence": []})

    for segment in segments:
        for point in getattr(segment, "key_points", None) or []:
            text = str(point.get("point") or "").strip()
            if not text:
                continue
            statements.append(
                {
                    "text": text,
                    "origin": "key_point",
                    "segment_index": getattr(segment, "index", None),
                    # An anchor alone is not a located quote, so it is not treated as a
                    # verbatim span; retrieval still runs. Recorded for display.
                    "anchor": point.get("anchor"),
                    "evidence": [],
                }
            )

    return statements


def build_takeaway(
    document: Dict[str, Any],
    segments: Sequence[Any],
    media_id: str,
    index: Any = None,
    model: Optional[str] = None,
    observed_at: Optional[str] = None,
    escalate_to_llm: Optional[bool] = None,
    transcript: Any = None,
) -> Dict[str, Any]:
    """Decompose, type, ground, check, and assemble the takeaway proposal.

    Escalation to the LLM tier is on by default. That is not the plan's preference — §7.2
    wanted small local entailment models — but no multilingual one is installed here, and the
    alternative is leaving most claims undecided, since abstractive takeaways rarely overlap
    the transcript enough for the lexical tier to settle them. Set `TAKEAWAY_ESCALATE=0` to
    get the cheap tiers only and accept `needs_review` for the remainder.
    """
    if escalate_to_llm is None:
        escalate_to_llm = (
            os.environ.get("TAKEAWAY_ESCALATE", "1").strip().lower()
            not in {"0", "false", "no"}
        )

    statements = collect_statements(document, segments)
    if not statements:
        return {
            "schema": "medialoom/verified-takeaway/v1",
            "status": "pending_human_review",
            "claims": [],
            "sections": {
                "what_the_video_claims": [],
                "what_holds_up": [],
                "what_to_verify": [],
                "what_is_unfalsifiable": [],
            },
            "reports": {},
            "note": "No document-level statements were available to verify.",
        }

    claims = decompose_and_type([s["text"] for s in statements], model=model)
    if not claims:
        return {
            "schema": "medialoom/verified-takeaway/v1",
            "status": "pending_human_review",
            "claims": [],
            "sections": {
                "what_the_video_claims": [],
                "what_holds_up": [],
                "what_to_verify": [],
                "what_is_unfalsifiable": [],
            },
            "reports": {},
            "note": "Claim decomposition returned nothing.",
        }

    grounded = assess_claims(
        claims,
        media_id,
        index=index,
        llm_model=model if escalate_to_llm else None,
        transcript=transcript,
    )
    checked = check_claims(
        grounded["claims"], model=model, observed_at=observed_at
    )
    final = checked["claims"]

    sections: Dict[str, List[Dict[str, Any]]] = {
        "what_the_video_claims": [],
        "what_holds_up": [],
        "what_to_verify": [],
        "what_is_unfalsifiable": [],
    }

    for claim in final:
        entry = {
            "claim": claim["claim"],
            "type": claim["type"],
            "groundedness": claim["groundedness"],
            "veracity": claim["veracity"],
        }

        # Section 1 is the record of what was said, with its span. Membership depends on
        # groundedness only — whether the video says it, not whether it is true.
        if claim["groundedness"]["verdict"] == "grounded":
            sections["what_the_video_claims"].append(entry)

        if claim["type"] in UNFALSIFIABLE_TYPES:
            sections["what_is_unfalsifiable"].append(entry)
        elif claim["veracity"]["verdict"] == "supported":
            sections["what_holds_up"].append(entry)
        elif claim["veracity"]["verdict"] in NEEDS_VERIFICATION:
            sections["what_to_verify"].append(entry)

    return {
        "schema": "medialoom/verified-takeaway/v1",
        "status": "pending_human_review",
        "observed_at": observed_at,
        "claims": final,
        "sections": sections,
        "reports": {
            "types": type_counts(final),
            "groundedness": grounded["report"],
            "veracity": checked["report"],
        },
        "note": (
            "Groundedness and veracity are separate verdicts. A claim can be perfectly "
            "grounded in the transcript and still be wrong, and 'unsupported' means no "
            "source addressed it — not that it is false."
        ),
    }
