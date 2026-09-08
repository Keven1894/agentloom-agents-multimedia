"""Verified takeaway: claim typing, groundedness, and veracity as separate verdicts.

See plan §7. The organizing rule is that "matches the transcript" and "is true" are different
questions with different methods and costs, and must never be merged into one verdict.
"""

from agentloom_media.verification.claims import (
    CHECKABLE_TYPES,
    CLAIM_TYPES,
    UNFALSIFIABLE_TYPES,
    decompose_and_type,
)
from agentloom_media.verification.groundedness import assess_claims, containment
from agentloom_media.verification.takeaway import build_takeaway
from agentloom_media.verification.veracity import check_claims

__all__ = [
    "CHECKABLE_TYPES",
    "CLAIM_TYPES",
    "UNFALSIFIABLE_TYPES",
    "assess_claims",
    "build_takeaway",
    "check_claims",
    "containment",
    "decompose_and_type",
]
