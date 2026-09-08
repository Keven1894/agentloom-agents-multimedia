"""Human review: evidence-level decisions over a proposal.

See plan §6. The organizing rule is that an evidence link is reviewable independently of the
claim it supports, so a correct claim with a wrong citation can be recorded as exactly that.
"""

from agentloom_media.review.decisions import (
    current_decisions,
    read_review,
    record_decision,
    review_path,
    summarize,
)
from agentloom_media.review.targets import build_review_items, coverage

__all__ = [
    "build_review_items",
    "coverage",
    "current_decisions",
    "read_review",
    "record_decision",
    "review_path",
    "summarize",
]
