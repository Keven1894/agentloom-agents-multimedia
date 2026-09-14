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
from agentloom_media.review.targets import (
    build_legacy_review_items,
    build_review_items,
    coverage,
    digest_path_for_proposal,
    media_id_from_source,
)

__all__ = [
    "build_legacy_review_items",
    "build_review_items",
    "coverage",
    "digest_path_for_proposal",
    "media_id_from_source",
    "current_decisions",
    "read_review",
    "record_decision",
    "review_path",
    "summarize",
]
