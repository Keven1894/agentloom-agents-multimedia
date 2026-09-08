"""Canonical transcript artifact: the one structure every downstream stage reads.

See `docs/plan/2026-09-07-core-pipeline-transcript-to-verified-takeaway.md` §1.1.
"""

from agentloom_media.transcripts.model import (
    PIPELINE_VERSION,
    SCHEMA_VERSION,
    Transcript,
    Utterance,
    Word,
)
from agentloom_media.transcripts.store import (
    load_transcript,
    media_dir,
    read_manifest,
    save_transcript,
)

__all__ = [
    "PIPELINE_VERSION",
    "SCHEMA_VERSION",
    "Transcript",
    "Utterance",
    "Word",
    "load_transcript",
    "media_dir",
    "read_manifest",
    "save_transcript",
]
