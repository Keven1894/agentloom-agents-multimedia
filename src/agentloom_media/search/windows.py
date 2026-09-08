"""Overlapping time windows: the unit that gets indexed and retrieved.

An ASR utterance is 2-8 seconds — too short to embed meaningfully and too short to answer
anything. Windows of ~45s with ~15s overlap are the retrieval unit instead, so a hit *is* a
timecode. The overlap exists so a topic boundary cannot split an answer into two halves that
are each individually unretrievable.

See plan §3.2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

DEFAULT_WINDOW_SECONDS = 45.0
DEFAULT_OVERLAP_SECONDS = 15.0

# Utterances are joined with a space; CJK text is unaffected and Latin text stays tokenizable.
TEXT_JOINER = " "


@dataclass
class Window:
    """A contiguous stretch of speech, wide enough to carry meaning on its own."""

    t0: float
    t1: float
    text: str
    utterance_ids: List[str]

    def to_row(self) -> Dict[str, Any]:
        return {
            "t0": self.t0,
            "t1": self.t1,
            "text": self.text,
            "utterance_ids": ",".join(self.utterance_ids),
        }


def build_windows(
    transcript: Any,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> List[Window]:
    """Slice a canonical transcript into overlapping windows.

    Boundaries land on utterance edges rather than exact second counts, so a window never
    quotes half a sentence and every window's `t0` is a real spoken moment that can be used
    as an anchor.

    Args:
        transcript: A `Transcript` (anything exposing `utterances`).
        window_seconds: Target window duration.
        overlap_seconds: How much consecutive windows share. Must be < `window_seconds`.
    """
    if overlap_seconds >= window_seconds:
        raise ValueError(
            f"overlap_seconds ({overlap_seconds}) must be smaller than "
            f"window_seconds ({window_seconds}); otherwise windows never advance."
        )

    utterances = [u for u in getattr(transcript, "utterances", []) if u.text.strip()]
    if not utterances:
        return []

    stride = window_seconds - overlap_seconds
    windows: List[Window] = []
    start = 0
    count = len(utterances)

    while start < count:
        anchor = utterances[start].t0
        end = start
        # Only take the next utterance if it still fits the budget. Checking the *current*
        # utterance instead would append one that overshoots, which lets a gap or a very long
        # utterance stretch a window far past its target.
        while end + 1 < count and utterances[end + 1].t1 - anchor <= window_seconds:
            end += 1

        chunk = utterances[start : end + 1]
        windows.append(
            Window(
                t0=chunk[0].t0,
                t1=chunk[-1].t1,
                text=TEXT_JOINER.join(u.text.strip() for u in chunk),
                utterance_ids=[u.id for u in chunk],
            )
        )

        # The last window already reaches the end of the transcript.
        if end + 1 >= count:
            break

        strided = start + 1
        while strided < count and utterances[strided].t0 < anchor + stride:
            strided += 1

        # Advance by the stride to create overlap, but never past the first utterance this
        # window did not include, or the transcript would end up with unsearchable holes.
        start = max(start + 1, min(strided, end + 1))

    return windows
