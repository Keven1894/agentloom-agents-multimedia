"""The canonical transcript structure.

Three levels of granularity over the same audio, all times in seconds:

- **Word** — precise seek targets. Only populated when the engine reports word timings.
- **Utterance** — the natural retrieval and quotation unit.
- **Document** — a flat string plus a character→utterance map, so any character offset an
  extractor produces can be resolved back to a time span. This is what makes grounded
  extraction possible in later stages.

`timing_granularity` records what the times are actually worth (`word`, `segment`, or `cue`).
Downstream stages must read it and degrade openly rather than imply precision the engine never
provided.
"""

from __future__ import annotations

import bisect
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

# Bump when a change alters transcript content for the same engine, invalidating caches.
PIPELINE_VERSION = "p1"

# Separator between utterances in the flat document text. Single char keeps offsets simple.
TEXT_JOINER = "\n"

TimingGranularity = str  # "word" | "segment" | "cue"


@dataclass
class Word:
    """One token with its own time span. Present only at word granularity."""

    w: str
    t0: float
    t1: float
    speaker: Optional[str] = None
    conf: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Word":
        return cls(
            w=data["w"],
            t0=float(data["t0"]),
            t1=float(data["t1"]),
            speaker=data.get("speaker"),
            conf=data.get("conf"),
        )


@dataclass
class Utterance:
    """A contiguous stretch of speech, and the anchor unit for citations.

    `text` is normalized (see `normalize`); `text_raw` is exactly what the engine returned and
    is never overwritten. `char_start`/`char_end` locate this utterance in the document text.
    """

    id: str
    t0: float
    t1: float
    text: str
    text_raw: str
    speaker: Optional[str] = None
    word_range: Optional[Tuple[int, int]] = None
    char_start: int = 0
    char_end: int = 0

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        if self.word_range is not None:
            data["word_range"] = list(self.word_range)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Utterance":
        word_range = data.get("word_range")
        return cls(
            id=data["id"],
            t0=float(data["t0"]),
            t1=float(data["t1"]),
            text=data["text"],
            text_raw=data.get("text_raw", data["text"]),
            speaker=data.get("speaker"),
            word_range=tuple(word_range) if word_range else None,
            char_start=int(data.get("char_start", 0)),
            char_end=int(data.get("char_end", 0)),
        )


@dataclass
class Transcript:
    """The canonical artifact. Construct via `from_segments` rather than by hand."""

    media_id: str
    utterances: List[Utterance]
    text: str
    text_raw: str
    language: Optional[str] = None
    timing_granularity: TimingGranularity = "segment"
    words: List[Word] = field(default_factory=list)
    duration: float = 0.0
    schema_version: int = SCHEMA_VERSION

    # ---- Derived views -------------------------------------------------------------

    @property
    def utterance_starts(self) -> List[float]:
        return [u.t0 for u in self.utterances]

    def to_segments(self) -> List[Dict[str, Any]]:
        """Flat `{start, end, text}` view, for the stages not yet ported to this model."""
        return [
            {"start": u.t0, "end": u.t1, "text": u.text} for u in self.utterances
        ]

    def resolve_char_span(self, char_start: int, char_end: int) -> Optional[Tuple[float, float]]:
        """Map a character range in `text` back to the time span that covers it.

        Returns None if the range falls outside the document. At segment granularity the span
        is the union of the overlapping utterances, which is the honest answer: we cannot
        locate a substring more precisely than the utterance that contains it.
        """
        if not self.utterances or char_end < char_start:
            return None

        starts = [u.char_start for u in self.utterances]
        first = max(0, bisect.bisect_right(starts, char_start) - 1)

        overlapping = [
            u
            for u in self.utterances[first:]
            if u.char_start <= char_end and u.char_end >= char_start
        ]
        if not overlapping:
            return None
        return (min(u.t0 for u in overlapping), max(u.t1 for u in overlapping))

    def utterance_at(self, seconds: float) -> Optional[Utterance]:
        """The utterance spoken at a given time, or the one starting just before it."""
        if not self.utterances:
            return None
        idx = bisect.bisect_right(self.utterance_starts, seconds) - 1
        if idx < 0:
            return None
        return self.utterances[idx]

    # ---- Serialization ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "media_id": self.media_id,
            "language": self.language,
            "timing_granularity": self.timing_granularity,
            "duration": self.duration,
            "text": self.text,
            "text_raw": self.text_raw,
            "utterances": [u.to_dict() for u in self.utterances],
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Transcript":
        return cls(
            media_id=data["media_id"],
            utterances=[Utterance.from_dict(u) for u in data.get("utterances", [])],
            text=data.get("text", ""),
            text_raw=data.get("text_raw", data.get("text", "")),
            language=data.get("language"),
            timing_granularity=data.get("timing_granularity", "segment"),
            words=[Word.from_dict(w) for w in data.get("words", [])],
            duration=float(data.get("duration", 0.0)),
            schema_version=int(data.get("schema_version", SCHEMA_VERSION)),
        )

    # ---- Construction -------------------------------------------------------------

    @classmethod
    def from_segments(
        cls,
        media_id: str,
        segments: List[Dict[str, Any]],
        timing_granularity: TimingGranularity = "segment",
        language: Optional[str] = None,
        normalizer: Optional[Any] = None,
        words: Optional[List[Dict[str, Any]]] = None,
    ) -> "Transcript":
        """Build a canonical transcript from engine output.

        Args:
            media_id: Stable id of the media item.
            segments: Engine segments with `start`, `end`, `text`.
            timing_granularity: What the times are worth. See module docstring.
            language: Language code, if the engine reported one.
            normalizer: Optional callable applied to each utterance's text. The raw text is
                always preserved alongside.
            words: Optional word-level timings.
        """
        utterances: List[Utterance] = []
        text_parts: List[str] = []
        raw_parts: List[str] = []
        cursor = 0

        for index, seg in enumerate(segments):
            raw_text = str(seg.get("text", "")).strip()
            if not raw_text:
                continue

            norm_text = normalizer(raw_text) if normalizer else raw_text
            t0 = float(seg.get("start", 0.0))
            t1 = float(seg.get("end", t0))

            char_start = cursor
            char_end = char_start + len(norm_text)

            utterances.append(
                Utterance(
                    id=f"u{index:05d}",
                    t0=t0,
                    t1=t1,
                    text=norm_text,
                    text_raw=raw_text,
                    speaker=seg.get("speaker"),
                    char_start=char_start,
                    char_end=char_end,
                )
            )
            text_parts.append(norm_text)
            raw_parts.append(raw_text)
            cursor = char_end + len(TEXT_JOINER)

        word_objs = [Word.from_dict(w) for w in (words or [])]
        duration = max((u.t1 for u in utterances), default=0.0)

        return cls(
            media_id=media_id,
            utterances=utterances,
            text=TEXT_JOINER.join(text_parts),
            text_raw=TEXT_JOINER.join(raw_parts),
            language=language,
            timing_granularity="word" if word_objs else timing_granularity,
            words=word_objs,
            duration=duration,
        )
