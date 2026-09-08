"""Semantic segmentation: real topic boundaries instead of mechanical batching.

This is what finally removes the defect P0 could only contain. Mechanical 5-minute batches
carry no topic meaning, so their boundaries could not be titled or anchored. Here, boundaries
are detected where the conversation actually changes subject, which makes a segment title a
description of real content and a segment's `t0` a meaningful anchor.

Method: chunk the transcript into short spans, embed them, and cut where the cosine distance
between consecutive chunks spikes. The threshold is **relative** (a percentile of the observed
distances) rather than an absolute cosine value, because absolute thresholds do not transfer
across languages, embedding models, or speaking styles — a number tuned on English prose would
silently over- or under-segment Chinese speech.

Author-published chapters are ground truth and are preferred when present; long chapters are
still subdivided.

**Boundary resolution equals the chunk size.** Cuts can only fall on chunk edges, so a chunk
that straddles a topic change contains both topics and the boundary lands within one chunk of
the true change. With the 15s default that is accurate enough to anchor a segment while
keeping embedding calls cheap; shrink `chunk_seconds` to trade cost for precision.

See plan §4.1.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Whisper segments run ~2s, which is too short and too noisy to embed meaningfully.
DEFAULT_CHUNK_SECONDS = 15.0

# Cut at distances in the top (100 - p) percent. 85 is a deliberately conservative starting
# point: over-segmenting is cheaper to review than merging two unrelated topics.
DEFAULT_PERCENTILE = 85.0

DEFAULT_MIN_SEGMENT_SECONDS = 60.0
DEFAULT_MAX_SEGMENT_SECONDS = 300.0


def _env_float(name: str, fallback: float) -> float:
    """Read a tuning knob from the environment, ignoring unusable values."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return fallback
    try:
        value = float(raw)
    except ValueError:
        return fallback
    return value if value > 0 else fallback


def segmentation_settings(
    chunk_seconds: Optional[float] = None,
    percentile_threshold: Optional[float] = None,
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> Dict[str, float]:
    """Resolve tuning knobs: explicit argument, then environment, then default."""
    settings = {
        "chunk_seconds": chunk_seconds
        if chunk_seconds is not None
        else _env_float("SEGMENT_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS),
        "percentile_threshold": percentile_threshold
        if percentile_threshold is not None
        else _env_float("SEGMENT_PERCENTILE", DEFAULT_PERCENTILE),
        "min_seconds": min_seconds
        if min_seconds is not None
        else _env_float("SEGMENT_MIN_SECONDS", DEFAULT_MIN_SEGMENT_SECONDS),
        "max_seconds": max_seconds
        if max_seconds is not None
        else _env_float("SEGMENT_MAX_SECONDS", DEFAULT_MAX_SEGMENT_SECONDS),
    }
    # A max below the min would make both guardrails unsatisfiable; the min is the harder
    # constraint for review quality, so it wins.
    if settings["max_seconds"] < settings["min_seconds"]:
        settings["max_seconds"] = settings["min_seconds"]
    return settings


@dataclass
class Chunk:
    """A short span of speech, the unit that gets embedded for boundary detection."""

    t0: float
    t1: float
    text: str
    utterance_ids: List[str]


@dataclass
class Segment:
    """A topic-coherent span of a media item."""

    index: int
    t0: float
    t1: float
    text: str
    utterance_ids: List[str]
    boundary_source: str
    title: Optional[str] = None
    key_points: List[str] = field(default_factory=list)
    analysis: Optional[str] = None
    # Cosine distance at this segment's opening boundary; None for the first segment and for
    # boundaries that came from author chapters.
    boundary_distance: Optional[float] = None

    @property
    def duration(self) -> float:
        return self.t1 - self.t0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "t0": self.t0,
            "t1": self.t1,
            "title": self.title,
            "boundary_source": self.boundary_source,
            "boundary_distance": self.boundary_distance,
            "utterance_ids": self.utterance_ids,
            "key_points": self.key_points,
            "analysis": self.analysis,
        }


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """1 - cosine similarity, clamped to [0, 2]."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return max(0.0, min(2.0, 1.0 - dot / (norm_a * norm_b)))


def percentile(values: Sequence[float], p: float) -> float:
    """Linear-interpolated percentile, so a handful of chunks still yields a usable cut."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (p / 100.0)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[int(position)]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def chunk_utterances(
    utterances: Sequence[Any], target_seconds: float = DEFAULT_CHUNK_SECONDS
) -> List[Chunk]:
    """Group utterances into non-overlapping chunks of roughly `target_seconds`."""
    chunks: List[Chunk] = []
    current: List[Any] = []

    for utterance in utterances:
        if not utterance.text.strip():
            continue
        current.append(utterance)
        if utterance.t1 - current[0].t0 >= target_seconds:
            chunks.append(_make_chunk(current))
            current = []

    if current:
        chunks.append(_make_chunk(current))
    return chunks


def _make_chunk(group: Sequence[Any]) -> Chunk:
    return Chunk(
        t0=group[0].t0,
        t1=group[-1].t1,
        text=" ".join(u.text.strip() for u in group),
        utterance_ids=[u.id for u in group],
    )


def _segment_from_chunks(
    chunks: Sequence[Chunk],
    index: int,
    boundary_source: str,
    boundary_distance: Optional[float],
) -> Segment:
    return Segment(
        index=index,
        t0=chunks[0].t0,
        t1=chunks[-1].t1,
        text=" ".join(c.text for c in chunks),
        utterance_ids=[uid for c in chunks for uid in c.utterance_ids],
        boundary_source=boundary_source,
        boundary_distance=boundary_distance,
    )


def _cut_points_to_groups(
    chunks: Sequence[Chunk], cuts: Sequence[int]
) -> List[Tuple[int, int]]:
    """Convert chunk-index cut points into inclusive (start, end) ranges."""
    bounds = [0, *sorted(set(cuts)), len(chunks)]
    groups = []
    for start, end in zip(bounds, bounds[1:]):
        if end > start:
            groups.append((start, end - 1))
    return groups


def _enforce_min_duration(
    chunks: Sequence[Chunk],
    groups: List[Tuple[int, int]],
    distances: Sequence[float],
    min_seconds: float,
) -> List[Tuple[int, int]]:
    """Merge segments shorter than `min_seconds` into the more similar neighbour.

    Merging toward the smaller boundary distance keeps the sharper topic change intact.
    """
    if len(groups) <= 1:
        return groups

    merged = list(groups)
    changed = True
    while changed and len(merged) > 1:
        changed = False
        for position, (start, end) in enumerate(merged):
            duration = chunks[end].t1 - chunks[start].t0
            if duration >= min_seconds:
                continue

            prev_distance = distances[start - 1] if start - 1 >= 0 else math.inf
            next_distance = (
                distances[end] if end < len(distances) else math.inf
            )

            if position == 0:
                target = position + 1
            elif position == len(merged) - 1:
                target = position - 1
            else:
                target = position - 1 if prev_distance <= next_distance else position + 1

            low = min(position, target)
            merged[low] = (merged[low][0], merged[low + 1][1])
            del merged[low + 1]
            changed = True
            break

    return merged


def _enforce_max_duration(
    chunks: Sequence[Chunk],
    groups: List[Tuple[int, int]],
    distances: Sequence[float],
    max_seconds: float,
    min_seconds: float,
) -> List[Tuple[int, int]]:
    """Split over-long segments at their strongest internal boundary."""
    result: List[Tuple[int, int]] = []
    queue = list(groups)

    while queue:
        start, end = queue.pop(0)
        duration = chunks[end].t1 - chunks[start].t0
        if duration <= max_seconds or end <= start:
            result.append((start, end))
            continue

        # Only consider splits that leave both halves above the minimum duration.
        best_index = None
        best_distance = -1.0
        for cut in range(start, end):
            left = chunks[cut].t1 - chunks[start].t0
            right = chunks[end].t1 - chunks[cut + 1].t0
            if left < min_seconds or right < min_seconds:
                continue
            if distances[cut] > best_distance:
                best_distance = distances[cut]
                best_index = cut

        if best_index is None:
            result.append((start, end))
            continue

        queue.insert(0, (best_index + 1, end))
        queue.insert(0, (start, best_index))

    return sorted(result)


def segment_by_similarity(
    chunks: Sequence[Chunk],
    embeddings: Sequence[Sequence[float]],
    percentile_threshold: float = DEFAULT_PERCENTILE,
    min_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
    max_seconds: float = DEFAULT_MAX_SEGMENT_SECONDS,
    base_index: int = 0,
    boundary_source: str = "semantic",
) -> List[Segment]:
    """Cut a chunk sequence where consecutive-chunk similarity drops."""
    if not chunks:
        return []
    if len(chunks) == 1:
        return [_segment_from_chunks(chunks, base_index, boundary_source, None)]

    distances = [
        cosine_distance(embeddings[i], embeddings[i + 1])
        for i in range(len(chunks) - 1)
    ]
    threshold = percentile(distances, percentile_threshold)
    cuts = [i + 1 for i, d in enumerate(distances) if d >= threshold and d > 0.0]

    groups = _cut_points_to_groups(chunks, cuts)
    groups = _enforce_min_duration(chunks, groups, distances, min_seconds)
    groups = _enforce_max_duration(chunks, groups, distances, max_seconds, min_seconds)

    segments = []
    for offset, (start, end) in enumerate(groups):
        opening_distance = distances[start - 1] if start > 0 else None
        segments.append(
            _segment_from_chunks(
                chunks[start : end + 1],
                base_index + offset,
                boundary_source,
                opening_distance,
            )
        )
    return segments


def segment_transcript(
    transcript: Any,
    chapters: Optional[List[Dict[str, Any]]] = None,
    embedder: Optional[Any] = None,
    chunk_seconds: Optional[float] = None,
    percentile_threshold: Optional[float] = None,
    min_seconds: Optional[float] = None,
    max_seconds: Optional[float] = None,
) -> Tuple[List[Segment], Dict[str, Any]]:
    """Segment a canonical transcript into topic-coherent spans.

    Author chapters take precedence, with long chapters subdivided semantically. Without
    chapters, boundaries are detected across the whole transcript.

    Returns `(segments, info)`. `info` records the method actually used, so a caller can never
    mistake a fallback for real semantic segmentation.
    """
    utterances = [u for u in transcript.utterances if u.text.strip()]
    if not utterances:
        return [], {"method": "empty", "chunks": 0, "embedded": False}

    settings = segmentation_settings(
        chunk_seconds, percentile_threshold, min_seconds, max_seconds
    )
    chunk_seconds = settings["chunk_seconds"]
    percentile_threshold = settings["percentile_threshold"]
    min_seconds = settings["min_seconds"]
    max_seconds = settings["max_seconds"]

    chapters = chapters or []

    if embedder is None:
        # No embedder: emit one segment per author chapter, or a single unsegmented span.
        # Never invent boundaries we cannot justify.
        if chapters:
            segments = _segments_from_chapters(utterances, chapters)
            return segments, {
                "method": "native_chapters",
                "chunks": 0,
                "embedded": False,
                "note": "No embedder; chapters used verbatim without subdivision.",
            }
        chunks = chunk_utterances(utterances, chunk_seconds)
        return [
            _segment_from_chunks(chunks, 0, "unsegmented", None)
        ], {
            "method": "unsegmented",
            "chunks": len(chunks),
            "embedded": False,
            "note": "No embedder available; the transcript was not segmented.",
        }

    if chapters:
        segments: List[Segment] = []
        total_chunks = 0
        for chapter in _segments_from_chapters(utterances, chapters):
            chapter_utterances = [
                u for u in utterances if u.id in set(chapter.utterance_ids)
            ]
            if chapter.duration <= max_seconds or len(chapter_utterances) < 2:
                chapter.index = len(segments)
                chapter.boundary_source = "native"
                segments.append(chapter)
                continue

            sub_chunks = chunk_utterances(chapter_utterances, chunk_seconds)
            total_chunks += len(sub_chunks)
            embeddings = embedder.embed([c.text for c in sub_chunks])
            subdivided = segment_by_similarity(
                sub_chunks,
                embeddings,
                percentile_threshold=percentile_threshold,
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                base_index=len(segments),
                boundary_source="native+semantic",
            )
            for sub in subdivided:
                # Keep the author's title on the first sub-segment only; the rest are ours.
                sub.title = chapter.title if sub.index == len(segments) else None
            segments.extend(subdivided)

        for position, segment in enumerate(segments):
            segment.index = position

        return segments, {
            "method": "native+semantic",
            "chunks": total_chunks,
            "embedded": total_chunks > 0,
        }

    chunks = chunk_utterances(utterances, chunk_seconds)
    if len(chunks) < 2:
        return [_segment_from_chunks(chunks, 0, "unsegmented", None)], {
            "method": "unsegmented",
            "chunks": len(chunks),
            "embedded": False,
            "note": "Too short to segment.",
        }

    embeddings = embedder.embed([c.text for c in chunks])
    segments = segment_by_similarity(
        chunks,
        embeddings,
        percentile_threshold=percentile_threshold,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
    )
    return segments, {
        "method": "semantic",
        "chunks": len(chunks),
        "embedded": True,
        "percentile": percentile_threshold,
        "chunk_seconds": chunk_seconds,
        "min_seconds": min_seconds,
        "max_seconds": max_seconds,
    }


def _segments_from_chapters(
    utterances: Sequence[Any], chapters: List[Dict[str, Any]]
) -> List[Segment]:
    """One segment per author chapter, anchored at its first real utterance."""
    segments: List[Segment] = []
    max_time = max(u.t1 for u in utterances)

    for index, chapter in enumerate(chapters):
        start = float(chapter.get("start_time", 0.0))
        end = float(chapter.get("end_time", max_time))
        inside = [u for u in utterances if u.t0 >= start and u.t0 < end]
        if not inside:
            continue
        segments.append(
            Segment(
                index=index,
                t0=inside[0].t0,
                t1=inside[-1].t1,
                text=" ".join(u.text.strip() for u in inside),
                utterance_ids=[u.id for u in inside],
                boundary_source="native",
                title=chapter.get("title") or None,
            )
        )
    return segments
