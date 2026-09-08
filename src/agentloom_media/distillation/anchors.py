"""Temporal anchor construction and validation.

Every claim distilled from a media item must point at a real spoken moment. This module
builds the timestamp-marked transcript handed to the LLM, and validates the anchors the LLM
returns against the set of times that actually exist in the transcript.

An anchor that cannot be resolved to a real utterance start is dropped rather than rendered,
per `behavior-builder-no-synthetic-segment-anchors`.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Anchors may land slightly off a segment boundary; snap within this window.
ANCHOR_SNAP_TOLERANCE_SECONDS = 3.0

# Target spoken duration per marked transcript line. ASR segments are ~2s, which would
# produce hundreds of markers and inflate the prompt without adding usable anchor targets.
DEFAULT_LINE_SECONDS = 20.0

_TIMESTAMP_PATTERN = re.compile(
    r"(?:(?P<h>\d{1,2}):)?(?P<m>\d{1,2}):(?P<s>\d{2})(?:\.(?P<frac>\d+))?"
)
_URL_SECONDS_PATTERN = re.compile(r"[?&#]t=(?P<sec>\d+(?:\.\d+)?)s?")
_BARE_SECONDS_PATTERN = re.compile(r"^\s*(?P<sec>\d+(?:\.\d+)?)\s*s?\s*$")
# Leftovers that mean the value was a span or a list rather than one moment.
_RANGE_HINT_PATTERN = re.compile(r"[-–—~,;/&]|\bto\b|\band\b")


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS, or HH:MM:SS past the hour mark."""
    total = int(seconds)
    h, m, s = total // 3600, (total % 3600) // 60, total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def anchor_url(video_url: str, seconds: float) -> str:
    """Build a shareable deep link. YouTube `t=` accepts whole seconds only."""
    joiner = "&" if "?" in video_url else "?"
    return f"{video_url}{joiner}t={int(seconds)}s"


def segment_start_times(segments: Iterable[Dict[str, Any]]) -> List[float]:
    """Sorted start times of every real transcript segment: the legal anchor targets."""
    times = {
        float(seg.get("start", 0.0))
        for seg in segments
        if seg.get("text") and str(seg.get("text")).strip()
    }
    return sorted(times)


def merge_segments_into_lines(
    segments: List[Dict[str, Any]],
    target_seconds: float = DEFAULT_LINE_SECONDS,
) -> List[Dict[str, Any]]:
    """Group short ASR segments into readable lines, each keeping a real start time."""
    lines: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", start))

        if current is None:
            current = {"t0": start, "t1": end, "parts": [text]}
            continue

        if end - current["t0"] >= target_seconds:
            current["parts"].append(text)
            current["t1"] = end
            lines.append(current)
            current = None
        else:
            current["parts"].append(text)
            current["t1"] = end

    if current is not None:
        lines.append(current)

    return [
        {"t0": ln["t0"], "t1": ln["t1"], "text": " ".join(ln["parts"]).strip()}
        for ln in lines
    ]


def build_timed_transcript(
    segments: List[Dict[str, Any]],
    target_seconds: float = DEFAULT_LINE_SECONDS,
) -> str:
    """Render the transcript with inline `[MM:SS]` markers the LLM must cite verbatim."""
    lines = merge_segments_into_lines(segments, target_seconds=target_seconds)
    return "\n".join(f"[{format_timestamp(ln['t0'])}] {ln['text']}" for ln in lines)


def parse_anchor_seconds(raw: Any) -> Optional[float]:
    """Parse a single moment in seconds, or None if the value is not one.

    Rejects spans and lists (`00:00-05:00`, `04:12; 09:30`) rather than quietly taking their
    first component: a span is not a citation of a moment, and taking its start would anchor
    the claim to wherever the batch happened to begin.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    url_match = _URL_SECONDS_PATTERN.search(text)
    if url_match:
        return float(url_match.group("sec"))

    bare_match = _BARE_SECONDS_PATTERN.match(text)
    if bare_match:
        return float(bare_match.group("sec"))

    matches = list(_TIMESTAMP_PATTERN.finditer(text))
    if len(matches) != 1:
        return None

    match = matches[0]
    remainder = (text[: match.start()] + text[match.end() :]).lower()
    if any(ch.isdigit() for ch in remainder):
        return None
    if _RANGE_HINT_PATTERN.search(remainder):
        return None

    hours = int(match.group("h") or 0)
    minutes = int(match.group("m"))
    seconds = int(match.group("s"))
    return float(hours * 3600 + minutes * 60 + seconds)


def resolve_anchor(
    raw: Any,
    valid_times: List[float],
    video_url: str,
    tolerance: float = ANCHOR_SNAP_TOLERANCE_SECONDS,
) -> Tuple[Optional[float], Optional[str], str]:
    """Validate one anchor against real transcript times.

    Returns `(seconds, url, status)` where status is one of:
    `ok` (snapped to a real segment start), `unparsable` (not a timestamp at all, e.g. the
    invented `00:00-05:00; 10:00-15:00` form), or `out_of_range` (parsed but no real segment
    starts near it, i.e. fabricated).
    """
    parsed = parse_anchor_seconds(raw)
    if parsed is None:
        return None, None, "unparsable"
    if not valid_times:
        return None, None, "out_of_range"

    nearest = min(valid_times, key=lambda t: abs(t - parsed))
    if abs(nearest - parsed) > tolerance:
        return None, None, "out_of_range"

    return nearest, anchor_url(video_url, nearest), "ok"


def resolve_anchors(
    raw_values: Iterable[Any],
    valid_times: List[float],
    video_url: str,
    tolerance: float = ANCHOR_SNAP_TOLERANCE_SECONDS,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Resolve several anchors, returning the valid ones plus a status tally."""
    resolved: List[Dict[str, Any]] = []
    stats = {"ok": 0, "unparsable": 0, "out_of_range": 0}

    for raw in raw_values:
        seconds, url, status = resolve_anchor(raw, valid_times, video_url, tolerance)
        stats[status] = stats.get(status, 0) + 1
        if status == "ok" and seconds is not None and url is not None:
            resolved.append(
                {
                    "seconds": seconds,
                    "timestamp": format_timestamp(seconds),
                    "url": url,
                    "raw": str(raw),
                }
            )

    return resolved, stats
