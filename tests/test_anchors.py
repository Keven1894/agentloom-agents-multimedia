"""Anchor validation tests.

The regression cases come from the 2026-09-07 Meta ads digest, where the distiller emitted
time ranges (`00:00-05:00`) and joined ranges (`00:00-05:00; 10:00-15:00`) that were rendered
as Markdown link targets. Those must now be rejected.
"""

from agentloom_media.distillation.anchors import (
    build_timed_transcript,
    format_timestamp,
    merge_segments_into_lines,
    parse_anchor_seconds,
    resolve_anchor,
    segment_start_times,
)
from agentloom_media.distillation.chapter_aligner import align_transcript_with_chapters

VIDEO_URL = "https://www.youtube.com/watch?v=abc123"

SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "第一句"},
    {"start": 4.0, "end": 9.5, "text": "第二句"},
    {"start": 9.5, "end": 22.0, "text": "第三句"},
    {"start": 22.0, "end": 30.0, "text": "第四句"},
    {"start": 305.0, "end": 312.0, "text": "後半段"},
]


def test_format_timestamp_crosses_hour():
    assert format_timestamp(0) == "00:00"
    assert format_timestamp(305) == "05:05"
    assert format_timestamp(3725) == "01:02:05"


def test_parse_accepts_real_forms():
    assert parse_anchor_seconds("05:05") == 305.0
    assert parse_anchor_seconds("01:02:05") == 3725.0
    assert parse_anchor_seconds("305") == 305.0
    assert parse_anchor_seconds("https://youtu.be/x?t=305s") == 305.0


def test_parse_rejects_non_timestamps():
    assert parse_anchor_seconds("") is None
    assert parse_anchor_seconds(None) is None
    assert parse_anchor_seconds("Chapter 3") is None


def test_range_anchor_is_rejected():
    """`00:00-05:00` parses as a time but names a span, not a moment."""
    valid = segment_start_times(SEGMENTS)
    _, url, status = resolve_anchor("00:00-05:00", valid, VIDEO_URL)
    # The leading 00:00 is a real segment start, so this must be caught as a range, not
    # silently snapped to zero.
    assert url is None, "a range must not produce an anchor link"
    assert status != "ok"


def test_joined_range_anchor_is_rejected():
    valid = segment_start_times(SEGMENTS)
    _, url, status = resolve_anchor("00:00-05:00; 10:00-15:00", valid, VIDEO_URL)
    assert url is None
    assert status != "ok"


def test_fabricated_time_is_rejected():
    valid = segment_start_times(SEGMENTS)
    _, url, status = resolve_anchor("99:00", valid, VIDEO_URL)
    assert url is None
    assert status == "out_of_range"


def test_real_time_resolves_and_snaps():
    valid = segment_start_times(SEGMENTS)
    seconds, url, status = resolve_anchor("05:05", valid, VIDEO_URL)
    assert status == "ok"
    assert seconds == 305.0
    assert url == f"{VIDEO_URL}&t=305s"

    # Two seconds off a real start snaps onto it.
    seconds, _, status = resolve_anchor("00:24", valid, VIDEO_URL)
    assert status == "ok"
    assert seconds == 22.0


def test_timed_transcript_markers_are_real_segment_starts():
    lines = merge_segments_into_lines(SEGMENTS, target_seconds=20.0)
    starts = set(segment_start_times(SEGMENTS))
    assert all(ln["t0"] in starts for ln in lines)

    text = build_timed_transcript(SEGMENTS, target_seconds=20.0)
    assert text.startswith("[00:00] ")


def test_no_chapters_yields_untitled_unanchored_batches():
    blocks = align_transcript_with_chapters(SEGMENTS, [], VIDEO_URL)
    assert blocks
    for block in blocks:
        assert block["boundary_source"] == "mechanical"
        assert block["title"] is None, "must not invent chapter titles"
        assert block["anchor_url"] is None, "batch boundaries are not anchors"


def test_native_chapters_are_kept_and_anchored():
    chapters = [
        {"title": "Intro", "start_time": 0.0, "end_time": 30.0},
        {"title": "Deep dive", "start_time": 300.0, "end_time": 320.0},
    ]
    blocks = align_transcript_with_chapters(SEGMENTS, chapters, VIDEO_URL)
    assert [b["title"] for b in blocks] == ["Intro", "Deep dive"]
    assert all(b["boundary_source"] == "native" for b in blocks)
    # Anchored at the first real utterance in the chapter, not the declared boundary.
    assert blocks[1]["start_time"] == 305.0
    assert blocks[1]["anchor_url"] == f"{VIDEO_URL}&t=305s"
