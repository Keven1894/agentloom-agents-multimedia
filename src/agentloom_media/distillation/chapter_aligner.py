"""Groups transcript segments into blocks for distillation.

Native chapters published by the author are treated as ground truth. When a video has no
chapters, this module batches the transcript mechanically for LLM context but does **not**
invent chapter titles or present the batch boundaries as topic boundaries — see
`behavior-builder-no-synthetic-segment-anchors`. Semantic segmentation is a separate,
later capability (plan P3).
"""

from typing import Any, Dict, List

from agentloom_media.distillation.anchors import (
    anchor_url,
    build_timed_transcript,
    format_timestamp,
)

# Mechanical batching target when no native chapters exist. This is a context-window
# concern only; it carries no claim about topic structure.
DEFAULT_BATCH_SECONDS = 300.0


def _collect_segments(
    segments: List[Dict[str, Any]], start: float, end: float
) -> List[Dict[str, Any]]:
    return [
        s
        for s in segments
        if (float(s.get("start", 0.0)) >= start and float(s.get("start", 0.0)) < end)
        or (float(s.get("end", 0.0)) > start and float(s.get("end", 0.0)) <= end)
    ]


def align_transcript_with_chapters(
    segments: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    video_url: str,
) -> List[Dict[str, Any]]:
    """Group transcript segments into blocks carrying timestamp-marked text.

    Args:
        segments: Segments with 'start', 'end', 'text'.
        chapters: Native chapters with 'title', 'start_time', 'end_time'. May be empty.
        video_url: Base URL of the video.

    Returns:
        Blocks with `boundary_source` set to 'native' or 'mechanical'. Mechanical blocks
        have `title=None`; callers must not render them as chapter names.
    """
    if not segments:
        return []

    max_time = max(
        float(s.get("end", s.get("start", 0.0))) for s in segments
    )

    if chapters:
        boundary_source = "native"
        spans = [
            {
                "title": ch.get("title") or None,
                "start": float(ch.get("start_time", 0.0)),
                "end": float(ch.get("end_time", max_time)),
            }
            for ch in chapters
        ]
    else:
        boundary_source = "mechanical"
        spans = []
        cursor = 0.0
        while cursor < max_time:
            nxt = min(cursor + DEFAULT_BATCH_SECONDS, max_time)
            spans.append({"title": None, "start": cursor, "end": nxt})
            cursor = nxt

    aligned: List[Dict[str, Any]] = []
    for span in spans:
        block_segments = _collect_segments(segments, span["start"], span["end"])
        if not block_segments:
            continue

        # Anchor the block at its first real utterance, not at the batch boundary.
        first_start = min(float(s.get("start", 0.0)) for s in block_segments)

        aligned.append(
            {
                "title": span["title"],
                "boundary_source": boundary_source,
                "start_time": first_start,
                "end_time": span["end"],
                "timestamp_str": format_timestamp(first_start),
                "anchor_url": anchor_url(video_url, first_start)
                if boundary_source == "native"
                else None,
                "text": " ".join(
                    str(s.get("text", "")).strip()
                    for s in block_segments
                    if s.get("text")
                ),
                "timed_text": build_timed_transcript(block_segments),
                "segment_count": len(block_segments),
            }
        )

    return aligned
