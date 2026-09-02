"""Aligns transcript segments with video chapter markers and attaches hyperlink anchors."""

from typing import Any, Dict, List


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{sec:02d}"
    return f"{m:02d}:{sec:02d}"


def align_transcript_with_chapters(
    segments: List[Dict[str, Any]],
    chapters: List[Dict[str, Any]],
    video_url: str,
) -> List[Dict[str, Any]]:
    """Group transcript segments into chapter buckets with clickable video timestamps.

    Args:
        segments: List of segments with 'start', 'end', 'text'.
        chapters: List of chapters with 'title', 'start_time', 'end_time'.
        video_url: Base URL of the video.

    Returns:
        List of aligned chapter blocks with combined text and timestamp anchors.
    """
    if not chapters:
        # Synthesize 5-minute chapters if none exist
        if not segments:
            return []
        max_time = max(s.get("end", s.get("start", 0)) for s in segments)
        interval = 300.0  # 5 minutes
        synthesized_chapters = []
        curr = 0.0
        while curr < max_time:
            nxt = min(curr + interval, max_time)
            synthesized_chapters.append({
                "title": f"Part ({format_timestamp(curr)} - {format_timestamp(nxt)})",
                "start_time": curr,
                "end_time": nxt,
            })
            curr = nxt
        chapters = synthesized_chapters

    aligned = []
    for ch in chapters:
        ch_start = ch.get("start_time", 0.0)
        ch_end = ch.get("end_time", float("inf"))
        ch_title = ch.get("title", "Untitled Section")

        # Collect segments falling into this chapter
        ch_segments = [
            s for s in segments
            if (s.get("start", 0.0) >= ch_start and s.get("start", 0.0) < ch_end)
            or (s.get("end", 0.0) > ch_start and s.get("end", 0.0) <= ch_end)
        ]

        combined_text = " ".join(s.get("text", "").strip() for s in ch_segments if s.get("text"))
        ts_str = format_timestamp(ch_start)
        anchor_url = f"{video_url}&t={int(ch_start)}s" if "?" in video_url else f"{video_url}?t={int(ch_start)}s"

        aligned.append({
            "title": ch_title,
            "start_time": ch_start,
            "end_time": ch_end,
            "timestamp_str": ts_str,
            "anchor_url": anchor_url,
            "text": combined_text,
            "segment_count": len(ch_segments),
        })

    return aligned
