"""Media probe for inspecting URLs, metadata, chapters, and stream availability."""

from typing import Any, Dict, List, Optional
import yt_dlp


def _iso_date(value: Optional[str]) -> Optional[str]:
    """Convert yt-dlp's YYYYMMDD to YYYY-MM-DD. Anything else is not a date we can use."""
    text = str(value or "").strip()
    if len(text) != 8 or not text.isdigit():
        return None
    return f"{text[0:4]}-{text[4:6]}-{text[6:8]}"


def probe_media(url: str) -> Dict[str, Any]:
    """Probe video/audio URL for metadata, chapters, subtitles, and audio streams without downloading.

    Args:
        url: The web URL or local path.

    Returns:
        Dictionary containing metadata, chapters, subtitle info, and audio stream details.
    """
    ydl_opts = {
        "simulate": True,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise ValueError(f"Could not extract info from URL: {url}")

        video_id = info.get("id")
        title = info.get("title", "Untitled")
        duration = info.get("duration", 0)
        channel = info.get("uploader", info.get("channel", "Unknown"))
        description = info.get("description", "")
        chapters = info.get("chapters") or []

        # Subtitles availability
        subtitles = list(info.get("subtitles", {}).keys())
        auto_subtitles = list(info.get("automatic_captions", {}).keys())
        has_subtitles = bool(subtitles or auto_subtitles)

        # Inspect format 140 (native m4a audio stream)
        formats = info.get("formats", [])
        m4a_format = None
        for f in formats:
            if f.get("format_id") == "140":
                m4a_format = {
                    "format_id": "140",
                    "ext": f.get("ext", "m4a"),
                    "filesize": f.get("filesize") or f.get("filesize_approx"),
                    "asr": f.get("asr"),
                    "tbr": f.get("tbr"),
                }
                break

        return {
            "id": video_id,
            "url": url,
            "title": title,
            "duration": duration,
            # Observation time for every claim distilled from this item. A platform-behaviour
            # claim is only meaningful relative to when it was made, so this feeds the KG's
            # `observed_at` axis; yt-dlp gives YYYYMMDD.
            "upload_date": _iso_date(info.get("upload_date")),
            "channel": channel,
            "description": description,
            "chapters": chapters,
            "subtitles": subtitles,
            "auto_subtitles": auto_subtitles,
            "has_subtitles": has_subtitles,
            "audio_stream": m4a_format,
        }
