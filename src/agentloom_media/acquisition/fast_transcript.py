"""Fast Track transcript extractor using official or automatic video captions."""

from typing import Any, Dict, List, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound


def fetch_fast_transcript(video_id: str, languages: Optional[List[str]] = None) -> Optional[List[Dict[str, Any]]]:
    """Fetch transcripts directly from YouTube's caption track without audio processing.

    Args:
        video_id: YouTube video ID.
        languages: Preferred languages in priority order.

    Returns:
        List of transcript entries with 'text', 'start', and 'duration', or None if unavailable.
    """
    if languages is None:
        languages = ["zh-TW", "zh-HK", "zh-Hant", "zh-Hans", "zh", "en", "es"]

    try:
        ytt_api = YouTubeTranscriptApi()
        transcript_list = ytt_api.list(video_id)
        transcript = transcript_list.find_transcript(languages)
        return transcript.fetch()
    except (TranscriptsDisabled, NoTranscriptFound, Exception):
        return None
