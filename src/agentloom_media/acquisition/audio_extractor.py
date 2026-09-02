"""Audio extractor using yt-dlp to download lightweight pure audio streams."""

import os
from pathlib import Path
from typing import Optional
import yt_dlp


def extract_audio_stream(url: str, output_dir: str) -> Path:
    """Download native pure audio stream (format 140 m4a) from URL.

    Args:
        url: The video URL.
        output_dir: Target directory to save audio.

    Returns:
        Path to the downloaded audio file.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "%(id)s.%(ext)s")

    ydl_opts = {
        "format": "140/ba[ext=m4a]/bestaudio",
        "outtmpl": out_template,
        "quiet": False,
        "no_warnings": True,
        "noplaylist": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get("id")
        ext = info.get("ext", "m4a")
        audio_path = Path(output_dir) / f"{video_id}.{ext}"

        if not audio_path.exists():
            # Search for any file matching video_id in output_dir
            for f in Path(output_dir).glob(f"{video_id}.*"):
                return f
        return audio_path
