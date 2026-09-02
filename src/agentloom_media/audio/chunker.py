"""Audio chunking and downsampling utilities using bundled FFmpeg."""

import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple
import imageio_ffmpeg


def get_audio_duration(audio_path: Path) -> float:
    """Get duration of audio file in seconds using ffprobe/ffmpeg."""
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    cmd = [
        ffmpeg_exe,
        "-i", str(audio_path),
        "-f", "null",
        "-"
    ]
    proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    # Parse time=HH:MM:SS.xx from stderr
    duration = 0.0
    for line in proc.stderr.splitlines():
        if "Duration:" in line:
            parts = line.split("Duration:")[1].split(",")[0].strip()
            # HH:MM:SS.xx
            h, m, s = parts.split(":")
            duration = float(h) * 3600 + float(m) * 60 + float(s)
            break
    return duration


def compress_audio_for_whisper(input_path: Path, target_bitrate: str = "32k") -> Path:
    """Compress audio to 16kHz mono MP3 at low bitrate (ideal for speech ASR).

    Args:
        input_path: Path to raw audio file.
        target_bitrate: Audio bitrate (e.g. '32k', '48k').

    Returns:
        Path to compressed MP3 file.
    """
    output_path = input_path.with_name(f"{input_path.stem}_16k.mp3")
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i", str(input_path),
        "-vn",
        "-ar", "16000",
        "-ac", "1",
        "-b:a", target_bitrate,
        str(output_path),
    ]

    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path


def prepare_audio_for_asr(input_path: Path, max_size_mb: float = 24.0) -> List[Tuple[Path, float]]:
    """Ensure audio is ready for Whisper API (<25MB), compressing and chunking if necessary.

    Returns:
        List of (file_path, offset_seconds) tuples.
    """
    file_size_mb = input_path.stat().st_size / (1024 * 1024)

    # 1. If already small enough, return as is
    if file_size_mb <= max_size_mb and input_path.suffix.lower() in [".mp3", ".m4a", ".wav"]:
        return [(input_path, 0.0)]

    # 2. Compress to 16kHz mono 32k mp3
    compressed = compress_audio_for_whisper(input_path)
    comp_size_mb = compressed.stat().st_size / (1024 * 1024)

    if comp_size_mb <= max_size_mb:
        return [(compressed, 0.0)]

    # 3. If still > 24MB (very long video > 1.5h), chunk by 15 minutes
    duration = get_audio_duration(compressed)
    chunk_sec = 900.0  # 15 minutes
    chunks = []
    curr = 0.0
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    idx = 0
    while curr < duration:
        chunk_out = input_path.with_name(f"{input_path.stem}_part{idx:03d}.mp3")
        cmd = [
            ffmpeg_exe,
            "-y",
            "-ss", str(curr),
            "-i", str(compressed),
            "-t", str(chunk_sec),
            "-acodec", "copy",
            str(chunk_out),
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        chunks.append((chunk_out, curr))
        curr += chunk_sec
        idx += 1

    return chunks
