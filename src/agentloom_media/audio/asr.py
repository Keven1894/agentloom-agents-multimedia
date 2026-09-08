"""ASR transcription module using Whisper API with chunking and compression support."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import openai

from agentloom_media.audio.chunker import prepare_audio_for_asr


def transcribe_audio_file(
    audio_path: Path,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe an audio file using OpenAI Whisper API.

    Automatically handles compression (<24MB) and chunking if audio is too large.

    Args:
        audio_path: Path to the raw audio file.
        api_key: OpenAI API key.
        language: Optional language code (e.g. 'zh', 'en').
        prompt: Optional prompt to guide vocabulary or style.

    Returns:
        Dict with full text, segment-level timestamps, and the engine identity used
        (needed so cached transcripts record which engine produced them).
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be provided or set in environment.")

    model = (os.environ.get("ASR_MODEL") or "whisper-1").strip()
    client = openai.OpenAI(api_key=key)

    # Prepare chunks (each < 24MB)
    chunks = prepare_audio_for_asr(audio_path, max_size_mb=24.0)

    combined_text = []
    combined_segments = []
    detected_language = language

    for chunk_file, offset_sec in chunks:
        with open(chunk_file, "rb") as f:
            response = client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
                language=language,
                prompt=prompt,
            )

        data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
        combined_text.append(data.get("text", ""))
        detected_language = detected_language or data.get("language")

        for seg in data.get("segments", []):
            seg_dict = dict(seg)
            seg_dict["start"] = seg_dict.get("start", 0.0) + offset_sec
            seg_dict["end"] = seg_dict.get("end", 0.0) + offset_sec
            combined_segments.append(seg_dict)

    return {
        "text": " ".join(combined_text),
        "segments": combined_segments,
        "engine": f"openai:{model}",
        "language": detected_language,
    }
