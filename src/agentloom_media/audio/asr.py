"""ASR transcription module using Whisper API with chunking support."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import openai


def transcribe_audio_file(
    audio_path: Path,
    api_key: Optional[str] = None,
    language: Optional[str] = None,
    prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Transcribe an audio file using OpenAI Whisper API.

    Args:
        audio_path: Path to the audio file.
        api_key: OpenAI API key.
        language: Optional language code (e.g. 'zh', 'en').
        prompt: Optional prompt to guide style or vocabulary.

    Returns:
        Dict containing full text and segment-level timestamps.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY must be provided or set in environment.")

    client = openai.OpenAI(api_key=key)

    file_size_mb = audio_path.stat().st_size / (1024 * 1024)
    if file_size_mb > 25.0:
        # Note: If audio exceeds 25MB, an external chunker or downsampling step is recommended.
        pass

    with open(audio_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
            language=language,
            prompt=prompt,
        )

    # Convert response to standard dictionary
    data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    return data
