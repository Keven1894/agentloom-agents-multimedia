"""ASR engine selection.

Engine choice is a strategy, not a constant: the right engine depends on whether a GPU is
available and on whether the work actually needs word-level timings. Every engine reports the
`timing_granularity` its output is worth, and nothing downstream may assume better.

Selected with `ASR_ENGINE`. See `docs/plan/2026-09-07-core-pipeline-transcript-to-verified-takeaway.md`
§2.2, including the v1 decision to defer local ASR.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_ASR_ENGINE = "whisper_api"


class AsrEngineUnavailable(RuntimeError):
    """Raised when the requested engine cannot run here, with what to do about it."""


# Whisper reports detected languages by English name ("chinese") while callers pass ISO
# codes ("zh"). Recording both spellings would put the same language under two labels in the
# manifests, which is the provenance sloppiness this layer exists to prevent.
_LANGUAGE_ALIASES = {
    "chinese": "zh",
    "mandarin": "zh",
    "english": "en",
    "japanese": "ja",
    "korean": "ko",
    "spanish": "es",
    "french": "fr",
    "german": "de",
    "portuguese": "pt",
    "russian": "ru",
    "italian": "it",
    "arabic": "ar",
    "hindi": "hi",
    "vietnamese": "vi",
    "thai": "th",
    "indonesian": "id",
}


def normalize_language(value: Optional[str]) -> Optional[str]:
    """Reduce a language name or code to an ISO 639-1 code where we know the mapping."""
    if not value:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return _LANGUAGE_ALIASES.get(text, text)


def resolve_asr_engine(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    return (os.environ.get("ASR_ENGINE") or "").strip() or DEFAULT_ASR_ENGINE


def _run_whisper_api(
    audio_path: Path,
    language: Optional[str],
    initial_prompt: Optional[str],
    **_: Any,
) -> Dict[str, Any]:
    """OpenAI `whisper-1`. Zero setup, segment-level timings only.

    Whisper's own segment boundaries drift by roughly a second. They are honest to ~2-10s,
    which is why this engine reports `segment` and not `word`.
    """
    from agentloom_media.audio.asr import transcribe_audio_file

    result = transcribe_audio_file(
        audio_path, language=language, prompt=initial_prompt
    )
    segments = [
        {
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": s.get("text", ""),
        }
        for s in result.get("segments", [])
    ]
    engine_id = result.get("engine", "openai:whisper-1")
    return {
        "segments": segments,
        "words": [],
        "language": normalize_language(result.get("language") or language),
        "engine": "whisper_api",
        "model": engine_id.split(":", 1)[-1],
        "engine_version": engine_id,
        "timing_granularity": "segment",
    }


def _run_faster_whisper(
    audio_path: Path,
    language: Optional[str],
    initial_prompt: Optional[str],
    model_size: Optional[str] = None,
    **_: Any,
) -> Dict[str, Any]:
    """Local `faster-whisper` (CTranslate2). Word timings come from Whisper's own attention,
    which is less reliable than forced alignment, so this reports `segment` unless word
    timestamps were requested and returned."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise AsrEngineUnavailable(
            "faster-whisper is not installed. Install it with "
            "`uv pip install -e '.[local-asr]'`, or set ASR_ENGINE=whisper_api."
        ) from exc

    size = model_size or (os.environ.get("ASR_MODEL_SIZE") or "large-v3").strip()
    compute_type = (os.environ.get("ASR_COMPUTE_TYPE") or "int8").strip()
    device = (os.environ.get("ASR_DEVICE") or "cpu").strip()
    want_words = (os.environ.get("ASR_WORD_TIMESTAMPS") or "1").strip() not in {
        "0",
        "false",
        "off",
        "no",
    }

    model = WhisperModel(size, device=device, compute_type=compute_type)
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        initial_prompt=initial_prompt,
        word_timestamps=want_words,
        vad_filter=True,
    )

    segments: List[Dict[str, Any]] = []
    words: List[Dict[str, Any]] = []
    for seg in segments_iter:
        segments.append(
            {"start": float(seg.start), "end": float(seg.end), "text": seg.text}
        )
        for word in getattr(seg, "words", None) or []:
            words.append(
                {
                    "w": word.word,
                    "t0": float(word.start),
                    "t1": float(word.end),
                    "conf": getattr(word, "probability", None),
                }
            )

    return {
        "segments": segments,
        "words": words,
        "language": normalize_language(getattr(info, "language", None) or language),
        "engine": "faster_whisper",
        "model": size,
        "engine_version": f"faster-whisper:{size}:{compute_type}",
        # Whisper-native word times are approximate; only claim `word` if we have them.
        "timing_granularity": "word" if words else "segment",
    }


def _unavailable(name: str, reason: str):
    def _run(*_: Any, **__: Any) -> Dict[str, Any]:
        raise AsrEngineUnavailable(f"ASR engine '{name}' is not wired up: {reason}")

    return _run


ENGINES = {
    "whisper_api": _run_whisper_api,
    "faster_whisper": _run_faster_whisper,
    "whisperx": _unavailable(
        "whisperx",
        "local ASR is deferred for v1 (see plan §2.2). WhisperX adds wav2vec2 forced "
        "alignment for <100ms word timings and is the intended default once a GPU box is "
        "available. Install with `uv pip install -e '.[whisperx]'` and implement the adapter "
        "before selecting it.",
    ),
    "whisper_cpp": _unavailable(
        "whisper_cpp",
        "requires a local whisper.cpp build with Metal/CoreML. Not implemented; use "
        "faster_whisper for local CPU transcription.",
    ),
}


def transcribe(
    audio_path: Path,
    engine: Optional[str] = None,
    language: Optional[str] = None,
    initial_prompt: Optional[str] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Transcribe with the selected engine.

    Returns `segments`, `words`, `language`, `engine`, `model`, `engine_version`, and
    `timing_granularity`.
    """
    name = resolve_asr_engine(engine)
    runner = ENGINES.get(name)
    if runner is None:
        raise AsrEngineUnavailable(
            f"Unknown ASR engine '{name}'. Available: {', '.join(sorted(ENGINES))}."
        )
    return runner(
        audio_path, language=language, initial_prompt=initial_prompt, **kwargs
    )
