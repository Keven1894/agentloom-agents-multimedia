"""Transcript acquisition: cache → captions → ASR, producing one canonical transcript.

Routes in ascending cost (plan §2.1). The important correction to the original pipeline is
that **the fast caption route is not automatically preferable**: caption cues carry no word or
speaker timing, so a captions-only item cannot support precise seeking. The route taken and the
timing granularity it yields are recorded rather than glossed over.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agentloom_media.transcripts import store
from agentloom_media.transcripts.model import PIPELINE_VERSION, Transcript
from agentloom_media.transcripts.normalize import (
    build_initial_prompt,
    build_script_normalizer,
    load_glossary,
)

Logger = Callable[[str], None]

ROUTE_AUTO = "auto"
ROUTE_FAST = "fast"
ROUTE_HEAVY = "heavy"


@dataclass
class Acquisition:
    """A canonical transcript plus how it was obtained."""

    transcript: Transcript
    route: str
    source: str
    engine: str
    model: Optional[str]
    timing_granularity: str
    reused: bool
    normalization: Dict[str, Any]
    variant: Optional[Dict[str, Any]] = None

    @property
    def segments(self) -> List[Dict[str, Any]]:
        """Flat view for stages not yet ported to the canonical model."""
        return self.transcript.to_segments()

    def provenance(self) -> Dict[str, Any]:
        """The subset of provenance that emitted artifacts must state."""
        return {
            "route": self.route,
            "source": self.source,
            "engine": self.engine,
            "model": self.model,
            "timing_granularity": self.timing_granularity,
            "language": self.transcript.language,
            "reused": self.reused,
            "normalization": self.normalization,
            "utterance_count": len(self.transcript.utterances),
            "pipeline_version": PIPELINE_VERSION,
        }

    def describe(self) -> str:
        prefix = "reused" if self.reused else self.route
        return (
            f"{prefix}: source={self.source}, engine={self.engine}, "
            f"model={self.model or 'n/a'}, timing={self.timing_granularity}, "
            f"language={self.transcript.language or 'unknown'}, "
            f"utterances={len(self.transcript.utterances)}"
        )


def resolve_route(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip().lower()
    return (os.environ.get("TRANSCRIPT_ROUTE") or "").strip().lower() or ROUTE_AUTO


def acquire_transcript(
    meta: Dict[str, Any],
    url: str,
    repo_root: Path,
    *,
    engine: Optional[str] = None,
    language: Optional[str] = None,
    route: Optional[str] = None,
    force: bool = False,
    log: Optional[Logger] = None,
) -> Acquisition:
    """Obtain a canonical transcript for a media item.

    Args:
        meta: Probe metadata; must carry `id`.
        url: Media URL, used by the heavy route.
        repo_root: Repo root; the transcript store lives under `data/transcripts/`.
        engine: ASR engine override. When set, a cached transcript from a *different* engine
            is not reused, so engine comparisons stay honest.
        language: Language hint passed to ASR.
        route: `auto` (captions then ASR), `fast` (captions only), `heavy` (ASR only).
        force: Ignore any cached transcript and re-acquire.
        log: Optional progress sink.
    """
    say: Logger = log or (lambda _msg: None)
    media_id = meta.get("id") or "media"
    chosen_route = resolve_route(route)
    normalizer, norm_info = build_script_normalizer()

    if norm_info.get("script") == "unavailable":
        say(
            "Script normalization unavailable (OpenCC not installed); "
            "transcript kept in its original script."
        )

    # 1. Reuse
    if not force:
        cached = store.load_transcript(
            repo_root, media_id, engine=engine, pipeline_version=PIPELINE_VERSION
        )
        if cached:
            variant = cached["variant"]
            return Acquisition(
                transcript=cached["transcript"],
                route=chosen_route,
                source=variant.get("source", "unknown"),
                engine=variant.get("engine", "unknown"),
                model=variant.get("model"),
                timing_granularity=cached["transcript"].timing_granularity,
                reused=True,
                normalization=variant.get("normalization", {}),
                variant=variant,
            )

        migrated = _migrate_legacy(
            repo_root, media_id, meta, url, normalizer, norm_info, say
        )
        if migrated:
            return migrated

    # 2. Fast route: author or automatic captions.
    if chosen_route in (ROUTE_AUTO, ROUTE_FAST) and meta.get("id"):
        from agentloom_media.acquisition.fast_transcript import fetch_fast_transcript

        say("Fast route: checking caption tracks...")
        raw_captions = fetch_fast_transcript(meta["id"])
        if raw_captions:
            segments = [
                {
                    "start": c["start"],
                    "end": c["start"] + c.get("duration", 0),
                    "text": c["text"],
                }
                for c in raw_captions
            ]
            transcript = Transcript.from_segments(
                media_id,
                segments,
                timing_granularity="cue",
                language=None,
                normalizer=normalizer,
            )
            say(
                f"Fast route succeeded: {len(transcript.utterances)} caption cues. "
                "Timing granularity is 'cue' — no word-level seeking."
            )
            variant = store.save_transcript(
                repo_root,
                transcript,
                source="captions",
                engine="youtube_captions",
                model=None,
                media_url=url,
                title=meta.get("title"),
                channel=meta.get("channel"),
                normalization=norm_info,
            )
            return Acquisition(
                transcript=transcript,
                route=ROUTE_FAST,
                source="captions",
                engine="youtube_captions",
                model=None,
                timing_granularity="cue",
                reused=False,
                normalization=norm_info,
                variant=variant,
            )

        if chosen_route == ROUTE_FAST:
            raise RuntimeError(
                "Fast route requested but no captions are available for this item. "
                "Re-run with --route heavy to transcribe the audio."
            )
        say("Captions unavailable; falling back to the heavy route.")

    # 3. Heavy route: extract audio and run ASR.
    from agentloom_media.acquisition.audio_extractor import extract_audio_stream
    from agentloom_media.audio import engines

    say("Heavy route: extracting audio stream...")
    audio_cache_dir = repo_root / ".cache" / "audio"
    audio_cache_dir.mkdir(parents=True, exist_ok=True)
    audio_file = extract_audio_stream(url, str(audio_cache_dir))
    size_mb = audio_file.stat().st_size / (1024 * 1024)

    engine_name = engines.resolve_asr_engine(engine)
    glossary = load_glossary(repo_root)
    initial_prompt = build_initial_prompt(repo_root)
    if glossary:
        say(f"Biasing ASR vocabulary with {len(glossary)} glossary terms.")

    say(f"Running ASR engine '{engine_name}' on {size_mb:.1f} MB of audio...")
    result = engines.transcribe(
        audio_file,
        engine=engine_name,
        language=language,
        initial_prompt=initial_prompt,
    )

    transcript = Transcript.from_segments(
        media_id,
        result["segments"],
        timing_granularity=result.get("timing_granularity", "segment"),
        language=result.get("language"),
        normalizer=normalizer,
        words=result.get("words") or None,
    )
    norm_info = {**norm_info, "asr_initial_prompt_terms": bool(initial_prompt)}

    say(
        f"ASR complete: {len(transcript.utterances)} utterances, "
        f"timing granularity '{transcript.timing_granularity}'."
    )

    variant = store.save_transcript(
        repo_root,
        transcript,
        source="asr",
        engine=result.get("engine", engine_name),
        model=result.get("model"),
        engine_version=result.get("engine_version"),
        media_url=url,
        title=meta.get("title"),
        channel=meta.get("channel"),
        normalization=norm_info,
    )

    return Acquisition(
        transcript=transcript,
        route=ROUTE_HEAVY,
        source="asr",
        engine=result.get("engine", engine_name),
        model=result.get("model"),
        timing_granularity=transcript.timing_granularity,
        reused=False,
        normalization=norm_info,
        variant=variant,
    )


def _migrate_legacy(
    repo_root: Path,
    media_id: str,
    meta: Dict[str, Any],
    url: str,
    normalizer: Optional[Callable[[str], str]],
    norm_info: Dict[str, Any],
    say: Logger,
) -> Optional[Acquisition]:
    """Promote a pre-P1 cache into the canonical store, keeping its provenance honest."""
    legacy = store.find_legacy_segments(repo_root, media_id)
    if not legacy:
        return None

    say(
        f"Migrating pre-P1 cached transcript into the canonical store "
        f"(source={legacy['source']}, engine={legacy['engine']})."
    )
    if legacy["engine"] == "unknown":
        say(
            "Warning: this transcript predates provenance tracking, so its engine and "
            "timing accuracy are unverified. Re-run with --force to re-transcribe."
        )

    transcript = Transcript.from_segments(
        media_id,
        legacy["segments"],
        timing_granularity="segment",
        language=legacy.get("language"),
        normalizer=normalizer,
    )
    variant = store.save_transcript(
        repo_root,
        transcript,
        source=legacy["source"],
        engine=legacy["engine"],
        model=None,
        engine_version="pre-p1-cache",
        media_url=url,
        title=meta.get("title"),
        channel=meta.get("channel"),
        normalization=norm_info,
    )
    return Acquisition(
        transcript=transcript,
        route="migrated",
        source=legacy["source"],
        engine=legacy["engine"],
        model=None,
        timing_granularity=transcript.timing_granularity,
        reused=True,
        normalization=norm_info,
        variant=variant,
    )
