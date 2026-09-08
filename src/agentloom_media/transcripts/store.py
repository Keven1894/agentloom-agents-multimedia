"""Persistence for canonical transcripts.

Layout (git-ignored; large and regenerable from the media URL plus the manifest):

```
data/transcripts/<media_id>/
├── transcript.<engine>.<model>.<pipeline_version>.json
└── manifest.json
```

The filename carries the engine identity so that switching engines cannot silently serve the
previous engine's output. `manifest.json` records every variant produced for the media item,
along with the normalization settings and a content hash, so an artifact can always state
which engine produced the timestamps it cites.

This supersedes the interim provenance-keyed cache from P0, which lived in
`acquisition/transcript_cache.py`; `find_legacy_segments` reads those files for migration.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentloom_media.transcripts.model import PIPELINE_VERSION, Transcript

MANIFEST_SCHEMA_VERSION = 1

# Prefer the most precise timings available when reusing a cached transcript.
GRANULARITY_RANK = {"word": 0, "segment": 1, "cue": 2}

# At equal precision, prefer a transcript whose engine we can name. A migrated pre-P1 cache
# must never outrank a real transcription just because it was written first.
SOURCE_RANK = {"asr": 0, "captions": 1, "legacy": 3}

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(value: Optional[str]) -> str:
    return _SAFE.sub("-", str(value or "unknown")).strip("-") or "unknown"


def default_root(repo_root: Path) -> Path:
    return repo_root / "data" / "transcripts"


def media_dir(repo_root: Path, media_id: str) -> Path:
    return default_root(repo_root) / _safe(media_id)


def variant_name(engine: str, model: Optional[str], pipeline_version: str) -> str:
    return f"transcript.{_safe(engine)}.{_safe(model or 'default')}.{_safe(pipeline_version)}.json"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_manifest(repo_root: Path, media_id: str) -> Dict[str, Any]:
    path = media_dir(repo_root, media_id) / "manifest.json"
    if not path.exists():
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "media_id": media_id,
            "variants": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "media_id": media_id,
            "variants": [],
        }
    data.setdefault("variants", [])
    return data


def save_transcript(
    repo_root: Path,
    transcript: Transcript,
    *,
    source: str,
    engine: str,
    model: Optional[str] = None,
    engine_version: Optional[str] = None,
    media_url: Optional[str] = None,
    title: Optional[str] = None,
    channel: Optional[str] = None,
    normalization: Optional[Dict[str, Any]] = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> Dict[str, Any]:
    """Write the transcript and record its provenance in the manifest.

    Returns the manifest variant entry that describes what was written.
    """
    target_dir = media_dir(repo_root, transcript.media_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    name = variant_name(engine, model, pipeline_version)
    payload = transcript.to_dict()
    serialized = json.dumps(payload, ensure_ascii=False, indent=2)

    with open(target_dir / name, "w", encoding="utf-8") as f:
        f.write(serialized)

    variant = {
        "file": name,
        "source": source,
        "engine": engine,
        "engine_version": engine_version,
        "model": model,
        "pipeline_version": pipeline_version,
        "language": transcript.language,
        "timing_granularity": transcript.timing_granularity,
        "utterance_count": len(transcript.utterances),
        "word_count": len(transcript.words),
        "duration": transcript.duration,
        "normalization": normalization or {},
        "transcript_sha256": _sha256(serialized),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    manifest = read_manifest(repo_root, transcript.media_id)
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["media_id"] = transcript.media_id
    if media_url:
        manifest["media_url"] = media_url
    if title:
        manifest["title"] = title
    if channel:
        manifest["channel"] = channel
    manifest["variants"] = [
        v for v in manifest["variants"] if v.get("file") != name
    ] + [variant]

    with open(target_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return variant


def _negated_timestamp(created_at: str) -> float:
    """Sort key that puts newer ISO timestamps first."""
    try:
        return -datetime.fromisoformat(created_at).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _load_variant(
    repo_root: Path, media_id: str, variant: Dict[str, Any]
) -> Optional[Transcript]:
    path = media_dir(repo_root, media_id) / str(variant.get("file", ""))
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return Transcript.from_dict(json.load(f))
    except (OSError, json.JSONDecodeError, KeyError):
        return None


def load_transcript(
    repo_root: Path,
    media_id: str,
    *,
    engine: Optional[str] = None,
    model: Optional[str] = None,
    pipeline_version: str = PIPELINE_VERSION,
) -> Optional[Dict[str, Any]]:
    """Load a cached transcript.

    With `engine` given, only that engine's output for this pipeline version is returned, so
    an engine comparison never gets another engine's transcript. Without it, the most precise
    available variant is returned; callers must surface the returned provenance, since it may
    not be what they would have produced.

    Returns `{"transcript": Transcript, "variant": {...}}`, or None.
    """
    manifest = read_manifest(repo_root, media_id)
    variants: List[Dict[str, Any]] = manifest.get("variants", [])
    if not variants:
        return None

    if engine:
        wanted = variant_name(engine, model, pipeline_version)
        variants = [v for v in variants if v.get("file") == wanted]
    else:
        variants = [
            v for v in variants if v.get("pipeline_version") == pipeline_version
        ] or variants

    def preference(variant: Dict[str, Any]) -> Any:
        return (
            GRANULARITY_RANK.get(variant.get("timing_granularity", "cue"), 9),
            SOURCE_RANK.get(variant.get("source", "legacy"), 3),
            # Newest first among otherwise equal candidates.
            _negated_timestamp(variant.get("created_at", "")),
        )

    variants = sorted(variants, key=preference)

    for variant in variants:
        transcript = _load_variant(repo_root, media_id, variant)
        if transcript and transcript.utterances:
            return {"transcript": transcript, "variant": variant}

    return None


def find_legacy_segments(
    repo_root: Path, media_id: str
) -> Optional[Dict[str, Any]]:
    """Read a pre-P1 cache from `.cache/transcripts/`, for one-time migration.

    Handles both the original bare-segment-list files and the P0 provenance wrapper. Returns
    `{"segments": [...], "source": ..., "engine": ..., "language": ...}` or None.
    """
    cache_dir = repo_root / ".cache" / "transcripts"
    if not cache_dir.exists():
        return None

    candidates = sorted(cache_dir.glob(f"{_safe(media_id)}__*.json"))
    legacy_bare = cache_dir / f"{media_id}_segments.json"
    if legacy_bare.exists():
        candidates.append(legacy_bare)

    for path in candidates:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue

        if isinstance(data, list) and data:
            return {
                "segments": data,
                "source": "legacy",
                "engine": "unknown",
                "language": None,
                "path": str(path),
            }
        if isinstance(data, dict) and data.get("segments"):
            return {
                "segments": data["segments"],
                "source": data.get("source", "legacy"),
                "engine": data.get("engine", "unknown"),
                "language": data.get("language"),
                "path": str(path),
            }

    return None
