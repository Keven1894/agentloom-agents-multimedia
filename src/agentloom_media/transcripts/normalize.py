"""Language normalization for transcripts.

Two defects surfaced in the 2026-09-07 DAOJIE run:

1. The video title is Simplified Chinese but Whisper returned Traditional, so the digest, the
   index, and the KG would hold two spellings of the same entity and never match.
2. Mixed zh/en domain speech ("Campaign Structure", "CBO", "Andromeda") was transliterated
   inconsistently. Whisper accepts an `initial_prompt` to bias vocabulary and we were not
   using it.

Normalization only ever produces an additional field. `text_raw` in the canonical transcript
keeps exactly what the engine said.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Default OpenCC conversion: Traditional → Simplified.
DEFAULT_OPENCC_CONFIG = "t2s"

DEFAULT_GLOSSARY_FILE = "config/asr-glossary.txt"


def _load_opencc(config: str):
    try:
        import opencc  # type: ignore
    except ImportError:
        return None
    try:
        return opencc.OpenCC(config)
    except Exception:
        return None


def build_script_normalizer(
    config: Optional[str] = None,
) -> tuple[Optional[Callable[[str], str]], Dict[str, Any]]:
    """Build the script normalizer and the metadata describing it.

    Returns `(normalizer_or_None, info)`. `info` always states whether normalization was
    actually applied, so a manifest never claims a conversion that silently did not happen.
    """
    if (os.environ.get("TRANSCRIPT_NORMALIZE_SCRIPT") or "").strip().lower() in {
        "0",
        "false",
        "off",
        "no",
    }:
        return None, {"script": "disabled", "opencc_config": None, "applied": False}

    resolved = (
        config
        or (os.environ.get("TRANSCRIPT_OPENCC_CONFIG") or "").strip()
        or DEFAULT_OPENCC_CONFIG
    )

    converter = _load_opencc(resolved)
    if converter is None:
        return None, {
            "script": "unavailable",
            "opencc_config": resolved,
            "applied": False,
            "note": "OpenCC not installed; transcript kept in its original script.",
        }

    def normalize(text: str) -> str:
        return converter.convert(text)

    return normalize, {
        "script": "simplified" if resolved.endswith("s") else resolved,
        "opencc_config": resolved,
        "applied": True,
    }


def load_glossary(repo_root: Optional[Path] = None) -> List[str]:
    """Domain terms used to bias ASR vocabulary.

    Sources, merged in order: the `ASR_GLOSSARY` env var (comma-separated), then the glossary
    file (`ASR_GLOSSARY_FILE`, default `config/asr-glossary.txt`).
    """
    terms: List[str] = []

    env_terms = (os.environ.get("ASR_GLOSSARY") or "").strip()
    if env_terms:
        terms.extend(t.strip() for t in env_terms.split(",") if t.strip())

    # An explicitly configured file is authoritative: falling back to the packaged default
    # would quietly bias ASR with terms the caller deliberately replaced.
    file_setting = (os.environ.get("ASR_GLOSSARY_FILE") or "").strip()
    if file_setting:
        candidates: List[Path] = [Path(file_setting)]
    else:
        candidates = []
        if repo_root:
            candidates.append(repo_root / DEFAULT_GLOSSARY_FILE)
        candidates.append(Path(__file__).resolve().parents[3] / DEFAULT_GLOSSARY_FILE)

    for path in candidates:
        if not path.exists():
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    terms.append(line)
        except OSError:
            continue
        break

    seen = set()
    unique = []
    for term in terms:
        if term not in seen:
            seen.add(term)
            unique.append(term)
    return unique


def build_initial_prompt(repo_root: Optional[Path] = None) -> Optional[str]:
    """Render the glossary as a Whisper `initial_prompt`.

    Whisper treats the prompt as preceding context rather than instructions, so a bare
    comma-separated term list biases the decoder better than a sentence telling it what to do.
    The API caps the prompt at 224 tokens, so keep the glossary short and domain-specific.
    """
    terms = load_glossary(repo_root)
    if not terms:
        return None
    return "、".join(terms) if any(_is_cjk(t) for t in terms) else ", ".join(terms)


def _is_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)
