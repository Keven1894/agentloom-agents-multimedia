"""Grounded extraction of keywords, technical items, and definitions.

Every extraction must quote the transcript verbatim. The quote is then located in the document
text and resolved to a time span through the canonical transcript's character map, so each
term arrives with an anchor rather than an assertion. Extractions whose quote cannot be found
are **dropped**, which makes this a structural anti-hallucination gate rather than a prompt
instruction the model may ignore.

Relation to plan §4.3: the design proposed Google's LangExtract for this, whose decisive
feature is exactly this `char_interval` grounding. It is not used here, for two measured
reasons. First, its alignment is word-based and tuned for space-delimited text, and this
corpus is unsegmented Chinese — the design itself flagged that alignment quality had to be
verified before depending on it. Second, it carries its own model-provider configuration,
which would add a second LLM credential path alongside the existing one. The grounding
mechanism it provides is reimplemented here in a way that is exact for CJK: substring search,
with whitespace-insensitive fallback because ASR inserts spaces the model does not reproduce.
LangExtract remains a reasonable swap for the extraction call if its CJK alignment is measured
and found good.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Both ASCII and CJK parentheses; models use either when glossing a term.
_PARENTHETICAL = re.compile(r"[（(]([^）)]*)[）)]")

from agentloom_media.distillation.anchors import format_timestamp
from agentloom_media.distillation.passes import _complete_json
from agentloom_media.distillation.distiller import resolve_distillation_model

TERM_TYPES = ("keyword", "technical_item", "definition", "metric", "entity")

TERMS_SYSTEM = """You extract terminology and knowledge points from a transcript.

For each extraction you MUST provide "quote": a span copied **verbatim** from the transcript,
character for character, that contains the term and shows how it is used. Do not paraphrase,
do not correct, do not translate the quote. A quote that does not appear in the transcript
exactly will be discarded along with its extraction.

Types: keyword | technical_item | definition | metric | entity

Return JSON:
{"terms": [{"term": "<string>", "type": "<one of the types>",
            "description": "<what it means here, in the transcript's language>",
            "quote": "<verbatim span from the transcript>"}]}

Extract 8 to 20 terms. Prefer domain-specific terms over generic vocabulary. A "metric" is a
specific figure the speaker cites."""


def _build_stripped_index(text: str) -> Tuple[str, List[int]]:
    """Whitespace-free copy of `text` plus a map back to original offsets."""
    chars: List[str] = []
    positions: List[int] = []
    for index, char in enumerate(text):
        if not char.isspace():
            chars.append(char)
            positions.append(index)
    return "".join(chars), positions


def find_quote_span(
    text: str,
    quote: str,
    stripped: Optional[Tuple[str, List[int]]] = None,
) -> Optional[Tuple[int, int]]:
    """Locate a quote in the document text, returning `(char_start, char_end)`.

    Tries an exact match first, then a whitespace-insensitive match: ASR output contains
    spaces that a model reproducing a Chinese quote will usually drop, and rejecting those
    quotes would discard correctly grounded extractions.
    """
    quote = (quote or "").strip()
    if not quote or not text:
        return None

    exact = text.find(quote)
    if exact != -1:
        return exact, exact + len(quote)

    stripped_text, positions = stripped or _build_stripped_index(text)
    stripped_quote = "".join(c for c in quote if not c.isspace())
    if not stripped_quote:
        return None

    found = stripped_text.find(stripped_quote)
    if found == -1:
        return None

    start = positions[found]
    end = positions[min(found + len(stripped_quote) - 1, len(positions) - 1)] + 1
    return start, end


def _squash(text: str) -> str:
    """Case-folded, whitespace-free form for containment checks."""
    return "".join(c for c in (text or "") if not c.isspace()).casefold()


def _term_surfaces(term: str) -> List[str]:
    """Surface forms of a term that would each count as literal support.

    Models annotate terms with a gloss or acronym — "广告投资回报（ROAS）" — which appears in
    no transcript verbatim even though both halves do. Accepting either half keeps the
    grounding real while not discarding a correct extraction over its formatting.
    """
    surfaces = [term]
    surfaces.append(_PARENTHETICAL.sub(" ", term))
    surfaces.extend(_PARENTHETICAL.findall(term))
    return [s for s in (surface.strip() for surface in surfaces) if s]


def term_supported_by_quote(term: str, quote: str) -> bool:
    """Whether the quote actually contains the term it is offered as evidence for.

    Without this check a term can drift from its own evidence: an extraction named
    "Maximized Conversion Value" was grounded by a quote reading "Maximized Conversion
    Volume". The quote was real, so quote-location alone accepted it, while the term itself
    was wrong. Requiring literal containment applies the same "better absent than wrong"
    rule as anchor validation.
    """
    squashed_quote = _squash(quote)
    if not squashed_quote:
        return False
    return any(
        _squash(surface) in squashed_quote
        for surface in _term_surfaces(term)
        if _squash(surface)
    )


def extract_grounded_terms(
    transcript: Any,
    video_url: str,
    model: Optional[str] = None,
    max_chars: int = 14000,
) -> Dict[str, Any]:
    """Extract terms and keep only those whose quote is present in the transcript.

    Returns `{"terms": [...], "stats": {...}}`. Each kept term carries `t0`, `t1`,
    `timestamp`, `anchor_url`, and the character span its quote occupies.
    """
    text = transcript.text or ""
    if not text.strip():
        return {
            "terms": [],
            "stats": {
                "extracted": 0,
                "grounded": 0,
                "dropped": 0,
                "dropped_reasons": {},
                "dropped_examples": [],
            },
        }

    model = resolve_distillation_model(model)
    payload = _complete_json(
        TERMS_SYSTEM,
        f"Transcript:\n\n{text[:max_chars]}",
        model,
        max_tokens=4096,
    )

    stripped = _build_stripped_index(text)
    grounded: List[Dict[str, Any]] = []
    dropped: List[Dict[str, str]] = []
    extracted = 0

    for item in payload.get("terms", []) or []:
        if not isinstance(item, dict) or not item.get("term"):
            continue
        extracted += 1

        term = str(item["term"]).strip()
        quote = str(item.get("quote") or "")
        span = find_quote_span(text, quote, stripped)
        if span is None:
            dropped.append({"term": term, "reason": "quote_not_found"})
            continue

        located = text[span[0] : span[1]]
        if not term_supported_by_quote(term, located):
            dropped.append({"term": term, "reason": "term_not_in_quote"})
            continue

        times = transcript.resolve_char_span(span[0], span[1])
        if times is None:
            dropped.append({"term": term, "reason": "span_not_resolvable"})
            continue

        t0, t1 = times
        term_type = str(item.get("type") or "keyword").strip()
        grounded.append(
            {
                "term": term,
                "type": term_type if term_type in TERM_TYPES else "keyword",
                "description": str(item.get("description") or "").strip(),
                "quote": text[span[0] : span[1]],
                "char_start": span[0],
                "char_end": span[1],
                "t0": t0,
                "t1": t1,
                "timestamp": format_timestamp(t0),
                "anchor_url": _anchor(video_url, t0),
            }
        )

    reasons: Dict[str, int] = {}
    for entry in dropped:
        reasons[entry["reason"]] = reasons.get(entry["reason"], 0) + 1

    return {
        "terms": grounded,
        "stats": {
            "extracted": extracted,
            "grounded": len(grounded),
            "dropped": len(dropped),
            "dropped_reasons": reasons,
            "dropped_examples": dropped[:10],
        },
    }


def _anchor(video_url: str, seconds: float) -> Optional[str]:
    if not video_url:
        return None
    joiner = "&" if "?" in video_url else "?"
    return f"{video_url}{joiner}t={int(seconds)}s"
