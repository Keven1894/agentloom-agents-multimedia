"""Typed tuple extraction, grounded per segment.

Extraction runs once per segment rather than once per document. That is ATOM's per-unit design
— segments are independent, a failure is contained to one segment, and evidence stays scoped
to a span we already know is real. Unlike ATOM, the source text is handed to the model
verbatim rather than as decontextualized paraphrase, so the quote it returns can be located in
the transcript and resolved to a time.

The grounding gate is the same one P3 established for terms: quote must be locatable, and must
contain what it is offered as evidence for. Edges that fail are dropped.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agentloom_media.distillation.anchors import format_timestamp
from agentloom_media.distillation.passes import _complete_json
from agentloom_media.distillation.distiller import resolve_distillation_model
from agentloom_media.distillation.terms import find_quote_span, term_supported_by_quote
from agentloom_media.kg.model import (
    EDGE_FAMILIES,
    Edge,
    Evidence,
    GraphProposal,
    MODEL_EMITTED_FAMILIES,
    PREDICATE_TO_FAMILY,
    is_atomic_surface,
    normalize_surface,
)

DEFAULT_MAX_WORKERS = 4

_PREDICATE_LIST = "\n".join(
    f"- {family}: {', '.join(EDGE_FAMILIES[family])}"
    for family in MODEL_EMITTED_FAMILIES
)

KG_SYSTEM = f"""You extract typed relations from one transcript segment.

Use ONLY these predicates, grouped by family:

{_PREDICATE_LIST}

For each relation provide:
- "source" and "target": short entity surfaces as spoken (a term, product, tactic, or metric).
  Not sentences. Not dates.
- "predicate": exactly one from the list above.
- "claim": one sentence stating what the speaker asserts, in the transcript's language.
- "quote": a span copied **verbatim** from the segment text, character for character, that
  states this relation. Do not paraphrase, correct, or translate it. A quote that is not found
  in the segment verbatim is discarded together with its relation.
- "valid_from" / "valid_until": only if the speaker states a period this holds for, as
  YYYY or YYYY-MM or YYYY-MM-DD. Use null when not stated. Do NOT guess from context.
- "supersedes": if the speaker says some previous practice or belief is now obsolete, the
  surface of that superseded thing. Otherwise null.

Return JSON: {{"relations": [{{"source": ..., "target": ..., "predicate": ...,
"claim": ..., "quote": ..., "valid_from": null, "valid_until": null, "supersedes": null}}]}}

Extract 3 to 10 relations. Prefer relations the speaker argues for over incidental mentions."""


def _anchor(video_url: str, seconds: float) -> Optional[str]:
    if not video_url:
        return None
    joiner = "&" if "?" in video_url else "?"
    return f"{video_url}{joiner}t={int(seconds)}s"


def _extract_for_segment(
    segment: Any, model: str
) -> Tuple[int, List[Dict[str, Any]], Optional[str]]:
    """One LLM call for one segment. Returns (segment_index, relations, error)."""
    try:
        payload = _complete_json(
            KG_SYSTEM,
            f"Segment [{format_timestamp(segment.t0)}–{format_timestamp(segment.t1)}] "
            f"titled {segment.title or '(untitled)'}:\n\n{segment.text[:6000]}",
            model,
            max_tokens=3072,
        )
        relations = [
            item for item in (payload.get("relations") or []) if isinstance(item, dict)
        ]
        return segment.index, relations, None
    except Exception as exc:
        return segment.index, [], str(exc)


def extract_graph(
    transcript: Any,
    segments: Sequence[Any],
    terms: Sequence[Dict[str, Any]],
    video_url: str,
    observed_at: Optional[str] = None,
    model: Optional[str] = None,
    max_workers: Optional[int] = None,
) -> GraphProposal:
    """Build a grounded, typed working graph for one media item.

    `terms` are the grounded terms from P3; they seed definitional nodes with evidence that has
    already passed the same gate, so the graph is not limited to what this pass rediscovers.
    """
    model = resolve_distillation_model(model)
    workers = max_workers or int(
        os.environ.get("KG_MAX_WORKERS") or DEFAULT_MAX_WORKERS
    )
    graph = GraphProposal(media_id=transcript.media_id)

    # Seed from grounded terms. These already carry verified spans.
    for term in terms:
        evidence = Evidence(
            media_id=transcript.media_id,
            t0=term["t0"],
            t1=term["t1"],
            quote=term["quote"],
            anchor_url=term.get("anchor_url"),
        )
        graph.add_node(
            term["term"],
            term.get("type") or "keyword",
            evidence=evidence,
            description=term.get("description"),
        )

    seeded = len(graph.nodes)
    segment_by_index = {segment.index: segment for segment in segments}

    # Per-segment extraction in parallel. Segments are independent by construction.
    results: List[Tuple[int, List[Dict[str, Any]], Optional[str]]] = []
    if segments:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            results = list(
                pool.map(lambda s: _extract_for_segment(s, model), segments)
            )

    proposed = 0
    dropped: Dict[str, int] = {}
    failed_segments: List[int] = []

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    for segment_index, relations, error in sorted(results, key=lambda r: r[0]):
        if error:
            failed_segments.append(segment_index)
            continue

        segment = segment_by_index[segment_index]
        for item in relations:
            proposed += 1

            source = str(item.get("source") or "").strip()
            target = str(item.get("target") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            quote = str(item.get("quote") or "")

            if not source or not target:
                drop("missing_endpoint")
                continue

            family = PREDICATE_TO_FAMILY.get(predicate)
            if family is None or family == "evidential":
                # An invented predicate cannot be placed in a reviewable family.
                drop("unknown_predicate")
                continue

            if normalize_surface(source) == normalize_surface(target):
                drop("self_loop")
                continue

            if not (is_atomic_surface(source) and is_atomic_surface(target)):
                drop("surface_not_atomic")
                continue

            # Ground the quote inside this segment, then map to a real transcript time.
            span = find_quote_span(segment.text, quote)
            if span is None:
                drop("quote_not_found")
                continue

            located = segment.text[span[0] : span[1]]

            # A real quote is not evidence for a relation it does not mention. Observed:
            # "CBO alternative_to ABO" — a correct relation — was grounded by a quote about
            # the speaker's own ad account, mentioning neither endpoint. Requiring both
            # surfaces in the quote is the edge analogue of `term_supported_by_quote`.
            if not (
                term_supported_by_quote(source, located)
                and term_supported_by_quote(target, located)
            ):
                drop("endpoints_not_in_quote")
                continue

            times = _resolve_time(transcript, located, segment)
            if times is None:
                drop("span_not_resolvable")
                continue

            t0, t1 = times
            evidence = Evidence(
                media_id=transcript.media_id,
                t0=t0,
                t1=t1,
                quote=located,
                anchor_url=_anchor(video_url, t0),
                segment_index=segment_index,
            )

            graph.add_node(source, "surface", evidence=evidence)
            graph.add_node(target, "surface")

            supersedes = str(item.get("supersedes") or "").strip() or None
            if supersedes:
                graph.add_node(supersedes, "surface")

            graph.edges.append(
                Edge(
                    source=f"surface:{normalize_surface(source)}",
                    target=f"surface:{normalize_surface(target)}",
                    predicate=predicate,
                    family=family,
                    evidence=evidence,
                    observed_at=observed_at,
                    valid_from=_clean_date(item.get("valid_from")),
                    valid_until=_clean_date(item.get("valid_until")),
                    supersedes=(
                        f"surface:{normalize_surface(supersedes)}"
                        if supersedes
                        else None
                    ),
                    claim=str(item.get("claim") or "").strip() or None,
                )
            )

    graph.stats = {
        "nodes": len(graph.nodes),
        "nodes_seeded_from_terms": seeded,
        "edges": len(graph.edges),
        "relations_proposed": proposed,
        "relations_dropped": sum(dropped.values()),
        "dropped_reasons": dropped,
        "segments_extracted": len(results) - len(failed_segments),
        "segments_failed": failed_segments,
    }
    return graph


def _resolve_time(
    transcript: Any, located: str, segment: Any
) -> Optional[Tuple[float, float]]:
    """Map a quote located within a segment onto real transcript times.

    The quote was found in the segment's own text, so it must be re-located against the
    document text to get character offsets the transcript can resolve. If that fails, fall
    back to the segment's own span, which is real, just coarser.
    """
    document_span = find_quote_span(transcript.text, located)
    if document_span is not None:
        times = transcript.resolve_char_span(*document_span)
        if times is not None:
            return times
    if segment is not None:
        return segment.t0, segment.t1
    return None


def _clean_date(value: Any) -> Optional[str]:
    """Keep only YYYY, YYYY-MM, or YYYY-MM-DD. Anything vaguer is not a date we can use."""
    text = str(value or "").strip()
    if not text or text.lower() in {"null", "none", "n/a", "unknown"}:
        return None
    parts = text.split("-")
    if not parts[0].isdigit() or len(parts[0]) != 4:
        return None
    if len(parts) > 3:
        return None
    for part in parts[1:]:
        if not part.isdigit():
            return None
    return text
