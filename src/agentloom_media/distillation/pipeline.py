"""Runs the typed distillation passes and reports what succeeded.

Each pass is isolated: a failure records itself and the remaining passes still run. The result
carries a `passes` report so a reader of the digest can tell which fields are missing because
a pass failed, rather than because the video had nothing to say.

See plan §4.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agentloom_media.distillation import passes as typed_passes
from agentloom_media.distillation.segmentation import Segment, segment_transcript
from agentloom_media.distillation.terms import extract_grounded_terms
from agentloom_media.kg.extract import extract_graph
from agentloom_media.kg.merge import suggest_merges
from agentloom_media.kg.model import GraphProposal
from agentloom_media.verification.takeaway import build_takeaway

Logger = Callable[[str], None]


@dataclass
class DistillationResult:
    """The distilled output plus a per-pass status report."""

    segments: List[Segment] = field(default_factory=list)
    segmentation: Dict[str, Any] = field(default_factory=dict)
    document: Dict[str, Any] = field(default_factory=dict)
    terms: List[Dict[str, Any]] = field(default_factory=list)
    term_stats: Dict[str, Any] = field(default_factory=dict)
    skill: Optional[Dict[str, Any]] = None
    skill_decision: Dict[str, Any] = field(default_factory=dict)
    graph: Optional[GraphProposal] = None
    takeaway: Optional[Dict[str, Any]] = None
    passes: Dict[str, str] = field(default_factory=dict)

    @property
    def failed_passes(self) -> List[str]:
        return [name for name, status in self.passes.items() if status != "ok"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segmentation": self.segmentation,
            "segments": [s.to_dict() for s in self.segments],
            "document": self.document,
            "terms": self.terms,
            "term_stats": self.term_stats,
            "skill": self.skill,
            "skill_decision": self.skill_decision,
            "graph": self.graph.to_dict() if self.graph else None,
            "takeaway": self.takeaway,
            "passes": self.passes,
        }


def distill(
    transcript: Any,
    meta: Dict[str, Any],
    embedder: Optional[Any] = None,
    model: Optional[str] = None,
    synthesis_model: Optional[str] = None,
    search_index: Optional[Any] = None,
    log: Optional[Logger] = None,
) -> DistillationResult:
    """Segment, title, summarize, synthesize, extract terms, build the graph, and verify.

    `search_index` is the P2 hybrid index. The takeaway pass uses it to retrieve candidate
    spans for document-level claims, which is how a takeaway that carried no anchor acquires
    one; without it, those claims are reported ungrounded rather than assumed.
    """
    say: Logger = log or (lambda _msg: None)
    result = DistillationResult()
    utterances = transcript.utterances

    # Pass 0: segmentation. Everything else is scoped to its output.
    try:
        segments, info = segment_transcript(transcript, meta.get("chapters"), embedder)
        result.segments = segments
        result.segmentation = info
        result.passes["segmentation"] = "ok"
        say(
            f"Segmentation: {len(segments)} segments via {info.get('method')} "
            f"({info.get('chunks', 0)} chunks embedded)."
        )
        if info.get("note"):
            say(f"  {info['note']}")
    except Exception as exc:
        result.passes["segmentation"] = f"failed: {exc}"
        say(f"Segmentation failed: {exc}")
        return result

    if not result.segments:
        return result

    # Pass 1: titles.
    try:
        titles = typed_passes.title_segments(result.segments, model=model)
        for segment in result.segments:
            if segment.index in titles:
                segment.title = titles[segment.index]
        untitled = sum(1 for s in result.segments if not s.title)
        result.passes["titles"] = "ok"
        say(
            f"Titles: {len(titles)} generated"
            + (f", {untitled} segments left untitled" if untitled else "")
        )
    except Exception as exc:
        result.passes["titles"] = f"failed: {exc}"
        say(f"Titling failed: {exc}")

    # Pass 2: per-segment summaries. One bad segment must not lose the others.
    summarized = 0
    failures: List[str] = []
    for segment in result.segments:
        try:
            summary = typed_passes.summarize_segment(segment, utterances, model=model)
            segment.key_points = summary["key_points"]
            segment.analysis = summary["analysis"]
            summarized += 1
        except Exception as exc:
            failures.append(f"segment {segment.index}: {exc}")

    result.passes["segment_summaries"] = (
        "ok" if not failures else f"partial: {len(failures)} of {len(result.segments)} failed"
    )
    say(
        f"Segment summaries: {summarized}/{len(result.segments)} succeeded"
        + (f"; first failure: {failures[0]}" if failures else "")
    )

    # Pass 3: document synthesis, on the stronger tier.
    try:
        result.document = typed_passes.synthesize_document(
            meta.get("title", ""),
            meta.get("channel", ""),
            result.segments,
            model=synthesis_model,
        )
        result.passes["synthesis"] = "ok"
        say("Document synthesis: ok")
    except Exception as exc:
        result.passes["synthesis"] = f"failed: {exc}"
        say(f"Document synthesis failed: {exc}")

    # Pass 4: grounded terms.
    try:
        extraction = extract_grounded_terms(
            transcript, meta.get("url", ""), model=model
        )
        result.terms = extraction["terms"]
        result.term_stats = extraction["stats"]
        result.passes["terms"] = "ok"
        stats = extraction["stats"]
        detail = ", ".join(
            f"{count} {reason}"
            for reason, count in sorted((stats.get("dropped_reasons") or {}).items())
        )
        say(
            f"Grounded terms: {stats['grounded']}/{stats['extracted']} kept"
            + (f" ({detail})." if detail else ".")
        )
    except Exception as exc:
        result.passes["terms"] = f"failed: {exc}"
        say(f"Term extraction failed: {exc}")

    # Pass 5: knowledge graph expansion. Runs after terms so it can seed from their
    # already-verified spans.
    try:
        graph = extract_graph(
            transcript,
            result.segments,
            result.terms,
            meta.get("url", ""),
            observed_at=meta.get("upload_date") or meta.get("publish_date"),
            model=model,
        )
        graph.merge_candidates = suggest_merges(graph, embedder)
        result.graph = graph
        result.passes["kg"] = "ok"
        stats = graph.stats
        detail = ", ".join(
            f"{count} {reason}"
            for reason, count in sorted((stats.get("dropped_reasons") or {}).items())
        )
        say(
            f"Knowledge graph: {stats['nodes']} surfaces, {stats['edges']} edges from "
            f"{stats['relations_proposed']} proposed"
            + (f" ({detail})" if detail else "")
            + (
                f"; {len(graph.merge_candidates)} merge candidates for review"
                if graph.merge_candidates
                else ""
            )
            + "."
        )
        if stats.get("segments_failed"):
            say(f"  KG extraction failed for segments {stats['segments_failed']}.")
    except Exception as exc:
        result.passes["kg"] = f"failed: {exc}"
        say(f"Knowledge graph extraction failed: {exc}")

    # Pass 6: verified takeaway. Last, because it verifies the synthesis the earlier passes
    # produced.
    if result.document:
        try:
            result.takeaway = build_takeaway(
                result.document,
                result.segments,
                transcript.media_id,
                index=search_index,
                model=model,
                observed_at=meta.get("upload_date") or meta.get("publish_date"),
                transcript=transcript,
            )
            result.passes["takeaway"] = "ok"
            reports = result.takeaway.get("reports", {})
            ground = reports.get("groundedness", {})
            verac = reports.get("veracity", {})
            say(
                f"Verified takeaway: {len(result.takeaway['claims'])} atomic claims, "
                f"types {reports.get('types', {})}"
            )
            say(
                f"  Groundedness {ground.get('verdicts', {})} via tiers "
                f"{ground.get('tiers', {})}"
                + ("" if ground.get("nli_available") else "; no NLI model configured")
            )
            say(
                f"  Veracity {verac.get('verdicts', {})}"
                + (
                    ""
                    if verac.get("retrieval_configured")
                    else "; no external retrieval configured, so checkable claims are "
                    "reported unverifiable rather than guessed"
                )
            )
        except Exception as exc:
            result.passes["takeaway"] = f"failed: {exc}"
            say(f"Verified takeaway failed: {exc}")

    # Pass 7: candidate skill, gated behind procedural detection.
    try:
        decision = typed_passes.extract_candidate_skill(
            meta.get("title", ""), result.segments, utterances, model=synthesis_model
        )
        result.skill = decision["skill"]
        result.skill_decision = {
            "extracted": decision["extracted"],
            "procedural_score": decision["procedural_score"],
            "reason": decision["reason"],
        }
        result.passes["skill"] = "ok"
        say(f"Candidate skill: {decision['reason']}")
    except Exception as exc:
        result.passes["skill"] = f"failed: {exc}"
        say(f"Skill extraction failed: {exc}")

    return result
