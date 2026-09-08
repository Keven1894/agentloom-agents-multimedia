"""Emits typed distillation output into AgentLoom 3-Track files and a proposal.

Differs from `emitter.py` in what it can promise. The old emitter received one blob of LLM
JSON; this one receives segments with per-point anchors, terms carrying verified character
spans, and a per-pass status report — so the digest can state which fields are missing because
a pass failed, and every rendered link has been validated against the transcript.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentloom_media.distillation.anchors import format_timestamp
from agentloom_media.proposals.emitter import (
    AnchorGate,
    _provenance_lines,
    slugify,
)

_REASON_LABELS = {
    "quote_not_found": "quote not in transcript",
    "term_not_in_quote": "quote did not contain the term",
    "span_not_resolvable": "span had no resolvable time",
}


def _reason_phrase(reasons: Dict[str, int]) -> str:
    return ", ".join(
        f"{count} {_REASON_LABELS.get(reason, reason)}"
        for reason, count in sorted(reasons.items())
    )


TYPE_LABELS = {
    "vendor_behavior": "platform behaviour",
    "mechanism": "mechanism",
    "quantitative_rule": "rule of thumb",
    "personal_result": "the speaker's own result",
    "recommendation": "recommendation",
}


def _takeaway_section(takeaway: Dict[str, Any], gate: Any) -> List[str]:
    """Render the four sections, keeping groundedness and veracity visibly separate."""
    sections = takeaway.get("sections") or {}
    reports = takeaway.get("reports") or {}
    ground = reports.get("groundedness") or {}
    verac = reports.get("veracity") or {}

    lines = [
        "## AI Takeaway (Candidate — Not Accepted)",
        "",
        "_Two separate verdicts per claim. **Groundedness** is whether the transcript says "
        "this; **veracity** is whether it is true. A claim can be perfectly grounded and "
        "still wrong._",
        "",
    ]

    if not verac.get("retrieval_configured"):
        lines += [
            "> **⚠ No external retrieval is configured**, so no claim was checked against "
            "outside sources. Checkable claims are reported `unverifiable`, which means "
            "*not checked* — it is not evidence for or against them.",
            "",
        ]
    if not ground.get("nli_available"):
        escalated = ground.get("escalated") or 0
        tail = (
            f"the {escalated} claims in the ambiguous band were escalated to an LLM "
            "entailment judge, shown as `llm` below"
            if escalated
            else "claims in the ambiguous band are reported `needs_review` rather than "
            "guessed at"
        )
        lines += [
            "> **No local entailment model is configured.** Cheap n-gram overlap against "
            f"retrieved spans settled the clear cases, and {tail}.",
            "",
        ]

    def render(entry: Dict[str, Any], show_veracity: bool = True) -> List[str]:
        g = entry.get("groundedness") or {}
        v = entry.get("veracity") or {}
        out = [f"- {entry['claim']}"]
        detail = [f"type: {TYPE_LABELS.get(entry['type'], entry['type'])}"]
        grounded = g.get("verdict") == "grounded"
        detail.append(
            f"grounded via {g.get('tier')}"
            if grounded
            else f"groundedness: {g.get('verdict')} ({g.get('tier')})"
        )
        if show_veracity:
            detail.append(f"veracity: {v.get('verdict')}")

        # Only a grounded claim gets a timestamp. The span behind an ungrounded claim is the
        # candidate we just rejected, so rendering it as a citation would repeat the P0
        # mistake: a link that looks authoritative while supporting nothing.
        link = None
        if grounded:
            t0 = (g.get("evidence") or {}).get("t0")
            if t0 is not None:
                link = gate.link(t0, "takeaway claim")

        out.append(f"  - {' · '.join(detail)}" + (f" — {link}" if link else ""))
        if not grounded and g.get("reason"):
            out.append(f"  - {g['reason']}")
        if show_veracity and v.get("reason"):
            out.append(f"  - {v['reason']}")
        if show_veracity and v.get("what_would_settle_it"):
            out.append(f"  - Would be settled by: {v['what_would_settle_it']}")
        for citation in v.get("citations") or []:
            if citation.get("url"):
                out.append(
                    f"  - Source: [{citation.get('title') or citation['url']}]"
                    f"({citation['url']})"
                )
        return out

    blocks = [
        (
            "### 1. What the video claims",
            "what_the_video_claims",
            "_Grounded in the transcript. This section records what was said, not whether "
            "it is correct._",
            False,
        ),
        (
            "### 2. What holds up",
            "what_holds_up",
            "_Supported by an external source, cited below._",
            True,
        ),
        (
            "### 3. What to verify",
            "what_to_verify",
            "_Flagged as contradicted, unsupported, time-sensitive, or unchecked, with what "
            "would settle each._",
            True,
        ),
        (
            "### 4. What is unfalsifiable",
            "what_is_unfalsifiable",
            "_The author's position and their own unaudited results. Presented as their "
            "claim, not as knowledge, and must not enter the knowledge graph as fact._",
            False,
        ),
    ]

    for heading, key, blurb, show_veracity in blocks:
        entries = sections.get(key) or []
        lines += [heading, "", blurb, ""]
        if entries:
            for entry in entries:
                lines += render(entry, show_veracity)
        else:
            lines.append("_Nothing in this category._")
        lines.append("")

    return lines


FAMILY_LABELS = {
    "causal": "Causal",
    "structural": "Structural",
    "procedural": "Procedural",
    "definitional": "Definitional",
    "temporal": "Temporal",
    "evidential": "Evidential",
}


def _graph_section(graph: Any) -> List[str]:
    """Render the typed graph, grouped by edge family, with evidence links."""
    stats = graph.stats or {}
    lines = [
        "## Knowledge Graph (Candidate)",
        "",
        f"_{stats.get('nodes', 0)} normalized surfaces and {stats.get('edges', 0)} typed "
        f"edges, from {stats.get('relations_proposed', 0)} proposed relations. Nodes are "
        "**normalized typed surfaces, not resolved identities** — aggregation is by exact "
        "normalized match and no coreference was performed, so homonyms may be merged._",
        "",
    ]

    by_family: Dict[str, List[Any]] = {}
    for edge in graph.edges:
        by_family.setdefault(edge.family, []).append(edge)

    node_label = {}
    for node in graph.nodes.values():
        node_label[node.id] = (
            node.surface_forms[0] if node.surface_forms else node.normalized_surface
        )

    for family in sorted(by_family):
        lines += [f"### {FAMILY_LABELS.get(family, family.title())}", ""]
        for edge in by_family[family]:
            source = node_label.get(edge.source, edge.source)
            target = node_label.get(edge.target, edge.target)
            anchor = edge.evidence.anchor_url
            stamp = format_timestamp(edge.evidence.t0)
            link = f"[{stamp}]({anchor})" if anchor else stamp
            lines.append(f"- **{source}** → `{edge.predicate}` → **{target}** — {link}")
            if edge.claim:
                lines.append(f"  - {edge.claim}")
            validity = _validity_phrase(edge)
            if validity:
                lines.append(f"  - {validity}")
            if edge.supersedes:
                superseded = node_label.get(edge.supersedes, edge.supersedes)
                lines.append(f"  - Speaker asserts this supersedes: **{superseded}**")
        lines.append("")

    if graph.merge_candidates:
        lines += [
            "### Merge Candidates (Not Merged)",
            "",
            "_Surfaces that look alike in embedding space. Similarity is not evidence of "
            "identity, so both remain in the graph pending review._",
            "",
        ]
        for candidate in graph.merge_candidates[:15]:
            lines.append(
                f"- **{candidate['a_surface']}** ~ **{candidate['b_surface']}** "
                f"(similarity {candidate['similarity']})"
            )
        lines.append("")

    return lines


def _validity_phrase(edge: Any) -> Optional[str]:
    """State the time axes explicitly, including which ones are unknown."""
    parts = []
    if edge.observed_at:
        parts.append(f"observed {edge.observed_at}")
    if edge.valid_from or edge.valid_until:
        parts.append(
            f"asserted valid {edge.valid_from or '?'} → {edge.valid_until or 'open'}"
        )
    elif edge.observed_at:
        # Silence here would read as "timeless"; it is not.
        parts.append("no validity period stated by the speaker")
    return "; ".join(parts).capitalize() if parts else None


SEGMENTATION_CAVEAT = {
    "semantic": "boundaries detected from embedding similarity; titles are LLM-proposed",
    "native": "boundaries are author-published chapters",
    "native+semantic": "author chapters, with long chapters subdivided semantically",
    "unsegmented": "NOT segmented — the whole item is one span",
    "empty": "no transcript content",
}


def emit_typed_distillation(
    meta: Dict[str, Any],
    result: Any,
    transcript: Any,
    repo_root: Path,
    model: Optional[str] = None,
    synthesis_model: Optional[str] = None,
    transcript_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write the Track 2 digest, an optional Track 3 candidate skill, and the proposal.

    Returns emitted paths plus `_anchor_report`.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(meta.get("title", "untitled-video"))
    base_name = f"{date_str}-{slug}"

    segments = list(result.segments)
    gate = AnchorGate(transcript.to_segments(), meta.get("url", ""))
    method = (result.segmentation or {}).get("method", "unknown")

    emitted: Dict[str, Any] = {}

    # ---- Track 2 digest ------------------------------------------------------------
    digests_dir = repo_root / "docs" / "digests"
    digests_dir.mkdir(parents=True, exist_ok=True)
    digest_path = digests_dir / f"{base_name}.md"

    document = result.document or {}
    lines: List[str] = [
        f"# {meta.get('title')}",
        "",
        f"- **Source Media**: [{meta.get('title')}]({meta.get('url')})",
        f"- **Channel / Author**: {meta.get('channel')}",
        f"- **Duration**: {int((meta.get('duration') or 0) // 60)} min "
        f"({meta.get('duration')}s)",
        f"- **Date Ingested**: {date_str}",
        *( [f"- **Distillation Model**: {model}"] if model else [] ),
        *( [f"- **Synthesis Model**: {synthesis_model}"] if synthesis_model else [] ),
        *_provenance_lines(transcript_provenance),
        f"- **Segmentation**: {method} — "
        f"{SEGMENTATION_CAVEAT.get(method, 'method unrecorded')}",
    ]

    if result.failed_passes:
        lines += [
            "",
            "> **⚠ Incomplete distillation.** These passes did not succeed, so the "
            "corresponding sections are missing rather than empty:",
            "",
            *[f"> - `{name}`: {result.passes[name]}" for name in result.failed_passes],
        ]

    lines += ["", "## Executive Summary", ""]
    if document.get("executive_summary"):
        lines.append(document["executive_summary"])
    else:
        lines.append("_Document synthesis did not run; see the pass report above._")

    if document.get("thesis"):
        lines += ["", f"**Thesis**: {document['thesis']}"]
    if document.get("structure"):
        lines += ["", f"**Argument structure**: {document['structure']}"]

    if document.get("takeaways"):
        lines += ["", "## Takeaways", ""]
        lines += [f"- {t}" for t in document["takeaways"]]

    # ---- Segments ------------------------------------------------------------------
    lines += ["", "## Segment Breakdown", ""]
    if method == "unsegmented":
        lines.append(
            "_This item was not segmented (no embedder available), so the breakdown below "
            "is a single span._"
        )
        lines.append("")

    for segment in segments:
        title = segment.title or f"Segment {segment.index + 1}"
        span = f"{format_timestamp(segment.t0)}–{format_timestamp(segment.t1)}"
        link = gate.link(segment.t0, f"segment '{title}'")
        header = f"### {title}"
        lines.append(header)
        lines.append("")
        lines.append(f"{link or span}" + (f"  ·  {span}" if link else ""))
        if segment.boundary_distance is not None:
            lines.append(
                f"[Boundary strength: {segment.boundary_distance:.3f} cosine distance]"
            )
        lines.append("")

        for point in segment.key_points:
            point_link = gate.link(point.get("anchor"), f"point in '{title}'")
            suffix = f" — {point_link}" if point_link else ""
            lines.append(f"- {point['point']}{suffix}")

        if not segment.key_points:
            lines.append("_No key points extracted for this segment._")

        if segment.analysis:
            lines += ["", f"> **Analysis**: {segment.analysis}"]
        lines.append("")

    # ---- Grounded terms ------------------------------------------------------------
    if result.terms:
        stats = result.term_stats or {}
        lines += [
            "## Grounded Terms & Knowledge Points",
            "",
            f"_{stats.get('grounded', 0)} of {stats.get('extracted', 0)} extractions "
            "are shown. Each is backed by a quote located verbatim in the transcript and "
            "containing the term itself; extractions failing either check were discarded"
            + (
                f" ({_reason_phrase(stats.get('dropped_reasons') or {})})."
                if stats.get("dropped")
                else "."
            )
            + "_",
            "",
        ]
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for term in result.terms:
            by_type.setdefault(term["type"], []).append(term)

        for term_type in sorted(by_type):
            lines += [f"**{term_type.replace('_', ' ').title()}**", ""]
            for term in by_type[term_type]:
                anchor = term.get("anchor_url")
                stamp = term.get("timestamp", "")
                located = f"[{stamp}]({anchor})" if anchor else stamp
                description = term.get("description") or ""
                lines.append(f"- **{term['term']}** — {description} {located}")
                lines.append(f"  - Quote: “{term['quote'].strip()}”")
            lines.append("")

    # ---- Verified takeaway ----------------------------------------------------------
    if result.takeaway and result.takeaway.get("claims"):
        lines += _takeaway_section(result.takeaway, gate)

    # ---- Knowledge graph ------------------------------------------------------------
    if result.graph and result.graph.edges:
        lines += _graph_section(result.graph)

    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    emitted["track2_digest"] = str(digest_path)

    # ---- Track 3 candidate skill ---------------------------------------------------
    if result.skill:
        skills_dir = repo_root / "agents" / "skills" / "domain" / "candidate"
        skills_dir.mkdir(parents=True, exist_ok=True)
        skill = result.skill
        skill_slug = slugify(skill.get("name", "candidate-skill"))
        skill_path = skills_dir / f"{skill_slug}.md"

        skill_lines = [
            f"# skill:candidate:{skill_slug}: {skill.get('name')}",
            "",
            f"**Category**: {skill.get('category') or 'Distilled Skill'}  ",
            "**Status**: Candidate (Propose-Review Pending)  ",
            f"**Source Media**: [{meta.get('title')}]({meta.get('url')})  ",
            f"**Date Distilled**: {date_str}  ",
            f"**Procedural Score**: {result.skill_decision.get('procedural_score')} "
            "(gate passed)  ",
            "",
            "## Purpose",
            "",
            skill.get("purpose") or skill.get("name") or "",
            "",
            "## Preconditions",
            "",
        ]
        for pre in skill.get("preconditions") or []:
            skill_lines.append(f"- {pre}")
        if not skill.get("preconditions"):
            skill_lines.append("_None stated._")

        skill_lines += ["", "## Steps", ""]
        for position, step in enumerate(skill.get("steps") or [], start=1):
            if not isinstance(step, dict):
                continue
            step_title = step.get("title") or f"Step {position}"
            step_link = gate.link(step.get("anchor"), f"skill step '{step_title}'")
            skill_lines.append(f"{position}. **{step_title}**")
            if step.get("detail"):
                skill_lines.append(f"   - {step['detail']}")
            if step_link:
                skill_lines.append(f"   - Evidence: {step_link}")

        skill_lines += [
            "",
            "## Verification",
            "",
            skill.get("verification") or "Not stated by the source.",
            "",
        ]

        with open(skill_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skill_lines))
        emitted["candidate_skill"] = str(skill_path)

    # ---- Proposal ------------------------------------------------------------------
    proposals_dir = repo_root / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    proposal_path = proposals_dir / f"proposal-{base_name}.json"

    payload = {
        "proposal_id": f"prop:{base_name}",
        "schema": "medialoom/typed-distillation/v1",
        "timestamp": datetime.now().isoformat(),
        "source": {
            "type": "multimedia",
            "url": meta.get("url"),
            "title": meta.get("title"),
            "channel": meta.get("channel"),
            "media_id": transcript.media_id,
        },
        "models": {"distillation": model, "synthesis": synthesis_model},
        "transcript_provenance": transcript_provenance or {},
        "segmentation": result.segmentation,
        "passes": result.passes,
        "document": result.document,
        "segments": [s.to_dict() for s in segments],
        "terms": result.terms,
        "term_stats": result.term_stats,
        "skill": result.skill,
        "skill_decision": result.skill_decision,
        "knowledge_graph": result.graph.to_dict() if result.graph else None,
        "takeaway": result.takeaway,
        "anchor_validation": gate.summary(),
        "status": "pending_human_review",
    }

    with open(proposal_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    emitted["agentloom_proposal"] = str(proposal_path)
    emitted["_anchor_report"] = gate.summary()

    return emitted
