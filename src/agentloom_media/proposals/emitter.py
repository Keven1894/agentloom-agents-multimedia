"""Emits distilled knowledge into AgentLoom 3-Track files and proposal JSONs."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentloom_media.distillation.anchors import resolve_anchor, segment_start_times


def slugify(text: str) -> str:
    """Convert string to slug."""
    text = re.sub(r"[^\w\s-]", "", str(text or "")).strip().lower()
    return re.sub(r"[-\s]+", "-", text)[:60]


def _as_text(value: Any) -> str:
    """Flatten LLM JSON values that may be str, list, or dict."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_as_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


TIMING_CAVEAT = {
    "word": "timestamps are word-aligned (<100ms)",
    "segment": "timestamps are segment-level, roughly 2-10s wide",
    "cue": "timestamps are caption-cue level; no word or speaker timing",
}


def _provenance_lines(provenance: Optional[Dict[str, Any]]) -> List[str]:
    """Digest header lines describing where the timestamps came from."""
    if not provenance:
        return []

    granularity = provenance.get("timing_granularity", "unknown")
    caveat = TIMING_CAVEAT.get(granularity, "timing accuracy unverified")
    lines = [
        f"- **Transcript Source**: {provenance.get('source', 'unknown')} "
        f"via `{provenance.get('engine', 'unknown')}`"
        + (f" ({provenance['model']})" if provenance.get("model") else ""),
        f"- **Timing Granularity**: {granularity} — {caveat}",
    ]
    normalization = provenance.get("normalization") or {}
    if normalization.get("applied"):
        lines.append(
            f"- **Script Normalization**: OpenCC `{normalization.get('opencc_config')}` "
            "applied (raw text preserved in the canonical transcript)"
        )
    if provenance.get("engine") == "unknown":
        lines.append(
            "- **⚠ Provenance Gap**: this transcript predates provenance tracking; "
            "its engine and timing accuracy are unverified"
        )
    return lines


class AnchorGate:
    """Validates LLM-emitted anchors against times that exist in the transcript.

    An anchor is rendered only if it resolves to a real utterance start. Everything else is
    dropped and counted, so a fabricated citation degrades into an unlinked claim instead of
    a link that points nowhere.
    """

    def __init__(self, segments: List[Dict[str, Any]], video_url: str):
        self.valid_times = segment_start_times(segments or [])
        self.video_url = video_url or ""
        self.stats = {"ok": 0, "unparsable": 0, "out_of_range": 0}
        self.rejected: List[str] = []

    @property
    def available(self) -> bool:
        return bool(self.valid_times and self.video_url)

    def link(self, raw: Any, label: str) -> Optional[str]:
        """Return a Markdown link for a valid anchor, else None."""
        if not self.available or raw in (None, ""):
            return None
        seconds, url, status = resolve_anchor(raw, self.valid_times, self.video_url)
        self.stats[status] = self.stats.get(status, 0) + 1
        if status != "ok" or url is None:
            self.rejected.append(f"{label}: {raw!r} ({status})")
            return None
        from agentloom_media.distillation.anchors import format_timestamp

        return f"[{format_timestamp(seconds)}]({url})"

    def summary(self) -> Dict[str, Any]:
        total = sum(self.stats.values())
        return {
            "anchors_seen": total,
            "anchors_valid": self.stats.get("ok", 0),
            "anchors_rejected_unparsable": self.stats.get("unparsable", 0),
            "anchors_rejected_out_of_range": self.stats.get("out_of_range", 0),
            "rejected_examples": self.rejected[:10],
        }


def emit_distillation_artifacts(
    meta: Dict[str, Any],
    distilled: Dict[str, Any],
    aligned_chapters: List[Dict[str, Any]],
    repo_root: Path,
    model: str | None = None,
    segments: Optional[List[Dict[str, Any]]] = None,
    transcript_provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write outputs to Track 2 (docs/digests/), Track 3 (agents/skills/), and proposals/.

    Args:
        meta: Video metadata.
        distilled: Distilled LLM result.
        aligned_chapters: Aligned transcript blocks.
        repo_root: Root path of the repo.
        model: Distillation model id, recorded for provenance.
        segments: Transcript segments used to validate every emitted anchor.
        transcript_provenance: Source, engine, and timing granularity of the transcript, so
            the digest can state what its timestamps are actually worth.

    Returns:
        Dict of paths to emitted files, plus an `_anchor_report` entry.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(meta.get("title", "untitled-video"))
    base_name = f"{date_str}-{slug}"

    gate = AnchorGate(segments or [], meta.get("url", ""))
    boundary_sources = {
        ch.get("boundary_source") for ch in (aligned_chapters or [])
    }
    boundary_source = (
        "native" if "native" in boundary_sources else "mechanical"
    )

    emitted = {}

    # 1. Track 2 Digest (docs/digests/YYYY-MM-DD-<slug>.md)
    digests_dir = repo_root / "docs" / "digests"
    digests_dir.mkdir(parents=True, exist_ok=True)
    digest_path = digests_dir / f"{base_name}.md"

    md_lines = [
        f"# {meta.get('title')}",
        "",
        f"- **Source Media**: [{meta.get('title')}]({meta.get('url')})",
        f"- **Channel / Author**: {meta.get('channel')}",
        f"- **Duration**: {int(meta.get('duration', 0) // 60)} min ({meta.get('duration', 0)}s)",
        f"- **Date Ingested**: {date_str}",
        *( [f"- **Distillation Model**: {model}"] if model else [] ),
        *_provenance_lines(transcript_provenance),
        f"- **Section Boundaries**: {boundary_source}"
        + (
            ""
            if boundary_source == "native"
            else " (mechanical batching — section titles are LLM-proposed, not author chapters)"
        ),
        "",
        "## Executive Summary",
        "",
        distilled.get("executive_summary", "No executive summary generated.")
        if isinstance(distilled.get("executive_summary"), str)
        else _as_text(distilled.get("executive_summary")),
        "",
        "## Section Breakdown & Key Takeaways",
        "",
    ]

    for idx, ch in enumerate(distilled.get("chapter_summaries", []) or []):
        if not isinstance(ch, dict):
            continue
        title = _as_text(ch.get("chapter_title")) or f"Section {idx + 1}"
        link = gate.link(ch.get("timestamp_anchor"), f"section '{title}'")
        md_lines.append(f"### {title}" + (f" — {link}" if link else ""))
        if not link and ch.get("timestamp_anchor"):
            md_lines.append("_No verifiable timestamp anchor for this section._")
        points = _as_list(ch.get("key_points", []))
        if points:
            for pt in points:
                md_lines.append(f"- {_as_text(pt)}")
        if ch.get("analysis"):
            md_lines.append(f"\n> **Analysis**: {_as_text(ch.get('analysis'))}\n")
        md_lines.append("")

    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    emitted["track2_digest"] = str(digest_path)

    # 2. Track 3 Candidate Skills (agents/skills/domain/candidate/)
    skills = distilled.get("candidate_skills", [])
    if skills:
        skills_dir = repo_root / "agents" / "skills" / "domain" / "candidate"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for i, sk in enumerate(skills):
            if not isinstance(sk, dict):
                continue
            sk_slug = slugify(sk.get("name", f"skill-{i+1}"))
            sk_path = skills_dir / f"{sk_slug}.md"
            sk_lines = [
                f"# skill:candidate:{sk_slug}: {_as_text(sk.get('name'))}",
                "",
                f"**Category**: {_as_text(sk.get('category', 'Distilled Skill'))}  ",
                "**Status**: Candidate (Propose-Review Pending)  ",
                f"**Source Media**: [{meta.get('title')}]({meta.get('url')})  ",
                f"**Date Distilled**: {date_str}  ",
                "",
                "## Purpose",
                "",
                _as_text(sk.get("purpose", sk.get("name"))),
                "",
                "## Preconditions",
                "",
            ]
            for pre in _as_list(sk.get("preconditions", [])):
                sk_lines.append(f"- {_as_text(pre)}")
            sk_lines.append("\n## Steps\n")
            for step in _as_list(sk.get("steps", [])):
                if not isinstance(step, dict):
                    sk_lines.append(f"### {_as_text(step)}")
                    sk_lines.append("")
                    continue
                step_title = _as_text(step.get("title", "Step"))
                sk_lines.append(f"### {step_title}")
                step_link = gate.link(step.get("anchor"), f"skill step '{step_title}'")
                if step_link:
                    sk_lines.append(f"- Anchor: {step_link}")
                if step.get("command"):
                    sk_lines.append(f"```bash\n{_as_text(step.get('command'))}\n```")
                sk_lines.append("")
            sk_lines.append("## Verification\n")
            sk_lines.append(_as_text(sk.get("verification", "Verify step execution manually.")))

            with open(sk_path, "w", encoding="utf-8") as f:
                f.write("\n".join(sk_lines))
            emitted[f"candidate_skill_{i+1}"] = str(sk_path)

    # 3. AgentLoom Proposal JSON (proposals/proposal-<slug>.json)
    proposals_dir = repo_root / "proposals"
    proposals_dir.mkdir(parents=True, exist_ok=True)
    proposal_file = proposals_dir / f"proposal-{base_name}.json"

    proposal_payload = {
        "proposal_id": f"prop:{base_name}",
        "timestamp": datetime.now().isoformat(),
        "source": {
            "type": "multimedia",
            "url": meta.get("url"),
            "title": meta.get("title"),
            "channel": meta.get("channel"),
        },
        "target_track": "Track 3 (Skills Track)",
        "model": model,
        "boundary_source": boundary_source,
        "transcript_provenance": transcript_provenance or {},
        "anchor_validation": gate.summary(),
        "candidate_kg_nodes": distilled.get("candidate_kg_nodes", []),
        "status": "pending_human_review",
    }

    with open(proposal_file, "w", encoding="utf-8") as f:
        json.dump(proposal_payload, f, indent=2, ensure_ascii=False)
    emitted["agentloom_proposal"] = str(proposal_file)
    emitted["_anchor_report"] = gate.summary()

    return emitted
