"""Emits distilled knowledge into AgentLoom 3-Track files and proposal JSONs."""

import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


def slugify(text: str) -> str:
    """Convert string to slug."""
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)[:60]


def emit_distillation_artifacts(
    meta: Dict[str, Any],
    distilled: Dict[str, Any],
    aligned_chapters: List[Dict[str, Any]],
    repo_root: Path,
) -> Dict[str, str]:
    """Write outputs to Track 2 (docs/digests/), Track 3 (agents/skills/), and proposals/.

    Args:
        meta: Video metadata.
        distilled: Distilled LLM result.
        aligned_chapters: Aligned chapters list.
        repo_root: Root path of the repo.

    Returns:
        Dict of paths to emitted files.
    """
    date_str = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(meta.get("title", "untitled-video"))
    base_name = f"{date_str}-{slug}"

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
        "",
        "## Executive Summary",
        "",
        distilled.get("executive_summary", "No executive summary generated."),
        "",
        "## Chapter Breakdown & Key Takeaways",
        "",
    ]

    for ch in distilled.get("chapter_summaries", []):
        md_lines.append(f"### [{ch.get('chapter_title')}]({ch.get('timestamp_anchor')})")
        points = ch.get("key_points", [])
        if isinstance(points, list):
            for pt in points:
                md_lines.append(f"- {pt}")
        else:
            md_lines.append(f"- {points}")
        if ch.get("analysis"):
            md_lines.append(f"\n> **Analysis**: {ch.get('analysis')}\n")
        md_lines.append("")

    with open(digest_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    emitted["track2_digest"] = str(digest_path)

    # 2. Track 3 Candidate Skills (agents/skills/candidate/)
    skills = distilled.get("candidate_skills", [])
    if skills:
        skills_dir = repo_root / "agents" / "skills" / "candidate"
        skills_dir.mkdir(parents=True, exist_ok=True)
        for i, sk in enumerate(skills):
            sk_slug = slugify(sk.get("name", f"skill-{i+1}"))
            sk_path = skills_dir / f"{sk_slug}.md"
            sk_lines = [
                f"# skill:candidate:{sk_slug}: {sk.get('name')}",
                "",
                f"**Category**: {sk.get('category', 'Distilled Skill')}  ",
                "**Status**: Candidate (Propose-Review Pending)  ",
                f"**Source Media**: [{meta.get('title')}]({meta.get('url')})  ",
                f"**Date Distilled**: {date_str}  ",
                "",
                "## Purpose",
                "",
                sk.get("purpose", sk.get("name")),
                "",
                "## Preconditions",
                "",
            ]
            for pre in sk.get("preconditions", []):
                sk_lines.append(f"- {pre}")
            sk_lines.append("\n## Steps\n")
            for step in sk.get("steps", []):
                sk_lines.append(f"### {step.get('title', 'Step')}")
                if step.get("anchor"):
                    sk_lines.append(f"- Anchor: {step.get('anchor')}")
                if step.get("command"):
                    sk_lines.append(f"```bash\n{step.get('command')}\n```")
                sk_lines.append("")
            sk_lines.append("## Verification\n")
            sk_lines.append(sk.get("verification", "Verify step execution manually."))

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
        "candidate_kg_nodes": distilled.get("candidate_kg_nodes", []),
        "status": "pending_human_review",
    }

    with open(proposal_file, "w", encoding="utf-8") as f:
        json.dump(proposal_payload, f, indent=2, ensure_ascii=False)
    emitted["agentloom_proposal"] = str(proposal_file)

    return emitted
