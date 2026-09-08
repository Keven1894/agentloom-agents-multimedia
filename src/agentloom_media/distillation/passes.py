"""Typed distillation passes.

The original pipeline made one call asking for every field at once, so a single malformed
field voided the whole distillation — and it attempted skill extraction on every video, which
is how a marketing commentary produced a "skill" whose steps were prose sentences in `bash`
fences.

Here each pass has its own schema and its own failure domain. A failed pass degrades that one
field; it does not lose the rest. Skill extraction is gated behind a procedural-content
detector rather than always running.

See plan §4.4.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence

from agentloom_media.distillation.anchors import build_timed_transcript, format_timestamp
from agentloom_media.distillation.distiller import (
    _uses_gpt5_sampling_contract,
    resolve_distillation_model,
)
from agentloom_media.distillation.segmentation import Segment

# Escalate only the passes that synthesize across the whole document. Per-segment work stays
# on the cheap tier.
DEFAULT_SYNTHESIS_MODEL = "gpt-5.6-terra"

# Titles must describe content. A title that is only a timestamp is the P0 defect returning.
_TIME_ONLY_TITLE = re.compile(r"^[\d\s:：\-–—~,;]+$")


class PassFailure(RuntimeError):
    """A single pass failed. Callers record it and continue with the other passes."""


def resolve_synthesis_model(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit.strip()
    return (
        os.environ.get("SYNTHESIS_MODEL") or ""
    ).strip() or DEFAULT_SYNTHESIS_MODEL


def _client():
    import openai

    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise PassFailure("OPENAI_API_KEY is not set.")
    return openai.OpenAI(api_key=key)


def _complete_json(
    system: str, user: str, model: str, max_tokens: int = 4096
) -> Dict[str, Any]:
    """One JSON-mode completion, with the sampling contract the model family expects."""
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens,
    }
    if _uses_gpt5_sampling_contract(model):
        kwargs["reasoning_effort"] = (
            os.environ.get("DISTILLATION_REASONING_EFFORT") or "low"
        ).strip()
    else:
        kwargs["temperature"] = 0.2

    response = _client().chat.completions.create(**kwargs)
    content = response.choices[0].message.content or "{}"
    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise PassFailure(f"Model returned invalid JSON: {exc}") from exc


def _segment_brief(segment: Segment, limit: int = 2600) -> str:
    text = segment.text if len(segment.text) <= limit else segment.text[:limit] + "…"
    return (
        f"[{format_timestamp(segment.t0)}–{format_timestamp(segment.t1)}]\n{text}"
    )


# ---- Pass 1: segment titles ---------------------------------------------------------

TITLE_SYSTEM = """You title transcript segments.

Rules:
- A title states the claim or topic actually discussed, in the transcript's own language.
- Never title a segment by its time span. "05:00-10:00" is not a title.
- 6 to 16 characters for Chinese, 3 to 10 words for English. No trailing punctuation.

Return JSON: {"titles": [{"index": <int>, "title": "<string>"}]}"""


def title_segments(
    segments: Sequence[Segment], model: Optional[str] = None
) -> Dict[int, str]:
    """Title each segment. Segments that already carry an author title are left alone."""
    pending = [s for s in segments if not s.title]
    if not pending:
        return {}

    model = resolve_distillation_model(model)
    blocks = "\n\n".join(
        f"### Segment {s.index}\n{_segment_brief(s, 1200)}" for s in pending
    )
    payload = _complete_json(
        TITLE_SYSTEM,
        f"Title these {len(pending)} segments.\n\n{blocks}",
        model,
    )

    titles: Dict[int, str] = {}
    for item in payload.get("titles", []) or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        title = str(item.get("title") or "").strip()
        # Reject time-only titles outright: that is the defect P0 fixed, reappearing.
        if not title or _TIME_ONLY_TITLE.match(title):
            continue
        titles[index] = title
    return titles


# ---- Pass 2: segment summaries ------------------------------------------------------

SEGMENT_SUMMARY_SYSTEM = """You summarize one transcript segment.

The transcript is given as lines prefixed with a real spoken time, e.g. `[04:12] ...`.

Rules:
- 2 to 5 key points, each a concrete claim made in this segment. No filler.
- Every key point carries an "anchor": a single marker copied verbatim from the transcript,
  in MM:SS or HH:MM:SS form, marking where that point is actually said.
- Never write a range ("00:00-05:00"), never join markers, never estimate an unseen time.
  Anchors that do not match a real marker are discarded and the point loses its evidence link.
- "analysis" is one or two sentences on why this matters, or null if there is nothing to add.

Return JSON: {"key_points": [{"point": "<string>", "anchor": "<MM:SS>"}],
              "analysis": "<string or null>"}"""


def summarize_segment(
    segment: Segment,
    utterances: Sequence[Any],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Summarize one segment with per-point anchors drawn from real markers."""
    model = resolve_distillation_model(model)
    ids = set(segment.utterance_ids)
    inside = [u for u in utterances if u.id in ids]
    timed = build_timed_transcript(
        [{"start": u.t0, "end": u.t1, "text": u.text} for u in inside]
    )

    payload = _complete_json(
        SEGMENT_SUMMARY_SYSTEM,
        f"Segment title: {segment.title or '(untitled)'}\n\n{timed}",
        model,
        max_tokens=2048,
    )

    points = []
    for item in payload.get("key_points", []) or []:
        if isinstance(item, dict) and item.get("point"):
            points.append(
                {"point": str(item["point"]).strip(), "anchor": item.get("anchor")}
            )
        elif isinstance(item, str) and item.strip():
            points.append({"point": item.strip(), "anchor": None})

    # Present points in the order they are spoken. Models return them in argument order,
    # which makes a reviewer following along with the video jump backwards.
    points = _sort_by_anchor(points, inside)

    analysis = payload.get("analysis")
    return {
        "key_points": points,
        "analysis": str(analysis).strip() if analysis else None,
    }


def _sort_by_anchor(
    points: List[Dict[str, Any]], utterances: Sequence[Any]
) -> List[Dict[str, Any]]:
    """Order points by anchor time, keeping unanchored points at the end in original order."""
    from agentloom_media.distillation.anchors import parse_anchor_seconds

    def sort_key(indexed):
        position, point = indexed
        seconds = parse_anchor_seconds(point.get("anchor"))
        if seconds is None:
            return (1, 0.0, position)
        return (0, seconds, position)

    return [point for _, point in sorted(enumerate(points), key=sort_key)]


# ---- Pass 3: document synthesis -----------------------------------------------------

SYNTHESIS_SYSTEM = """You synthesize a whole media item from its segment summaries.

Return JSON:
{"executive_summary": "<one paragraph, the item's actual argument>",
 "thesis": "<one sentence: the central claim>",
 "structure": "<one or two sentences on how the argument is built>",
 "takeaways": ["<string>"]}

Write in the transcript's language. State what the speaker claims; do not endorse it."""


def synthesize_document(
    title: str,
    channel: str,
    segments: Sequence[Segment],
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Cross-segment synthesis. This is the one pass worth escalating to a stronger model."""
    model = resolve_synthesis_model(model)
    outline = []
    for segment in segments:
        points = "; ".join(p["point"] for p in segment.key_points[:5])
        outline.append(
            f"- [{format_timestamp(segment.t0)}] {segment.title or '(untitled)'}: {points}"
        )

    payload = _complete_json(
        SYNTHESIS_SYSTEM,
        f"Title: {title}\nChannel: {channel}\n\nSegments:\n" + "\n".join(outline),
        model,
        max_tokens=2048,
    )
    return {
        "executive_summary": str(payload.get("executive_summary") or "").strip(),
        "thesis": str(payload.get("thesis") or "").strip() or None,
        "structure": str(payload.get("structure") or "").strip() or None,
        "takeaways": [
            str(t).strip()
            for t in (payload.get("takeaways") or [])
            if str(t).strip()
        ],
    }


# ---- Pass 4: procedural detection and candidate skills ------------------------------

# Imperative and step-marking language, in both scripts. Presence of a few of these is weak
# evidence; the detector exists to stop skill extraction on pure commentary, not to be clever.
_PROCEDURAL_MARKERS = [
    "第一步", "第二步", "第三步", "步骤", "首先", "然后", "接下来", "最后",
    "设置", "設定", "配置", "打开", "打開", "关掉", "關掉", "点击", "點擊",
    "创建", "創建", "新建", "启用", "啟用", "先", "再",
    "step 1", "step 2", "first,", "then ", "next,", "finally",
    "click", "open ", "create ", "enable ", "configure", "set up", "install",
]

DEFAULT_PROCEDURAL_THRESHOLD = 6


def procedural_score(text: str) -> int:
    """How many distinct step-marking cues appear. Cheap, deterministic, no LLM call."""
    lowered = text.lower()
    return sum(1 for marker in _PROCEDURAL_MARKERS if marker.lower() in lowered)


def is_procedural(text: str, threshold: Optional[int] = None) -> bool:
    limit = (
        threshold
        if threshold is not None
        else int(
            os.environ.get("SKILL_PROCEDURAL_THRESHOLD")
            or DEFAULT_PROCEDURAL_THRESHOLD
        )
    )
    return procedural_score(text) >= limit


SKILL_SYSTEM = """You extract an executable procedure from a transcript, or decline.

Decline by returning {"skill": null} unless the transcript describes a concrete, repeatable
procedure that a practitioner could follow. Commentary, opinion, and analysis are NOT skills.

If it does, return JSON:
{"skill": {"name": "<string>", "category": "<string>", "purpose": "<string>",
           "preconditions": ["<string>"],
           "steps": [{"title": "<string>", "detail": "<string>", "anchor": "<MM:SS>"}],
           "verification": "<how to tell it worked>"}}

Rules:
- "detail" is prose describing the action. Do NOT invent shell commands; this is a UI
  procedure, not a script. Never put prose inside a code fence.
- "anchor" is a single marker copied verbatim from the transcript, in MM:SS form."""


def extract_candidate_skill(
    title: str,
    segments: Sequence[Segment],
    utterances: Sequence[Any],
    model: Optional[str] = None,
    threshold: Optional[int] = None,
) -> Dict[str, Any]:
    """Extract a skill only when the content is actually procedural.

    Returns `{"skill": ..., "procedural_score": int, "extracted": bool, "reason": str}`.
    """
    full_text = " ".join(s.text for s in segments)
    score = procedural_score(full_text)

    if not is_procedural(full_text, threshold):
        return {
            "skill": None,
            "procedural_score": score,
            "extracted": False,
            "reason": (
                f"Procedural score {score} is below the threshold; the content reads as "
                "commentary rather than a repeatable procedure."
            ),
        }

    model = resolve_synthesis_model(model)
    timed = build_timed_transcript(
        [{"start": u.t0, "end": u.t1, "text": u.text} for u in utterances]
    )
    payload = _complete_json(
        SKILL_SYSTEM,
        f"Title: {title}\n\n{timed[:14000]}",
        model,
        max_tokens=3072,
    )

    skill = payload.get("skill")
    if not isinstance(skill, dict) or not skill.get("name"):
        return {
            "skill": None,
            "procedural_score": score,
            "extracted": False,
            "reason": "The model declined to extract a procedure.",
        }

    return {
        "skill": skill,
        "procedural_score": score,
        "extracted": True,
        "reason": f"Procedural score {score} met the threshold.",
    }
