"""Classify how a stretch was delivered: lecture, screen walkthrough, or promo.

The Overview needs this so a client watches demos instead of reading a conceptual paraphrase.
This module does not look anything up and does not read pixels. Metaphorical "seeing"
("让 Meta 看得见所有广告") is lecture.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from agentloom_media.distillation.anchors import (
    build_timed_transcript,
    format_timestamp,
    parse_anchor_seconds,
)
from agentloom_media.distillation.distiller import resolve_distillation_model

SPEECH_MODES = {
    "lecture",
    "walkthrough",
    "reading_screen",
    "promo",
    "mixed",
}
WATCH_MODES = {"walkthrough", "reading_screen", "mixed"}

# Point-at-screen language. Deliberately no bare 看 / 看见 — those fire on "让算法看见广告".
_DEIXIS = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"来看一看",
        r"再来看一看",
        r"再来看看",
        r"再给大家看",
        r"我给大家看",
        r"给大家看",
        r"大家现在看到的是",
        r"现在看到的是",
        r"大家可以看到",
        r"我们可以看到",
        r"大家看到这边",
        r"大家看到",
        r"这边的图示",
        r"这个图示",
        r"那我们看Ads",
        r"我们看Ads",
        r"看Ads\b",
        r"看这个账户",
        r"看这个账号",
        r"看一看这个",
        r"看一看这样",
        r"往下看",
        r"打开你的",
        r"打开后台",
        r"回到刚刚这个",
        r"回到刚刚",
        r"Lifetime到",
        r"let me show",
        r"as you can see",
        r"you can see",
        r"on the screen",
        r"look at this",
        r"look at the",
    )
]


def deixis_cues(text: str) -> List[str]:
    """Surface forms that usually mean the creator is pointing at the frame."""
    if not text:
        return []
    found: List[str] = []
    seen = set()
    for pattern in _DEIXIS:
        match = pattern.search(text)
        if match:
            cue = match.group(0)
            key = cue.lower()
            if key not in seen:
                seen.add(key)
                found.append(cue)
    return found


def watch_from_mode(mode: str, spans: Optional[Sequence[Any]] = None) -> bool:
    if mode in WATCH_MODES:
        return True
    return bool(spans)


def _clip_span(
    t0: Optional[float],
    t1: Optional[float],
    seg_t0: float,
    seg_t1: float,
) -> Optional[Dict[str, float]]:
    if t0 is None:
        return None
    start = max(float(seg_t0), min(float(t0), float(seg_t1)))
    end = float(t1) if t1 is not None else min(float(seg_t1), start + 20.0)
    end = max(start, min(end, float(seg_t1)))
    if end - start < 2.0:
        end = min(float(seg_t1), start + 15.0)
    if end <= start:
        return None
    return {"t0": start, "t1": end}


def _parse_span(raw: Any, seg_t0: float, seg_t1: float) -> Optional[Dict[str, Any]]:
    if not isinstance(raw, dict):
        return None
    t0 = parse_anchor_seconds(raw.get("t0") if raw.get("t0") is not None else raw.get("start"))
    t1 = parse_anchor_seconds(raw.get("t1") if raw.get("t1") is not None else raw.get("end"))
    clipped = _clip_span(t0, t1, seg_t0, seg_t1)
    if clipped is None:
        return None
    reason = str(raw.get("reason") or "").strip() or None
    return {
        "t0": clipped["t0"],
        "t1": clipped["t1"],
        "label": f"{format_timestamp(clipped['t0'])}–{format_timestamp(clipped['t1'])}",
        "reason": reason,
    }


def normalize_speech_mode(
    raw: Any,
    *,
    t0: float,
    t1: float,
    cues: Optional[Sequence[str]] = None,
    source: str = "llm",
) -> Dict[str, Any]:
    """Clamp a model or heuristic payload onto the contract stored on the segment."""
    payload = raw if isinstance(raw, dict) else {}
    mode = str(payload.get("speech_mode") or payload.get("mode") or "").strip().lower()
    if mode not in SPEECH_MODES:
        mode = "walkthrough" if cues else "lecture"

    spans: List[Dict[str, Any]] = []
    for item in payload.get("watch_spans") or []:
        parsed = _parse_span(item, t0, t1)
        if parsed:
            spans.append(parsed)

    reason = str(payload.get("watch_reason") or payload.get("reason") or "").strip()
    watch = bool(payload.get("watch")) if payload.get("watch") is not None else watch_from_mode(
        mode, spans
    )
    if mode in WATCH_MODES:
        watch = True
    if not reason:
        if watch:
            reason = "口播在指屏幕或读数，建议看这一段，不要只读文字摘要。"
        elif mode == "promo":
            reason = "这段主要是宣传或花絮，可以跳过。"
        else:
            reason = "这段是口播原则，看概述即可。"

    return {
        "speech_mode": mode,
        "watch": watch,
        "watch_reason": reason,
        "watch_spans": spans,
        "deixis_cues": list(cues or payload.get("deixis_cues") or []),
        "source": source,
    }


def heuristic_speech_mode(text: str, t0: float, t1: float) -> Dict[str, Any]:
    cues = deixis_cues(text)
    if cues:
        return normalize_speech_mode(
            {
                "speech_mode": "walkthrough",
                "watch": True,
                "watch_reason": "转写里有明确的指屏用语。",
            },
            t0=t0,
            t1=t1,
            cues=cues,
            source="heuristic",
        )
    return normalize_speech_mode(
        {"speech_mode": "lecture", "watch": False},
        t0=t0,
        t1=t1,
        cues=[],
        source="heuristic",
    )


def _view(segment: Any) -> Dict[str, Any]:
    if isinstance(segment, dict):
        return segment
    return {
        "index": segment.index,
        "t0": segment.t0,
        "t1": segment.t1,
        "title": segment.title,
        "text": segment.text,
        "utterance_ids": list(segment.utterance_ids or []),
    }


def _utterances_for(view: Dict[str, Any], utterances: Sequence[Any]) -> List[Any]:
    ids = set(view.get("utterance_ids") or [])
    if ids:
        inside = [u for u in utterances if getattr(u, "id", None) in ids]
        if inside:
            return inside
    t0 = float(view.get("t0") or 0.0)
    t1 = float(view.get("t1") or t0)
    return [
        u
        for u in utterances
        if getattr(u, "t0", 0.0) < t1 + 0.05 and getattr(u, "t1", 0.0) > t0 - 0.05
    ]


def _segment_text(view: Dict[str, Any], inside: Sequence[Any]) -> str:
    if view.get("text"):
        return str(view["text"])
    return "\n".join(str(getattr(u, "text", "")).strip() for u in inside if str(getattr(u, "text", "")).strip())


SPEECH_MODE_SYSTEM = """You label how each transcript segment was delivered on camera.

Modes (pick one):
- lecture: talking through principles, no need to watch the frame
- walkthrough: pointing at a UI, diagram, account, or "look at this"
- reading_screen: reciting numbers or labels that are on screen
- promo: CTA, intro branding, song, course pitch
- mixed: lecture plus a real screen demo in the same segment

Rules:
- Metaphor is lecture. "让 Meta 看得见所有广告" / "你需要看的指标" is not a demo.
- Deixis such as "来看一看这个账户", "大家可以看到", "这边的图示", "往下看", "Lifetime到" is a demo.
- watch=true only when a client would miss something by skipping the video.
- watch_spans: if mixed (or a short demo inside lecture), give t0/t1 copied from the
  [MM:SS] markers in the transcript. Omit watch_spans when the whole segment is the demo.
- Do not invent times. Do not describe what the screen "must have shown".

Return JSON:
{"segments":[{"index":0,"speech_mode":"lecture","watch":false,"watch_reason":"...",
              "watch_spans":[{"t0":"08:19","t1":"08:50","reason":"..."}]}]}"""


def classify_speech_modes(
    segments: Sequence[Any],
    utterances: Sequence[Any],
    model: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """Return {segment_index: normalized speech-mode payload}."""
    views = [_view(segment) for segment in segments]
    prepared: List[Dict[str, Any]] = []
    for view in views:
        inside = _utterances_for(view, utterances)
        text = _segment_text(view, inside)
        prepared.append(
            {
                "view": view,
                "inside": inside,
                "text": text,
                "cues": deixis_cues(text),
            }
        )

    heuristics = {
        int(item["view"]["index"]): heuristic_speech_mode(
            item["text"],
            float(item["view"].get("t0") or 0.0),
            float(item["view"].get("t1") or 0.0),
        )
        for item in prepared
        if item["view"].get("index") is not None
    }

    blocks = []
    for item in prepared:
        view = item["view"]
        timed = build_timed_transcript(
            [
                {"start": u.t0, "end": u.t1, "text": u.text}
                for u in item["inside"]
            ]
        )
        cue_line = (
            f"Deixis hints: {', '.join(item['cues'])}"
            if item["cues"]
            else "Deixis hints: none"
        )
        blocks.append(
            f"### Segment {view.get('index')}\n"
            f"Title: {view.get('title') or '(untitled)'}\n"
            f"{cue_line}\n\n{timed or item['text']}"
        )

    try:
        from agentloom_media.distillation.passes import _complete_json

        payload = _complete_json(
            SPEECH_MODE_SYSTEM,
            "Label these segments.\n\n" + "\n\n".join(blocks),
            resolve_distillation_model(model),
            max_tokens=4096,
        )
    except Exception:
        return heuristics

    by_index: Dict[int, Dict[str, Any]] = dict(heuristics)
    for item in payload.get("segments") or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        match = next((p for p in prepared if p["view"].get("index") == index), None)
        if match is None:
            continue
        by_index[index] = normalize_speech_mode(
            item,
            t0=float(match["view"].get("t0") or 0.0),
            t1=float(match["view"].get("t1") or 0.0),
            cues=match["cues"],
            source="llm",
        )
    return by_index


def apply_speech_mode(segment: Any, info: Dict[str, Any]) -> None:
    """Write the contract onto a Segment object."""
    segment.speech_mode = info.get("speech_mode")
    segment.watch = bool(info.get("watch"))
    segment.watch_reason = info.get("watch_reason")
    segment.watch_spans = list(info.get("watch_spans") or [])


def apply_speech_modes_to_proposal(
    proposal: Dict[str, Any],
    by_index: Dict[int, Dict[str, Any]],
) -> int:
    """Patch proposal['segments'] in place. Returns how many segments were updated."""
    updated = 0
    for raw in proposal.get("segments") or []:
        if not isinstance(raw, dict):
            continue
        try:
            index = int(raw.get("index"))
        except (TypeError, ValueError):
            continue
        info = by_index.get(index)
        if not info:
            continue
        raw["speech_mode"] = info["speech_mode"]
        raw["watch"] = info["watch"]
        raw["watch_reason"] = info["watch_reason"]
        raw["watch_spans"] = info["watch_spans"]
        updated += 1
    passes = dict(proposal.get("passes") or {})
    passes["speech_mode"] = "ok"
    proposal["passes"] = passes
    return updated
