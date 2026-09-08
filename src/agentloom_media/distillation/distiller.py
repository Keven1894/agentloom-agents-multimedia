"""Multi-perspective distillation using LLM to generate Track 2 digests and Track 3 KG/Skills."""

import json
import os
from typing import Any, Dict, List, Optional
import openai

DEFAULT_DISTILLATION_MODEL = "gpt-5.6-luna"
DEFAULT_REASONING_EFFORT = "low"


def resolve_distillation_model(explicit: Optional[str] = None) -> str:
    """Resolve the distillation model: CLI/UI override, then env, then Luna."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    env = (os.environ.get("DISTILLATION_MODEL") or "").strip()
    return env or DEFAULT_DISTILLATION_MODEL


def _uses_gpt5_sampling_contract(model: str) -> bool:
    name = model.lower()
    return name.startswith("gpt-5") or name.startswith("o1") or name.startswith("o3")


DISTILLATION_SYSTEM_PROMPT = """You are MediaLoom, an expert research analyst and knowledge engineer.
Your task is to distill video transcripts into structured intelligence adhering to the AgentLoom 3-Track framework:
1. Track 2 Digest: Executive summary, section-by-section deep-dive with timestamp anchors, critical arguments, and key takeaways.
2. Track 3 Knowledge Graph: Key concepts, entities, and causal relationships.
3. Track 3 Candidate Skills: If the video describes actionable, step-by-step technical procedures or workflows, formulate an executable skill.

TIMESTAMP RULES (strictly enforced by a downstream validator):
- The transcript is given as lines prefixed with a real spoken time, e.g. `[04:12] ...`.
- Every 'timestamp_anchor' and every step 'anchor' MUST be a single marker copied verbatim
  from the transcript, in `MM:SS` or `HH:MM:SS` form. Copy the marker of the line where the
  supporting speech actually occurs.
- Never write a range (`00:00-05:00`), never join several markers (`04:12; 09:30`), and never
  estimate a time that has no marker. Anchors that do not match a real marker are discarded,
  and the claim then appears with no evidence link.
- Section titles must describe the actual topic discussed. Never label a section by its time
  span alone.

Output MUST be valid JSON with keys:
- 'executive_summary': string
- 'chapter_summaries': list of objects { 'chapter_title', 'timestamp_anchor', 'key_points', 'analysis' }
- 'candidate_kg_nodes': list of objects { 'id', 'name', 'type', 'description', 'relationships': [{ 'target', 'relation' }] }
- 'candidate_skills': list of objects { 'name', 'category', 'preconditions', 'steps': [{ 'title', 'command', 'anchor' }], 'verification' }
"""


def distill_aligned_chapters(
    video_title: str,
    channel: str,
    aligned_chapters: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """Distill aligned chapter transcripts into structured intelligence.

    Args:
        video_title: Title of the video.
        channel: Author or channel name.
        aligned_chapters: List of aligned chapter dictionaries.
        api_key: Optional OpenAI API key.
        model: Model name for distillation. Defaults to DISTILLATION_MODEL or gpt-5.6-luna.

    Returns:
        Distilled structured dictionary.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required for distillation.")

    model = resolve_distillation_model(model)
    client = openai.OpenAI(api_key=key)

    # Prepare the transcript payload. Mechanical blocks have no title, so they are labelled
    # as batches rather than presented to the model as topic sections.
    content_blocks = []
    for ch in aligned_chapters:
        if ch.get("title"):
            header = f"### Author chapter: {ch['title']} (from {ch['timestamp_str']})"
        else:
            header = (
                f"### Transcript batch from {ch['timestamp_str']} "
                "(mechanical split, NOT a topic boundary)"
            )
        body = ch.get("timed_text") or ch.get("text", "")
        content_blocks.append(f"{header}\n{body}\n")
    combined_transcript = "\n".join(content_blocks)

    user_prompt = f"""Video Title: {video_title}
Creator/Channel: {channel}

Aligned Transcript Content:
{combined_transcript}

Perform structured distillation according to the system prompt and return valid JSON."""

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
    }
    # GPT-5.x rejects custom temperature; use a cheap reasoning tier instead.
    if _uses_gpt5_sampling_contract(model):
        create_kwargs["reasoning_effort"] = (
            os.environ.get("DISTILLATION_REASONING_EFFORT") or DEFAULT_REASONING_EFFORT
        ).strip()
    else:
        create_kwargs["temperature"] = 0.2

    response = client.chat.completions.create(**create_kwargs)

    raw_json = response.choices[0].message.content or "{}"
    return json.loads(raw_json)
