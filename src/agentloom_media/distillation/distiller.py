"""Multi-perspective distillation using LLM to generate Track 2 digests and Track 3 KG/Skills."""

import json
import os
from typing import Any, Dict, List, Optional
import openai


DISTILLATION_SYSTEM_PROMPT = """You are MediaLoom, an expert research analyst and knowledge engineer.
Your task is to distill video transcripts into structured intelligence adhering to the AgentLoom 3-Track framework:
1. Track 2 Digest: Executive summary, chapter-by-chapter deep-dive with timestamp anchors, critical arguments, and key takeaways.
2. Track 3 Knowledge Graph: Key concepts, entities, and causal relationships.
3. Track 3 Candidate Skills: If the video describes actionable, step-by-step technical procedures or workflows, formulate an executable skill.

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
    model: str = "gpt-4o-mini",
) -> Dict[str, Any]:
    """Distill aligned chapter transcripts into structured intelligence.

    Args:
        video_title: Title of the video.
        channel: Author or channel name.
        aligned_chapters: List of aligned chapter dictionaries.
        api_key: Optional OpenAI API key.
        model: Model name for distillation.

    Returns:
        Distilled structured dictionary.
    """
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY is required for distillation.")

    client = openai.OpenAI(api_key=key)

    # Prepare chapter text payload
    content_blocks = []
    for ch in aligned_chapters:
        content_blocks.append(
            f"### [{ch['timestamp_str']} {ch['title']}]({ch['anchor_url']})\n{ch['text']}\n"
        )
    combined_transcript = "\n".join(content_blocks)

    user_prompt = f"""Video Title: {video_title}
Creator/Channel: {channel}

Aligned Transcript Content:
{combined_transcript}

Perform structured distillation according to the system prompt and return valid JSON."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": DISTILLATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw_json = response.choices[0].message.content or "{}"
    return json.loads(raw_json)
