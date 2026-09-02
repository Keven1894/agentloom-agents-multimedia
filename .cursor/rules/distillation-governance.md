# Knowledge Distillation & Provenance Governance

**Category**: Governance Helix  
**Audience**: MediaLoom Agent  

---

## 1. Provenance Anchoring Rule

Every claim, conclusion, or step extracted from a multimedia artifact **MUST** include an exact temporal anchor:
- Format: `[MM:SS Chapter Title](https://youtu.be/<video_id>?t=<seconds>)`
- Do not emit unanchored architectural claims. If an agent or human needs to verify a technical nuance, they must be able to click directly to the precise second in the original video.

---

## 2. Multi-Perspective Distillation Targets

When distilling a technical video, generate outputs across three distinct targets:

1. **Track 2: Chronological Digest (`docs/digests/`)**
   - Summary aligned to native video chapters.
   - Core thesis, technical background, timeline of events.

2. **Track 3: Candidate Knowledge Graph Nodes (`proposals/`)**
   - Entities, causal relationships, system definitions.
   - Formatted as AgentLoom proposal JSON for review on Dashboard (`:8000`).

3. **Track 3: Candidate Executable Skills (`agents/skills/candidate/`)**
   - Extracted if the video demonstrates a hands-on tutorial or developer workflow.
   - Must follow standard AgentLoom Skill format:
     - Frontmatter (name, category, author, source URL)
     - Preconditions
     - Step-by-step commands
     - Verification step
