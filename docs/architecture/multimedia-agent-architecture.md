# AgentLoom Multimedia Knowledge Ingestion Agent — Architecture & Design

**Status**: Proposed Architecture & Implementation Blueprint  
**Date**: 2026-09-02  
**Author**: Dr. Boyuan (Keven) Guan (@Keven1894) & Envita (Builder Mode)  
**Parent Framework**: [AgentLoom Framework (v3.0)](https://github.com/Keven1894/AgentLoom)  
**Repository**: `Keven1894/agentloom-agents-multimedia`  
**License Alignment**: Code under MIT License, Documentation & Knowledge under CC BY-NC 4.0 (identical to AgentLoom).

---

## 1. Executive Summary & Problem Statement

### 1.1 The Human Knowledge Transfer Bottleneck
In modern technical and academic research, a rapidly increasing proportion of high-value knowledge is published not as structured text or formal whitepapers, but as **multimedia artifacts**:
- Architecture talks and tech keynotes (YouTube, Bilibili, Vimeo)
- System walk-throughs, screencasts, and developer tutorials
- Conference presentations and academic workshops (PEARC, IEEE, UCGIS, AGU)
- Deep-dive technical podcasts and developer discussions

Currently, integrating this knowledge into an AI agent's long-term memory requires a **labor-intensive human intermediary**:
1. A human researcher must spend 30–60 minutes watching/listening to the video.
2. The human takes manual notes or writes a partial summary.
3. The human passes the text to the AI agent or manually drafts a Knowledge Graph (KG) entry.

This human-in-the-loop translation is the single largest throughput bottleneck in maintaining up-to-date agent intelligence.

### 1.2 The Solution: Autonomous Multi-Media Knowledge Agent
The goal of **`agentloom-agents-multimedia`** is to build a dedicated, autonomous AgentLoom instance that:
1. Ingests video and audio sources directly (URLs or local files).
2. Extracts high-fidelity audio streams and transcriptions (with dual-path fast/heavy ASR).
3. Preserves exact temporal anchors (`?t=timestamp`) for every extracted claim and diagram.
4. Performs structured, multi-perspective distillation (conceptual, procedural, and chronological).
5. Emits formal AgentLoom proposals (candidate KG nodes, candidate Skills, and Track 2 documentation) through the **Propose → Review → Accept** governance gate.

---

## 2. Positioning in the AgentLoom Ecosystem

AgentLoom operates on a four-tier ecosystem divided by lifecycle phase and scope:

```
┌────────────────────────────────────────────────────────────────────────┐
│ 1. Governance Spec: co-agenticOS (Rules, safety, memory boundaries)    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ inherits
┌───────────────────────────────────▼────────────────────────────────────┐
│ 2. Build Framework: AgentLoom (Authoring time, KG, Tier-A validators)   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ powers
┌───────────────────────────────────▼────────────────────────────────────┐
│ 3. Runtime Library: agentloom-runtime (Layer 0 memory, search, sync)   │
└───────────────┬───────────────────────────────────────┬────────────────┘
                │                                       │
          instantiates agents                     instantiates agents
                │                                       │
┌───────────────▼──────────────────────┐ ┌──────────────▼───────────────────────────┐
│ Other AgentLoom Instances            │ │ agentloom-agents-multimedia                │
│ - Domain-specific capabilities       │ │ - Autonomous media learning agent         │
│ - Agent-native UI and HITL portal    │ │ - YouTube/Podcast -> KG/Skill             │
│ - Independent governed memory        │ │ - Open-source implementation              │
│                                      │ │ - Agent-native UI and HITL portal         │
└──────────────────────────────────────┘ └───────────────────────────────────────────┘
```

### Implementation Principles
- **Clean Namespace**: The repository lives under `Keven1894/agentloom-agents-multimedia`.
- **Portable Infrastructure**: It supports open APIs, local models such as Whisper/Ollama, and file-based or SQLite AgentLoom memory.
- **Framework Feedback**: Enhancements made to multi-modal parsing and knowledge distillation flow back into the core `AgentLoom` framework as reusable patterns.

---

## 3. Dual-Helix & 3-Track Architecture

Like all governed AgentLoom instances, `agentloom-agents-multimedia` implements the dual-helix pattern:

### 3.1 The Learning Helix (Knowledge Extraction Loop)
1. **Raw Signal Ingestion**: Video/audio URL or media file.
2. **Acoustic / Textual Processing**: Dual-path transcription with speaker separation and timestamp preservation.
3. **Semantic Distillation**: Extraction of key concepts, technical definitions, execution recipes, and conceptual relations.
4. **Vector & Graph Weaving**: Chunking and embedding generation for semantic search; candidate RDF/JSON graph node generation.

### 3.2 The Governance Helix (Safety & Verification Loop)
1. **Source Provenance Guarantee**: Every extracted entity or skill must link back to its verifiable media source and timestamp offset (`source_url#t=...`).
2. **Hallucination & Speculation Filtering**: Tier-A validators verify that claims made in the summary are directly supported by the transcript text.
3. **Propose-Review Protocol**: Extracted knowledge never silently overwrites canonical memory. It enters `proposals/` and must be accepted via the AgentLoom Dashboard (`:8000`).
4. **License & Redistribution Rules**: Adheres to fair-use and attribution constraints; stores distilled insights rather than redistributing raw copyrighted video streams.

### 3.3 3-Track Organization & The Two-Role Graph Separation (Builder vs. Domain)

In accordance with AgentLoom v3 core architecture, knowledge is strictly split into two decoupled roles, each carrying its own 3-track components:

1. **Role 1 — `role-builder` (The MediaLoom Ingestion Engine / "How I Learn")**:
   - Meta-knowledge driving the agent: how to probe media, audio stream formats, 16kHz downsampling heuristics, Whisper ASR budget thresholds, chapter alignment, and timestamp hyperlink anchoring.
   - Low-frequency evolution: updated only when media handling tools, ASR APIs, or distillation prompts are upgraded.
2. **Role 2 — `role-domain` (Harvested Multimedia Knowledge / "What I Learned")**:
   - The substantive concepts, causal theories, historical analogies, and tutorial SOP skills extracted from ingested media.
   - High-frequency evolution: grows dynamically every time a video or podcast is processed. Governed strictly via the **Propose → Review → Accept** pipeline.

| Role | Sub-Track | Path | Canonical Contents |
|---|---|---|---|
| **Builder (Engine)** | Knowledge | `agents/knowledge-graphs/builder-knowledge-graph.json` | Engine architecture, dual-path ASR strategy, 24MB size boundaries, chapter scaffolding |
| **Builder (Engine)** | Skills | `agents/skills/builder/` & `builder-skills-graph.json` | Executable engine skills: `probe-media-stream`, `audio-chunking-compression`, `whisper-asr-transcription`, `chapter-alignment-anchoring` |
| **Builder (Engine)** | Behaviors | `agents/behaviors/builder/` & `builder-behaviors-graph.json` | Operational rules: `audio-budget-constraint`, `timestamp-anchoring-rule` |
| **Domain (Harvested)** | Knowledge | `agents/knowledge-graphs/domain-knowledge-graph.json` | Harvested concepts, entity networks, and causal theories (e.g. AI Bubble, Netscape IPO analogy) |
| **Domain (Harvested)** | Skills | `agents/skills/domain/` & `domain-skills-graph.json` | Harvested executable procedures from tutorials (`candidate/` for proposals, `accepted/` post-review) |
| **Domain (Harvested)** | Behaviors | `agents/behaviors/domain/` & `domain-behaviors-graph.json` | Domain policies and constraints derived from digested content |
| **Master Index** | Master | `agents/knowledge-graphs/master-graph.json` | Central registry connecting `role-builder` and `role-domain` roots for dashboard inspection |

---

## 4. End-to-End Pipeline Architecture

```
 ┌────────────────────────────────────────────────────────┐
 │ Stage 1: Acquisition & Probing                         │
 │ - Input: URL (YouTube, Vimeo, Podcast) or Local Media   │
 │ - Probe: Title, author, duration, chapter markers      │
 └───────────────────────────┬────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
 ┌──────────────┐   ┌──────────────┐   ┌──────────────────────┐
 │ 2A Fast Track│   │ 2B Heavy Path│   │ 2C Slow Track        │
 │ Official /   │   │ yt-dlp audio │   │ Browser playback +   │
 │ auto captions│   │ format 140   │   │ in-page audio capture│
 │ 0 cost, 2-5s │   │ seconds–mins │   │ wall-clock ≈ T/rate  │
 └──────┬───────┘   └──────┬───────┘   └──────────┬───────────┘
        │ captions OK      │ extract OK           │ 2A and 2B fail
        │                  ▼                      │ (cipher, login,
        │           ┌──────────────┐              │  blob/MSE only)
        │           │ Preprocess   │              ▼
        │           │ 16kHz / 24MB │       ┌──────────────┐
        │           └──────┬───────┘       │ Play in Chromium│
        │                  │               │ playbackRate    │
        │                  │               │ 1.5x–2.0x       │
        │                  │               │ captureStream() │
        │                  │               └──────┬─────────┘
        │                  │                      │ rescale t *= rate
        │                  ▼                      ▼
        │           ┌──────────────────────────────────────┐
        │           │ ASR Engine (Whisper API / Local)     │
        │           └──────────────────┬───────────────────┘
        └──────────────────────────────┘
                                       │
 ┌───────────────────────────▼────────────────────────────┐
 │ Stage 3: Structural Alignment & Chunking               │
 │ - Align transcript text with native chapter markers    │
 │ - Attach hyperlinked timestamp anchors (?t=seconds)    │
 │ - Punctuation & sentence boundary correction           │
 └───────────────────────────┬────────────────────────────┘
                             │
 ┌───────────────────────────▼────────────────────────────┐
 │ Stage 4: Multi-Perspective LLM Distillation            │
 │ ├─ Perspective 1: Chronological Chapter Digest         │
 │ ├─ Perspective 2: Conceptual Graph (Entities/Relations)│
 │ ├─ Perspective 3: Procedural Recipes (Executable SOPs) │
 │ └─ Perspective 4: Caveats, Risks & Counter-Arguments   │
 └───────────────────────────┬────────────────────────────┘
                             │
 ┌───────────────────────────▼────────────────────────────┐
 │ Stage 5: AgentLoom Governance & Ingestion Gate         │
 │ - Run Tier-A verification (source citations, integrity)│
 │ - Emit proposal via `agentloom.kg.propose_node`        │
 │ - Review & Accept on Dashboard (:8000)                 │
 └────────────────────────────────────────────────────────┘
```

---

## 5. Real-World Case Study: Empirical Test with `R_PMTlFn0TQ`

To validate this architecture against real-world production conditions, we ran live probes against the user-provided sample video:
- **URL**: `https://www.youtube.com/watch?v=R_PMTlFn0TQ`
- **Title**: `被各平台禁止討論！三大證據令人毛骨悚然：2026像極了千禧年災難前的1998！2027大崩盤真的要來了嗎？能让你保命的一期！深度推演AI泡沫破裂時間線 [She's 小烏]`
- **Duration**: 2,118 seconds (~35.3 minutes)

### 5.1 Critical Finding 1: The "Subtitle Disabled" Reality Check
When attempting Stage 2A (Fast Track via `youtube-transcript-api`), the request failed immediately:
```text
youtube_transcript_api._errors.TranscriptsDisabled:
Subtitles are disabled for this video
```
**Architectural Takeaway**: Any tool relying solely on text-caption extraction will fail on millions of creator videos where subtitles are disabled or restricted. **A dedicated audio-stream extraction and ASR engine is mandatory, not an optional fallback.**

### 5.2 Critical Finding 2: Audio Stream Sizing & The 25MB ASR Boundary
Using `yt-dlp` to probe native audio streams without downloading the full high-resolution video:
- **Format Selected**: Format 140 (native AAC / m4a container, ~128 kbps).
- **Extracted Audio Size**: `34,284,170` bytes (~32.7 MB).
- **ASR Constraint**: Standard cloud ASR endpoints (e.g., OpenAI Whisper API) enforce a strict **25 MB file limit**.
- **Architectural Takeaway**:
  1. For short clips (< 20 min), direct transmission works.
  2. For medium/long videos (30+ min), the pipeline must either:
     - Perform **lossless audio chunking** into ~15-minute segments (e.g., splitting at silence or chapter boundaries).
     - Or **downsample** to 16kHz mono 32kbps MP3/OGG (reducing 34MB to ~8MB, well within the 25MB boundary).

### 5.3 Critical Finding 3: Native Chapter Markers as Structural Scaffolding
The video metadata already contains rich temporal anchors:
- `00:00` 科技奇點 VS AI泡沫忌日
- `01:54` 崩盤先兆？
- `08:12` 千禧年網景奇蹟 (Netscape 1995–1998)
- `18:46` 一地雞毛 (2000 Dot-com crash)
- `25:33` 歷史重演？ (2024–2026 AI infrastructure capex comparison)
- `30:40` 「這次不一樣」 (Arguments for/against AI monetization gap)
- `34:08` 適者生存 (Survival and hedging strategies)

**Architectural Takeaway**: When chapters exist, the agent aligns the transcribed sentences directly into these chapter buckets. The resulting markdown report provides instant navigation:
```markdown
### [08:12 千禧年網景奇蹟](https://www.youtube.com/watch?v=R_PMTlFn0TQ&t=492s)
- **Historical Analogy**: 1995 Netscape IPO sparked the commercial web boom...
```

### 5.4 Stage 2C Slow Track: Browser Playback Capture (Authorized Personal Study Bridge)

Fast Track and Heavy Path both assume the agent can fetch a machine-readable caption file or a downloadable audio object via public protocols. That assumption fails on modern educational platforms, paid developer tutorials, and specialized learning apps where:
- The platform employs Media Source Extensions (MSE), Blob URLs, rotating tokens, or app-specific web players specifically designed to prevent unauthorized scraping and video piracy.
- Universal command-line downloaders like `yt-dlp` cannot extract the audio streams.
- However, the user **has legitimately purchased the tutorial/course and possesses full authorization to watch, study, and take personal notes**.

**Slow Track acts as an Authorized Personal Study Bridge (合法购买教程与个人学习知识桥梁)**. It does not attempt to bypass paywalls or circumvent encryption (which is neither feasible nor intended). Instead, it operates entirely within the user's authorized browser session:
- As the user plays their purchased course, the agent captures the **decoded in-page audio element** directly (`HTMLMediaElement.captureStream()`), optionally accelerated at 1.5×–2.0× `playbackRate`.
- It transcribes and distills the lesson into structured study notes, timestamped chapter outlines, and personal knowledge graphs.
- This empowers users who legitimately invest in educational content to directly connect their purchased knowledge into their personal AgentLoom brain for note-taking, revision, and agentic skills synthesis.

Preferred capture stack (isolated, no system-wide loopback):
1. Playwright / Chromium opens the authorized course URL (with user session/cookies) and waits for `HTMLMediaElement`.
2. Set `video.playbackRate` (default **1.5**, aggressive **2.0**; never above 2.0 for Chinese speech).
3. `video.captureStream()` → take the audio track → `MediaRecorder` (`audio/webm;codecs=opus`).
4. After `ended`, downsample to 16 kHz mono and send to ASR.
5. **Invariant**: rescale every ASR timestamp back onto the original timeline: `t_original = t_asr × playbackRate`. Without this step, chapter alignment and `?t=` evidence links are wrong.

Fallbacks only if `captureStream()` is blocked:
- Chromium tab-audio capture (extension / CDP).
- OS loopback (WASAPI on Windows) as last resort — it mixes other desktop audio and is not the default.

**Why speed-up is worth it**: Slow Track wall-clock is bounded by play time. A 35-minute video at 2.0× finishes capture in ~18 minutes, and cloud Whisper bills the **uploaded** duration, so ASR cost also halves. Pitch-preserving `playbackRate` is required; do not use crude resampling that raises pitch.

**Routing rule**: Slow Track is the fallback bridge, activated when Fast Track and Heavy Path cannot extract streams, bridging legitimately accessible course lessons into personal agent memory.

---

## 6. The Agent-Native UI & HITL Review Portal: The AgentLoom HITL Paradigm

A fundamental architectural tenet of the AgentLoom ecosystem is that **AgentLoom is inherently a Human-in-the-Loop (HITL) Agentic-AI framework**. In AgentLoom, autonomous LLM processing never operates as an unmonitored black box. Knowledge and executable procedures cannot bypass human evaluation to overwrite canonical memory.

To support this governance principle at the micro-agent level, **every AgentLoom agent ships with its own self-contained, native web UI**. Rather than relying solely on monolithic centralized control planes, each agent is an independently deployable micro-harness equipped with a built-in interactive portal (`agentloom-media ui`).

```
 ┌────────────────────────────────────────────────────────────────────────┐
 │           Agent-Native Web UI Portal (`agentloom-media ui`)            │
 ├──────────────────┬──────────────────┬─────────────────┬────────────────┤
 │  1. Agent        │  2. Data &       │  3. Interactive │  4. Human      │
 │     Capabilities │     Digest       │     Knowledge   │     Review     │
 │     Showcase     │     Explorer     │     Graph       │     Portal     │
 ├──────────────────┼──────────────────┼─────────────────┼────────────────┤
 │ • Profile & Role │ • Media History  │ • Master Graph  │ • Proposal     │
 │ • Pipeline Stats │ • Clickable      │ • Builder vs.   │   Queue        │
 │ • Active Models  │   Transcript     │   Domain views  │ • Evidence     │
 │ • Toolset &      │ • Track 2 Digest │ • Sub-track     │   Provenance   │
 │   Heuristics     │   Viewer         │   Filters       │ • Approve /    │
 │                  │                  │                 │   Reject Gate  │
 └──────────────────┴──────────────────┴─────────────────┴────────────────┘
```

### 6.1 Pillar 1: Self-Profile & Capabilities Showcase
The agent's built-in UI introduces itself, explaining its role, operational envelope, and available toolsets:
- **Identity & Status**: Displays agent name (`MediaLoom`), role definition, framework version, and environment readiness (e.g., OpenAI API Key status, `ffmpeg` binary detection, cache disk usage).
- **Pipeline Architecture & Capabilities**: Interactive visual walkthrough of the three-path ingestion pipeline (Fast Track captions, Heavy Path stream extraction, Slow Track browser capture) and distillation capabilities.
- **Active Skills & Behaviors Catalog**: Live directory of operational `builder` skills (probing, chunking, transcribing, chapter anchoring) and runtime constraints.

### 6.2 Pillar 2: Data & Digest Explorer
The UI functions as a comprehensive multimedia library and distilled knowledge explorer:
- **Ingestion History**: Overview of all processed video and audio streams, including titles, durations, channel provenance, audio format specs, and processing timestamps.
- **Clickable Interactive Transcripts**: Transcripts rendered with synchronized time markers, allowing reviewers to click any sentence to seek directly to the source media at that exact second.
- **Track 2 Digest Viewer**: Rich Markdown rendering of structured chapter digests, executive takeaways, and thematic deep dives.

### 6.3 Pillar 3: Interactive Dual-Role Knowledge Graph
Built on topology visualization (e.g., Cytoscape.js / D3), this view renders the agent's multi-graph structure:
- **Dual-Role Navigation**: Switch seamlessly between **Builder Graph** (the agent's operational meta-knowledge: how it chunks audio and preserves budget) and **Domain Graph** (the substantive knowledge harvested from digested content).
- **3-Track Filtering**: Toggle between `Knowledge` (concepts and entities), `Skills` (executable procedures), and `Behaviors` (governing rules).
- **Node Inspector**: Click any node to inspect its category, tags, markdown definition, and outbound relations.

### 6.4 Pillar 4: Human Review Portal (The Core HITL Gate — HIGHLIGHT)
**This is the central anchor of AgentLoom's governance helix.** The agent never automatically mutates canonical knowledge graphs or registers new operational skills without explicit human approval.

1. **Pending Proposal Queue**:
   - Lists all candidate proposals generated during the distillation phase (`proposals/proposal-*.json`).
   - Categorized into **Candidate Knowledge Nodes**, **Candidate Skills (SOPs)**, and **Candidate Behaviors**.
2. **Evidence & Provenance Auditing**:
   - Each proposed item displays the exact source transcript excerpt and a clickable timestamp hyperlink (e.g., `https://youtu.be/...&t=492s`).
   - The human reviewer can verify in seconds whether the distilled conclusion is factual or an LLM hallucination.
3. **Decisive HITL Action Gates**:
   - **`Approve`**: Automatically promotes the candidate. A candidate skill is moved from `agents/skills/domain/candidate/` to `agents/skills/domain/accepted/`, and candidate KG nodes are merged into `agents/knowledge-graphs/domain-knowledge-graph.json`.
   - **`Edit & Approve`**: Allows the human reviewer to refine descriptions, adjust step commands, or fix terminology before merging.
   - **`Reject`**: Discards speculative or redundant proposals, logging the rejection rationale to prevent the agent from re-proposing identical invalid concepts.
4. **Immutable Audit Trail**:
   - Every acceptance or rejection is recorded with timestamp, reviewer ID, and diff summary, satisfying enterprise and research data governance standards.

---

## 7. Repository Scaffold (`agentloom-agents-multimedia`)

The planned open-source repository layout strictly adheres to AgentLoom's **3-Track Architecture & Built-in UI Portal**:

```text
agentloom-agents-multimedia/
├── .cursor/ rules/                        # Track 1: Guidance Track
│   ├── core/identity.md                  # Agent identity & mode definition
│   ├── audio-processing-budget.md        # Size limits, chunking, ASR routing
│   └── distillation-governance.md        # Timestamp attribution requirements
├── docs/                                  # Track 2: Knowledge Track
│   ├── architecture/
│   │   └── multimedia-agent-architecture.md
│   ├── guides/
│   │   └── START_HERE.md
│   └── digests/                           # Timestamped video digests & research memos
│       └── YYYY-MM-DD-<slug>.md
├── agents/                                # Track 3: Skills Track (Executable Knowledge)
│   ├── skills/
│   │   ├── builder/                      # Engine skills (probe, chunk, transcribe, align)
│   │   └── domain/                       # Harvested skills from tutorials
│   │       ├── candidate/                # Ingested candidate SOPs pending review
│   │       └── accepted/                 # Reviewed & approved SOP skills
│   ├── behaviors/
│   │   ├── builder/                      # Engine operational rules & constraints
│   │   └── domain/                       # Domain behavioral rules
│   └── knowledge-graphs/
│       ├── master-graph.json             # Master graph registry
│       ├── builder-knowledge-graph.json  # Engine architecture graph
│       ├── builder-skills-graph.json     # Engine skills graph
│       ├── builder-behaviors-graph.json  # Engine behaviors graph
│       ├── domain-knowledge-graph.json   # Harvested concepts & entities graph
│       ├── domain-skills-graph.json      # Harvested skills graph
│       └── domain-behaviors-graph.json   # Domain behaviors graph
├── src/agentloom_media/                   # Python Core Package & CLI
│   ├── __init__.py
│   ├── cli.py                            # `agentloom-media ingest` & `agentloom-media ui`
│   ├── ui/                               # Agent-Native Built-in UI & HITL Portal
│   │   ├── server.py                     # FastAPI backend (Stats, KGs, Proposals, Review API)
│   │   └── static/                       # Lightweight reactive frontend (Tailwind/Alpine)
│   │       ├── index.html                # Single-page UI with 4 core pillars
│   │       └── app.js                    # Dynamic graph visualizer & proposal review logic
│   ├── acquisition/
│   │   ├── probe.py                      # Probes video metadata, chapters, streams
│   │   ├── fast_transcript.py            # Subtitle extraction (YouTube/Vimeo)
│   │   └── audio_extractor.py            # yt-dlp format 140 audio extraction
│   ├── audio/
│   │   ├── chunker.py                    # Audio chunking for >25MB streams
│   │   └── asr.py                        # Whisper API / Local ASR adapter
│   ├── distillation/
│   │   ├── chapter_aligner.py            # Chapter & timestamp alignment
│   │   └── distiller.py                  # Multi-perspective LLM extraction
│   └── proposals/
│       └── emitter.py                    # Emits proposals for human review
├── proposals/                             # Pending proposals for human review
├── tests/
├── pyproject.toml                         # Packaging and dependencies
└── README.md
```

---

## 8. Open-Source Implementation Roadmap

### Phase 1: Core Ingestion & Transcribe CLI (MVP) - [Completed]
- Implemented `agentloom-media ingest <URL>`:
  - Probe metadata and check for subtitles.
  - If subtitles exist, download and clean with timestamps.
  - If subtitles disabled, extract audio format 140, downsample/chunk, and transcribe via Whisper.
- Output raw timestamped transcript JSON and formatted markdown digest.

### Phase 2: AgentLoom Distillation & Propose-Review Pipeline - [Completed]
- Built prompt pipelines for the 3 distillation targets:
  1. **Concept & Architecture Memo** (Markdown report).
  2. **Candidate Knowledge Graph Nodes** (`proposals/proposal-<slug>.json`).
  3. **Candidate Executable Skills** (Markdown skill format with preconditions & verification in `agents/skills/domain/candidate/`).

### Phase 3: Agent-Native UI & HITL Review Portal - [In Progress]
- Implement built-in FastAPI web server (`agentloom-media ui`):
  - Agent Profile & Capabilities dashboard.
  - Interactive transcript & digest explorer.
  - Multi-graph visualization for `builder` and `domain` roles.
  - Interactive **Human Review Portal** with one-click `Approve` / `Reject` / `Edit` actions to merge candidate knowledge and skills into canonical storage.

### Phase 4: Multimodal Vision & Diagram Capture (Advanced)
- For technical presentations with architecture slides, extract keyframes at topic shifts using visual frame difference analysis.
- Run Vision LLM (e.g., Gemini 2.5/3 Pro, GPT-4o) on extracted slide images to extract system architecture diagrams directly into Mermaid/SVG format.

---

## 9. Conclusion

By defining that **every AgentLoom agent ships with its own native UI and HITL portal**, we solidify AgentLoom as a premier human-in-the-loop framework:
1. **Explainable & Autonomous**: The agent communicates what it is, what tools it possesses, and what it has processed.
2. **True Human-in-the-Loop Governance**: Knowledge never mutates without human verification, with direct timestamp links grounding every claim back to empirical video/audio evidence.
3. **Zero Configuration**: A developer or researcher clones the agent and immediately has both an autonomous CLI pipeline and an intuitive human review dashboard out-of-the-box.
