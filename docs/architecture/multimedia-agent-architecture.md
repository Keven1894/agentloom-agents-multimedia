# AgentLoom Multimedia Knowledge Ingestion Agent — Architecture & Design

**Status**: Proposed Architecture & Implementation Blueprint  
**Date**: 2026-09-02  
**Author**: Dr. Boyuan (Keven) Guan (@Keven1894) & Envita (Builder Mode)  
**Parent Framework**: [AgentLoom Framework (v3.0)](https://github.com/Keven1894/AgentLoom)  
**Project Ownership & Asset Classification**:
- **Personal Open-Source Research Software** (Repository: `Keven1894/agentloom-agents-multimedia`)
- **Local Path**: `C:\projects\03_personal-agents\agentloom-agents-multimedia`
- **NOT an FIU Institutional Asset**: Completely decoupled from FIU production databases (`envita_prod`), FIU internal networking (`10.100.118.x`), and FIU enterprise storage.
- **License Alignment**: Code under MIT License, Documentation & Knowledge under CC BY-NC 4.0 (identical to AgentLoom).

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
     instantiates (Enterprise / Private)     instantiates (Personal / Open-Source)
                │                                       │
┌───────────────▼──────────────────────┐ ┌──────────────▼───────────────────────────┐
│ FIU EnviStor / Envita Platform       │ │ agentloom-agents-multimedia (NEW)         │
│ - Private research data curation     │ │ - Autonomous media learning agent         │
│ - MySQL envita_prod, MCP :8765       │ │ - YouTube/Podcast -> KG/Skill             │
│ - FIU institutional asset            │ │ - Open-source repo (Keven1894)            │
│                                      │ │ - Path: C:\projects\03_personal-agents\...│
└──────────────────────────────────────┘ └───────────────────────────────────────────┘
```

### Key Decoupling Principles
- **Clean Namespace**: The repository lives under `Keven1894/agentloom-agents-multimedia` at `C:\projects\03_personal-agents\agentloom-agents-multimedia`.
- **Zero Proprietary Infrastructure**: It relies solely on open APIs (or local models like Whisper / Ollama) and file-based/SQLite AgentLoom memory.
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

### 3.3 3-Track Organization in `agentloom-agents-multimedia`

AgentLoom strictly defines the three tracks of agent knowledge:
- **Track 1**: Guidance Track (`.cursor/`, `.clinerules/`) — Rules and prompt behaviors governing how the agent executes.
- **Track 2**: Domain Knowledge Track (`docs/`) — Long-form architecture, research summaries, and timestamped video digests.
- **Track 3**: Skills Track (`agents/`) — Executable procedures, behaviors, and knowledge graphs that the agent directly runs.

| Track | Directory | Canonical Contents | Role |
|---|---|---|---|
| **Track 1: Guidance Track** | `.cursor/` or `.clinerules/` | Rules for audio chunking thresholds, ASR routing, and distillation prompts | How the builder agent works |
| **Track 2: Knowledge Track** | `docs/sources/` & `docs/digests/` | Cleaned transcript records, timestamped chapter breakdowns, deep-dive technical syntheses | Domain documentation & reference |
| **Track 3: Skills Track** | `agents/` (`agents/skills/`, `agents/behaviors/`, `agents/knowledge-graphs/`) | Executable skills (step-by-step procedures extracted from tutorials), behavior validators, and distilled knowledge graph JSONs | Executable intelligence & skills |

---

## 4. End-to-End Pipeline Architecture

```
 ┌────────────────────────────────────────────────────────┐
 │ Stage 1: Acquisition & Probing                         │
 │ - Input: URL (YouTube, Vimeo, Podcast) or Local Media   │
 │ - Probe: Title, author, duration, chapter markers      │
 └───────────────────────────┬────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            ▼                                 ▼
 ┌──────────────────────┐          ┌──────────────────────┐
 │ Stage 2A: Fast Track │          │ Stage 2B: Heavy Path │
 │ Official / Auto-Subs │          │ Direct Audio Stream  │
 │ (0 cost, 2-5 sec)    │          │ (Format 140 / m4a)   │
 └──────────┬───────────┘          └──────────┬───────────┘
            │ Subtitles Available             │ Subtitles Disabled / Custom Audio
            │                                 ▼
            │                      ┌──────────────────────┐
            │                      │ Audio Preprocessor   │
            │                      │ - Stream chunking    │
            │                      │ - Downsample / 16kHz │
            │                      └──────────┬───────────┘
            │                                 ▼
            │                      ┌──────────────────────┐
            │                      │ ASR Engine           │
            │                      │ Whisper API / Local  │
            │                      └──────────┬───────────┘
            └────────────────┬────────────────┘
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

---

## 6. Repository Scaffold (`agentloom-agents-multimedia`)

The planned open-source repository layout strictly adheres to AgentLoom's **3-Track Architecture**:

```text
agentloom-agents-multimedia/
├── .cursor/ rules/                        # Track 1: Guidance Track
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
│   ├── skills/                            # Distilled executable skills (tutorials -> SOPs)
│   │   └── candidate/
│   ├── behaviors/                         # Agent verification rules & constraints
│   └── knowledge-graphs/                  # Conceptual knowledge graphs & entity networks
│       ├── domain-concepts-graph.json
│       └── media-provenance-graph.json
├── src/agentloom_media/                   # Python Core Package & CLI
│   ├── __init__.py
│   ├── cli.py                            # `agentloom-media ingest <url>` entrypoint
│   ├── acquisition/
│   │   ├── probe.py                      # Probes video metadata, chapters, streams
│   │   ├── fast_transcript.py            # Subtitle extraction (YouTube/Vimeo)
│   │   └── audio_extractor.py            # yt-dlp format 140 audio extraction
│   ├── audio/
│   │   ├── chunker.py                    # Audio chunking for >25MB streams
│   │   └── asr.py                        # Whisper API / Local ASR adapter
│   ├── distillation/
│   │   ├── chapter_aligner.py            # Chapter & timestamp alignment
│   │   ├── concept_extractor.py          # Concepts & entities extraction (for KG)
│   │   └── skill_synthesizer.py          # Procedures & commands extraction (for Skills)
│   └── proposals/
│       └── emitter.py                    # Emits proposals to AgentLoom Dashboard
├── proposals/                             # Pending proposals for human review
├── tests/
├── pyproject.toml                         # Packaging and dependencies
└── README.md
```

---

## 7. Open-Source Implementation Roadmap

### Phase 1: Core Ingestion & Transcribe CLI (MVP)
- Implement `agentloom-media ingest <URL>`:
  - Probe metadata and check for subtitles.
  - If subtitles exist, download and clean with timestamps.
  - If subtitles disabled, extract audio format 140, downsample/chunk, and transcribe via Whisper.
- Output raw timestamped transcript JSON and formatted markdown digest.

### Phase 2: AgentLoom Distillation & Propose-Review Integration
- Build prompt pipelines for the 3 distillation targets:
  1. **Concept & Architecture Memo** (Markdown report).
  2. **Candidate Knowledge Graph Nodes** (`proposals/node_<slug>.json`).
  3. **Candidate Executable Skills** (Markdown skill format with preconditions & verification).
- Integrate with `agentloom.kg.propose_node` so extractions can be reviewed and accepted on the AgentLoom Dashboard (`:8000`).

### Phase 3: Multimodal Vision & Diagram Capture (Advanced)
- For technical presentations with architecture slides, extract keyframes at topic shifts using visual frame difference analysis.
- Run Vision LLM (e.g., Gemini 2.5/3 Pro, GPT-4o) on extracted slide images to extract system architecture diagrams directly into Mermaid/SVG format.

---

## 8. Conclusion

By separating `agentloom-agents-multimedia` as an independent open-source repository under `Keven1894` located at `C:\projects\03_personal-agents\agentloom-agents-multimedia`, we:
1. Provide a general-purpose, reusable tool for the entire AI builder community.
2. Maintain strict asset isolation from FIU institutional projects.
3. Solve the single most time-consuming bottleneck in knowledge transfer: empowering agents to independently "listen", "watch", and "learn" from multimedia sources with full provenance and governance.
