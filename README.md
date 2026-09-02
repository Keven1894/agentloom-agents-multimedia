# agentloom-agents-multimedia

**Autonomous multimedia knowledge ingestion, transcription, and multi-perspective distillation agent** for the [AgentLoom](https://github.com/Keven1894/AgentLoom) governance framework.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: AgentLoom](https://img.shields.io/badge/Framework-AgentLoom%20v3.0-blue)](https://github.com/Keven1894/AgentLoom)

---

## Overview

In modern research and engineering, vast amounts of critical knowledge are presented as video tutorials, architecture walk-throughs, technical conference presentations (PEARC, IEEE, AGU), and developer discussions. 

Historically, ingesting this knowledge required a human to watch the media, draft notes, and pass them to an AI agent. **`agentloom-agents-multimedia`** automates this pipeline end-to-end:
1. **Acquires** media streams (YouTube, Vimeo, local audio/video) with dual-path transcription (Fast Subtitle Track vs. Heavy ASR Stream).
2. **Aligns** speech with native video chapters and hyperlinked timestamp anchors (`?t=timestamp`).
3. **Distills** content across multiple analytical perspectives:
   - Chronological Chapter Digest
   - Conceptual Graph (Entities & Causal Relationships)
   - Procedural Recipes (Step-by-step executable SOPs)
   - Caveats, Trade-offs & Critical Analyses
4. **Governs** and outputs candidate knowledge via AgentLoom's **3-Track Architecture** and Propose-Review protocol.

---

## 3-Track Architecture & The Two-Role Graph Separation

Following AgentLoom core architecture, knowledge is strictly divided into two roles:
1. **Builder Role (`role-builder`) — "How I Learn"**: Meta-knowledge, skills, and behaviors driving the MediaLoom Ingestion Engine (probing, chunking, Whisper ASR, chapter alignment).
2. **Domain Role (`role-domain`) — "What I Learned"**: Harvested substantive concepts, causal theories, and candidate skills distilled from media content, governed via propose-review.

| Role | Sub-Track | Path | Canonical Contents |
|---|---|---|---|
| **Builder (Engine)** | Knowledge | `agents/knowledge-graphs/builder-knowledge-graph.json` | Engine architecture, dual-path ASR strategy, 24MB boundaries |
| **Builder (Engine)** | Skills | `agents/skills/builder/` & `builder-skills-graph.json` | Engine skills: `probe-media-stream`, `audio-chunking-compression`, `whisper-asr-transcription`, `chapter-alignment-anchoring` |
| **Builder (Engine)** | Behaviors | `agents/behaviors/builder/` & `builder-behaviors-graph.json` | Operational rules: `audio-budget-constraint`, `timestamp-anchoring-rule` |
| **Domain (Harvested)** | Knowledge | `agents/knowledge-graphs/domain-knowledge-graph.json` | Harvested concepts and causal networks (e.g. AI bubble, Dot-com analogies) |
| **Domain (Harvested)** | Skills | `agents/skills/domain/` & `domain-skills-graph.json` | Harvested SOP skills (`candidate/` for proposals, `accepted/` post-review) |
| **Domain (Harvested)** | Behaviors | `agents/behaviors/domain/` & `domain-behaviors-graph.json` | Domain policies and constraints derived from digested content |
| **Master Index** | Master | `agents/knowledge-graphs/master-graph.json` | Central registry connecting builder and domain roots for dashboard inspection |

---

## Repository Structure

```text
agentloom-agents-multimedia/
├── .cursor/ rules/                        # Track 1: Guidance Track
│   ├── core/identity.md                  # Agent identity & mode definition
│   ├── audio-processing-budget.md        # Audio chunking thresholds & ASR budgets
│   └── distillation-governance.md        # Provenance & timestamp citation rules
├── docs/                                  # Track 2: Knowledge Track
│   ├── architecture/                     # Architecture & system design
│   │   └── multimedia-agent-architecture.md
│   ├── guides/                           # Getting started & operation guides
│   │   └── START_HERE.md
│   └── digests/                          # Timestamped digests & research memos
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
├── src/agentloom_media/                  # Python Framework & CLI
│   ├── cli.py                            # `agentloom-media ingest` & `agentloom-media ui`
│   ├── ui/                               # Built-in Agent Portal & Review Webapp
│   │   ├── server.py                     # FastAPI REST API (Profile, Data, KG, Review)
│   │   └── static/                       # Reactive single-page web portal (index.html)
│   ├── acquisition/                      # Probing, fast subtitles, stream extraction
│   ├── audio/                            # Audio chunker & Whisper/ASR adapters
│   ├── distillation/                     # Chapter alignment, concepts, and skills
│   └── proposals/                        # AgentLoom proposal emitter
├── proposals/                            # Output directory for candidate KG proposals
├── pyproject.toml
└── README.md
```

---

## Agent-Native UI & Human-in-the-Loop (HITL) Portal

A core architectural principle of AgentLoom is that **AgentLoom is inherently a Human-in-the-Loop (HITL) Agentic-AI framework**. Every AgentLoom agent ships with its own self-contained, out-of-the-box web portal (`agentloom-media ui`):

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

1. **Self-Profile & Capabilities Showcase**: Introduces the agent's identity, role definitions (`builder` vs. `domain`), environment readiness, and live capability status.
2. **Data & Digest Explorer**: Archives all processed multimedia, providing interactive clickable-transcript readers and Track 2 structured chapter digests.
3. **Interactive 3-Track Knowledge Graph**: Explores `role-builder` (operational meta-knowledge) and `role-domain` (harvested concepts and causal models) across Knowledge, Skills, and Behaviors.
4. **Human Review Portal (THE CORE HITL GATE)**: Queues all candidate extractions in `proposals/`. Human reviewers audit claims against source video timestamps (`?t=...s`) and execute one-click **Approve** (auto-promoted into canonical graphs and accepted skills) or **Reject** decisions.

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/Keven1894/agentloom-agents-multimedia.git
cd agentloom-agents-multimedia
python -m venv .venv

# Windows
.venv\Scripts\pip install -e .

# Linux / macOS
source .venv/bin/activate && pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env` and provide your API keys:

```bash
cp .env.example .env
```

Ensure `OPENAI_API_KEY` is set.

### 3. Ingest a Video

```bash
# Ingest and distill a YouTube video
agentloom-media ingest "https://www.youtube.com/watch?v=R_PMTlFn0TQ"

# Output:
# - Track 2 Digest: docs/digests/YYYY-MM-DD-<slug>.md
# - Track 3 Candidate KG / Skills: proposals/proposal-<id>.json
```

### 4. Launch Built-in UI & HITL Review Portal

```bash
agentloom-media ui --port 8000
# Open http://localhost:8000 in your browser to inspect and audit extractions
```

---

## Governance & Ecosystem Alignment

This project is an **independent personal open-source research agent** created by [Dr. Boyuan (Keven) Guan (@Keven1894)](https://github.com/Keven1894). It is powered by the [AgentLoom Framework](https://github.com/Keven1894/AgentLoom) and is completely decoupled from FIU institutional production systems.

- **License**: Code is licensed under [MIT](LICENSE). Documentation & Knowledge are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
