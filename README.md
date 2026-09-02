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

## 3-Track Architecture

This repository strictly organizes knowledge and agent instructions into AgentLoom's 3-Track convention:

| Track | Directory | Role & Description |
|---|---|---|
| **Track 1: Guidance Track** | `.cursor/` & `.clinerules/` | Rules, thresholds, audio processing budgets, and distillation prompts for the builder agent. |
| **Track 2: Knowledge Track** | `docs/` | System architecture, start guides, and timestamped video digests (`docs/digests/`). |
| **Track 3: Skills Track** | `agents/` | Executable skills (`agents/skills/`), behavior validators (`agents/behaviors/`), and distilled conceptual knowledge graphs (`agents/knowledge-graphs/`). |

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
│   ├── skills/                           # Executable skills (tutorials -> SOPs)
│   │   └── candidate/
│   ├── behaviors/                        # Agent behavior rules & constraints
│   └── knowledge-graphs/                 # Conceptual knowledge graphs & entity networks
│       └── domain-concepts-graph.json
├── src/agentloom_media/                  # Python Framework & CLI
│   ├── cli.py                            # `agentloom-media ingest <url>` entrypoint
│   ├── acquisition/                      # Probing, fast subtitles, stream extraction
│   ├── audio/                            # Audio chunker & Whisper/ASR adapters
│   ├── distillation/                     # Chapter alignment, concepts, and skills
│   └── proposals/                        # AgentLoom proposal emitter
├── proposals/                            # Output directory for candidate KG proposals
├── pyproject.toml
└── README.md
```

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

---

## Governance & Ecosystem Alignment

This project is an **independent personal open-source research agent** created by [Dr. Boyuan (Keven) Guan (@Keven1894)](https://github.com/Keven1894). It is powered by the [AgentLoom Framework](https://github.com/Keven1894/AgentLoom) and is completely decoupled from FIU institutional production systems.

- **License**: Code is licensed under [MIT](LICENSE). Documentation & Knowledge are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
