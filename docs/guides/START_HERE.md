# Getting Started with agentloom-agents-multimedia

Welcome to **`agentloom-agents-multimedia`**, the multimedia knowledge ingestion and distillation agent for the AgentLoom framework.

---

## 1. Prerequisites

- Python 3.10+
- An OpenAI API Key (or Groq / local Whisper model)
- Optional: `ffmpeg` installed on your system PATH for advanced local audio downsampling / transcoding.

---

## 2. Environment Setup

```bash
git clone https://github.com/Keven1894/agentloom-agents-multimedia.git
cd agentloom-agents-multimedia

# Setup virtual environment
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows

# Install in editable mode
pip install -e .
```

Create your `.env`:

```bash
cp .env.example .env
# Edit .env and insert OPENAI_API_KEY
```

---

## 3. Running Your First Ingestion

```bash
# Ingest a YouTube video
agentloom-media ingest "https://www.youtube.com/watch?v=R_PMTlFn0TQ"
```

The pipeline will:
1. Probe video metadata and chapters.
2. Check for native subtitles (Fast Track).
3. If subtitles are disabled, extract audio format 140 (Heavy Track) and perform ASR.
4. Align the transcript with native chapters.
5. Generate:
   - A Track 2 digest in `docs/digests/`.
   - Track 3 candidate skills in `agents/skills/candidate/` (if practical steps exist).
   - Candidate KG nodes in `proposals/`.

---

## 4. Governance & Human Review

In accordance with AgentLoom v3:
1. Candidate KG proposals are staged in `proposals/`.
2. Inspect the proposed nodes on the AgentLoom Dashboard (`http://127.0.0.1:8000`).
3. Accept proposals into canonical knowledge:
   ```bash
   python -m agentloom.kg.accept_proposal --file proposals/proposal-xyz.json
   ```
