# MediaLoom Cost Optimization, One-Click Ingestion UI & Multi-Platform Expansion Plan

**Date**: 2026-09-02  
**Status**: 📋 Planned / Ready for Execution  
**Owner**: Keven / Builder Envita  
**Scope**: In-depth cost economics modeling and 98% reduction strategy for multimedia knowledge ingestion; one-click URL submission and live progress tracking web UI; Slow Track browser-playback audio capture with 1.5×–2.0× acceleration; expansion roadmap for Bilibili, podcasts/RSS, Loom, and local file ingestion.  
**Canonical Architecture**: [`docs/architecture/multimedia-agent-architecture.md`](../architecture/multimedia-agent-architecture.md)  
**Parent Framework**: [AgentLoom Framework (v3.0)](https://github.com/Keven1894/AgentLoom)  
**Target Repository**: `Keven1894/agentloom-agents-multimedia` (`C:\projects\03_personal-agents\agentloom-agents-multimedia`)  
**Tags**: `agentloom`, `medialoom`, `multimedia`, `cost-optimization`, `whisper`, `hitl-ui`, `bilibili`, `podcast`, `plan`

---

## 0. Executive Summary & Strategic Context

In Session 2026-09-02, the initial autonomous multimedia agent `MediaLoom` was successfully scaffolded, verified on real-world long-form video (`R_PMTlFn0TQ`, 35.3 minutes), structured into AgentLoom's dual-role (`role-builder` vs. `role-domain`) 3-track architecture, and equipped with an agent-native Web UI and Human-in-the-Loop (HITL) review portal.

To transition MediaLoom from an engineering prototype to a daily personal intelligence production line, four core challenges must be resolved:
1. **Economic Viability**: Understanding unit costs across speech recognition and LLM reasoning, followed by deploying zero-cost/tiered hybrid routing to slash operational expenses by >90%.
2. **Operational Ergonomics**: Transitioning from terminal command-line execution (`agentloom-media ingest <URL>`) to a dead-simple web interface where the user drops a video URL, watches a real-time progress bar through discrete pipeline stages, and seamlessly transitions into the HITL Review Portal upon completion.
3. **Coverage / Slow Track**: When captions and protocol extraction both fail, fall back to browser playback + in-page audio capture, optionally at 1.5×–2.0×, then rescale ASR timestamps back to the original timeline.
4. **Platform Extensibility**: Generalizing beyond YouTube to ingest knowledge from Chinese technical video ecosystems (Bilibili), developer walkthroughs (Loom), high-density audio podcasts (Xiaoyuzhou/Apple Podcasts/RSS), and confidential local recordings.

---

## 1. Pillar 1: Multimedia Learning Cost Economics & Optimization Strategy

### 1.1 Baseline Unit Pricing Structure (2026 Standards)

| Pipeline Stage | Engine / Model | Unit Pricing Metric | Normalized Hourly Rate |
|---|---|---|---|
| **ASR (Cloud API)** | OpenAI `whisper-1` | **$0.006 / minute** | **$0.360 / hour** |
| **ASR (Local GPU)** | `faster-whisper` (large-v3 / distil-large-v3) | Local compute (RTX 3060/4090 / Apple Silicon) | **$0.000 / hour** (0 API cost) |
| **ASR (Local CPU)** | `whisper.cpp` / `faster-whisper-cpu` | Multi-core CPU compute (quantized INT8) | **$0.000 / hour** (0 API cost) |
| **LLM Reasoning (Flagship)** | OpenAI `gpt-4o` | Input: $2.50 / 1M tokens<br>Output: $10.00 / 1M tokens | ~$0.050 – $0.080 / video |
| **LLM Reasoning (Lightweight)** | OpenAI `gpt-4o-mini` | Input: $0.15 / 1M tokens<br>Output: $0.60 / 1M tokens | ~$0.003 – $0.005 / video |
| **LLM Reasoning (Open Tier)** | `DeepSeek-V3` / `Qwen-2.5-72B` | Input: ~$0.14 / 1M tokens<br>Output: ~$0.28 / 1M tokens | ~$0.002 – $0.004 / video |

---

### 1.2 Empirical Cost Breakdown: Benchmark Case (`R_PMTlFn0TQ`)
- **Video Specifications**: 2,118 seconds (~35.3 minutes).
- **Transcript Volume**: ~13,200 Chinese characters (~15,500 input tokens across prompt wrappers).
- **Distilled Output**: Structured JSON + Markdown summary (~2,100 output tokens).

```
┌────────────────────────────────────────────────────────────────────────┐
│ Comparative Unit Economics per 35-Minute Video                         │
├───────────────────────────────────┬──────────────┬─────────────────────┤
│ Execution Strategy                │ Unit Cost    │ Cost in CNY (RMB)   │
├───────────────────────────────────┼──────────────┼─────────────────────┤
│ 1. Current Cloud Heavy Path       │ $0.270       │ ~1.95 元            │
│    (Whisper API + GPT-4o)         │              │                     │
│ 2. Cloud Fast Track (Subtitles)   │ $0.058       │ ~0.42 元            │
│    (0 ASR + GPT-4o)               │              │                     │
│ 3. Cloud Tiered Fast Track        │ $0.004       │ ~0.03 元 (3 分钱)   │
│    (0 ASR + GPT-4o-mini)          │              │                     │
│ 4. Optimized Hybrid Heavy Path    │ $0.005       │ ~0.035 元 (3.5 分钱)│
│    (Local Faster-Whisper + mini)  │              │                     │
└───────────────────────────────────┴──────────────┴─────────────────────┘
```

> **Key Economic Finding**:
> In the Heavy Path (where subtitles are disabled by video creators), **ASR accounts for 78.5% of the total processing cost ($0.212 / $0.270)**. Replacing cloud ASR with local inference eliminates nearly 80% of costs immediately.

---

### 1.3 The 98% Cost Reduction Playbook

1. **Local Faster-Whisper Integration (Primary Cost Killer)**:
   - Introduce `local-asr` adapter powered by `faster-whisper` using `int8_float16` quantization.
   - For a 35-minute audio stream, local GPU inference executes in 90–120 seconds with identical Word Error Rate (WER).
   - **Cost Saving**: Reduces ASR cost from **$0.212 to $0.000**.
2. **Tiered LLM Distillation (Two-Pass Distillation)**:
   - **Pass 1 (Extraction & Filtering)**: Route chapter summarization, noise removal, and preliminary entity identification to `gpt-4o-mini` or `DeepSeek-V3`.
   - **Pass 2 (Synthesis & Validation)**: Call `gpt-4o` only when synthesizing formal candidate executable skills or complex causal relationships in knowledge graph nodes.
   - **Cost Saving**: Reduces LLM inference cost from **$0.058 to $0.004** (93% reduction).
3. **Persistent Audio & Segment Fingerprint Cache (Already Active)**:
   - Hash video ID and audio stream content. Never re-transcribe an identical video when prompts or distillation models are re-tested.
4. **Combined Impact**:
   - Total cost per video drops from **$0.270 (~1.95 RMB) down to $0.005 (~0.035 RMB)**.
   - Enables batch processing of 100+ research videos for under **$0.50**.

---

## 1.4 Pillar 1b: Stage 2C Slow Track — Browser Playback Capture (Authorized Personal Study Bridge)

Fast Track and Heavy Path assume public caption files or downloadable video objects exist. In real-world educational scenarios, users frequently purchase legitimate access to premium developer courses, professional workshops, or specialized learning platforms (e.g., GeekBang/极客时间, Dedao/得到, Udemy, proprietary LMS portals). These platforms employ Media Source Extensions (MSE), Blob URLs, token rotations, and encrypted player wrappers strictly to prevent video scraping and piracy.

For the user, however, **they have legitimately purchased the content and are fully authorized to watch, learn, take study notes, and summarize the material for personal use**. Universal tools like `yt-dlp` cannot download from these players.

**Slow Track serves as the Authorized Personal Study Bridge (合法购买教程与个人学习知识桥梁)**:
- It runs inside the user's authenticated browser session (Chromium/Playwright).
- It does not attempt to bypass paywalls or crack encryption (which is neither feasible nor intended).
- Instead, as the user plays the lesson they legitimately own, it captures the **decoded in-page audio stream** directly (`HTMLMediaElement.captureStream()`), optionally accelerated at 1.5×–2.0× `playbackRate`.
- The captured audio is passed to Whisper and LLM distillation, converting the purchased video lesson into structured notes, chapter summaries, and personal knowledge graphs.
- **Result**: Users can seamlessly bridge their legitimately acquired learning materials into their personal AgentLoom brain.

### When it fires
1. Fast Track captions unavailable.
2. Heavy Path `yt-dlp` extract fails (HTTP 403, cipher, MSE/Blob-only stream, no format 140, login wall).
3. The URL is an authorized course page the operator is entitled to watch.

### Capture stack (ranked)
| Priority | Method | Isolation | Automation | Notes |
|---|---|---|---|---|
| **1 (default)** | Playwright + `HTMLMediaElement.captureStream()` + `MediaRecorder` | Tab/element only | High | Records decoded audio without speakers or virtual cables |
| 2 | Chromium tab-audio capture (extension / CDP) | Tab only | Medium | Use when `captureStream()` is blocked by the page |
| 3 | OS loopback (Windows WASAPI / macOS BlackHole / Pulse monitor) | Whole desktop | Low | Mixes other apps; last resort only |

Do **not** default to analog/loopback recording. Element capture is cleaner, scriptable, and does not require muting the rest of the machine.

### Speed-up (`playbackRate`) — this is the point of Slow Track
HTML5 `playbackRate` is pitch-preserving (browser time-stretch). That makes it usable for ASR.

| Rate | 35.3 min video wall-clock | Whisper-billed minutes | Cloud ASR cost (`whisper-1`) | Quality (Chinese speech) |
|---|---|---|---|---|
| 1.0× | ~35 min | 35.3 | $0.212 | Baseline |
| **1.5× (default)** | ~24 min | 23.5 | **$0.141** | Recommended |
| **2.0× (budget)** | ~18 min | 17.7 | **$0.106** | Acceptable on clear speech |
| 3.0× | ~12 min | 11.8 | $0.071 | Rejected — WER collapses on Chinese |

**Hard invariant**: after ASR, rescale every segment timestamp onto the original timeline:

```
t_original = t_asr × playbackRate
```

Without this, chapter alignment and `?t=` evidence URLs point at the wrong second.

Optional second acceleration (Heavy Path or already-captured 1.0× files): ffmpeg `atempo=1.5` before Whisper. This saves ASR minutes but **does not** save Slow Track wall-clock; prefer in-player `playbackRate` during capture.

### Cost position vs other paths (same 35-min video, GPT-4o distillation still $0.058)
| Path | Wall-clock | ASR $ | Total $ |
|---|---|---|---|
| Fast Track captions | seconds | $0 | $0.058 |
| Heavy Path + local Whisper | 1–3 min | $0 | $0.058 |
| Heavy Path + cloud Whisper | 1–3 min | $0.212 | $0.270 |
| Slow Track 1.5× + cloud Whisper | ~24 min | $0.141 | $0.199 |
| Slow Track 2.0× + **local** Whisper | ~18 min | $0 | $0.058 |

Slow Track never wins on speed against Heavy Path. It wins on **coverage**. Pair it with local Whisper so the extra wall-clock is the only penalty.

### Routing rule
```
probe → Fast Track
      → else Heavy Path
      → else Slow Track (browser capture @ 1.5×, rescale timestamps)
      → else fail closed and ask the operator
```

Scope: operator-accessible watch pages only. Slow Track is not a DRM circumvention tool.

---

## 2. Pillar 2: One-Click URL Ingestion & Live Progress Tracking Web UI

### 2.1 User Experience Flow
```
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      MediaLoom Ingestion Station                       │
 ├────────────────────────────────────────────────────────────────────────┤
 │ [ https://www.youtube.com/watch?v=...                           ] [Run]│
 ├────────────────────────────────────────────────────────────────────────┤
 │ Status: Processing... [=====================>              ] 65%      │
 │                                                                        │
 │ [✓] Stage 1: Probing metadata & checking captions (Fast Track)         │
 │ [✓] Stage 2: Extracting format 140 audio & 16kHz chunking (8.2 MB)     │
 │ [▶] Stage 3: Acoustic Whisper ASR Transcribing (Chunk 1/2)...          │
 │ [ ] Stage 4: Aligning speech segments with native chapters             │
 │ [ ] Stage 5: Multi-perspective LLM Distillation (GPT-4o)               │
 │ [ ] Stage 6: Emit Proposals -> Direct Link to Human Review Portal      │
 │                                                                        │
 │ Console Stream: [View Live Log Terminal v]                             │
 └────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Backend Architecture & Event-Stream Specification
1. **Background Job Manager (`agentloom_media.ui.jobs`)**:
   - In-memory async job registry with persistent execution history in `.cache/jobs/`.
   - Thread-safe stage emission with state persistence: `queued` → `probing` → `extracting` → `transcribing` → `aligning` → `distilling` → `completed` | `failed`.
2. **Server-Sent Events (SSE) Streaming Endpoint**:
   - `POST /api/ingest/submit`: Accepts `{ "url": str, "model": str, "asr_engine": "auto"|"local"|"api" }`. Returns `job_id`.
   - `GET /api/ingest/stream/{job_id}`: Real-time event stream emitting typed JSON events:
     ```json
     {
       "event": "stage_update",
       "job_id": "job_20260902_173000",
       "stage": "transcribing",
       "percent": 50,
       "message": "Transcribing audio segment 1 of 2 via Whisper...",
       "timestamp": "2026-09-02T17:30:15"
     }
     ```
   - `GET /api/ingest/jobs`: Returns list of historical and active ingestion jobs.

### 2.3 Frontend Implementation Components
1. **Hero Input Card**:
   - Placed at the top of the portal, visible across tabs.
   - Clean URL input box with automatic platform badge detection (YouTube / Bilibili / RSS / Loom).
   - Engine selector toggle (`Auto`, `Local Whisper (Free)`, `Cloud API`).
2. **Stepped Visual Indicator**:
   - 6 sequential milestone nodes that animate dynamically as backend stages complete.
3. **Collapsible Real-Time Log Drawer**:
   - Live streaming terminal log for developers who want to inspect chunk boundaries, download speeds, and token consumption in real time.
4. **Completion Action Gateway**:
   - On completion, emits a toast notification and displays a primary action button: **"Audit 3 New Proposals in HITL Portal →"**, guiding the reviewer directly to the review gate.

---

## 3. Pillar 3: Multi-Platform Ingestion Expansion Matrix

```
┌────────────────────────────────────────────────────────────────────────┐
│                     MediaLoom Multi-Source Architecture                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
    ┌──────────────┬────────────────┼────────────────┬────────────────┐
    ▼              ▼                ▼                ▼                ▼
┌────────┐   ┌───────────┐   ┌─────────────┐   ┌───────────┐   ┌────────────┐
│YouTube │   │ Bilibili  │   │ Podcast/RSS │   │   Loom    │   │ Local File │
└────────┘   └───────────┘   └─────────────┘   └───────────┘   └────────────┘
```

### 3.1 Target Platforms & Integration Specifications

| Platform | Domain / Use Case | Technical Extraction Path | Subtitle Availability |
|---|---|---|---|
| **YouTube** | Global tech talks, academic keynotes, AI updates | `yt-dlp` format 140 m4a + `youtube-transcript-api` | Mixed (Native, Auto, or Disabled) |
| **Bilibili (B站)** | Chinese developer tutorials, university CS courses, AI breakdowns | `yt-dlp` format 30280 (audio) + Bilibili Web CC subtitle API (`https://api.bilibili.com/x/player/v2`) | High (frequent CC subtitles, enables Fast Track) |
| **Podcasts & RSS** | Tech founder interviews, deep-dive architectural discussions (Xiaoyuzhou, Apple Podcasts, Spotify) | Feed parser (`feedparser` / XML reader) extracting MP3/M4A `enclosure` URL | Pure audio (Zero video download overhead; directly enters 16kHz chunker) |
| **Loom** | Internal developer walkthroughs, PR screen recordings, bug repros | Loom page video stream extraction (`https://www.loom.com/share/...`) | Auto-captions available via Loom GraphQL API |
| **Local File Drop** | Confidential lectures, meeting recordings, offline MP4/WAV/M4A | Direct browser drag-and-drop to `.cache/uploads/` | None (Direct Heavy Path through local ASR) |

---

## 4. Phased Implementation Roadmap & Work Breakdown Structure (WBS)

### Phase 1: Cost Optimization & Hybrid ASR Engine
- [ ] **Task 1.1: Local Faster-Whisper Adapter**  
  Implement `agentloom_media.audio.local_whisper.py` wrapping `faster-whisper`. Support `large-v3` with automatic GPU fallback to CPU INT8.
- [ ] **Task 1.2: Tiered LLM Distillation Routing**  
  Add model parameterization in `distiller.py`. Test extraction quality between `gpt-4o`, `gpt-4o-mini`, and `DeepSeek-V3`.
- [ ] **Task 1.3: Cost Tracking & Telemetry Reporting**  
  Record exact token usage and compute duration in every proposal JSON (`cost_audit` object with total USD/CNY estimated).
- [x] **Task 1.4: Slow Track Browser Capture Adapter (Authorized Personal Study Bridge)**  
  Implemented `agentloom_media.acquisition.browser_capture.py` using Playwright + `HTMLMediaElement.captureStream()` + `MediaRecorder`. Provides legitimate personal study bridge for purchased course lessons.
- [x] **Task 1.5: Playback-Rate Acceleration & Timestamp Rescale**  
  Implemented `rescale_timestamps()` enforcing `t_original = t_asr * playbackRate` to maintain fine-grained temporal provenance.
- [ ] **Task 1.6: Three-Path Router**  
  Update `cli.py` / job worker: Fast → Heavy → Slow. Surface the chosen path and `playbackRate` in the UI stepper and in `cost_audit`.

### Phase 2: One-Click Ingestion & Live Progress Tracking Web UI
- [x] **Task 2.1: Async Job Manager Backend**  
  Implemented `agentloom_media.ui.jobs.JobManager` supporting asynchronous queueing, stage tracking (Probe → ASR → Align → Distill → Emit), and in-memory log buffer.
- [x] **Task 2.2: SSE Streaming API**  
  Implemented `/api/ingest/submit` and `/api/ingest/stream/{job_id}` in `agentloom_media.ui.server`.
- [x] **Task 2.3: Frontend Hero Ingestion Card & Stepper**  
  Updated `index.html` with URL input, live step indicators, animated progress percentage bar, and live terminal stream console.
- [x] **Task 2.4: Review Portal Integration**  
  Completed jobs trigger reactive UI refresh on the "Human Review Portal" tab with one-click review navigation and instant badge update.

### Phase 3: Platform Expansion & Local File Upload
- [ ] **Task 3.1: Bilibili Ingestion Adapter**  
  Support Bilibili `BV...` and `av...` URLs, including Bilibili CC subtitle extraction for instant Fast Track parsing.
- [ ] **Task 3.2: Podcast RSS Ingestion Adapter**  
  Support direct podcast RSS XML feeds and episode links, directly streaming audio enclosures into the chunker.
- [ ] **Task 3.3: Local Media File Drag-and-Drop**  
  Add multipart file upload endpoint (`POST /api/ingest/upload`) to let users drag-and-drop local MP4, MKV, MP3, and WAV files directly.

---

## 5. Verification & Acceptance Criteria

1. **Cost Benchmark Test**:
   - Re-running a 30+ minute video with `--asr-engine local --model gpt-4o-mini` must complete with **$0.00 ASR cost and < $0.01 total cost**.
2. **Web UI End-to-End Test**:
   - Submitting a YouTube URL via the web input must stream real-time progression from 0% to 100% without UI freeze.
   - Upon completion, the new proposal must appear immediately in the Human Review Portal with intact timestamp evidence links.
3. **Multi-Source Ingestion Test**:
   - Successfully ingest and distill at least one Bilibili video and one podcast episode, producing compliant Track 2 digests and Track 3 candidate proposals.
4. **Slow Track Fallback Test**:
   - On a URL where captions and `yt-dlp` extract are forced to fail, browser capture at 1.5× must produce a transcript whose chapter anchors match the original video timeline within ±2 seconds.
5. **Registry & Memory Synchronization**:
   - Plan registered in `docs/plan/plan_registry.json` and validated clean via `python Scripts/plan_memory/validate_plan_memory.py`.
