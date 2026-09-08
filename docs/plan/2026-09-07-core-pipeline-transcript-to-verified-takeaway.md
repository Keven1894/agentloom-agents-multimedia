# Core Pipeline Design: Transcript → Searchable Video → Distillation → KG → Grounded Review → Verified Takeaway

**Date**: 2026-09-07
**Status**: 📋 Design proposal, awaiting author approval
**Scope**: The six core capabilities of MediaLoom, replacing the current single-shot
`probe → captions/ASR → 5-minute buckets → one JSON call → files` flow.
**Supersedes on distillation detail**: `docs/architecture/multimedia-agent-architecture.md` §4
(the pipeline stages remain valid; the alignment and distillation stages are redesigned here).
**Related**: `docs/plan/2026-09-02-medialoom-cost-optimization-one-click-ui-and-platform-expansion-plan.md`
(cost tiering, Slow Track), `docs/architecture/llm-selection-and-tiered-pricing-guide.md` (model routing).

---

## 0. The defect that motivates this redesign

`align_transcript_with_chapters()` synthesizes **fixed 5-minute buckets** when a video has no
native chapters (`chapter_aligner.py:32-48`). The DAOJIE ingestion run on 2026-09-07 had
`Chapters found: 0`, so every "chapter" in the emitted digest was a 5-minute bucket, and the
distiller was handed `Part (00:00 - 05:00)` as if it were a topic.

Two consequences, both governance failures rather than cosmetic ones:

1. The emitted anchors (`00:00-05:00`) are **not** anchors. `behavior-builder-timestamp-anchoring-rule`
   requires that a reviewer can "click directly to the precise second". A 5-minute bucket is a
   300-second haystack.
2. Because the LLM received bucket labels as topic labels, it invented composite anchor strings
   such as `00:00-05:00; 10:00-15:00`. That is a fabricated citation format that no validator
   currently rejects.

Everything below is built on the premise that **the atomic unit of this system is a timed span,
and every derived artifact must be traceable to one**.

---

## 1. Foundational data model (do this before any stage)

All six stages read and write one canonical structure. Today the pipeline passes ad-hoc dicts
between functions and persists only a digest and a proposal, so nothing downstream can be
re-derived or audited.

### 1.1 Canonical transcript artifact

Persist one file per media item, git-ignored (large, regenerable):

```
data/transcripts/<media_id>/
├── transcript.json         # canonical, versioned
└── manifest.json           # provenance + hashes + engine versions
```

`transcript.json` carries three levels, all with times in seconds (float):

| Level | Field | Purpose |
|---|---|---|
| Word | `words[]`: `{w, t0, t1, speaker, conf}` | Precise seek targets, forced-alignment output |
| Utterance | `utterances[]`: `{id, t0, t1, text, speaker, word_range}` | Natural retrieval and quotation unit |
| Document | `text` + `char_index[]` | Flat string for span-based extractors, with a char→word map |

The `char_index` array is load-bearing: it lets any character offset produced by a downstream
extractor be resolved back into `(t0, t1)`. This is what makes Stage 3's grounding work.

### 1.2 Working index vs canonical knowledge

Keep the existing `durable vs ephemeral` split explicit:

- **Ephemeral / regenerable** (git-ignored): audio cache, transcripts, embeddings, SQLite index,
  any third-party graph store. Rebuildable from the media URL plus the manifest.
- **Durable / canonical** (git-tracked, human-accepted): `agents/knowledge-graphs/domain-*.json`,
  `agents/skills/domain/accepted/`, `docs/digests/`.

Proposals (`proposals/`) are the queue between them. No stage may write canonical files directly.

### 1.3 Cache key correctness

`cli.py` currently caches transcripts as `<video_id>_segments.json`. The key omits the ASR engine
and model, so switching from `whisper-1` to WhisperX silently reuses inferior output. Change the
key to `<media_id>.<asr_engine>.<model>.<pipeline_version>.json` and record the same tuple in
`manifest.json`.

---

## 2. Stage 1 — Transcript acquisition and generation

**Goal**: a word-level, speaker-labelled, language-normalized transcript, at the lowest cost that
still supports second-accurate anchors.

### 2.1 Three acquisition routes, ordered by cost

| Route | Mechanism | Word-level times? | Cost | When |
|---|---|---|---|---|
| Fast | `youtube-transcript-api` captions | No (caption cue granularity) | $0 | Captions exist and cue granularity is acceptable |
| Heavy | `yt-dlp` format 140 → local ASR | Yes | $0 local / $0.006 per min API | Default for anything to be anchored precisely |
| Slow | Browser playback capture (already designed in the 09-02 plan) | Yes | $0 + wall clock | Fast and Heavy both blocked |

Important correction to the current routing: **the Fast Track is not automatically preferable.**
Caption cues are typically 2–8 seconds and carry no speaker or word timing, so a Fast-Track-only
item cannot support word-level seeking. Record `timing_granularity: "cue" | "word"` in the manifest
and let downstream stages degrade honestly rather than pretend to precision they do not have.

### 2.2 ASR engine: WhisperX as the default local engine

[WhisperX](https://github.com/m-bain/whisperX) (BSD-2-Clause, ~22k stars, INTERSPEECH 2023) is the
mature answer and we should not build any of it ourselves. It is a pipeline over three stages:

1. **VAD** (Silero) — skip silence, chunk on speech boundaries, enable batching.
2. **Transcription** — `faster-whisper` (CTranslate2) backend.
3. **Forced alignment** — wav2vec2, bringing timestamp drift from Whisper's ~1s down to **<100ms**.
4. **Diarization** — optional `pyannote.audio` for speaker labels.

Trade-offs to accept knowingly:
- ~30–40% slower per audio minute than bare `faster-whisper`; the alignment pass is the cost of
  knowing *when* each word was said. For this system that is the whole point.
- Diarization requires accepting the `pyannote` license and supplying a Hugging Face token. Make
  diarization **opt-in** (`--diarize`), since a single-narrator video like the DAOJIE channel does
  not need it and it adds a gated dependency.
- CUDA-oriented. On this machine (Apple Silicon, no CUDA) WhisperX runs CPU-only and slowly.

Therefore engine selection must be a strategy, not a constant:

| `ASR_ENGINE` | Implementation | Use case |
|---|---|---|
| `whisperx` | WhisperX, word-level + optional diarization | Default when a GPU box is available |
| `faster_whisper` | `faster-whisper` int8 | CPU-bound machines, segment-level acceptable |
| `whisper_cpp` | `whisper.cpp` Metal/CoreML | Apple Silicon local default |
| `whisper_api` | Current OpenAI `whisper-1` | Zero-setup fallback (segment-level only) |

Keep `whisper_api` working (it is what runs today), but stop treating its segment timings as
word-level. If precise anchors are required and only `whisper_api` is available, run a separate
forced-alignment pass rather than trusting Whisper's own timings.

**v1 decision (2026-09-07)**: local ASR is deferred, so `whisper_api` stays the default and v1
ships with `timing_granularity: "segment"`. The consequence must be accepted explicitly rather
than discovered later: **word-level transcript highlighting and sub-second evidence seeking are
out of scope for v1.** Segment-level anchors are roughly 2–10 seconds wide, which is honest and
still a ~30× improvement over today's 300-second buckets. Every stage below is designed to read
`timing_granularity` and degrade openly, so upgrading to WhisperX later is a swap of one engine
plus a re-index, not a redesign.

### 2.3 Language normalization (a real defect found in the DAOJIE run)

The video title is Simplified Chinese; the Whisper output and therefore the entire digest came back
in **Traditional** Chinese, with terms drifting between scripts. Add a normalization step:

- Script normalization via OpenCC (`t2s`) so the index and the KG do not hold two spellings of the
  same entity.
- A domain glossary passed as the ASR `initial_prompt` (Whisper supports this and we already ignore
  the capability). For this channel: `Meta`, `Andromeda`, `CBO`, `ABO`, `Ad Set`, `ROAS`,
  `Prospecting`, `Remarketing`. This measurably reduces transliteration noise on mixed zh/en speech.
- Keep the raw, unnormalized text in `transcript.json` alongside the normalized field. Never
  destroy the original.

---

## 3. Stage 2 — Make an unsearchable video searchable

**Goal**: a query returns **time ranges in specific videos**, not documents.

### 3.1 Storage: one SQLite file, no server

Use SQLite with **FTS5** (built-in, BM25) for lexical search and **`sqlite-vec`** (pip-installable
extension by Alex Garcia) for vector search, fused with **Reciprocal Rank Fusion** in a single SQL
statement. This is a well-documented pattern and runs in single-digit milliseconds on a
million-row index.

Why not Meilisearch / Qdrant / a vector DB: they add a service to operate for no capability we need
at personal-agent scale, and Meilisearch is known to use several times more disk than FTS5 for the
same corpus. A single `data/medialoom.db` also matches the repo's stated "portable infrastructure"
principle and stays trivially regenerable.

### 3.2 Retrieval unit: overlapping time windows, not ASR segments

An ASR utterance is 2–8 seconds — too short to embed meaningfully, too short to answer anything.
Build a second layer:

- **Windows** of ~45s with ~15s overlap (tune empirically), each storing `(media_id, t0, t1,
  text, utterance_ids)`.
- Windows are the embedded and BM25-indexed unit. A hit therefore *is* a timecode.
- Overlap prevents a topic boundary from splitting an answer across two unretrievable halves.

### 3.3 Chinese tokenization (blocking issue for FTS5)

FTS5's default `unicode61` tokenizer splits on whitespace and punctuation, so a Chinese transcript
becomes a handful of enormous tokens and BM25 collapses. Two workable options:

1. **`tokenize='trigram'`** — built into FTS5, language-agnostic, no dependency. Larger index,
   good enough recall for CJK. **Recommended starting point.**
2. **jieba pre-segmentation** — insert space-delimited tokens into a shadow column. Better
   precision, adds a dependency and a vocabulary to maintain.

Start with trigram; revisit only if measured recall is poor.

### 3.4 Embedding model

Mixed Chinese/English content rules out English-only models.

| Option | Notes |
|---|---|
| **Qwen3-Embedding-0.6B** (Apache 2.0, local) | Strong C-MTEB/MMTEB results at 0.6B, configurable dims, 32k context. **Recommended default.** |
| BGE-M3 | Uniquely gives dense + sparse + ColBERT in one pass; pick this if we later want its sparse vectors to replace FTS5 |
| OpenAI `text-embedding-3-large` | Zero ops, per-token cost, sends transcripts off-machine |

Recommendation: local Qwen3-Embedding-0.6B as the target default (no per-video cost, no data
egress), with the OpenAI embedder as a configured fallback. Store the embedding model id per row —
an index built with two different embedders is silently corrupt.

**v1 decision (2026-09-07)**: consistent with the API-first choice, v1 uses the OpenAI embedder so
no model download is required. Because the model id is stored per row, switching to Qwen3 later is
a re-embed of existing windows rather than a schema change. Note the trade-off being accepted:
embedding means the full transcript text leaves the machine, which is fine for public YouTube
content but would need revisiting before ingesting private recordings or paid course material.

### 3.5 Deliverable

`agentloom-media search "Andromeda 之後還要不要拆 Campaign"` returns ranked
`(video title, MM:SS–MM:SS, matched quote, deep link)`. That single command is what converts an
opaque video library into a searchable corpus, and it is worth shipping on its own before any
distillation work.

---

## 4. Stage 3 — Distillation into summary, segments, keywords, knowledge points

**Goal**: replace the one-shot "give me all the JSON" call with staged, individually
verifiable passes.

### 4.1 Semantic segmentation replaces the 5-minute bucket

Compute embeddings per utterance, then detect topic boundaries where consecutive-window similarity
drops below a threshold (semantic chunking, ~0.7 cosine distance is the commonly cited starting
point). Output real segments with real `(t0, t1)` and an LLM-generated title.

This directly removes the §0 defect: segment titles become descriptive, and anchors become the
segment's true start second.

When a video *does* publish native chapters, prefer them — they are author ground truth — but still
run boundary detection inside long chapters.

### 4.2 Hierarchical summarization (RAPTOR-shaped)

For anything beyond ~20 minutes, a flat "summarize the whole transcript" call loses structure and
will not scale to multi-episode courses. Adopt the RAPTOR pattern: embed → cluster → summarize,
recursively, producing a tree whose leaves are segments and whose internal nodes are progressively
more abstract summaries.

Two refinements from the 2025/2026 follow-up work, worth taking because they are cheap:
- **Semantic chunking for leaves** instead of fixed-token chunks (we already need this for §4.1).
- **Leiden community detection with layer-adaptive resolution** instead of GMM clustering,
  reported to cut the number of summary nodes by up to 76% while improving accuracy — fewer nodes
  means a cheaper build and a shorter human review queue.

For a single 16-minute video the tree is 2 levels and this is overkill; implement it behind a
duration/《series》threshold so short videos stay one cheap pass.

### 4.3 Grounded extraction with LangExtract

For keywords, knowledge points, technical items, and definitions, use
[LangExtract](https://github.com/google/langextract) (Google, v1.6.0) rather than free-form JSON.
The decisive feature: every extraction carries a `char_interval` (`start_pos`/`end_pos`) into the
source text, and **extractions that cannot be located in the source are returned with
`char_interval = None`** so they can be dropped:

```python
grounded = [e for e in result.extractions if e.char_interval]
```

That is a structural anti-hallucination gate, not a prompt instruction. Combined with §1.1's
`char_index`, every extracted term resolves to `(t0, t1)` automatically — which is exactly what
`behavior-builder-timestamp-anchoring-rule` demands and what the current pipeline cannot do.

It also handles long documents natively (chunking, parallel workers, `extraction_passes` for
recall) and offsets are rebased onto the original document, so we do not maintain that logic.

Caveat to verify during implementation: LangExtract's alignment is word-based and tuned for
space-delimited text; CJK alignment quality must be measured before we depend on it, with a
fallback of exact substring search into the normalized transcript.

### 4.4 Typed passes and model routing

Split the single call into passes with separate schemas, so one malformed field cannot void the
whole distillation:

| Pass | Output | Model tier |
|---|---|---|
| P1 Segment titles | title per segment | Luna |
| P2 Segment summary | key points + analysis, anchored | Luna |
| P3 Grounded terms | keywords, technical items, definitions with spans | Luna via LangExtract |
| P4 Document synthesis | executive summary, thesis, structure | Terra / Sol |
| P5 Candidate skills | executable SOP, only if procedural content detected | Terra |

`gpt-5.6-luna` is already the configured default. Escalate only P4/P5, and gate P5 behind a
detector — the current pipeline attempts skill extraction on every video, which is how a marketing
commentary produced a "skill" whose steps are prose sentences in `bash` fences.

---

## 5. Stage 4 — Multi-dimensional knowledge graph expansion

**Goal**: turn distilled nodes into a graph with several distinct edge families, incrementally,
without handing canonical memory to a third-party store.

### 5.1 Library choice: incremental is the hard constraint

| Tool | Model | Incremental | Verdict |
|---|---|---|---|
| Microsoft GraphRAG | Batch index, Leiden communities, global summaries | No (reindex) | **Reject.** MediaLoom ingests one video at a time; an LLM re-reading the whole corpus per video is the wrong cost curve |
| LightRAG | Dual-level entity/theme graph | Native | Viable for retrieval-oriented graph at ~1/10 the index cost |
| Graphiti | Bi-temporal agent memory, Neo4j | Native, real-time | Strong temporal model; adds Neo4j |
| ATOM (formerly iText2KG) | Parallel 5-tuple extraction, dual-time, cosine-based merging | Native | **Recommended for extraction.** Avoids serial entity-then-relation LLM calls; reported 93.8% latency cut vs Graphiti |

Recommendation: use **ATOM for extraction and merging**, and treat its store as an *ephemeral
working graph*. Its output is normalized into AgentLoom proposal JSON; the canonical graph stays in
`agents/knowledge-graphs/domain-*.json` behind the HITL gate. We adopt the library for the
mechanical work — tuple extraction, entity resolution, temporal reconciliation — and keep
governance ours.

Optionally add LightRAG later as a query-time index over accepted nodes. Do not run both as sources
of truth.

### 5.2 Edge families ("multi-dimensional")

One extraction pass, several typed relations, each independently reviewable:

| Dimension | Edge | Example from the DAOJIE video |
|---|---|---|
| Causal | `enables`, `causes`, `reduces_need_for` | Andromeda → audience automation |
| Evidential | `supported_by_span` | "85% of businesses" → `(t0, t1)` |
| Structural | `part_of`, `alternative_to` | CBO ↔ ABO |
| Procedural | `step_of`, `precondition_of` | consolidate → set objective → enable CBO |
| Definitional | `defined_as` | "Winning Ads Team" |
| Temporal | `valid_from`, `observed_at`, `superseded_by` | "Winning Ads 已死" is valid only post-Andromeda |

The temporal dimension is not optional for this content. A platform-behavior claim from a
2026-09 video is not a timeless fact, and a graph that cannot express "this was true as of the
observation date" will accumulate silent contradictions as the next video updates the advice. This
is precisely the bi-temporal case ATOM and Graphiti are built for: keep observation time separate
from validity period.

### 5.3 Node identity: normalized surfaces, not resolved identities

Reuse the lesson already paid for in the SESAME/JCDL work: KG nodes should be **normalized typed
entity surfaces** (case-fold, strip punctuation, collapse whitespace, first-seen type, no
coreference) and must not be described as resolved identities. Aggregating by surface can merge
homonyms, and a lexical or embedding similarity score does not settle identity. State this in the
node schema so no downstream consumer over-claims.

---

## 6. Stage 5 — Human review with embedded video and evidence seek

**Goal**: for every proposed node, the reviewer plays the exact moments it came from — plural.

### 6.1 Evidence is many-to-many and independently labelled

```json
{
  "node_id": "kg:campaign-consolidation",
  "evidence": [
    { "media_id": "EtU45PsaL1M", "t0": 337.2, "t1": 352.8, "quote": "...", "score": 0.83 },
    { "media_id": "EtU45PsaL1M", "t0": 611.0, "t1": 628.4, "quote": "...", "score": 0.71 }
  ]
}
```

Review must be able to reject **an evidence link** while keeping the node, and vice versa. That
distinction is what made the SESAME identity audit informative: label the decision, not the item.
A node that is correct with a wrong citation is still a governance failure.

### 6.2 Player integration and one live constraint

Use the YouTube **IFrame Player API** with `enablejsapi=1` and `origin`, then
`player.seekTo(seconds, true)` per evidence chip. Two facts to design around:

- **Deep-link `start=` accepts whole seconds only**, while in-app `seekTo()` accepts floats.
  Keep both: integer-second URLs for shareable links in Markdown digests, fractional API seeks for
  precise in-portal review.
- Since the **embed redesign of late March 2026**, every programmatic `seekTo()` wakes the full
  player chrome for ~4 seconds, and there is no documented flag to suppress it (`controls: 0`,
  `modestbranding`, and post-seek `pauseVideo()`/`playVideo()` all fail to). So do not build a UI
  whose usability depends on silent seeking — e.g. avoid auto-advancing through evidence chips on a
  timer. For local media files, `<video>.currentTime` has no such constraint.

### 6.3 Transcript panel

Virtualized transcript list, binary search from `currentTime` to word index for live highlighting,
click-any-word to seek. Word-level highlighting is only honest when Stage 1 produced
`timing_granularity: "word"`; fall back to utterance highlighting otherwise.

---

## 7. Stage 6 — AI takeaway with verification

**Goal**: a takeaway that separates "what the speaker said" from "what holds up", without
pretending to adjudicate the unfalsifiable.

### 7.1 Two checks that must never be conflated

| Check | Question | Method | Cost |
|---|---|---|---|
| **Groundedness** | Does our summary match the transcript? | NLI entailment, local encoders | Near-zero |
| **Veracity** | Is what the speaker said actually true? | External retrieval + citation | Search + LLM |

Failing to separate these is the classic error: a perfectly grounded summary of a wrong claim reads
as verified.

### 7.2 Groundedness: small local models, not an LLM judge

Run atomic-claim decomposition, then entailment against the cited span. The relevant benchmark
result: on RAGTruth, a dual ensemble of two small open models (HHEM-2.1-open + MiniCheck-Flan-T5-Large)
matched a frontier model as judge — **AUROC 0.844 vs 0.846 at roughly 250× lower per-call cost**.
There is no reason to pay a frontier model for this step.

Libraries worth evaluating rather than reimplementing:
- **`ragwarden`** — atomic-claim decomposition plus a cost-tiered cascade (retrieval heuristics →
  encoder NLI → uncertainty sampling → LLM judge only for the ambiguous remainder) and a policy
  engine emitting `ALLOW / REDACT_CLAIMS / RETRY / ABSTAIN / ESCALATE`. Closest fit to our gate.
- **`verifiable-rag`** — sentence-level citations to exact spans, per-claim NLI, calibrated
  refusal, and a self-contained HTML audit report per query.
- **`ragground`** — sub-millisecond deterministic tier plus quantized cross-encoder NLI, with
  inline citation injection.
- **`dokis`** — zero-LLM provenance middleware producing a `claim → chunk → source` map;
  explicitly English-oriented, which matters for our Chinese content.

Selection criterion for our case: **multilingual capability**, since most of this corpus is
Chinese. Several of these are English-oriented; measure on our own transcripts before adopting, and
be prepared to use a multilingual NLI cross-encoder directly.

### 7.3 Veracity: claim typing first, then targeted checking

Most claims in a marketing or strategy video are not checkable propositions, and a system that
labels them true/false will be confidently wrong. Type each claim before attempting verification:

| Claim type | Example | Checkable? |
|---|---|---|
| Vendor behavior | "Andromeda changed how Meta matches audiences" | Partly — against vendor docs and dated reporting |
| Mechanism | "Splitting ad sets fragments learning signal" | Partly — against documented platform mechanics |
| Personal result | "$800k spend, ROAS 6.12 across 300 accounts" | **No.** Single-operator, unaudited |
| Recommendation | "Use one campaign for 85% of businesses" | No — a judgment, not a fact |
| Quantitative rule of thumb | "~50 conversions in 14 days" | Partly — against published guidance |

Only the checkable types go to external retrieval. The verdict vocabulary is
`supported / contradicted / unsupported / unverifiable / time-sensitive` — deliberately **not**
`true / false`, and "unsupported" must never be rendered as "wrong".

### 7.4 Output shape

The takeaway is itself a **proposal**, never auto-accepted:

1. **What the video claims** — anchored, grounded, per-claim spans.
2. **What holds up** — supported claims with external citations.
3. **What to verify** — contradicted, unsupported, and time-sensitive claims, each with the reason
   it was flagged and what evidence would settle it.
4. **What is unfalsifiable** — personal results and recommendations, presented as the author's
   position rather than as knowledge.

Section 4 is what makes this trustworthy rather than sycophantic. An honest takeaway on the DAOJIE
video says the mechanism argument is consistent with documented platform behavior, and that the
ROAS 6.12 figure is a single unaudited operator claim that should not enter the knowledge graph as
a fact.

---

## 8. Governance additions

New behaviors to author under `agents/behaviors/builder/`:

| Behavior | Rule |
|---|---|
| `behavior-no-synthetic-segment-anchors` | Never present a synthesized time bucket as a chapter or anchor. If no semantic segmentation ran, emit no segment titles. |
| `behavior-evidence-required-for-node` | No candidate node, skill step, or takeaway claim without at least one `(media_id, t0, t1)` span. Ungrounded extractions are dropped, not softened. |
| `behavior-separate-groundedness-from-veracity` | Never merge "matches the transcript" with "is true". Every claim carries both verdicts or neither. |
| `behavior-claim-type-before-verdict` | Assign a claim type before any veracity verdict; unfalsifiable claims are attributed, not adjudicated. |

Also amend `.cursor/rules/distillation-governance.md`, whose anchor format
`[MM:SS Chapter Title](...?t=<seconds>)` is currently satisfied by the fake bucket labels.

---

## 9. Phasing

Each phase is independently shippable and independently useful.

| Phase | Deliverable | Depends on |
|---|---|---|
| **P0** ✅ | Kill synthetic 5-minute chapters; emit no anchor rather than a fake one. Fix the transcript cache key. | — |
| **P1** ✅ | Canonical transcript artifact (§1.1), WhisperX + engine strategy, zh normalization + glossary | P0 |
| **P2** ✅ | `agentloom-media search`: SQLite FTS5 + sqlite-vec + RRF, 45s windows, Qwen3 embeddings | P1 |
| **P3** | Semantic segmentation, typed distillation passes, LangExtract grounded terms | P1, P2 (reuses embeddings) |
| **P4** | KG expansion via ATOM into proposal JSON, six edge families, temporal fields | P3 |
| **P5** | Review portal: embedded player, evidence chips with `seekTo`, per-evidence labels | P3, P4 |
| **P6** | Verified takeaway: groundedness cascade, claim typing, external checking | P5 |

### P0 as shipped (2026-09-07)

- `distillation/anchors.py`: builds the `[MM:SS]`-marked transcript given to the LLM, and parses
  anchors as single moments. Spans and lists (`00:00-05:00`, `04:12; 09:30`) are rejected
  instead of collapsing to their first component.
- `proposals/emitter.py::AnchorGate`: validates every anchor against real segment starts (3s
  snap tolerance) before rendering. Rejects are dropped, counted, and reported in the CLI
  output and in `anchor_validation` inside the proposal JSON.
- `distillation/chapter_aligner.py`: with no author chapters, the transcript is batched for
  context only — `boundary_source: mechanical`, `title: null`, no anchor from the boundary.
  Native chapters anchor at their first real utterance, not the declared boundary.
- `acquisition/transcript_cache.py`: cache keyed on video id + source + engine + language,
  with provenance stored in the payload. Pre-existing caches load as `source: legacy` with a
  warning that their engine is unknown.
- `tests/test_anchors.py`: 10 tests, with the bad anchors from the 2026-09-07 Meta digest as
  regression cases.

Re-run on the same video (`EtU45PsaL1M`) went from 4 range-valued anchors rendered as broken
links to 17/17 anchors resolving to real transcript times.

### P1 as shipped (2026-09-08)

- `transcripts/model.py`: the canonical `Transcript` — word, utterance, and document levels,
  with `char_start`/`char_end` per utterance and `resolve_char_span()` mapping any character
  range back to a time span. That is the hook Stage 3 grounding needs. `timing_granularity`
  travels with the data and is only `word` when word timings actually exist.
- `transcripts/store.py`: `data/transcripts/<media_id>/` holding one file per engine variant
  plus a `manifest.json` recording source, engine, version, language, granularity,
  normalization settings, and a content hash. Reuse prefers precision first, then a *named*
  engine, then recency, so a migrated pre-P1 cache cannot outrank a real transcription.
- `transcripts/normalize.py` + `config/asr-glossary.txt`: OpenCC `t2s` normalization with the
  raw text always preserved, and the glossary rendered into Whisper's `initial_prompt`.
- `audio/engines.py`: engine dispatch on `ASR_ENGINE`. `whisper_api` and `faster_whisper` are
  implemented; `whisperx` and `whisper_cpp` are registered and raise with what to install and
  why they are deferred, rather than being silently absent.
- `transcripts/acquire.py`: one route implementation (`auto` / `fast` / `heavy`) shared by the
  CLI and the UI, replacing the duplicated stage-2 blocks in `cli.py` and `ui/jobs.py`.
- Provenance now reaches the reader: the digest header states the engine and what its
  timestamps are worth, and flags a migrated transcript as unverified.
- `tests/test_transcripts.py`, `tests/test_asr_engines.py`: 27 further tests.

Measured on `EtU45PsaL1M`: the transcript cached before P1 carried **1090 Traditional-form
characters** despite a Simplified source, all of which OpenCC converts. Re-transcribing with
`--language zh` plus the glossary returned **0** Traditional-form characters at the source, so
the two defenses work independently.

**Still true after P1**: `timing_granularity` is `segment`, per the v1 decision. Word-level
seeking arrives with the WhisperX adapter, which is now a single module to write rather than a
change to the pipeline.

### P2 as shipped (2026-09-08)

- `search/windows.py`: 45s windows, 15s overlap, boundaries snapped to utterance edges so a
  window never quotes half a sentence and every `t0` is usable as an anchor.
- `search/index.py`: `data/medialoom.db` with FTS5 (`tokenize='trigram'`) + `sqlite-vec`,
  fused by RRF (k=60). Re-indexing replaces an item's windows rather than accumulating stale
  ones. Mixing embedding models is refused, since vectors from different models silently fail
  to be comparable.
- `search/embed.py`: OpenAI embedder, `text-embedding-3-small` by default (a 16-minute video
  is roughly 2k tokens, so the cheaper tier costs nothing meaningful here).
- CLI: `agentloom-media index` and `agentloom-media search`; `ingest` now indexes as stage 2b.

**Three defects found by testing, all fixed.** These are recorded because each one silently
degrades retrieval rather than failing loudly:

1. **The lexical half contributed nothing to real questions.** Quoting the whole query as one
   FTS5 phrase only matches verbatim strings, so BM25 never fired on a natural-language
   question and RRF quietly degenerated to vector-only — meaning search would return nothing
   at all without an API key. Fixed by OR-ing terms and expanding long CJK runs into 3-grams
   (`build_fts_query`).
2. **Trigram FTS5 cannot match queries shorter than 3 characters**, and 2-character words
   dominate Chinese ("广告", "预算", "结构"). Fixed with a term-ranked substring fallback.
   This is the "revisit if measured recall is poor" case from §3.3; recall *was* poor, and
   the fallback closes it without taking on jieba and a vocabulary to maintain.
3. **Window construction had a coverage bug.** The builder appended the next utterance without
   checking whether it fit, so a silent gap could stretch one window across 340 seconds; the
   naive fix then let the stride skip utterances entirely, leaving them permanently
   unsearchable. Coverage is now asserted over uniform, gapped, oversized, and ragged
   transcripts.

Measured on the two DAOJIE videos (60 windows, 60 vectors): the query
`"Andromeda 之后还要不要拆 Campaign"` returns `05:13–05:57` as the top hit with
`keyword #3, vector #1`, i.e. both rankers contributing, and pulls a relevant window from the
*other* video at rank 2. The two-character query `"预算"` returns a genuinely relevant window
via the substring fallback. With `SEARCH_DISABLE_EMBEDDINGS=1` the results stay relevant and
the output labels itself `lexical only` rather than implying a hybrid ranking.

**Deviation from §3.4**: v1 uses `text-embedding-3-small` rather than
`text-embedding-3-large`. The model id is stored per row, so moving to Qwen3-Embedding-0.6B
locally, or to the large model, is a re-embed rather than a schema change.

P2 is deliberately early. A searchable video library is valuable on its own and every later stage
reuses its embeddings and windows, so it is the highest-leverage next step rather than a
prerequisite to be rushed through.

## 10. Decisions and open questions

**Decided 2026-09-07 (author):**

- **Compute route: API-first for v1.** Local ASR and local embeddings are deferred. `whisper_api`
  and the OpenAI embedder are the v1 defaults; WhisperX and Qwen3-Embedding remain the documented
  upgrade path. See the v1 notes in §2.2 and §3.4 for the two consequences accepted here
  (segment-level anchors only; transcript text leaves the machine at embedding time).
- **Plan status: under author review**, no implementation started.

**Still open:**

1. **Corpus scope** — single videos, or full course series? Series ingestion is what justifies the
   RAPTOR hierarchy in §4.2 and changes the segmentation defaults. A 16-minute single video does
   not need the tree at all.
2. **Whether Stage 6 does external retrieval in v1**, or ships as groundedness plus claim typing
   only, deferring web verification. Claim typing alone already prevents the worst failure
   (treating an unaudited personal result as a fact).
3. **Diarization** — currently proposed as opt-in and unnecessary for single-narrator channels.
   Confirm no near-term interviews or panel recordings are in scope, since `pyannote` adds a
   license acceptance and a gated model download.
4. **Groundedness library for Chinese** — `ragwarden` / `verifiable-rag` / `ragground` / `dokis`
   are all at least partly English-oriented. This needs a measurement on our own transcripts
   before one is adopted; the fallback is a multilingual NLI cross-encoder used directly.
