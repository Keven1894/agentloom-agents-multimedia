# One review path, and Data & Digests as the client overview

**Date**: 2026-09-12
**Status**: ✅ Implemented locally (2026-09-12) — Overview tab + review queue; no new LLM pass
**Trigger**: The portal currently has three places that look like "review". A client
should read an overview first, then audit claims against the video — not approve the
same object in three UIs.
**Related**: P3 already writes `document.executive_summary` / `thesis` / segment
`analysis`; P5 writes evidence decisions to `reviews/`. Cloud tenancy is
`docs/plan/2026-09-12-cloud-mvp-accounts-quota-and-storage.md`.

---

## 0. What is wrong today

Two generations of UI are stacked:

| Surface | Intended job | What it actually writes |
|---|---|---|
| **Data & Digests** | Read the Track 2 markdown | Nothing. But it dumps the whole operator digest (KG, verified takeaway, provenance). |
| **Human Review Portal** | Whole-proposal Approve & Canonize (v0.1 AgentLoom gate) | Proposal status + move old `candidate_kg_nodes` / skills into canonical files. **Does not read** `reviews/*.json`. |
| **`/review/<proposal>.json`** (Ingest → Review, or "Review Evidence") | Per-claim / per-citation accept against the video | `reviews/review-*.json` |

Typed proposals no longer populate `candidate_kg_nodes` / `chapter_summaries` the way the
Human Review cards expect, so that tab looks empty *and* still offers Approve. The two
approve buttons do not share a ledger.

For the learning-platform MVP (one person, one video) this is one workflow wearing three
costumes.

---

## 1. Target UX

```
Ingest (or open a finished job)
    → Client overview   (what the video said, in order)
    → Evidence review   (the only place accept / reject exists)
```

- **Overview first.** Before a client jumps into 200 claims they need: one whole-video
  summary, then a sectioned summary they can skim with timestamps.
- **One approve surface.** Evidence review is the only writer of verdicts.
- **Canonize is later.** Promoting into AgentLoom canonical memory is a separate, explicit
  step — not a green button next to "Review Evidence".

---

## 2. Data & Digests = client overview (not a second review)

The LLM work for this **already exists**. Do not add a fourth summarizer.

P3 already emits, on every typed ingest:

- Whole video: `document.executive_summary`, `document.thesis`, `document.structure`,
  `document.takeaways`
- Per segment: `title`, `[t0, t1]`, `analysis`, plus dated key points

The digest markdown already renders these as `## Executive Summary` and
`## Segment Breakdown`. The Data tab's failure is presentation: it shows the entire
operator file (terms, KG, takeaway cascade, skill, provenance).

### Do this

- [x] New overview view (keep the tab, rename it — e.g. **Overview** or **Summary**)
- [x] Header: title, channel, duration, link to source video
- [x] Block 1 — **整片概述**: executive summary + thesis (and structure if it helps orientation)
- [x] Block 2 — **分段概述**: each segment as a card: title, `t0–t1`, 2–4 sentence `analysis`,
      click seeks the player *or* deep-links YouTube. Key points stay collapsed; they belong
      to evidence review
- [x] Primary CTA: **Review this video** → `/review/<proposal>.json`
- [x] Hide from this view: knowledge graph, verified-takeaway sections, grounded-term dump,
      skill candidate, pipeline provenance (move to an "Operator / raw digest" disclosure)

### Only add an LLM pass if

The existing `analysis` / executive summary are too long or too "operator" for a client.
Then add one **overview rewrite** pass that takes those fields and returns shorter
client-facing copy — still grounded on the existing text, not a new read of the transcript.
Do not re-summarize the raw transcript from scratch.

---

## 3. Collapse the other two "review" doors

### Ingest tab

- [x] Keep job history
- [x] Actions: **Overview** (Data tab for that media) and **Review** (evidence workbench)
- [x] Remove any wording that sounds like a third approve

### Human Review Portal

- [x] v1: turn it into a **queue of videos to review** — title, pending evidence counts
      from `reviews/`, button that opens the evidence workbench. No Approve & Canonize
- [x] Hide or delete the old whole-proposal approve/reject buttons
- [x] Leave `/api/proposals/.../review` in the server but unused, until we really want
      AgentLoom canonize (it still talks to `candidate_kg_nodes`, which typed emit no longer fills)

### Evidence workbench (`review.html`)

- [x] Unchanged as the only accept / reject / unsure writer
- [x] Add a link back to the client overview for that video
- [ ] Optional later: show the segment `analysis` as a sticky blurb so the reviewer
      remembers "this stretch is about X" while they listen

---

## 4. Execution order (when we pick this up)

1. Overview view on Data tab, fed by existing `document` + `segments` (no new LLM).
2. Strip Approve & Canonize from Human Review; make that tab a queue.
3. Align Ingest actions with Overview + Review only.
4. Measure whether clients still bounce — only then add a rewrite pass.
5. Canonize-to-AgentLoom as its own plan, reading `reviews/*.json` rather than the v0.1
   candidate lists.

Local file layout does not change. This is a UI and copy-contract change, not a storage
migration.
