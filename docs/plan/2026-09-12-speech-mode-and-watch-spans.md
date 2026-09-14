# Speech mode: mark demos so the client watches, do not look them up

**Date**: 2026-09-12
**Status**: ✅ Implemented locally (2026-09-12) — speech_mode pass + Overview badges; no vision, no lookup
**Trigger**: The Overview reads as all-conceptual even when the creator is pointing at Ads
Manager. Distillation kept the thesis and dropped the deixis (`来看一看这个账户`,
`大家可以看到Lifetime`).
**Related**: Overview surface is `docs/plan/2026-09-12-review-surfaces-and-client-overview.md`.
World-knowledge lookup stays with P6 veracity — not this pass.

---

## 0. Two jobs, do not merge them

| Question | This pass | Not this pass |
|---|---|---|
| Is she walking a screen, reading numbers off it, lecturing, or pitching? | Tag the span. Send the client to watch. | Do not ask an LLM to invent what the dashboard showed. |
| Is Andromeda / “14 days 50 conversions” a public Meta fact? | Leave it. | Later veracity / retrieval. Never used to fill in her account. |

No vision / OCR in this slice. No web search. A later pass may OCR *only* the
already-tagged walkthrough spans.

---

## 1. What we write on each segment

```
speech_mode: lecture | walkthrough | reading_screen | promo | mixed
watch: bool          # derived: walkthrough | reading_screen | mixed, or any watch_span
watch_reason: str    # one sentence, why a client should (or need not) watch
watch_spans: [{t0, t1, reason}]   # demo islands inside a mixed/lecture segment
```

- Whole-segment demo: `watch=true`, `watch_spans` empty (the section times are enough).
- Mixed: `speech_mode=mixed`, `watch_spans` are the moments to seek.
- Metaphor is lecture: “让 Meta 看得见所有广告” is not a demo.

Times must land inside the segment. Unresolvable spans are dropped.

---

## 2. How we decide (transcript only)

1. Cheap deixis scan (Chinese/English point-at-screen cues) as hints.
2. One LLM JSON pass over the titled segments + timed transcript (cheap distillation model).
3. If the LLM fails, keep the heuristic: any real deixis → `walkthrough` + watch.

Do not add this into `summarize_segment`. That pass writes claims; this pass writes
*how they were shown*.

---

## 3. Surfaces

- **Proposal JSON** — source of truth on `segments[]`.
- **Overview** — badge `建议看` / `口播` / `宣传`; watch-reason line; seek chips for
  `watch_spans`. A short “worth watching” list at the top.
- **Raw digest** — one `**建议看**` line under the section header. No new dump.
- **Evidence workbench** — unchanged this slice.
- **CLI** — `agentloom-media tag-speech` backfills an existing proposal without re-ingest.

---

## 4. Execution

- [x] `speech_mode` module + Segment fields + pipeline pass
- [x] Overview API/UI badges and seek chips
- [x] `tag-speech` CLI; run it on the kept DAOJIE proposal
- [x] Tests: deixis vs metaphor; span clipping; mocked LLM; overview reshape
- [ ] Vision OCR and public-fact lookup: later plans, not this one
