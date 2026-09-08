# behavior:builder:evidence-required-for-node: Grounded Extraction & Typed Passes

**Category**: Governance Helix  
**Role**: role-builder  
**Status**: Enforced  

## Rule

1. **Evidence Or Nothing**: No candidate node, term, skill step, or takeaway is emitted without evidence tying it to a real moment in the source. Ungrounded extractions are **dropped, not softened** — no hedging adverb substitutes for a span.
2. **Quotes Are Located, Not Trusted**: Every extracted term carries a quote that must be found in the canonical transcript by substring search (whitespace-insensitive, since ASR inserts spaces a model will not reproduce). The located span is resolved to `(t0, t1)` through the transcript character map. A quote that cannot be located drops its extraction (`distillation/terms.py`).
3. **A Quote Must Support Its Own Term**: Locating the quote is not sufficient; the quote must also contain the term it is offered as evidence for. Observed failure: "Maximized Conversion Value" was grounded by a real quote reading "Maximized Conversion *Volume*". A term glossed with an acronym or translation is supported when either surface appears.
4. **Segment Boundaries Declare Their Origin**: Segments record `boundary_source` as `native`, `native+semantic`, `semantic`, or `unsegmented`, and the digest states the method. Boundary resolution equals the embedding chunk size, so a boundary is accurate to within one chunk and must not be presented as exact. When no embedder is available, the transcript is left as one span rather than cut on invented boundaries.
5. **Passes Fail Independently**: Distillation runs as typed passes with separate schemas (`distillation/passes.py`). A failed pass degrades its own field and records itself in the `passes` report; it never voids the others. The digest prints the failed passes, so a missing section is distinguishable from a source that had nothing to say.
6. **Skill Extraction Is Gated**: A candidate skill is only extracted when the content is actually procedural, decided by a deterministic step-cue detector before any LLM call. Commentary produces no skill, and the decision plus its score is recorded. Prose must never be emitted inside a code fence.
7. **Points Are Ordered As Spoken**: Key points within a segment are sorted by anchor time, because a reviewer follows the recording forwards.

## Rationale

The purpose of grounding is to bound how much a reviewer must trust the model. A claim with a
verified span can be checked in seconds; a claim without one requires re-watching the source,
which is the work the pipeline was supposed to remove. So the gates here are structural rather
than prompt-level: the model is asked to quote, and the quote is then independently located and
matched against its own term. A prompt instruction the model may ignore is not a control.

The same logic applies to boundaries and passes. A segment title over an invented boundary, or
an empty section that looks like an editorial judgement but was actually an API error, both
transfer the model's uncertainty onto the reviewer without telling them. Declaring the method
and reporting the failures keeps that uncertainty visible where it can be acted on.
