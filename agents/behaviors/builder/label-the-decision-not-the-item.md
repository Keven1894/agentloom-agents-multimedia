# behavior:builder:label-the-decision-not-the-item: Evidence-Level Review

**Category**: Governance Helix  
**Role**: role-builder  
**Status**: Enforced  

## Rule

1. **An Evidence Link Is Reviewable On Its Own**: Every citation is a decision target independent of the claim it supports. Rejecting a citation must not reject the claim, and accepting a claim must not launder its citations. Decisions are keyed by `(kind, target_id, evidence_index)`, where a null index means the claim itself.
2. **Split Verdicts Are Reported, Not Reconciled**: A claim accepted while one of its citations is rejected is a valid and important state — a correct claim with a wrong citation is still a governance failure. The review summary surfaces these as `accepted_with_rejected_evidence` rather than resolving them silently.
3. **The Log Is Append-Only**: A changed verdict appends; it never overwrites. The audit trail is the record of what a reviewer thought and when, not only the final state. `current_decisions` collapses the log for display.
4. **Closed Verdict Vocabulary**: `accept`, `reject`, `unsure`. An unknown verdict or target kind is rejected at the API boundary — a typo'd verdict that persisted would corrupt the audit trail it exists to provide.
5. **Ungrounded Claims Are Shown, Not Hidden**: A target with no evidence span is listed and marked as uncheckable. Omitting it would misrepresent how much of a proposal can actually be verified. Coverage is reported per kind.
6. **Target Ids Come From Content**: Ids are content hashes, not list positions, so re-running a distillation and re-reviewing cannot reattach an old verdict to a different claim.
7. **Honest Highlighting**: Word-level transcript highlighting is only offered when Stage 1 produced `timing_granularity: "word"`. Otherwise the panel highlights whole utterances and says so.
8. **No Seek-Driven Automation**: Since the embed redesign of late March 2026, every programmatic `seekTo()` wakes the full player chrome for roughly four seconds, with no documented flag to suppress it. No UI may depend on silent seeking — in particular, no timer-driven auto-advance through evidence chips.

## Rationale

Review is where the pipeline's claims meet a human's limited attention, so the design question
is what a reviewer can decide in bounded time. Playing the exact moment a claim came from is a
few seconds of work; re-watching a sixteen-minute video is not. That is why an unciteable claim
is marked rather than dropped: the reviewer needs to know which claims they are being asked to
take on trust.

Separating the citation verdict from the claim verdict is what makes the resulting record
informative rather than a rubber stamp. An all-or-nothing approve button collapses "this is
true and well-sourced" and "this is true but the citation is wrong" into the same signal, and
the second case is the one that predicts future extraction failures. This is the same lesson as
the SESAME identity audit: the value was in labelling what kind of error each case was, not in
counting how many passed.
