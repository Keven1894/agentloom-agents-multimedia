# Two verdicts, never one

**Track:** Behaviors · **Owner:** builder · **Status:** active · **Phase:** P6

## Rule

Every claim carries two independent verdicts, and they are never merged into a single
"verified" flag.

- **Groundedness** — does the transcript say this? Answered against retrieved spans.
- **Veracity** — is it true? Answered against external sources, or not at all.

A claim can be perfectly grounded and completely wrong. Merging the two produces the failure
this rule exists to prevent: a faithful summary of a false claim, rendered as verified.

## Type before you judge

Most claims in a strategy or marketing video are not checkable propositions. Type each claim
first; only three of the five types ever reach external retrieval.

| Type | Checkable | Why |
|---|---|---|
| `vendor_behavior` | partly | against vendor docs and dated reporting |
| `mechanism` | partly | against documented platform mechanics |
| `quantitative_rule` | partly | against published guidance |
| `personal_result` | **no** | single-operator and unaudited |
| `recommendation` | **no** | a judgment, not a fact |

An unrecognized type falls back to `recommendation`, because the conservative error is routing
a claim *away* from a verdict we cannot justify.

A figure the speaker reports about their own account is `personal_result` however precise it
sounds. `$800k spend, ROAS 6.12` does not become checkable by being specific, and it must not
enter the knowledge graph as fact.

## Verdict vocabulary

`supported` · `contradicted` · `unsupported` · `unverifiable` · `time-sensitive`

Deliberately **not** true/false. Three distinctions are load-bearing:

- **`unsupported` is never rendered as "wrong".** It means no source addressed the claim.
- **`unverifiable` is not `unsupported`.** It means we did not check. Collapsing them turns
  "we didn't look" into "we looked and found nothing".
- **A supported platform-behavior claim becomes `time-sensitive`.** It was true on some date,
  and the platform can change under us.

## Cost-tiered cascade, honestly labelled

Groundedness runs cheap tiers first and escalates only the ambiguous remainder. Every verdict
records which tier decided it, so a reader can tell an n-gram match from an entailment
judgement:

`verbatim_span` → `lexical` → `nli` → `llm`

Two constraints on the cascade:

- **An unavailable tier degrades to undecided, never to a guess.** With no entailment model
  configured, the ambiguous band is reported `needs_review`.
- **A failed escalation leaves the claim undecided.** A judge that errors out must not let a
  claim fall through to `grounded`.

## Read the neighborhood, cite the window

A retrieval window is a fixed-length slice, so it cuts sentences in half. One measured case:
a 45-second window ended at `投7万9` and stranded `能达到差不多46万的这个收益 看我们这个亚洲地区的`
in the next window, so no single window contained the whole statement. The judge reading one
window correctly said the excerpt did not state the claim — and three true claims were
rejected, with the market attribution appearing to be invented when it was simply next door.

So the reading unit and the citation unit are different:

- The **judge reads a neighborhood** — the window padded by one window stride on each side,
  built from transcript utterances rather than stitched from overlapping windows, which would
  repeat text at every seam.
- The **citation stays the tight window**, so the anchor a reviewer clicks is still precise.

The lexical screen keeps both units for different jobs. A **clear match** is decided on the
tight window, because high overlap inside 45 seconds is strong evidence while the same words
scattered over 105 seconds may be coincidence. A **clear miss** is decided on the
neighborhood, so truncation cannot hard-reject a claim the transcript does state before the
judge ever sees it.

## No link without support

Only a grounded claim gets a timestamp. The span behind an ungrounded claim is the candidate
the pipeline just rejected; rendering it as a citation would repeat the P0 mistake — a link
that looks authoritative while supporting nothing. The same applies to review targets: an
ungrounded claim is offered for review with no evidence, so no reviewer can accept a citation
the pipeline already refused.

## The takeaway is a proposal

`status: pending_human_review`, always. Section 4 — what is unfalsifiable — is what makes the
output trustworthy rather than sycophantic. An honest takeaway says the mechanism argument is
consistent with documented platform behavior *and* that the ROAS figure is one unaudited
operator's claim.
