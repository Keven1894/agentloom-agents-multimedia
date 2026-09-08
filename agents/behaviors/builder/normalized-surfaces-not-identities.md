# behavior:builder:normalized-surfaces-not-identities: KG Node Identity & Edge Grounding

**Category**: Governance Helix  
**Role**: role-builder  
**Status**: Enforced  

## Rule

1. **Surfaces, Not Identities**: A KG node is a normalized typed *surface* — NFKC-folded, case-folded, punctuation-stripped, whitespace-collapsed — carrying the type it was first seen with. It is not a resolved entity. No downstream consumer may describe these nodes as coreference-resolved, deduplicated entities, and the serialized node states `identity_claim: "none"` so the limitation travels with the data.
2. **Aggregate On Exact Match Only**: Surfaces combine only when their normalized forms are identical. No stemming, no synonym mapping, no transliteration — each would assert an identity the surface does not establish. Every distinct spelling is retained in `surface_forms` so a reviewer can audit what the aggregation merged.
3. **Similarity Suggests, Humans Merge**: Embedding similarity produces `merge_candidates` with `decision: pending_human_review`. It never merges. An unmerged duplicate is visible and fixable; a wrong merge is silent and destroys the losing surface. Both surfaces stay in the graph until a human decides.
4. **Closed Predicate Set**: Predicates come from the six families (causal, structural, procedural, definitional, temporal, evidential). An unrecognized predicate is dropped — invented relation types make the families unreviewable. Evidential edges are constructed by the grounding gate and are never accepted as model output.
5. **A Quote Must Mention Both Endpoints**: An edge is grounded only if its quote is locatable in the transcript *and* contains both endpoint surfaces. Observed failure: `CBO alternative_to ABO` — a correct relation — was grounded by a real quote about the speaker's own ad account, mentioning neither endpoint. A link that sends a reviewer to a moment that does not state the relation spends their trust.
6. **Endpoints Are Atomic**: A node surface must be short and clause-free. Clauses cannot be merged across videos and are unreadable as graph labels; edges with clause endpoints are dropped.
7. **Three Time Axes, Never Collapsed**: Every edge records media time (`evidence.t0/t1`, where it is said), observation time (`observed_at`, the publish date), and validity (`valid_from` / `valid_until`, only when the speaker states it). Validity is never inferred from the observation date. When no validity is stated, the digest says so — silence would read as "timeless", and a platform-behaviour claim is not.

## Rationale

The temporal rule is what keeps the graph from accumulating silent contradictions. Advice about
an advertising platform is true relative to a platform version and a date; the next video will
update it. A graph that cannot say "asserted on 2026-09-06, no validity period given" will
present two contradictory claims as equally current and force the reviewer to rediscover which
is newer.

The identity rule is a lesson already paid for in the SESAME/JCDL work. Surface aggregation is
cheap, deterministic, and auditable; identity resolution is none of those. Presenting the
former as the latter creates a graph that looks authoritative and quietly merges homonyms, and
the over-claim is in the *description*, not the data — which is why the constraint is written
into the node schema rather than left to documentation.
