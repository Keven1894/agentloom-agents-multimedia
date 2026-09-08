# ATOM (iText2KG) rejected as the KG extraction engine

**Date**: 2026-09-08
**Status**: Decided
**Supersedes**: plan §5.1, which recommended ATOM for extraction and merging

## Decision

Do not adopt `itext2kg` / ATOM. Implement typed tuple extraction in `agentloom_media/kg/`,
adopting ATOM's *design* — parallel per-unit extraction, bi-temporal fields, cosine-based
merge suggestion — over our own grounded structures.

The plan recommended ATOM on the strength of its incremental model and reported 93.8% latency
reduction versus Graphiti. Those claims are not in dispute. The library was installed
(`itext2kg` 1.1.0) and inspected, and two properties make it unusable for *this* corpus under
*our* governance rules.

## Evidence

### 1. Chinese entity types are destroyed

`itext2kg/atom/models/entity.py` normalizes labels with an ASCII-only character class:

```python
LABEL_PATTERN = re.compile(r'[^a-zA-Z0-9]+')  # For cleaning labels
self.label = LABEL_PATTERN.sub("_", self.label).replace("&", "and").lower()
```

Measured:

| Input label | Input name | Resulting label |
|---|---|---|
| `Methodology` | `Campaign Structure` | `methodology` |
| `营销方法` | `广告结构` | `_` |
| `平台特性` | `Andromeda` | `_` |

Every Chinese label collapses to `_`. Because `Entity.__eq__` and `__hash__` are defined over
`(name, label)`, the type axis silently disappears for Chinese-typed entities — and the six
edge families in plan §5.2 are typed relations over typed nodes. The corpus this pipeline was
built for is Chinese-primary.

### 2. Tuples cannot carry evidence spans

ATOM's flow is `extract_atomic_facts` → `extract_quintuples(atomic_facts)`. The first step
rewrites source text into "decontextualized" atomic factoids, explicitly instructing the model
to replace pronouns and rephrase. Tuples are then extracted *from those paraphrases*, so every
tuple is two LLM generations away from the source, and `Entity` carries only `label`, `name`,
and an embedding — no character offsets, no source location. The best provenance available is
`add_atomic_facts_to_relationships`, which attaches the paraphrase.

This conflicts with `behavior-evidence-required-for-node`: no node or edge without a
`(media_id, t0, t1)` span. The conflict is structural, not a missing feature — the
decontextualization step is designed to discard the surface form we ground against.

### 3. Its temporal model has the wrong axes for a recording

`t_start` / `t_end` are calendar strings (`'18-06-2024'`) describing when a relationship holds.
That is one useful axis, but a claim extracted from a recording needs three:

| Axis | Meaning | ATOM |
|---|---|---|
| Media time | where in the recording it is said, `(t0, t1)` | absent |
| Observation time | when the claim was made — the publish date | conflated into `t_start` |
| Validity time | the period the claim is asserted to hold | `t_start` / `t_end` |

Media time is what makes a claim checkable in bounded time, and it is the axis ATOM lacks.

### 4. Secondary costs

`langchain`, `langchain-experimental`, `langchain-openai`, `neo4j`, `scikit-learn`, `pymupdf`,
`pypdf`, `openpyxl`, and a `numpy<2.0.0` pin, for a store the plan already designated ephemeral.
A second model-provider configuration path alongside the existing one. English-centric
extraction prompts throughout. None of these would be decisive alone; they remove any
reason to work around 1–3.

## What we take from ATOM

The paper's design is sound and is adopted:

- **Per-unit parallel extraction** rather than serial entity-then-relation passes. We extract
  typed tuples per segment, so segments are independent and one failure is contained.
- **Bi-temporal separation** of observation from validity, extended with the media-time axis.
- **Cosine similarity for merge *suggestion***. Our surfaces merge on exact normalized match
  only; embedding similarity produces `merge_candidates` for a human, never an automatic merge
  (see `behavior-normalized-surfaces-not-identities`).

## Revisit if

ATOM gains span-preserving extraction (offsets surviving factoid rewriting), or its label
normalization becomes Unicode-aware. Either alone is insufficient; both would make it viable.
LightRAG as a query-time index over *accepted* nodes remains open and is unaffected.
