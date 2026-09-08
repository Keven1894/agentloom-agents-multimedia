"""Groundedness: does our claim actually match the transcript?

This is **not** a truth check. Groundedness asks only whether the transcript says what we
claim it says. Conflating it with veracity produces the classic failure — a perfectly grounded
summary of a false claim reading as verified — so the two verdicts are computed separately and
never merged (plan §7.1).

Cost-tiered cascade, in the spirit of `ragwarden`: a cheap deterministic tier decides the clear
cases, and only the ambiguous remainder escalates. Every result records which tier decided it,
so a reader can tell a lexical match from an entailment judgement.

| Tier | Method | Decides |
|---|---|---|
| `lexical` | character-n-gram containment against retrieved spans | clear matches and clear misses |
| `nli` | multilingual NLI cross-encoder, if installed | the ambiguous band |
| `llm` | LLM entailment, explicitly labelled | the remainder, only when enabled |

Candidate spans come from the P2 hybrid index, which is what lets a takeaway that carried no
anchor acquire one: retrieve the windows that could support it, then check.

**Deliberate deviation from plan §7.2.** The plan proposed HHEM-2.1-open plus
MiniCheck-Flan-T5-Large on the strength of RAGTruth results (AUROC 0.844 vs 0.846 for a
frontier judge, ~250x cheaper). Both are English-only, and this corpus is Chinese — the plan
itself set multilingual capability as the selection criterion and said to measure before
adopting. Neither is wired in. The NLI tier is a plug point for a multilingual cross-encoder
(e.g. mDeBERTa-v3-base-xnli), left uninstalled rather than pulling ~2GB of torch into the
dependency tree unasked. Until it is installed, the ambiguous band is reported as
`needs_review` rather than guessed at.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

DEFAULT_MAX_WORKERS = 8

# Containment at or above this is a match without further checking.
GROUNDED_THRESHOLD = 0.62
# Containment at or below this has nothing to entail from.
UNGROUNDED_THRESHOLD = 0.18

VERDICTS = ("grounded", "needs_review", "ungrounded")

_LATIN = re.compile(r"[A-Za-z0-9]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff]")


def _shingles(text: str) -> set:
    """Character bigrams for CJK plus lowercased Latin words.

    Word tokenization is meaningless for unsegmented Chinese, and character bigrams are a
    reasonable proxy for overlap without a segmenter. Mixed-script speech needs both.
    """
    if not text:
        return set()
    tokens = {m.group(0).casefold() for m in _LATIN.finditer(text)}
    cjk = [c for c in text if _CJK.match(c)]
    tokens |= {cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)}
    if len(cjk) == 1:
        tokens.add(cjk[0])
    return tokens


def containment(claim: str, evidence: str) -> float:
    """Fraction of the claim's shingles present in the evidence.

    Containment rather than Jaccard: evidence is usually much longer than the claim, and
    Jaccard would penalize that length rather than measuring support.
    """
    claim_shingles = _shingles(claim)
    if not claim_shingles:
        return 0.0
    evidence_shingles = _shingles(evidence)
    return len(claim_shingles & evidence_shingles) / len(claim_shingles)


NLI_SYSTEM = """You judge whether a transcript excerpt supports a claim about it.

This is NOT a truth check. Judge only whether the excerpt states the claim. A claim that is
true in the world but absent from the excerpt is "not_supported".

Return JSON: {"verdict": "supported" | "not_supported" | "contradicted",
              "reason": "<one short sentence>"}"""


def llm_entailment(claim: str, evidence: str, model: str) -> Dict[str, Any]:
    """Escalation tier. Reported as `llm`, never as NLI."""
    from agentloom_media.distillation.passes import _complete_json

    payload = _complete_json(
        NLI_SYSTEM,
        f"Excerpt:\n{evidence}\n\nClaim:\n{claim}",
        model,
        max_tokens=512,
    )
    verdict = str(payload.get("verdict") or "").strip()
    if verdict not in {"supported", "not_supported", "contradicted"}:
        verdict = "not_supported"
    return {"verdict": verdict, "reason": str(payload.get("reason") or "").strip() or None}


def load_nli_scorer() -> Optional[Callable[[str, str], float]]:
    """Return a multilingual NLI entailment scorer, or None if unavailable.

    Kept as a lookup rather than a hard dependency: installing it is a deliberate choice
    (~2GB of torch), and its absence must degrade to `needs_review`, not to a guess.
    """
    model_name = (os.environ.get("NLI_MODEL") or "").strip()
    if not model_name:
        return None
    try:
        from transformers import pipeline  # type: ignore
    except ImportError:
        return None

    try:
        classifier = pipeline("text-classification", model=model_name, top_k=None)
    except Exception:
        return None

    def score(claim: str, evidence: str) -> float:
        results = classifier(dict(text=evidence, text_pair=claim))
        rows = results[0] if isinstance(results, list) and results and isinstance(results[0], list) else results
        for row in rows:
            if str(row.get("label", "")).lower().startswith("entail"):
                return float(row.get("score", 0.0))
        return 0.0

    return score


# Windows are 45s with 15s overlap, so a 30s stride. Padding by one stride on each side
# guarantees the neighbouring windows' content is included.
DEFAULT_PAD_SECONDS = 30.0


def neighborhood(
    transcript: Any, t0: float, t1: float, pad: float = DEFAULT_PAD_SECONDS
) -> Dict[str, Any]:
    """Text of every utterance overlapping `[t0 - pad, t1 + pad]`.

    A retrieval window is a fixed-length slice, so it cuts sentences in half. One real case:
    a window ended at "投7万9" and left "能达到差不多46万的这个收益" to the next window, so no
    single window contained the whole statement and an entailment judge reading one window
    correctly said the excerpt did not state the claim. Reading the neighbourhood fixes that.

    Built from transcript utterances rather than by concatenating adjacent windows, because
    windows overlap and stitching them would repeat 15 seconds of text at every seam.
    """
    utterances = getattr(transcript, "utterances", None) or []
    low, high = t0 - pad, t1 + pad
    covered = [u for u in utterances if u.t1 > low and u.t0 < high]
    if not covered:
        return {"text": "", "t0": t0, "t1": t1}
    return {
        "text": "\n".join(u.text for u in covered),
        "t0": covered[0].t0,
        "t1": covered[-1].t1,
    }


def retrieve_candidates(
    index: Any, media_id: str, claim: str, limit: int = 4
) -> List[Dict[str, Any]]:
    """Candidate spans for a claim, from the hybrid index, restricted to this media item."""
    if index is None:
        return []
    try:
        # `search` filters by media server-side and returns (hits, ranker_info).
        hits, _info = index.search(claim, limit=limit, media_id=media_id)
    except Exception:
        return []
    return [
        {
            "window_id": getattr(hit, "window_id", None),
            "media_id": getattr(hit, "media_id", media_id),
            "t0": getattr(hit, "t0", None),
            "t1": getattr(hit, "t1", None),
            "text": getattr(hit, "text", "") or "",
        }
        for hit in hits
    ]


def assess_claim(
    claim: str,
    media_id: str,
    index: Any = None,
    nli_scorer: Optional[Callable[[str, str], float]] = None,
    llm_model: Optional[str] = None,
    known_evidence: Optional[Sequence[Dict[str, Any]]] = None,
    transcript: Any = None,
) -> Dict[str, Any]:
    """Assess one claim's groundedness, returning the verdict, tier, and best span.

    `known_evidence` short-circuits retrieval for claims that already carry a verified span
    from P3/P4 — those passed a stricter gate (quote located verbatim in the transcript) than
    anything measured here.
    """
    if known_evidence:
        best = known_evidence[0]
        return {
            "verdict": "grounded",
            "tier": "verbatim_span",
            "score": 1.0,
            "evidence": best,
            "reason": (
                "Carries a span whose quote was located verbatim in the transcript by the "
                "extraction gate."
            ),
        }

    candidates = retrieve_candidates(index, media_id, claim)
    if not candidates:
        return {
            "verdict": "ungrounded",
            "tier": "retrieval",
            "score": 0.0,
            "evidence": None,
            "reason": "No transcript span could be retrieved for this claim.",
        }

    scored: List[Tuple[float, Dict[str, Any]]] = sorted(
        ((containment(claim, c.get("text", "")), c) for c in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    score, best = scored[0]

    # A tight window is the right test for a clear match: high overlap inside 45 seconds is
    # strong evidence, whereas the same words scattered across a wider span may be
    # coincidence. So "grounded" is decided on the window alone.
    if score >= GROUNDED_THRESHOLD:
        return {
            "verdict": "grounded",
            "tier": "lexical",
            "score": round(score, 4),
            "evidence": best,
            "reason": f"{round(score * 100)}% of the claim's n-grams appear in this span.",
        }

    # A clear miss is judged on the neighbourhood instead, because window truncation can
    # strand half a sentence next door and drop the window score far enough to hard-reject a
    # claim the transcript does state.
    context = neighborhood(transcript, best["t0"], best.get("t1") or best["t0"])
    context_score = containment(claim, context["text"]) if context["text"] else score

    if max(score, context_score) <= UNGROUNDED_THRESHOLD:
        return {
            "verdict": "ungrounded",
            "tier": "lexical",
            "score": round(score, 4),
            "evidence": best,
            "reason": (
                f"Only {round(max(score, context_score) * 100)}% of the claim's n-grams "
                "appear anywhere near the best retrieved span; there is nothing here to "
                "entail from."
            ),
        }

    # Ambiguous band: lexical overlap cannot settle it. The judge reads the neighbourhood,
    # but the citation stays the tight window so the anchor a reader clicks is still precise.
    reading = context["text"] or best.get("text", "")
    padded = bool(context["text"]) and context["text"] != best.get("text", "")

    if nli_scorer is not None:
        entailment = nli_scorer(claim, reading)
        return {
            "verdict": "grounded" if entailment >= 0.5 else "ungrounded",
            "tier": "nli",
            "score": round(float(entailment), 4),
            "evidence": best,
            "context_span": [context["t0"], context["t1"]] if padded else None,
            "reason": f"Multilingual NLI entailment probability {entailment:.2f}.",
        }

    if llm_model:
        judged = llm_entailment(claim, reading, llm_model)
        return {
            "verdict": "grounded" if judged["verdict"] == "supported" else "ungrounded",
            "tier": "llm",
            "score": None,
            "evidence": best,
            "context_span": [context["t0"], context["t1"]] if padded else None,
            "reason": judged["reason"]
            or f"LLM entailment judged the excerpt {judged['verdict']}.",
        }

    return {
        "verdict": "needs_review",
        "tier": "lexical",
        "score": round(score, 4),
        "evidence": best,
        "reason": (
            f"Lexical overlap {round(score * 100)}% falls in the ambiguous band and no "
            "entailment model is configured, so groundedness is undecided rather than guessed."
        ),
    }


def assess_claims(
    claims: Sequence[Dict[str, Any]],
    media_id: str,
    index: Any = None,
    llm_model: Optional[str] = None,
    use_nli: bool = True,
    max_workers: int = DEFAULT_MAX_WORKERS,
    transcript: Any = None,
) -> Dict[str, Any]:
    """Assess a batch of claims. Returns claims annotated with `groundedness` plus a report.

    Two phases, so the escalation actually behaves like a cascade. The cheap tiers run over
    everything first; only the leftover ambiguous band goes to the expensive tier, and it goes
    in parallel because that band is large for abstractive claims — a takeaway is a rewording
    of the transcript, so character overlap is structurally weak evidence about it and defers
    most of the work here.
    """
    nli_scorer = load_nli_scorer() if use_nli else None
    annotated: List[Dict[str, Any]] = []

    for claim in claims:
        enriched = dict(claim)
        enriched["groundedness"] = assess_claim(
            claim["claim"],
            media_id,
            index=index,
            nli_scorer=nli_scorer,
            # Escalation is deferred to the parallel phase below.
            llm_model=None,
            known_evidence=claim.get("evidence"),
            transcript=transcript,
        )
        annotated.append(enriched)

    escalated = 0
    if llm_model:
        pending = [
            item
            for item in annotated
            if item["groundedness"]["verdict"] == "needs_review"
            and (item["groundedness"].get("evidence") or {}).get("text")
        ]
        escalated = len(pending)

        def judge(item: Dict[str, Any]) -> None:
            assessment = item["groundedness"]
            evidence = assessment["evidence"]
            # Same split as the single-claim path: read the neighbourhood, cite the window.
            context = neighborhood(
                transcript, evidence["t0"], evidence.get("t1") or evidence["t0"]
            )
            reading = context["text"] or evidence["text"]
            padded = bool(context["text"]) and context["text"] != evidence["text"]
            try:
                judged = llm_entailment(item["claim"], reading, llm_model)
            except Exception as exc:
                # A failed judgement leaves the claim undecided. It must not silently
                # become "grounded".
                assessment["reason"] = (
                    f"{assessment['reason']} Escalation failed: {exc}"
                )
                return
            item["groundedness"] = {
                "verdict": "grounded" if judged["verdict"] == "supported" else "ungrounded",
                "tier": "llm",
                "score": None,
                "evidence": evidence,
                "context_span": [context["t0"], context["t1"]] if padded else None,
                "reason": judged["reason"]
                or f"LLM entailment judged the excerpt {judged['verdict']}.",
                "lexical_score": assessment.get("score"),
            }

        if pending:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                list(pool.map(judge, pending))

    tiers: Dict[str, int] = {}
    verdicts: Dict[str, int] = {}
    for item in annotated:
        assessment = item["groundedness"]
        tiers[assessment["tier"]] = tiers.get(assessment["tier"], 0) + 1
        verdicts[assessment["verdict"]] = verdicts.get(assessment["verdict"], 0) + 1

    return {
        "claims": annotated,
        "report": {
            "verdicts": verdicts,
            "tiers": tiers,
            "escalated": escalated,
            "nli_available": nli_scorer is not None,
            "note": (
                "Groundedness measures agreement with the transcript only. It says nothing "
                "about whether the speaker is correct."
            ),
        },
    }
