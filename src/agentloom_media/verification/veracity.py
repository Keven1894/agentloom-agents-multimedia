"""Veracity: is what the speaker said actually true?

Separate from groundedness by construction. Groundedness asks whether the transcript says what
we claim; veracity asks whether the speaker is right. A grounded summary of a false claim is
the classic failure this split exists to prevent (plan §7.1).

Two rules shape the vocabulary:

- The verdicts are `supported / contradicted / unsupported / unverifiable / time-sensitive`,
  deliberately **not** true/false.
- **`unsupported` must never be rendered as "wrong".** It means we found nothing either way.
  `unverifiable` means we could not check at all, which is a different statement again — and
  collapsing the two would turn "we didn't look" into "we looked and found nothing".

Only claims typed as checkable reach retrieval. Personal results and recommendations are
returned as the author's position, unadjudicated, because a system that grades a single
operator's unaudited ROAS figure as "supported" has laundered it into knowledge.

**Deliberate gap.** No external retrieval provider is configured in this environment, so every
checkable claim currently returns `unverifiable` with that reason attached. The alternative —
asking an LLM from memory — is precisely the "confidently wrong" mode §7.3 warns about, and
would produce verdicts with no citation behind them. The provider interface below is the plug
point; `SEARCH_PROVIDER` plus a key enables it.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from agentloom_media.verification.claims import CHECKABLE_TYPES

VERDICTS = (
    "supported",
    "contradicted",
    "unsupported",
    "unverifiable",
    "time-sensitive",
)

# Never present these as adjudicable.
UNFALSIFIABLE_VERDICT = "unverifiable"

# Claim types whose subject can change under us: a platform-behaviour claim true on the
# observation date may be false now, so a bare "supported" would overstate it.
TIME_SENSITIVE_TYPES = {"vendor_behavior", "quantitative_rule"}

RetrievalProvider = Callable[[str], List[Dict[str, Any]]]


def resolve_provider() -> Optional[RetrievalProvider]:
    """Return a web retrieval callable, or None when none is configured.

    None is a first-class outcome. It must produce `unverifiable`, not a guess.
    """
    provider = (os.environ.get("SEARCH_PROVIDER") or "").strip().lower()
    if not provider:
        return None

    if provider == "tavily":
        key = os.environ.get("TAVILY_API_KEY")
        if not key:
            return None

        def tavily(query: str) -> List[Dict[str, Any]]:
            import urllib.request
            import json as _json

            request = urllib.request.Request(
                "https://api.tavily.com/search",
                data=_json.dumps(
                    {"api_key": key, "query": query, "max_results": 5}
                ).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = _json.loads(response.read().decode("utf-8"))
            return [
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "snippet": item.get("content"),
                }
                for item in payload.get("results") or []
            ]

        return tavily

    return None


VERACITY_SYSTEM = """You assess a claim against retrieved external sources.

Return JSON:
{"verdict": "supported" | "contradicted" | "unsupported",
 "reason": "<one or two sentences>",
 "citations": [{"url": "...", "title": "..."}],
 "what_would_settle_it": "<what evidence would decide this, or null>"}

Rules:
- "supported" requires a source that states the claim. Cite it.
- "contradicted" requires a source that states the opposite. Cite it.
- "unsupported" means the sources say nothing either way. This is NOT the same as false, and
  you must not phrase it as though the claim were wrong.
- Never rely on your own memory. If the sources do not address the claim, say "unsupported"."""


def check_claim(
    claim: Dict[str, Any],
    provider: Optional[RetrievalProvider],
    model: Optional[str] = None,
    observed_at: Optional[str] = None,
) -> Dict[str, Any]:
    """Assess one claim's veracity. Never returns true/false."""
    claim_type = claim.get("type", "recommendation")

    if claim_type not in CHECKABLE_TYPES:
        return {
            "verdict": UNFALSIFIABLE_VERDICT,
            "checked": False,
            "reason": (
                "Personal results and recommendations are the author's position, not "
                "propositions that external sources can settle."
                if claim_type in {"personal_result", "recommendation"}
                else "This claim type is not externally checkable."
            ),
            "citations": [],
            "what_would_settle_it": (
                "An independently audited account would be required."
                if claim_type == "personal_result"
                else None
            ),
        }

    if provider is None:
        return {
            "verdict": "unverifiable",
            "checked": False,
            "reason": (
                "No external retrieval provider is configured, so this claim was not "
                "checked. This is not evidence for or against it."
            ),
            "citations": [],
            "what_would_settle_it": (
                "Vendor documentation or dated reporting covering this behaviour."
            ),
        }

    from agentloom_media.distillation.passes import _complete_json
    from agentloom_media.distillation.distiller import resolve_distillation_model

    try:
        results = provider(claim["claim"])
    except Exception as exc:
        return {
            "verdict": "unverifiable",
            "checked": False,
            "reason": f"External retrieval failed: {exc}",
            "citations": [],
            "what_would_settle_it": None,
        }

    if not results:
        return {
            "verdict": "unsupported",
            "checked": True,
            "reason": "Retrieval returned no sources addressing this claim.",
            "citations": [],
            "what_would_settle_it": "A source that discusses this behaviour directly.",
        }

    sources = "\n\n".join(
        f"[{i}] {r.get('title')} — {r.get('url')}\n{r.get('snippet')}"
        for i, r in enumerate(results)
    )
    payload = _complete_json(
        VERACITY_SYSTEM,
        f"Claim ({claim_type}):\n{claim['claim']}\n\nSources:\n{sources}",
        resolve_distillation_model(model),
        max_tokens=1024,
    )

    verdict = str(payload.get("verdict") or "").strip()
    if verdict not in {"supported", "contradicted", "unsupported"}:
        verdict = "unsupported"

    # A platform-behaviour claim that checks out is still only true as of some date.
    if verdict == "supported" and claim_type in TIME_SENSITIVE_TYPES:
        verdict = "time-sensitive"

    return {
        "verdict": verdict,
        "checked": True,
        "reason": str(payload.get("reason") or "").strip() or None,
        "citations": [
            c for c in (payload.get("citations") or []) if isinstance(c, dict)
        ],
        "what_would_settle_it": str(payload.get("what_would_settle_it") or "").strip()
        or None,
        "observed_at": observed_at,
    }


def check_claims(
    claims: Sequence[Dict[str, Any]],
    model: Optional[str] = None,
    observed_at: Optional[str] = None,
    provider: Optional[RetrievalProvider] = None,
) -> Dict[str, Any]:
    """Assess a batch. Returns claims annotated with `veracity` plus a report."""
    provider = provider if provider is not None else resolve_provider()
    annotated: List[Dict[str, Any]] = []
    verdicts: Dict[str, int] = {}
    checked = 0

    for claim in claims:
        assessment = check_claim(claim, provider, model=model, observed_at=observed_at)
        enriched = dict(claim)
        enriched["veracity"] = assessment
        annotated.append(enriched)
        verdicts[assessment["verdict"]] = verdicts.get(assessment["verdict"], 0) + 1
        if assessment["checked"]:
            checked += 1

    return {
        "claims": annotated,
        "report": {
            "verdicts": verdicts,
            "checked": checked,
            "retrieval_configured": provider is not None,
            "note": (
                "'unsupported' means no source addressed the claim; it does not mean the "
                "claim is false. 'unverifiable' means it was not checked at all."
            ),
        },
    }
