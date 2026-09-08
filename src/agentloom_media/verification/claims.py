"""Atomic claim decomposition and claim typing.

Typing comes **before** any verification attempt, because most claims in a strategy or
marketing video are not checkable propositions at all. A system that labels them true or false
will be confidently wrong — and confidently wrong is worse than silent, since it launders a
single operator's unaudited results into apparent knowledge.

The five types are from plan §7.3. Only three of them can be checked against anything external;
the other two are the author's position and are presented as such, never adjudicated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from agentloom_media.distillation.passes import _complete_json
from agentloom_media.distillation.distiller import resolve_distillation_model

# type -> whether external checking can say anything useful
CLAIM_TYPES: Dict[str, bool] = {
    # "Andromeda changed how Meta matches audiences" — against vendor docs and dated reporting
    "vendor_behavior": True,
    # "Splitting ad sets fragments learning signal" — against documented platform mechanics
    "mechanism": True,
    # "~50 conversions in 14 days" — against published guidance
    "quantitative_rule": True,
    # "$800k spend, ROAS 6.12" — single-operator and unaudited. Not checkable, ever.
    "personal_result": False,
    # "Use one campaign for 85% of businesses" — a judgment, not a fact
    "recommendation": False,
}

CHECKABLE_TYPES = {t for t, checkable in CLAIM_TYPES.items() if checkable}
UNFALSIFIABLE_TYPES = {t for t, checkable in CLAIM_TYPES.items() if not checkable}

DEFAULT_TYPE = "recommendation"

CLAIMS_SYSTEM = """You decompose a takeaway into atomic claims and type each one.

An atomic claim states exactly one thing and is understandable on its own. Split compound
statements. Do not add anything the source does not say. Keep the source language.

Type each claim with exactly one of:
- vendor_behavior: what a platform or product does or changed
- mechanism: a causal or technical explanation of how something works
- quantitative_rule: a threshold or rule of thumb with numbers
- personal_result: the speaker's own results, spend, or performance figures
- recommendation: advice, judgment, or what someone should do

Typing rules that matter:
- A figure the speaker reports about their OWN account is `personal_result`, not
  `quantitative_rule`, however precise it sounds.
- "You should X" is `recommendation` even when justified by a mechanism. Split the mechanism
  into its own claim if one is stated.

Return JSON: {"claims": [{"claim": "<string>", "type": "<one of the five>"}]}"""


def decompose_and_type(
    statements: Sequence[str], model: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Split statements into typed atomic claims.

    Returns a list of `{"claim", "type", "checkable", "source_statement"}`. An unrecognized
    type falls back to `recommendation`, the most conservative option: it routes the claim away
    from external checking rather than toward a verdict we cannot justify.
    """
    statements = [s for s in (str(x).strip() for x in statements) if s]
    if not statements:
        return []

    model = resolve_distillation_model(model)
    numbered = "\n".join(f"[{i}] {s}" for i, s in enumerate(statements))
    payload = _complete_json(
        CLAIMS_SYSTEM,
        f"Decompose and type these {len(statements)} statements. Keep track of which "
        f"statement each claim came from by including its index as \"from\".\n\n{numbered}",
        model,
        max_tokens=4096,
    )

    # JSON mode is supposed to guarantee an object, but this model family sometimes returns a
    # bare array of claims. Accept either rather than losing the whole pass to the wrapper.
    if isinstance(payload, list):
        items = payload
    else:
        items = payload.get("claims") or payload.get("atomic_claims") or []

    claims: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not item.get("claim"):
            continue
        claim_type = str(item.get("type") or "").strip()
        if claim_type not in CLAIM_TYPES:
            claim_type = DEFAULT_TYPE

        source_index = item.get("from")
        try:
            source_index = int(source_index)
        except (TypeError, ValueError):
            source_index = None

        claims.append(
            {
                "claim": str(item["claim"]).strip(),
                "type": claim_type,
                "checkable": CLAIM_TYPES[claim_type],
                "source_statement": (
                    statements[source_index]
                    if source_index is not None and 0 <= source_index < len(statements)
                    else None
                ),
            }
        )
    return claims


def type_counts(claims: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for claim in claims:
        counts[claim["type"]] = counts.get(claim["type"], 0) + 1
    return counts
