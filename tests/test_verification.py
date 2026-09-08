"""Claim typing, groundedness, and veracity — and the distinctions between them."""

import pytest

from agentloom_media.verification import claims as claims_module
from agentloom_media.verification import groundedness as ground_module
from agentloom_media.verification import takeaway as takeaway_module
from agentloom_media.verification import veracity as veracity_module
from agentloom_media.verification.claims import (
    CHECKABLE_TYPES,
    CLAIM_TYPES,
    UNFALSIFIABLE_TYPES,
    decompose_and_type,
)
from agentloom_media.verification.groundedness import (
    GROUNDED_THRESHOLD,
    assess_claim,
    containment,
)
from agentloom_media.verification.veracity import check_claim, check_claims

# ---- claim typing ------------------------------------------------------------------


def test_only_three_types_are_checkable():
    assert CHECKABLE_TYPES == {"vendor_behavior", "mechanism", "quantitative_rule"}
    assert UNFALSIFIABLE_TYPES == {"personal_result", "recommendation"}
    assert set(CLAIM_TYPES) == CHECKABLE_TYPES | UNFALSIFIABLE_TYPES


def test_unknown_type_falls_back_to_the_conservative_option(monkeypatch):
    """An unrecognized type must route away from external checking, not toward it."""
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: {
            "claims": [{"claim": "某个说法", "type": "totally_made_up", "from": 0}]
        },
    )
    result = decompose_and_type(["某个说法"])
    assert result[0]["type"] == "recommendation"
    assert result[0]["checkable"] is False


def test_decomposition_tracks_source_statements(monkeypatch):
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: {
            "claims": [
                {"claim": "A", "type": "mechanism", "from": 1},
                {"claim": "B", "type": "personal_result", "from": 0},
                {"claim": "C", "type": "mechanism", "from": 99},
            ]
        },
    )
    result = decompose_and_type(["first", "second"])
    assert result[0]["source_statement"] == "second"
    assert result[1]["source_statement"] == "first"
    # An out-of-range index must not raise or silently mislabel.
    assert result[2]["source_statement"] is None


def test_bare_array_response_is_accepted(monkeypatch):
    """JSON mode should guarantee an object, but this model family returns arrays sometimes."""
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: [{"claim": "拆分分散信号", "type": "mechanism", "from": 0}],
    )
    result = decompose_and_type(["拆分分散信号"])
    assert len(result) == 1
    assert result[0]["type"] == "mechanism"


def test_alternate_wrapper_key_is_accepted(monkeypatch):
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: {"atomic_claims": [{"claim": "x", "type": "mechanism"}]},
    )
    assert len(decompose_and_type(["x"])) == 1


def test_empty_input_needs_no_llm_call():
    assert decompose_and_type([]) == []
    assert decompose_and_type(["", "   "]) == []


# ---- groundedness ------------------------------------------------------------------


def test_containment_handles_chinese_without_a_segmenter():
    evidence = "CBO 会自动分配预算给表现好的广告"
    assert containment("CBO 会自动分配预算", evidence) > GROUNDED_THRESHOLD
    assert containment("完全不相关的内容在讲别的东西", evidence) < 0.2


def test_containment_is_directional_not_jaccard():
    """A short claim inside a long span must score high; Jaccard would punish the length."""
    short = "CBO 分配预算"
    long_evidence = "CBO 分配预算 " + "其他内容 " * 60
    assert containment(short, long_evidence) > 0.9


def test_containment_edge_cases():
    assert containment("", "anything") == 0.0
    assert containment("something", "") == 0.0


class FakeHit:
    """Mirrors SearchHit: an attribute object, not a dict."""

    def __init__(self, media_id, text, t0, t1=None, window_id="w0"):
        self.window_id = window_id
        self.media_id = media_id
        self.text = text
        self.t0 = t0
        self.t1 = t1 if t1 is not None else t0 + 45.0


class FakeIndex:
    """Mirrors SearchIndex.search: returns `(hits, info)` and filters by media itself."""

    def __init__(self, hits):
        self._hits = [FakeHit(**h) if isinstance(h, dict) else h for h in hits]

    def search(self, query, limit=10, media_id=None, depth=None):
        hits = [h for h in self._hits if media_id is None or h.media_id == media_id]
        return hits[:limit], {"lexical": len(hits), "vector": 0, "vector_used": False}


def test_verbatim_span_short_circuits_retrieval():
    """A claim already carrying a located quote passed a stricter gate than anything here."""
    result = assess_claim(
        "CBO 会自动分配预算",
        "vid",
        index=FakeIndex([]),
        known_evidence=[{"t0": 20.0, "t1": 30.0, "quote": "CBO 会自动分配预算"}],
    )
    assert result["verdict"] == "grounded"
    assert result["tier"] == "verbatim_span"


def test_no_retrieval_means_ungrounded_not_grounded():
    result = assess_claim("任意说法", "vid", index=FakeIndex([]))
    assert result["verdict"] == "ungrounded"
    assert result["tier"] == "retrieval"


def test_lexical_tier_decides_clear_matches():
    hits = [{"media_id": "vid", "text": "CBO 会自动分配预算给表现好的广告", "t0": 10.0}]
    result = assess_claim("CBO 会自动分配预算", "vid", index=FakeIndex(hits))
    assert result["verdict"] == "grounded"
    assert result["tier"] == "lexical"
    assert result["evidence"]["t0"] == 10.0


def test_lexical_tier_decides_clear_misses():
    hits = [{"media_id": "vid", "text": "今天天气不错我们去公园散步吧", "t0": 10.0}]
    result = assess_claim("CBO 会自动分配预算", "vid", index=FakeIndex(hits))
    assert result["verdict"] == "ungrounded"
    assert result["tier"] == "lexical"


def test_other_media_hits_are_filtered_out():
    hits = [{"media_id": "other", "text": "CBO 会自动分配预算给表现好的广告", "t0": 5.0}]
    result = assess_claim("CBO 会自动分配预算", "vid", index=FakeIndex(hits))
    assert result["verdict"] == "ungrounded"
    assert result["tier"] == "retrieval"


def test_ambiguous_band_is_undecided_not_guessed():
    """Without an entailment model, the middle band must abstain."""
    # Roughly half the claim's n-grams present: inside the ambiguous band.
    hits = [{"media_id": "vid", "text": "CBO 会自动 处理别的事情完全不同的话题内容", "t0": 1.0}]
    result = assess_claim("CBO 会自动分配预算", "vid", index=FakeIndex(hits))
    assert result["verdict"] == "needs_review"
    assert "guessed" in result["reason"]


def test_nli_tier_resolves_the_ambiguous_band():
    hits = [{"media_id": "vid", "text": "CBO 会自动 处理别的事情完全不同的话题内容", "t0": 1.0}]
    result = assess_claim(
        "CBO 会自动分配预算",
        "vid",
        index=FakeIndex(hits),
        nli_scorer=lambda claim, evidence: 0.91,
    )
    assert result["verdict"] == "grounded"
    assert result["tier"] == "nli"


def test_escalation_only_touches_the_ambiguous_band(monkeypatch):
    """The cascade must not pay for claims the cheap tier already settled."""
    calls = []

    def fake_entailment(claim, evidence, model):
        calls.append(claim)
        return {"verdict": "supported", "reason": "states it"}

    monkeypatch.setattr(ground_module, "llm_entailment", fake_entailment)

    clear_hit = FakeHit("vid", "CBO 会自动分配预算给表现好的广告", 10.0)
    ambiguous_hit = FakeHit("vid", "CBO 会自动 处理别的事情完全不同的话题内容", 20.0)

    class PerClaimIndex:
        def search(self, query, limit=10, media_id=None, depth=None):
            hit = clear_hit if "分配预算" in query else ambiguous_hit
            return [hit], {}

    result = ground_module.assess_claims(
        [{"claim": "CBO 会自动分配预算"}, {"claim": "CBO 会自动分配算法"}],
        "vid",
        index=PerClaimIndex(),
        llm_model="m",
    )
    assert result["report"]["escalated"] == 1
    assert calls == ["CBO 会自动分配算法"]
    assert result["report"]["tiers"] == {"lexical": 1, "llm": 1}


def test_failed_escalation_leaves_the_claim_undecided(monkeypatch):
    def boom(claim, evidence, model):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(ground_module, "llm_entailment", boom)
    hits = [{"media_id": "vid", "text": "CBO 会自动 处理别的事情完全不同的话题内容", "t0": 1.0}]
    result = ground_module.assess_claims(
        [{"claim": "CBO 会自动分配预算"}], "vid", index=FakeIndex(hits), llm_model="m"
    )
    assessment = result["claims"][0]["groundedness"]
    # Must not become "grounded" just because the judge fell over.
    assert assessment["verdict"] == "needs_review"
    assert "rate limited" in assessment["reason"]


def test_escalation_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TAKEAWAY_ESCALATE", "0")
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: {"claims": [{"claim": "x", "type": "mechanism", "from": 0}]},
    )

    def boom(*a, **k):
        raise AssertionError("escalation should be off")

    monkeypatch.setattr(ground_module, "llm_entailment", boom)
    takeaway_module.build_takeaway({"takeaways": ["x"]}, [], "vid", index=FakeIndex([]))


# ---- window truncation -------------------------------------------------------------


class FakeUtterance:
    def __init__(self, t0, t1, text):
        self.t0 = t0
        self.t1 = t1
        self.text = text


class FakeTranscript:
    def __init__(self, utterances):
        self.utterances = [FakeUtterance(*u) for u in utterances]


# The real case: a 45s window ended mid-sentence at "投7万9", stranding the rest next door.
TRUNCATED = FakeTranscript(
    [
        (430.0, 445.0, "一共我们投了10万美金 然后收回的是78万"),
        (445.0, 460.0, "所以30天就能达到投7万9"),
        (460.0, 475.0, "能达到差不多46万的这个收益 看我们这个亚洲地区的"),
    ]
)


def test_neighborhood_spans_the_window_boundary():
    context = ground_module.neighborhood(TRUNCATED, 430.0, 460.0)
    assert "投7万9" in context["text"]
    # The half-sentence that the window cut off must be present.
    assert "46万" in context["text"]
    assert context["t0"] == 430.0
    assert context["t1"] == 475.0


def test_neighborhood_does_not_duplicate_overlapping_text():
    """Built from utterances, not by stitching overlapping windows at their seams."""
    context = ground_module.neighborhood(TRUNCATED, 430.0, 475.0)
    assert context["text"].count("投7万9") == 1


def test_neighborhood_without_a_transcript_is_empty():
    assert ground_module.neighborhood(None, 0.0, 45.0)["text"] == ""


def test_judge_reads_the_neighborhood_but_cites_the_window(monkeypatch):
    seen = {}

    def capture(claim, evidence, model):
        seen["evidence"] = evidence
        return {"verdict": "supported", "reason": "states it"}

    monkeypatch.setattr(ground_module, "llm_entailment", capture)

    truncated_window = "一共我们投了10万美金 然后收回的是78万 所以30天就能达到投7万9"
    hits = [FakeHit("vid", truncated_window, 430.0, 460.0)]
    result = assess_claim(
        "亚洲市场30天投7万9带来约46万收益",
        "vid",
        index=FakeIndex(hits),
        llm_model="m",
        transcript=TRUNCATED,
    )

    # The judge saw the stranded half-sentence.
    assert "46万" in seen["evidence"]
    # But the citation is still the tight window, so the anchor stays precise.
    assert result["evidence"]["t0"] == 430.0
    assert result["evidence"]["t1"] == 460.0
    assert result["context_span"] == [430.0, 475.0]
    assert result["verdict"] == "grounded"


def test_truncation_cannot_cause_a_silent_hard_reject(monkeypatch):
    """A window score below the floor must still escalate if the neighborhood has more."""
    escalated = []

    def capture(claim, evidence, model):
        escalated.append(claim)
        return {"verdict": "supported", "reason": "states it"}

    monkeypatch.setattr(ground_module, "llm_entailment", capture)

    # The window holds almost none of the claim; the neighbouring utterance holds the rest.
    hits = [FakeHit("vid", "一共我们投了10万美金", 430.0, 445.0)]
    result = assess_claim(
        "30天投7万9带来差不多46万的收益",
        "vid",
        index=FakeIndex(hits),
        llm_model="m",
        transcript=TRUNCATED,
    )
    assert escalated, "truncation dropped the window score and hard-rejected the claim"
    assert result["verdict"] == "grounded"


def test_a_genuine_miss_is_still_rejected_on_the_neighborhood():
    """Widening the reading window must not turn every miss into an escalation."""
    hits = [FakeHit("vid", "一共我们投了10万美金", 430.0, 445.0)]
    result = assess_claim(
        "今天天气很好我们去公园散步吧",
        "vid",
        index=FakeIndex(hits),
        llm_model="m",
        transcript=TRUNCATED,
    )
    assert result["verdict"] == "ungrounded"
    assert result["tier"] == "lexical"


def test_clear_match_is_still_decided_on_the_tight_window():
    """High overlap inside 45s is strong evidence; widening it would invite coincidence."""
    hits = [FakeHit("vid", "CBO 会自动分配预算给表现好的广告", 10.0, 55.0)]
    result = assess_claim(
        "CBO 会自动分配预算", "vid", index=FakeIndex(hits), transcript=TRUNCATED
    )
    assert result["tier"] == "lexical"
    assert result["verdict"] == "grounded"
    assert result.get("context_span") is None


def test_llm_tier_is_labelled_as_llm_not_nli(monkeypatch):
    monkeypatch.setattr(
        ground_module,
        "llm_entailment",
        lambda claim, evidence, model: {"verdict": "supported", "reason": "says so"},
    )
    hits = [{"media_id": "vid", "text": "CBO 会自动 处理别的事情完全不同的话题内容", "t0": 1.0}]
    result = assess_claim(
        "CBO 会自动分配预算", "vid", index=FakeIndex(hits), llm_model="m"
    )
    assert result["tier"] == "llm"
    assert result["verdict"] == "grounded"


def test_missing_nli_model_is_not_an_error(monkeypatch):
    monkeypatch.delenv("NLI_MODEL", raising=False)
    assert ground_module.load_nli_scorer() is None


# ---- veracity ----------------------------------------------------------------------


def test_unfalsifiable_claims_are_never_adjudicated():
    for claim_type in sorted(UNFALSIFIABLE_TYPES):
        result = check_claim({"claim": "x", "type": claim_type}, provider=None)
        assert result["verdict"] == "unverifiable"
        assert result["checked"] is False


def test_personal_results_say_what_would_settle_them():
    result = check_claim(
        {"claim": "我们的 ROAS 是 6.12", "type": "personal_result"}, provider=None
    )
    assert "audited" in result["what_would_settle_it"]


def test_unconfigured_retrieval_is_unverifiable_not_unsupported():
    """The crux: "we didn't look" must not be reported as "we found nothing"."""
    result = check_claim({"claim": "Andromeda 改变了匹配方式", "type": "vendor_behavior"}, provider=None)
    assert result["verdict"] == "unverifiable"
    assert result["verdict"] != "unsupported"
    assert "not checked" in result["reason"]


def test_empty_retrieval_results_are_unsupported_not_contradicted():
    result = check_claim(
        {"claim": "Andromeda 改变了匹配方式", "type": "vendor_behavior"},
        provider=lambda q: [],
    )
    assert result["verdict"] == "unsupported"
    assert result["checked"] is True


def test_retrieval_failure_is_unverifiable():
    def broken(query):
        raise RuntimeError("network down")

    result = check_claim(
        {"claim": "x", "type": "mechanism"}, provider=broken
    )
    assert result["verdict"] == "unverifiable"
    assert "network down" in result["reason"]


def test_supported_platform_claims_become_time_sensitive(monkeypatch):
    """A platform-behaviour claim that checks out is still only true as of some date."""
    monkeypatch.setattr(
        veracity_module,
        "_complete_json",
        lambda *a, **k: {
            "verdict": "supported",
            "reason": "vendor docs confirm",
            "citations": [{"url": "https://example.com", "title": "docs"}],
        },
        raising=False,
    )
    import agentloom_media.distillation.passes as passes_module

    monkeypatch.setattr(
        passes_module,
        "_complete_json",
        lambda *a, **k: {
            "verdict": "supported",
            "reason": "vendor docs confirm",
            "citations": [{"url": "https://example.com", "title": "docs"}],
        },
    )

    result = check_claim(
        {"claim": "Andromeda 改变了匹配方式", "type": "vendor_behavior"},
        provider=lambda q: [{"title": "docs", "url": "https://example.com", "snippet": "..."}],
    )
    assert result["verdict"] == "time-sensitive"

    # A mechanism claim is not inherently dated, so it stays "supported".
    mechanism = check_claim(
        {"claim": "拆分会分散学习信号", "type": "mechanism"},
        provider=lambda q: [{"title": "docs", "url": "https://example.com", "snippet": "..."}],
    )
    assert mechanism["verdict"] == "supported"


def test_verdict_vocabulary_excludes_true_and_false():
    assert "true" not in veracity_module.VERDICTS
    assert "false" not in veracity_module.VERDICTS
    assert set(veracity_module.VERDICTS) == {
        "supported",
        "contradicted",
        "unsupported",
        "unverifiable",
        "time-sensitive",
    }


def test_batch_report_flags_missing_retrieval():
    result = check_claims(
        [{"claim": "a", "type": "mechanism"}, {"claim": "b", "type": "recommendation"}],
        provider=None,
    )
    assert result["report"]["retrieval_configured"] is False
    assert result["report"]["checked"] == 0
    assert "does not mean the claim is false" in result["report"]["note"]


def test_provider_absent_without_configuration(monkeypatch):
    monkeypatch.delenv("SEARCH_PROVIDER", raising=False)
    assert veracity_module.resolve_provider() is None
    monkeypatch.setenv("SEARCH_PROVIDER", "tavily")
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    # Named but keyless: still None, so we abstain rather than fail mid-run.
    assert veracity_module.resolve_provider() is None


# ---- takeaway assembly -------------------------------------------------------------


class FakeSegment:
    def __init__(self, index, key_points):
        self.index = index
        self.key_points = key_points


def test_statements_are_collected_from_document_and_segments():
    statements = takeaway_module.collect_statements(
        {"takeaways": ["t1", "t2"], "thesis": "th"},
        [FakeSegment(0, [{"point": "kp1", "anchor": "01:00"}])],
    )
    origins = [s["origin"] for s in statements]
    assert origins == ["takeaway", "takeaway", "thesis", "key_point"]


def test_unfalsifiable_claims_never_land_in_what_holds_up(monkeypatch):
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: {
            "claims": [
                {"claim": "我们 ROAS 6.12", "type": "personal_result", "from": 0},
                {"claim": "应该用一个 Campaign", "type": "recommendation", "from": 0},
                {"claim": "拆分分散信号", "type": "mechanism", "from": 0},
            ]
        },
    )
    result = takeaway_module.build_takeaway(
        {"takeaways": ["混合陈述"]}, [], "vid", index=FakeIndex([])
    )

    sections = result["sections"]
    unfalsifiable = {c["claim"] for c in sections["what_is_unfalsifiable"]}
    assert unfalsifiable == {"我们 ROAS 6.12", "应该用一个 Campaign"}
    assert sections["what_holds_up"] == []
    # The mechanism claim was checkable but unverifiable, so it goes to "verify".
    assert [c["claim"] for c in sections["what_to_verify"]] == ["拆分分散信号"]


def test_section_one_depends_on_groundedness_not_veracity(monkeypatch):
    """"What the video claims" records what was said, regardless of whether it is true."""
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: {
            "claims": [{"claim": "CBO 会自动分配预算", "type": "mechanism", "from": 0}]
        },
    )
    hits = [{"media_id": "vid", "text": "CBO 会自动分配预算给表现好的广告", "t0": 12.0}]
    result = takeaway_module.build_takeaway(
        {"takeaways": ["CBO 会自动分配预算"]}, [], "vid", index=FakeIndex(hits)
    )

    claimed = result["sections"]["what_the_video_claims"]
    assert len(claimed) == 1
    assert claimed[0]["groundedness"]["verdict"] == "grounded"
    # Grounded but unchecked: it is in section 1 and also in section 3.
    assert claimed[0]["veracity"]["verdict"] == "unverifiable"
    assert len(result["sections"]["what_to_verify"]) == 1


def test_takeaway_is_a_proposal_never_auto_accepted(monkeypatch):
    monkeypatch.setattr(
        claims_module,
        "_complete_json",
        lambda *a, **k: {"claims": [{"claim": "x", "type": "mechanism", "from": 0}]},
    )
    result = takeaway_module.build_takeaway(
        {"takeaways": ["x"]}, [], "vid", index=FakeIndex([])
    )
    assert result["status"] == "pending_human_review"
    assert "still be wrong" in result["note"]


# ---- rendering ---------------------------------------------------------------------


class FakeGate:
    def link(self, t0, label):
        return f"[{t0}](url&t={int(t0)}s)"


def _render(entry_overrides):
    from agentloom_media.proposals.emitter_v2 import _takeaway_section

    entry = {
        "claim": "CBO 会自动分配预算",
        "type": "mechanism",
        "groundedness": {"verdict": "grounded", "tier": "llm", "evidence": {"t0": 30.0}},
        "veracity": {"verdict": "unverifiable", "reason": "not checked"},
    }
    entry.update(entry_overrides)
    return "\n".join(
        _takeaway_section(
            {
                "sections": {
                    "what_the_video_claims": [entry],
                    "what_holds_up": [],
                    "what_to_verify": [entry],
                    "what_is_unfalsifiable": [],
                },
                "reports": {
                    "groundedness": {"nli_available": False, "escalated": 3},
                    "veracity": {"retrieval_configured": False},
                },
            },
            FakeGate(),
        )
    )


def test_ungrounded_claims_get_no_timestamp_link():
    """A rejected candidate span must not be rendered as a citation (the P0 mistake)."""
    grounded = _render({})
    assert "url&t=30s" in grounded

    ungrounded = _render(
        {
            "groundedness": {
                "verdict": "ungrounded",
                "tier": "lexical",
                "evidence": {"t0": 30.0},
                "reason": "nothing to entail from",
            }
        }
    )
    assert "url&t=30s" not in ungrounded
    assert "nothing to entail from" in ungrounded


def test_unfalsifiable_section_does_not_show_a_veracity_verdict():
    rendered = _render({})
    unfalsifiable_block = rendered.split("### 4.")[1]
    assert "veracity:" not in unfalsifiable_block


def test_escalation_is_reported_accurately_in_the_header():
    rendered = _render({})
    assert "escalated to an LLM entailment judge" in rendered
    assert "needs_review` rather than" not in rendered


def test_missing_retrieval_is_flagged_in_the_header():
    rendered = _render({})
    assert "No external retrieval is configured" in rendered
    assert "not checked" in rendered


def test_takeaway_claims_become_review_targets():
    from agentloom_media.review.targets import build_review_items

    items = build_review_items(
        {
            "document": {"takeaways": ["raw takeaway"]},
            "takeaway": {
                "claims": [
                    {
                        "claim": "grounded claim",
                        "type": "mechanism",
                        "groundedness": {
                            "verdict": "grounded",
                            "tier": "llm",
                            "evidence": {"t0": 10.0, "t1": 55.0, "text": "span text"},
                        },
                        "veracity": {"verdict": "unverifiable"},
                    },
                    {
                        "claim": "ungrounded claim",
                        "type": "recommendation",
                        "groundedness": {
                            "verdict": "ungrounded",
                            "tier": "lexical",
                            "evidence": {"t0": 99.0, "text": "rejected span"},
                        },
                        "veracity": {"verdict": "unverifiable"},
                    },
                ]
            },
        }
    )
    claims = [i for i in items if i["kind"] == "claim"]
    assert len(claims) == 2
    assert len(claims[0]["evidence"]) == 1
    assert claims[0]["evidence"][0]["t0"] == 10.0
    # The rejected candidate must not be offered to a reviewer as a citation.
    assert claims[1]["evidence"] == []
    # Claims replace the raw takeaway, which a reviewer cannot check against anything.
    assert not [i for i in items if i["kind"] == "takeaway"]


def test_raw_takeaways_still_listed_without_a_takeaway_pass():
    from agentloom_media.review.targets import build_review_items

    items = build_review_items({"document": {"takeaways": ["raw takeaway"]}})
    takeaways = [i for i in items if i["kind"] == "takeaway"]
    assert len(takeaways) == 1
    assert takeaways[0]["evidence"] == []


def test_no_statements_yields_empty_sections():
    result = takeaway_module.build_takeaway({}, [], "vid")
    assert result["claims"] == []
    assert all(v == [] for v in result["sections"].values())
    assert result["status"] == "pending_human_review"
