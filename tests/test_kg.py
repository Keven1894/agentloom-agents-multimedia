"""Knowledge graph extraction, surface normalization, and merge suggestion."""

import pytest

from agentloom_media.distillation.segmentation import Segment
from agentloom_media.kg import extract as kg_extract
from agentloom_media.kg.merge import merge_threshold, suggest_merges
from agentloom_media.kg.model import (
    EDGE_FAMILIES,
    Evidence,
    GraphProposal,
    PREDICATE_TO_FAMILY,
    is_atomic_surface,
    normalize_surface,
)
from agentloom_media.transcripts.model import Transcript

SEGMENTS = [
    {"start": 0.0, "end": 10.0, "text": "Andromeda 让 Meta AI 自动完成受众匹配"},
    {"start": 10.0, "end": 20.0, "text": "所以不再需要拆分 Ad Set 来定向受众"},
    {"start": 20.0, "end": 30.0, "text": "CBO 是 Campaign 层级的预算分配方式"},
]


def _transcript():
    return Transcript.from_segments("vid", SEGMENTS)


def _segment(transcript, index=0):
    return Segment(
        index=index,
        t0=transcript.utterances[0].t0,
        t1=transcript.utterances[-1].t1,
        text=transcript.text,
        utterance_ids=[u.id for u in transcript.utterances],
        boundary_source="semantic",
        title="测试段落",
    )


# ---- surface normalization ---------------------------------------------------------


def test_normalization_is_case_and_punctuation_insensitive():
    assert normalize_surface("Campaign Structure") == "campaign structure"
    assert normalize_surface("Campaign-Structure!") == "campaignstructure"
    assert normalize_surface("  CBO  ") == "cbo"
    assert normalize_surface("") == ""


def test_normalization_handles_cjk_and_fullwidth():
    """The failure that ruled out ATOM: Chinese must survive normalization intact."""
    assert normalize_surface("广告结构") == "广告结构"
    assert normalize_surface("营销方法") == "营销方法"
    # Full-width Latin, common in mixed Chinese/English speech, folds to ASCII.
    assert normalize_surface("ＣＢＯ") == "cbo"
    assert normalize_surface("广告（结构）") == "广告结构"


def test_normalization_does_not_assert_synonymy():
    """No stemming or synonym mapping: those would assert identity we cannot establish."""
    assert normalize_surface("campaigns") != normalize_surface("campaign")
    assert normalize_surface("广告系列") != normalize_surface("campaign")


def test_clauses_are_not_atomic_surfaces():
    """Observed: the model returned clauses as endpoints, which cannot serve as nodes."""
    assert is_atomic_surface("CBO")
    assert is_atomic_surface("Campaign Structure")
    assert is_atomic_surface("广告结构")
    assert is_atomic_surface("Advantage+")

    assert not is_atomic_surface("Meta Andromeda以后85%以上的Ads Account都应该使用的结构")
    assert not is_atomic_surface("图片,视频以及跟它们搭配的文案")
    assert not is_atomic_surface("这个方法很好。它有效")
    assert not is_atomic_surface("")


def test_non_atomic_endpoints_are_dropped(monkeypatch):
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "Andromeda 让 Meta AI 自动完成受众匹配,所以不再需要拆分",
                "target": "受众匹配",
                "predicate": "enables",
                "quote": "Andromeda 让 Meta AI 自动完成受众匹配",
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    assert graph.edges == []
    assert graph.stats["dropped_reasons"] == {"surface_not_atomic": 1}


def test_predicate_registry_covers_all_six_families():
    assert set(EDGE_FAMILIES) == {
        "causal",
        "structural",
        "procedural",
        "definitional",
        "temporal",
        "evidential",
    }
    assert PREDICATE_TO_FAMILY["enables"] == "causal"
    assert PREDICATE_TO_FAMILY["part_of"] == "structural"
    assert PREDICATE_TO_FAMILY["defined_as"] == "definitional"
    assert PREDICATE_TO_FAMILY["superseded_by"] == "temporal"


# ---- node semantics ----------------------------------------------------------------


def test_first_seen_type_wins_and_surfaces_accumulate():
    graph = GraphProposal(media_id="vid")
    graph.add_node("Campaign Structure", "technical_item")
    graph.add_node("campaign structure", "keyword")
    graph.add_node("CAMPAIGN STRUCTURE", "entity")

    assert len(graph.nodes) == 1
    node = graph.nodes["campaign structure"]
    assert node.first_seen_type == "technical_item"
    # Every spelling is retained so a reviewer can audit the aggregation.
    assert node.surface_forms == [
        "Campaign Structure",
        "campaign structure",
        "CAMPAIGN STRUCTURE",
    ]


def test_node_dict_refuses_to_claim_identity():
    graph = GraphProposal(media_id="vid")
    graph.add_node("CBO", "technical_item")
    payload = graph.nodes["cbo"].to_dict()
    assert payload["identity_claim"] == "none"
    assert "not a resolved identity" in payload["identity_note"]
    assert payload["id"] == "surface:cbo"


def test_empty_surface_is_not_a_node():
    graph = GraphProposal(media_id="vid")
    assert graph.add_node("   ", "keyword") is None
    assert graph.add_node("!!!", "keyword") is None
    assert graph.nodes == {}


# ---- extraction and grounding ------------------------------------------------------


def _fake_relations(monkeypatch, relations):
    def fake(system, user, model, max_tokens=4096):
        return {"relations": relations}

    monkeypatch.setattr(kg_extract, "_complete_json", fake)


def test_grounded_edge_carries_evidence_and_time_axes(monkeypatch):
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "Andromeda",
                "target": "受众匹配",
                "predicate": "enables",
                "claim": "Andromeda 让 Meta AI 自动完成受众匹配",
                "quote": "Andromeda 让 Meta AI 自动完成受众匹配",
                "valid_from": "2026-09",
                "valid_until": None,
                "supersedes": None,
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(
        t, [_segment(t)], [], "https://youtu.be/vid", observed_at="2026-09-07"
    )

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.family == "causal"
    assert edge.predicate == "enables"
    # Media time, observation time, and validity time are all distinct and all present.
    assert edge.evidence.t0 == 0.0 and edge.evidence.t1 == 10.0
    assert edge.evidence.anchor_url == "https://youtu.be/vid?t=0s"
    assert edge.observed_at == "2026-09-07"
    assert edge.valid_from == "2026-09"
    assert edge.valid_until is None


def test_ungrounded_relation_is_dropped(monkeypatch):
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "Andromeda",
                "target": "受众匹配",
                "predicate": "enables",
                "quote": "这句话不在转写里出现过",
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    assert graph.edges == []
    assert graph.stats["dropped_reasons"] == {"quote_not_found": 1}


def test_quote_must_mention_both_endpoints(monkeypatch):
    """Observed in production: a correct relation grounded by an unrelated real quote.

    "CBO alternative_to ABO" is true, but the quote cited mentioned neither endpoint, so the
    evidence link pointed a reviewer at a moment that does not state the relation.
    """
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "CBO",
                "target": "ABO",
                "predicate": "alternative_to",
                "quote": "Andromeda 让 Meta AI 自动完成受众匹配",
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    assert graph.edges == []
    assert graph.stats["dropped_reasons"] == {"endpoints_not_in_quote": 1}


def test_quote_mentioning_only_one_endpoint_is_dropped(monkeypatch):
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "CBO",
                "target": "受众匹配",
                "predicate": "enables",
                "quote": "CBO 是 Campaign 层级的预算分配方式",
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    assert graph.edges == []
    assert graph.stats["dropped_reasons"] == {"endpoints_not_in_quote": 1}


def test_invented_predicate_is_dropped(monkeypatch):
    """An unknown predicate cannot be placed in a reviewable family."""
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "CBO",
                "target": "Campaign",
                "predicate": "is_vaguely_related_to",
                "quote": "CBO 是 Campaign 层级的预算分配方式",
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    assert graph.edges == []
    assert graph.stats["dropped_reasons"] == {"unknown_predicate": 1}


def test_evidential_predicate_cannot_come_from_the_model(monkeypatch):
    """Evidential edges are built by the grounding gate, never asserted by the model."""
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "CBO",
                "target": "Campaign",
                "predicate": "supported_by_span",
                "quote": "CBO 是 Campaign 层级的预算分配方式",
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    assert graph.edges == []
    assert graph.stats["dropped_reasons"] == {"unknown_predicate": 1}


def test_self_loops_and_missing_endpoints_are_dropped(monkeypatch):
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "CBO",
                "target": "cbo",
                "predicate": "part_of",
                "quote": "CBO 是 Campaign 层级的预算分配方式",
            },
            {
                "source": "",
                "target": "Campaign",
                "predicate": "part_of",
                "quote": "CBO 是 Campaign 层级的预算分配方式",
            },
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    assert graph.edges == []
    assert graph.stats["dropped_reasons"] == {"self_loop": 1, "missing_endpoint": 1}


def test_superseded_surface_becomes_a_node(monkeypatch):
    _fake_relations(
        monkeypatch,
        [
            {
                "source": "Ad Set",
                "target": "受众",
                "predicate": "superseded_by",
                "quote": "所以不再需要拆分 Ad Set 来定向受众",
                "supersedes": "Ad Set 拆分",
            }
        ],
    )
    t = _transcript()
    graph = kg_extract.extract_graph(t, [_segment(t)], [], "https://youtu.be/vid")
    edge = graph.edges[0]
    assert edge.family == "temporal"
    assert edge.supersedes == "surface:ad set 拆分"
    assert "ad set 拆分" in graph.nodes


def test_terms_seed_nodes_with_verified_spans(monkeypatch):
    _fake_relations(monkeypatch, [])
    t = _transcript()
    terms = [
        {
            "term": "CBO",
            "type": "technical_item",
            "description": "预算分配",
            "quote": "CBO 是 Campaign 层级的预算分配方式",
            "t0": 20.0,
            "t1": 30.0,
            "anchor_url": "https://youtu.be/vid?t=20s",
        }
    ]
    graph = kg_extract.extract_graph(t, [_segment(t)], terms, "https://youtu.be/vid")
    assert "cbo" in graph.nodes
    node = graph.nodes["cbo"]
    assert node.first_seen_type == "technical_item"
    assert node.evidence[0].t0 == 20.0
    assert graph.stats["nodes_seeded_from_terms"] == 1


def test_segment_failure_is_contained(monkeypatch):
    """One failing segment must not lose the others."""
    calls = {"n": 0}

    def flaky(system, user, model, max_tokens=4096):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("rate limited")
        return {
            "relations": [
                {
                    "source": "CBO",
                    "target": "Campaign",
                    "predicate": "part_of",
                    "quote": "CBO 是 Campaign 层级的预算分配方式",
                }
            ]
        }

    monkeypatch.setattr(kg_extract, "_complete_json", flaky)
    t = _transcript()
    graph = kg_extract.extract_graph(
        t, [_segment(t, 0), _segment(t, 1)], [], "https://youtu.be/vid", max_workers=1
    )
    assert len(graph.edges) == 1
    assert graph.stats["segments_failed"] == [0]
    assert graph.stats["segments_extracted"] == 1


def test_vague_validity_dates_are_rejected():
    assert kg_extract._clean_date("2026") == "2026"
    assert kg_extract._clean_date("2026-09") == "2026-09"
    assert kg_extract._clean_date("2026-09-07") == "2026-09-07"
    for bad in ["recently", "post-Andromeda", "null", "unknown", "", None, "09-2026"]:
        assert kg_extract._clean_date(bad) is None, bad


# ---- merge suggestion --------------------------------------------------------------


class PairEmbedder:
    """Returns near-identical vectors for surfaces sharing a prefix marker."""

    def embed(self, texts):
        vectors = []
        for text in texts:
            if text.startswith("A"):
                # ~0.995 cosine similarity to the "B" vector: above the 0.88 default, below
                # a deliberately strict threshold.
                vectors.append([1.0, 0.1, 0.0])
            elif text.startswith("B"):
                vectors.append([1.0, 0.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return vectors


def test_similar_surfaces_are_suggested_never_merged():
    graph = GraphProposal(media_id="vid")
    graph.add_node("A campaign", "keyword")
    graph.add_node("B campaign", "keyword")
    graph.add_node("unrelated", "keyword")

    candidates = suggest_merges(graph, PairEmbedder())

    assert len(candidates) == 1
    assert candidates[0]["decision"] == "pending_human_review"
    assert "not evidence" in candidates[0]["note"]
    # The graph is untouched: both surfaces survive until a human decides.
    assert len(graph.nodes) == 3


def test_no_embedder_yields_no_suggestions():
    graph = GraphProposal(media_id="vid")
    graph.add_node("A campaign", "keyword")
    graph.add_node("B campaign", "keyword")
    assert suggest_merges(graph, None) == []


def test_merge_threshold_is_configurable(monkeypatch):
    monkeypatch.delenv("KG_MERGE_THRESHOLD", raising=False)
    assert merge_threshold() == 0.88
    monkeypatch.setenv("KG_MERGE_THRESHOLD", "0.95")
    assert merge_threshold() == 0.95
    monkeypatch.setenv("KG_MERGE_THRESHOLD", "not-a-number")
    assert merge_threshold() == 0.88
    assert merge_threshold(0.5) == 0.5


def test_high_threshold_suppresses_suggestions():
    graph = GraphProposal(media_id="vid")
    graph.add_node("A campaign", "keyword")
    graph.add_node("B campaign", "keyword")
    assert suggest_merges(graph, PairEmbedder(), threshold=0.999) == []


def test_oversized_graph_refuses_pairwise_comparison(monkeypatch):
    monkeypatch.setattr("agentloom_media.kg.merge.MAX_SURFACES_FOR_PAIRWISE", 3)
    graph = GraphProposal(media_id="vid")
    for index in range(5):
        graph.add_node(f"surface {index}", "keyword")
    assert suggest_merges(graph, PairEmbedder()) == []
    assert "exceeds the pairwise limit" in graph.stats["merge_suggestion_skipped"]


# ---- serialization -----------------------------------------------------------------


def test_proposal_declares_semantics_and_axes():
    graph = GraphProposal(media_id="vid")
    graph.add_node("CBO", "technical_item")
    payload = graph.to_dict()

    assert "not resolved identities" in payload["node_semantics"]
    assert set(payload["edge_families"]) == set(EDGE_FAMILIES)
    assert set(payload["time_axes"]) == {
        "media_time",
        "observation_time",
        "validity_time",
    }
