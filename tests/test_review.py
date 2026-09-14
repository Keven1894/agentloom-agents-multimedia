"""Evidence-level review: decision log and target flattening."""

import json

import pytest

from agentloom_media.review.decisions import (
    current_decisions,
    read_review,
    record_decision,
    review_path,
    summarize,
)
from agentloom_media.review.targets import (
    build_legacy_review_items,
    build_review_items,
    coverage,
    digest_path_for_proposal,
    media_id_from_source,
)

PROPOSAL = {
    "source": {"media_id": "vid", "title": "T", "url": "https://youtu.be/vid"},
    "segments": [
        {
            "index": 0,
            "title": "开场",
            "key_points": [
                {"point": "结构应该简化", "anchor": "01:04"},
                {"point": "没有锚点的要点", "anchor": None},
                {"point": "锚点是区间", "anchor": "01:00-02:00"},
            ],
        }
    ],
    "terms": [
        {
            "term": "CBO",
            "type": "technical_item",
            "description": "预算分配",
            "quote": "CBO 会自动分配预算",
            "t0": 20.0,
            "t1": 30.0,
            "anchor_url": "https://youtu.be/vid?t=20s",
        }
    ],
    "knowledge_graph": {
        "nodes": [
            {
                "id": "surface:cbo",
                "normalized_surface": "cbo",
                "surface_forms": ["CBO", "cbo"],
                "first_seen_type": "technical_item",
                "evidence": [
                    {"t0": 20.0, "t1": 30.0, "quote": "CBO 会自动分配预算"},
                    {"t0": 60.0, "t1": 70.0, "quote": "再说一次 CBO"},
                ],
            },
            {
                "id": "surface:abo",
                "normalized_surface": "abo",
                "surface_forms": ["ABO"],
                "first_seen_type": "surface",
                "evidence": [],
            },
        ],
        "edges": [
            {
                "source": "surface:cbo",
                "target": "surface:abo",
                "predicate": "alternative_to",
                "family": "structural",
                "claim": "CBO 与 ABO 互为替代",
                "observed_at": "2026-09-06",
                "valid_from": None,
                "valid_until": None,
                "evidence": {"t0": 20.0, "t1": 30.0, "quote": "CBO 会自动分配预算"},
            }
        ],
        "merge_candidates": [],
    },
    "document": {"takeaways": ["一个没有证据的 takeaway"]},
    "skill": None,
}


# ---- target flattening -------------------------------------------------------------


def test_every_kind_is_flattened():
    items = build_review_items(PROPOSAL)
    kinds = {i["kind"] for i in items}
    assert kinds == {"key_point", "term", "node", "edge", "takeaway"}


def test_ids_are_content_derived_and_stable():
    first = {i["id"] for i in build_review_items(PROPOSAL)}
    second = {i["id"] for i in build_review_items(json.loads(json.dumps(PROPOSAL)))}
    assert first == second


def test_reordering_key_points_does_not_move_verdicts():
    """Ids come from content, so a re-run that reorders points keeps verdicts attached."""
    reordered = json.loads(json.dumps(PROPOSAL))
    reordered["segments"][0]["key_points"].reverse()

    original = {i["id"]: i["label"] for i in build_review_items(PROPOSAL) if i["kind"] == "key_point"}
    shuffled = {i["id"]: i["label"] for i in build_review_items(reordered) if i["kind"] == "key_point"}
    assert original == shuffled


def test_ungrounded_targets_are_listed_not_hidden():
    items = build_review_items(PROPOSAL)
    takeaway = next(i for i in items if i["kind"] == "takeaway")
    assert takeaway["evidence"] == []

    # A key point whose anchor was a range has no usable evidence, and must still be shown.
    ranged = next(i for i in items if i["label"] == "锚点是区间")
    assert ranged["evidence"] == []
    unanchored = next(i for i in items if i["label"] == "没有锚点的要点")
    assert unanchored["evidence"] == []

    node = next(i for i in items if i["id"] == "surface:abo")
    assert node["evidence"] == []


def test_multiple_evidence_links_are_preserved():
    """Evidence is many-to-many; a node with two citations must expose both."""
    items = build_review_items(PROPOSAL)
    node = next(i for i in items if i["id"] == "surface:cbo")
    assert len(node["evidence"]) == 2
    assert [e["t0"] for e in node["evidence"]] == [20.0, 60.0]


def test_edge_carries_time_axes_for_display():
    items = build_review_items(PROPOSAL)
    edge = next(i for i in items if i["kind"] == "edge")
    assert edge["meta"]["observed_at"] == "2026-09-06"
    assert edge["meta"]["valid_from"] is None
    assert edge["context"] == "structural"


def test_coverage_counts_grounding_honestly():
    items = build_review_items(PROPOSAL)
    report = coverage(items)
    assert report["total"] == len(items)
    assert report["grounded"] + report["ungrounded"] == report["total"]
    assert report["by_kind"]["takeaway"]["ungrounded"] == 1
    assert report["by_kind"]["term"]["grounded"] == 1


def test_media_id_falls_back_to_the_youtube_url():
    assert media_id_from_source({"url": "https://www.youtube.com/watch?v=VFjup6AbQOM"}) == "VFjup6AbQOM"
    assert media_id_from_source({"url": "https://youtu.be/VFjup6AbQOM"}) == "VFjup6AbQOM"
    assert media_id_from_source({"media_id": "EtU45PsaL1M", "url": "https://youtu.be/other"}) == "EtU45PsaL1M"
    assert media_id_from_source({"url": "https://example.com/not-youtube"}) is None


def test_digest_path_matches_proposal_filename(tmp_path):
    digest = tmp_path / "docs" / "digests"
    digest.mkdir(parents=True)
    (digest / "2026-09-08-andromeda.md").write_text("x", encoding="utf-8")
    found = digest_path_for_proposal(tmp_path, "proposal-2026-09-08-andromeda.json")
    assert found == digest / "2026-09-08-andromeda.md"
    assert digest_path_for_proposal(tmp_path, "proposal-missing.json") is None


def test_legacy_digest_chapters_become_seekable_items():
    items = build_legacy_review_items(
        {"candidate_kg_nodes": [{"id": "cbo", "name": "CBO", "type": "term", "description": "x"}]},
        "### 不要只关低效广告 — [02:29](https://youtu.be/x?t=149s)\n\nbody\n",
    )
    chapters = [i for i in items if i["kind"] == "key_point"]
    nodes = [i for i in items if i["kind"] == "node"]
    assert chapters[0]["evidence"][0]["t0"] == 149.0
    assert nodes[0]["evidence"] == []
    assert nodes[0]["label"] == "CBO"


def test_typed_proposal_is_not_replaced_by_legacy_nodes():
    """Legacy fallback is only for empty typed output; this must stay a no-op here."""
    typed = build_review_items(PROPOSAL)
    assert typed
    assert all(not i["id"].startswith("legacy-") for i in typed)


def test_empty_proposal_yields_no_targets():
    assert build_review_items({}) == []
    assert coverage([])["total"] == 0


# ---- decision log ------------------------------------------------------------------


def test_decision_is_persisted_and_reloaded(tmp_path):
    record_decision(tmp_path, "proposal-x.json", "term", "term:abc", "accept", reviewer="bg")
    review = read_review(tmp_path, "proposal-x.json")

    assert review_path(tmp_path, "proposal-x.json").exists()
    assert len(review["decisions"]) == 1
    entry = review["decisions"][0]
    assert entry["verdict"] == "accept"
    assert entry["scope"] == "item"
    assert entry["reviewer"] == "bg"


def test_claim_and_its_citation_are_decided_independently(tmp_path):
    """The §6.1 requirement: reject a citation while keeping the claim."""
    record_decision(tmp_path, "p.json", "edge", "edge:1", "accept")
    record_decision(tmp_path, "p.json", "edge", "edge:1", "reject", evidence_index=0,
                    note="quote does not mention either endpoint")

    latest = current_decisions(read_review(tmp_path, "p.json"))
    assert latest["edge|edge:1|-1"]["verdict"] == "accept"
    assert latest["edge|edge:1|0"]["verdict"] == "reject"
    assert "endpoint" in latest["edge|edge:1|0"]["note"]


def test_split_verdicts_are_surfaced_in_the_summary(tmp_path):
    record_decision(tmp_path, "p.json", "edge", "edge:1", "accept")
    record_decision(tmp_path, "p.json", "edge", "edge:1", "reject", evidence_index=0)
    record_decision(tmp_path, "p.json", "term", "term:2", "accept")

    summary = summarize(read_review(tmp_path, "p.json"))
    assert summary["counts"]["item"]["accept"] == 2
    assert summary["counts"]["evidence"]["reject"] == 1
    assert summary["accepted_with_rejected_evidence"] == ["edge|edge:1"]


def test_history_is_kept_but_latest_wins(tmp_path):
    record_decision(tmp_path, "p.json", "node", "surface:cbo", "accept")
    record_decision(tmp_path, "p.json", "node", "surface:cbo", "reject")

    review = read_review(tmp_path, "p.json")
    assert len(review["decisions"]) == 2  # the log is append-only
    latest = current_decisions(review)
    assert len(latest) == 1
    assert latest["node|surface:cbo|-1"]["verdict"] == "reject"


def test_unknown_vocabulary_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unknown verdict"):
        record_decision(tmp_path, "p.json", "term", "term:1", "looks-fine")
    with pytest.raises(ValueError, match="Unknown target kind"):
        record_decision(tmp_path, "p.json", "widget", "w:1", "accept")
    with pytest.raises(ValueError, match="target_id is required"):
        record_decision(tmp_path, "p.json", "term", "  ", "accept")
    # Nothing was written for any of the rejected calls.
    assert not review_path(tmp_path, "p.json").exists()


def test_corrupt_review_file_does_not_lose_new_decisions(tmp_path):
    path = review_path(tmp_path, "p.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    record_decision(tmp_path, "p.json", "term", "term:1", "accept")
    review = read_review(tmp_path, "p.json")
    assert len(review["decisions"]) == 1


def test_empty_review_is_summarizable(tmp_path):
    summary = summarize(read_review(tmp_path, "missing.json"))
    assert summary["decided_targets"] == 0
    assert summary["accepted_with_rejected_evidence"] == []
