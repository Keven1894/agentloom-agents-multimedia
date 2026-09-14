"""Client overview is a reshape of P3 fields, not a new summary pass."""

from agentloom_media.review.decisions import read_review, record_decision
from agentloom_media.review.targets import build_review_items
from agentloom_media.ui.overview import (
    client_overview,
    duration_seconds,
    evidence_progress,
    list_overview_cards,
    overview_card,
    resolve_proposal_filename,
)


TYPED = {
    "source": {
        "title": "Meta广告变了",
        "channel": "DAOJIE",
        "url": "https://www.youtube.com/watch?v=EtU45PsaL1M",
        "media_id": "EtU45PsaL1M",
    },
    "document": {
        "executive_summary": "整片主张简化 Campaign。",
        "thesis": "85% 账户应整合结构。",
        "structure": "先讲 Andromeda，再讲结构。",
        "takeaways": ["少拆 Campaign", "多做创意"],
    },
    "segments": [
        {
            "index": 0,
            "t0": 0.0,
            "t1": 232.5,
            "title": "结构与创意",
            "analysis": "重点从拆账户转向给算法更多素材。",
            "speech_mode": "walkthrough",
            "watch": True,
            "watch_reason": "开场在指三个账户。",
            "watch_spans": [{"t0": 0.0, "t1": 40.0, "reason": "第一个账户"}],
            "key_points": [
                {"point": "高转化不代表要更多 Campaign", "anchor": "01:04"},
                {"point": "没有时间的要点", "anchor": None},
            ],
        },
        {
            "index": 1,
            "t0": 232.5,
            "t1": 330.0,
            "title": "三项标准",
            "analysis": "结构是测试与学习的机制。",
            "key_points": [],
        },
    ],
    "terms": [
        {
            "term": "CBO",
            "type": "technical_item",
            "quote": "CBO 会自动分配预算",
            "t0": 20.0,
            "t1": 30.0,
        }
    ],
}


def test_resolve_proposal_filename_peels_percent_encoding():
    name = "proposal-2026-09-08-meta广告变了.json"
    from urllib.parse import quote

    assert resolve_proposal_filename(name) == name
    assert resolve_proposal_filename(quote(name, safe="")) == name
    assert resolve_proposal_filename(quote(quote(name, safe=""), safe="")) == name
    assert resolve_proposal_filename(f"/tmp/{quote(name)}") == name


def test_duration_falls_back_to_last_segment():
    assert duration_seconds(TYPED) == 330.0
    assert duration_seconds({"source": {"duration": 972}, "segments": []}) == 972


def test_client_overview_uses_existing_document_and_segments():
    briefing = client_overview(TYPED, "proposal-demo.json")
    assert briefing["has_structured_overview"] is True
    assert briefing["document"]["executive_summary"].startswith("整片主张")
    assert briefing["document"]["thesis"].startswith("85%")
    assert briefing["source"]["duration_label"] == "05:30"
    assert briefing["source"]["media_id"] == "EtU45PsaL1M"
    assert len(briefing["segments"]) == 2

    first = briefing["segments"][0]
    assert first["title"] == "结构与创意"
    assert first["span_label"] == "00:00–03:52"
    assert first["url"].endswith("t=0s")
    assert first["analysis"].startswith("重点从拆账户")
    assert first["key_points"][0]["label"] == "01:04"
    assert "t=64s" in first["key_points"][0]["url"]
    assert first["key_points"][1]["url"] is None
    assert first["watch"] is True
    assert first["speech_mode"] == "walkthrough"
    assert briefing["watch_count"] == 1
    assert first["watch_spans"][0]["label"] == "00:00–00:40"


def test_legacy_proposal_is_marked_unstructured():
    briefing = client_overview(
        {"source": {"title": "Old", "url": "https://youtu.be/abcdefghijk"}},
        "proposal-old.json",
    )
    assert briefing["has_structured_overview"] is False
    assert briefing["segments"] == []
    assert briefing["source"]["media_id"] == "abcdefghijk"


def test_evidence_progress_counts_undecided_items(tmp_path):
    progress = evidence_progress(TYPED, {})
    assert progress["total"] >= 3  # two key points + one term
    assert progress["decided"] == 0
    assert progress["pending"] == progress["total"]
    assert progress["legacy"] is False

    first = build_review_items(TYPED)[0]
    record_decision(
        tmp_path,
        "proposal-demo.json",
        kind=first["kind"],
        target_id=first["id"],
        verdict="accept",
    )
    after = evidence_progress(TYPED, read_review(tmp_path, "proposal-demo.json"))
    assert after["decided"] == 1
    assert after["pending"] == progress["pending"] - 1


def test_overview_card_and_list(tmp_path):
    proposals = tmp_path / "proposals"
    proposals.mkdir()
    (proposals / "proposal-demo.json").write_text(
        __import__("json").dumps(TYPED, ensure_ascii=False),
        encoding="utf-8",
    )
    cards = list_overview_cards(tmp_path)
    assert len(cards) == 1
    assert cards[0]["title"] == "Meta广告变了"
    assert cards[0]["segment_count"] == 2
    assert cards[0]["review"]["pending"] > 0

    card = overview_card(TYPED, "proposal-demo.json")
    assert "executive_summary" not in card
    assert card["channel"] == "DAOJIE"
