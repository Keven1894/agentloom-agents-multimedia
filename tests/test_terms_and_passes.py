"""Grounded term extraction and the typed-pass guards."""

import pytest

from agentloom_media.distillation import terms as terms_module
from agentloom_media.distillation.passes import (
    _TIME_ONLY_TITLE,
    is_procedural,
    procedural_score,
    resolve_synthesis_model,
)
from agentloom_media.distillation.terms import extract_grounded_terms, find_quote_span
from agentloom_media.transcripts.model import Transcript

SEGMENTS = [
    {"start": 0.0, "end": 10.0, "text": "今天我们讲 Campaign Structure 的简化"},
    {"start": 10.0, "end": 20.0, "text": "大约 85% 的业务适合 One Campaign"},
    {"start": 20.0, "end": 30.0, "text": "CBO 会自动分配预算"},
]


def _transcript():
    return Transcript.from_segments("vid", SEGMENTS)


# ---- quote location ----------------------------------------------------------------


def test_exact_quote_is_located():
    t = _transcript()
    span = find_quote_span(t.text, "CBO 会自动分配预算")
    assert span is not None
    assert t.text[span[0] : span[1]] == "CBO 会自动分配预算"


def test_quote_without_asr_spaces_still_matches():
    """ASR inserts spaces a model reproducing a Chinese quote will usually drop."""
    t = _transcript()
    span = find_quote_span(t.text, "CBO会自动分配预算")
    assert span is not None
    assert "自动分配预算" in t.text[span[0] : span[1]]


def test_fabricated_quote_is_not_located():
    t = _transcript()
    assert find_quote_span(t.text, "这句话从来没有出现过") is None


def test_empty_inputs_are_safe():
    assert find_quote_span("", "abc") is None
    assert find_quote_span("abc", "") is None
    assert find_quote_span("abc", None) is None


def test_located_quote_resolves_to_a_time_span():
    t = _transcript()
    span = find_quote_span(t.text, "CBO 会自动分配预算")
    times = t.resolve_char_span(*span)
    assert times == (20.0, 30.0)


# ---- grounding gate ----------------------------------------------------------------


def _fake_llm(monkeypatch, payload):
    monkeypatch.setattr(terms_module, "_complete_json", lambda *a, **k: payload)


def test_key_points_are_ordered_by_anchor_time():
    """A reviewer follows the video forwards; points must not jump backwards."""
    from agentloom_media.distillation.passes import _sort_by_anchor

    points = [
        {"point": "third", "anchor": "02:50"},
        {"point": "first", "anchor": "01:04"},
        {"point": "unanchored", "anchor": None},
        {"point": "second", "anchor": "01:47"},
        {"point": "bad anchor", "anchor": "01:00-02:00"},
    ]
    ordered = [p["point"] for p in _sort_by_anchor(points, [])]
    assert ordered[:3] == ["first", "second", "third"]
    # Points we cannot place keep their relative order, at the end.
    assert ordered[3:] == ["unanchored", "bad anchor"]


def test_ungrounded_terms_are_dropped(monkeypatch):
    """A term whose quote is not in the transcript must not reach the digest."""
    _fake_llm(
        monkeypatch,
        {
            "terms": [
                {
                    "term": "CBO",
                    "type": "technical_item",
                    "description": "自动预算分配",
                    "quote": "CBO 会自动分配预算",
                },
                {
                    "term": "幻觉术语",
                    "type": "keyword",
                    "description": "编造的",
                    "quote": "这段话根本不存在于转写里",
                },
            ]
        },
    )

    result = extract_grounded_terms(_transcript(), "https://youtu.be/vid")
    assert [t["term"] for t in result["terms"]] == ["CBO"]
    assert result["stats"]["extracted"] == 2
    assert result["stats"]["grounded"] == 1
    assert result["stats"]["dropped"] == 1
    assert result["stats"]["dropped_examples"] == [
        {"term": "幻觉术语", "reason": "quote_not_found"}
    ]


def test_term_must_appear_in_its_own_quote(monkeypatch):
    """A real quote is not evidence for a term the quote does not contain.

    Observed in production: "Maximized Conversion Value" was grounded by a quote reading
    "Maximized Conversion Volume". The quote existed, so quote-location alone accepted it.
    """
    _fake_llm(
        monkeypatch,
        {
            "terms": [
                {
                    "term": "CBO 预算分配",
                    "type": "technical_item",
                    "description": "名称与引文不符",
                    "quote": "今天我们讲 Campaign Structure 的简化",
                }
            ]
        },
    )
    result = extract_grounded_terms(_transcript(), "https://youtu.be/vid")
    assert result["terms"] == []
    assert result["stats"]["dropped_reasons"] == {"term_not_in_quote": 1}


def test_term_containment_ignores_spacing_and_case():
    assert terms_module.term_supported_by_quote("One Campaign", "采用 onecampaign 结构")
    # ASR splits acronyms; ignoring spacing is what makes these match rather than a leak.
    assert terms_module.term_supported_by_quote("CBO", "所以 C B O 会分配预算")
    assert not terms_module.term_supported_by_quote("ROAS", "所以 CBO 会分配预算")
    assert not terms_module.term_supported_by_quote("", "任何引文")


def test_glossed_terms_are_supported_by_either_half():
    """"广告投资回报（ROAS）" appears verbatim nowhere, but both halves do."""
    assert terms_module.term_supported_by_quote(
        "广告投资回报（ROAS）", "现在的广告投资回报也超过了6倍"
    )
    assert terms_module.term_supported_by_quote(
        "广告投资回报（ROAS）", "这个 ROAS 在 6.22"
    )
    # The gloss must not become a wildcard: neither half present is still a drop.
    assert not terms_module.term_supported_by_quote(
        "广告投资回报（ROAS）", "今天我们讲 Campaign Structure 的简化"
    )


def test_drop_reasons_are_counted_separately(monkeypatch):
    _fake_llm(
        monkeypatch,
        {
            "terms": [
                {"term": "甲", "quote": "根本不存在的引文", "description": "x"},
                {"term": "乙", "quote": "CBO 会自动分配预算", "description": "x"},
                {"term": "CBO", "quote": "CBO 会自动分配预算", "description": "x"},
            ]
        },
    )
    result = extract_grounded_terms(_transcript(), "https://youtu.be/vid")
    assert [t["term"] for t in result["terms"]] == ["CBO"]
    assert result["stats"]["dropped_reasons"] == {
        "quote_not_found": 1,
        "term_not_in_quote": 1,
    }


def test_grounded_term_carries_anchor_and_span(monkeypatch):
    _fake_llm(
        monkeypatch,
        {
            "terms": [
                {
                    "term": "One Campaign",
                    "type": "technical_item",
                    "description": "单一 Campaign 结构",
                    "quote": "大约 85% 的业务适合 One Campaign",
                }
            ]
        },
    )

    result = extract_grounded_terms(_transcript(), "https://youtu.be/vid")
    term = result["terms"][0]
    assert term["t0"] == 10.0 and term["t1"] == 20.0
    assert term["timestamp"] == "00:10"
    assert term["anchor_url"] == "https://youtu.be/vid?t=10s"
    assert term["char_end"] > term["char_start"]


def test_unknown_term_type_is_normalized(monkeypatch):
    _fake_llm(
        monkeypatch,
        {
            "terms": [
                {
                    "term": "CBO",
                    "type": "made_up_type",
                    "quote": "CBO 会自动分配预算",
                    "description": "x",
                }
            ]
        },
    )
    result = extract_grounded_terms(_transcript(), "https://youtu.be/vid")
    assert result["terms"][0]["type"] == "keyword"


def test_terms_without_a_quote_are_dropped(monkeypatch):
    _fake_llm(monkeypatch, {"terms": [{"term": "CBO", "description": "no quote given"}]})
    result = extract_grounded_terms(_transcript(), "https://youtu.be/vid")
    assert result["terms"] == []
    assert result["stats"]["dropped"] == 1


def test_empty_transcript_needs_no_llm_call():
    result = extract_grounded_terms(Transcript.from_segments("vid", []), "u")
    assert result["terms"] == []
    assert result["stats"]["extracted"] == 0


# ---- typed pass guards -------------------------------------------------------------


def test_time_only_titles_are_recognized():
    """These are exactly the fake titles P0 removed; they must never come back."""
    for bad in ["00:00-05:00", "05:00", "00:00-05:00; 10:00-15:00", " 12:00 – 15:00 "]:
        assert _TIME_ONLY_TITLE.match(bad.strip()), bad
    for good in ["Campaign 结构简化", "Why CBO wins", "85% 的业务"]:
        assert not _TIME_ONLY_TITLE.match(good), good


def test_procedural_detector_separates_commentary_from_procedure():
    commentary = (
        "我觉得这个平台的变化很有意思 大家的看法可能不一样 "
        "这只是我的观察和分析 市场情绪也在变"
    )
    procedure = (
        "第一步 打开广告后台 然后 创建一个新的 Campaign "
        "接下来 设置 Campaign Objective 再 启用 CBO 最后 点击发布"
    )
    assert procedural_score(procedure) > procedural_score(commentary)
    assert is_procedural(procedure, threshold=5)
    assert not is_procedural(commentary, threshold=5)


def test_procedural_threshold_is_configurable(monkeypatch):
    monkeypatch.setenv("SKILL_PROCEDURAL_THRESHOLD", "1")
    assert is_procedural("首先我们看数据")
    monkeypatch.setenv("SKILL_PROCEDURAL_THRESHOLD", "99")
    assert not is_procedural("首先我们看数据")


def test_synthesis_model_defaults_and_overrides(monkeypatch):
    monkeypatch.delenv("SYNTHESIS_MODEL", raising=False)
    assert resolve_synthesis_model() == "gpt-5.6-terra"
    monkeypatch.setenv("SYNTHESIS_MODEL", "gpt-5.6-sol")
    assert resolve_synthesis_model() == "gpt-5.6-sol"
    assert resolve_synthesis_model("gpt-5.6-luna") == "gpt-5.6-luna"
