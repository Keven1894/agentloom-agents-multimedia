"""Speech-mode tags: deixis is a demo, metaphor is lecture, spans stay inside the segment."""

from types import SimpleNamespace

from agentloom_media.distillation.speech_mode import (
    apply_speech_modes_to_proposal,
    classify_speech_modes,
    deixis_cues,
    heuristic_speech_mode,
    normalize_speech_mode,
)


def test_deixis_catches_pointing_not_metaphor():
    assert deixis_cues("那你来看一看这个Meta账户")
    assert deixis_cues("所以大家看到这边的图示")
    assert deixis_cues("大家可以看到Lifetime到")
    assert not deixis_cues("好的Campaign Structure必须让Meta看得见所有广告")
    assert not deixis_cues("你需要看的指标 Tracking Metric 完全不一样")
    assert not deixis_cues("是应该要让客户看到你的品牌")


def test_heuristic_marks_walkthrough_when_pointing():
    info = heuristic_speech_mode("那我们再来看一看这样的一个广告账户", 0.0, 60.0)
    assert info["speech_mode"] == "walkthrough"
    assert info["watch"] is True
    assert info["source"] == "heuristic"


def test_normalize_clips_spans_to_the_segment():
    info = normalize_speech_mode(
        {
            "speech_mode": "mixed",
            "watch": True,
            "watch_reason": "中间在读账户数字",
            "watch_spans": [{"t0": "00:10", "t1": "12:00", "reason": "读 ROAS"}],
        },
        t0=400.0,
        t1=620.0,
    )
    assert info["watch"] is True
    assert len(info["watch_spans"]) == 1
    span = info["watch_spans"][0]
    assert span["t0"] == 400.0
    assert span["t1"] == 620.0
    assert span["label"].startswith("06:40")


def test_unknown_mode_falls_back_using_cues():
    with_cues = normalize_speech_mode({"speech_mode": "demo"}, t0=0, t1=10, cues=["来看一看"])
    assert with_cues["speech_mode"] == "walkthrough"
    without = normalize_speech_mode({"speech_mode": "nope"}, t0=0, t1=10, cues=[])
    assert without["speech_mode"] == "lecture"
    assert without["watch"] is False


def test_classify_uses_llm_payload_and_falls_back(monkeypatch):
    from agentloom_media.distillation import speech_mode as module

    utterances = [
        SimpleNamespace(id="u1", t0=0.0, t1=8.0, text="那你来看一看这个Meta账户"),
        SimpleNamespace(id="u2", t0=60.0, t1=70.0, text="好的结构应让Meta看得见所有广告"),
    ]
    segments = [
        {
            "index": 0,
            "t0": 0.0,
            "t1": 50.0,
            "title": "开场账户",
            "utterance_ids": ["u1"],
        },
        {
            "index": 1,
            "t0": 50.0,
            "t1": 90.0,
            "title": "原则",
            "utterance_ids": ["u2"],
        },
    ]

    monkeypatch.setattr(
        "agentloom_media.distillation.passes._complete_json",
        lambda *a, **k: {
            "segments": [
                {
                    "index": 0,
                    "speech_mode": "reading_screen",
                    "watch": True,
                    "watch_reason": "对着账户念数字",
                    "watch_spans": [],
                },
                {
                    "index": 1,
                    "speech_mode": "lecture",
                    "watch": False,
                    "watch_reason": "在讲原则",
                    "watch_spans": [],
                },
            ]
        },
    )
    modes = classify_speech_modes(segments, utterances)
    assert modes[0]["speech_mode"] == "reading_screen"
    assert modes[0]["watch"] is True
    assert modes[0]["source"] == "llm"
    assert modes[1]["speech_mode"] == "lecture"
    assert modes[1]["watch"] is False

    monkeypatch.setattr(
        "agentloom_media.distillation.passes._complete_json",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")),
    )
    fallback = classify_speech_modes(segments, utterances)
    assert fallback[0]["source"] == "heuristic"
    assert fallback[0]["watch"] is True
    assert fallback[1]["watch"] is False


def test_apply_patches_proposal_segments():
    proposal = {
        "segments": [{"index": 0, "title": "A"}, {"index": 1, "title": "B"}],
        "passes": {"synthesis": "ok"},
    }
    updated = apply_speech_modes_to_proposal(
        proposal,
        {
            0: {
                "speech_mode": "walkthrough",
                "watch": True,
                "watch_reason": "指后台",
                "watch_spans": [],
            }
        },
    )
    assert updated == 1
    assert proposal["segments"][0]["watch"] is True
    assert "speech_mode" not in proposal["segments"][1]
    assert proposal["passes"]["speech_mode"] == "ok"
