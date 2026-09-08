"""Canonical transcript model, store, and normalization tests."""

import json

import pytest

from agentloom_media.transcripts import store
from agentloom_media.transcripts.model import PIPELINE_VERSION, Transcript

SEGMENTS = [
    {"start": 0.0, "end": 4.0, "text": "廣告投放邏輯改變了"},
    {"start": 4.0, "end": 9.5, "text": "Campaign Structure 要簡化"},
    {"start": 9.5, "end": 22.0, "text": "預算不是越大越複雜"},
]


def _t2s(text: str) -> str:
    """Stand-in normalizer so these tests do not depend on OpenCC being installed."""
    table = str.maketrans({"廣": "广", "邏": "逻", "輯": "辑", "簡": "简", "預": "预", "複": "复", "雜": "杂", "變": "变", "應": "应"})
    return text.translate(table)


def test_char_offsets_round_trip_to_times():
    t = Transcript.from_segments("vid", SEGMENTS)
    for utterance in t.utterances:
        assert t.text[utterance.char_start : utterance.char_end] == utterance.text
        span = t.resolve_char_span(utterance.char_start, utterance.char_end)
        assert span == (utterance.t0, utterance.t1)


def test_char_span_across_utterances_unions_times():
    t = Transcript.from_segments("vid", SEGMENTS)
    span = t.resolve_char_span(t.utterances[0].char_start, t.utterances[1].char_end)
    assert span == (0.0, 9.5)


def test_char_span_out_of_range_returns_none():
    t = Transcript.from_segments("vid", SEGMENTS)
    assert t.resolve_char_span(10**6, 10**6 + 5) is None


def test_normalizer_preserves_raw_text():
    t = Transcript.from_segments("vid", SEGMENTS, normalizer=_t2s)
    assert t.utterances[0].text == "广告投放逻辑改变了"
    assert t.utterances[0].text_raw == "廣告投放邏輯改變了"
    assert "廣" in t.text_raw and "廣" not in t.text


def test_granularity_defaults_to_segment_and_upgrades_with_words():
    assert Transcript.from_segments("vid", SEGMENTS).timing_granularity == "segment"

    words = [{"w": "廣告", "t0": 0.0, "t1": 0.6}]
    with_words = Transcript.from_segments("vid", SEGMENTS, words=words)
    assert with_words.timing_granularity == "word"
    assert with_words.words[0].t1 == 0.6


def test_utterance_at_finds_containing_utterance():
    t = Transcript.from_segments("vid", SEGMENTS)
    assert t.utterance_at(5.0).id == t.utterances[1].id
    assert t.utterance_at(-1.0) is None


def test_empty_segments_yield_empty_transcript():
    t = Transcript.from_segments("vid", [])
    assert t.utterances == [] and t.text == "" and t.duration == 0.0
    assert t.resolve_char_span(0, 1) is None


def test_serialization_round_trip():
    original = Transcript.from_segments("vid", SEGMENTS, normalizer=_t2s, language="zh")
    restored = Transcript.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored.to_dict() == original.to_dict()


def test_store_keys_on_engine_so_engines_do_not_collide(tmp_path):
    api = Transcript.from_segments("vid", SEGMENTS)
    local = Transcript.from_segments("vid", SEGMENTS[:2])

    store.save_transcript(tmp_path, api, source="asr", engine="whisper_api", model="whisper-1")
    store.save_transcript(
        tmp_path, local, source="asr", engine="faster_whisper", model="large-v3"
    )

    got_api = store.load_transcript(tmp_path, "vid", engine="whisper_api", model="whisper-1")
    got_local = store.load_transcript(
        tmp_path, "vid", engine="faster_whisper", model="large-v3"
    )
    assert len(got_api["transcript"].utterances) == 3
    assert len(got_local["transcript"].utterances) == 2

    # An engine that was never run must not be served another engine's output.
    assert store.load_transcript(tmp_path, "vid", engine="whisperx") is None


def test_store_prefers_most_precise_when_no_engine_requested(tmp_path):
    cue = Transcript.from_segments("vid", SEGMENTS, timing_granularity="cue")
    seg = Transcript.from_segments("vid", SEGMENTS, timing_granularity="segment")

    store.save_transcript(tmp_path, cue, source="captions", engine="youtube_captions")
    store.save_transcript(tmp_path, seg, source="asr", engine="whisper_api", model="whisper-1")

    best = store.load_transcript(tmp_path, "vid")
    assert best["variant"]["timing_granularity"] == "segment"


def test_known_engine_outranks_migrated_legacy_at_equal_precision(tmp_path):
    """A migrated pre-P1 cache must not keep winning just because it was written first."""
    t = Transcript.from_segments("vid", SEGMENTS)
    store.save_transcript(
        tmp_path, t, source="legacy", engine="unknown", engine_version="pre-p1-cache"
    )
    store.save_transcript(tmp_path, t, source="asr", engine="whisper_api", model="whisper-1")

    best = store.load_transcript(tmp_path, "vid")
    assert best["variant"]["engine"] == "whisper_api"


def test_manifest_records_provenance_and_hash(tmp_path):
    t = Transcript.from_segments("vid", SEGMENTS, language="zh")
    variant = store.save_transcript(
        tmp_path,
        t,
        source="asr",
        engine="whisper_api",
        model="whisper-1",
        engine_version="openai:whisper-1",
        media_url="https://youtu.be/vid",
        normalization={"opencc_config": "t2s", "applied": True},
    )
    assert variant["pipeline_version"] == PIPELINE_VERSION
    assert len(variant["transcript_sha256"]) == 64

    manifest = store.read_manifest(tmp_path, "vid")
    assert manifest["media_url"] == "https://youtu.be/vid"
    assert manifest["variants"][0]["engine"] == "whisper_api"
    assert manifest["variants"][0]["normalization"]["applied"] is True


def test_resaving_same_variant_replaces_rather_than_duplicates(tmp_path):
    t = Transcript.from_segments("vid", SEGMENTS)
    for _ in range(3):
        store.save_transcript(tmp_path, t, source="asr", engine="whisper_api", model="whisper-1")
    assert len(store.read_manifest(tmp_path, "vid")["variants"]) == 1


def test_legacy_bare_list_cache_is_found_for_migration(tmp_path):
    cache_dir = tmp_path / ".cache" / "transcripts"
    cache_dir.mkdir(parents=True)
    (cache_dir / "vid_segments.json").write_text(
        json.dumps(SEGMENTS), encoding="utf-8"
    )

    legacy = store.find_legacy_segments(tmp_path, "vid")
    assert legacy["source"] == "legacy"
    assert legacy["engine"] == "unknown", "unknown provenance must not be invented"
    assert len(legacy["segments"]) == 3


def test_legacy_p0_wrapper_cache_keeps_its_provenance(tmp_path):
    cache_dir = tmp_path / ".cache" / "transcripts"
    cache_dir.mkdir(parents=True)
    (cache_dir / "vid__asr__openai-whisper-1__zh__v1.json").write_text(
        json.dumps(
            {"source": "asr", "engine": "openai:whisper-1", "language": "zh", "segments": SEGMENTS}
        ),
        encoding="utf-8",
    )

    legacy = store.find_legacy_segments(tmp_path, "vid")
    assert legacy["engine"] == "openai:whisper-1"
    assert legacy["language"] == "zh"


def test_no_cache_returns_none(tmp_path):
    assert store.load_transcript(tmp_path, "missing") is None
    assert store.find_legacy_segments(tmp_path, "missing") is None
