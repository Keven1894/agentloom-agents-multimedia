"""Semantic segmentation tests.

A deterministic fake embedder stands in for the API: topics are encoded as orthogonal basis
vectors, so the correct boundaries are known exactly and the assertions are about the
algorithm rather than about an embedding model's judgement.
"""

import pytest

from agentloom_media.distillation.segmentation import (
    Segment,
    chunk_utterances,
    cosine_distance,
    percentile,
    segment_transcript,
)
from agentloom_media.transcripts.model import Transcript


class TopicEmbedder:
    """Maps a leading topic marker to a one-hot vector; identical topics are identical."""

    model = "fake-topic-embedder"

    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 6
            for index in range(6):
                if f"T{index}" in text:
                    vector[index] = 1.0
            if not any(vector):
                vector[0] = 1.0
            vectors.append(vector)
        return vectors


def _topic_transcript(topics_and_counts, seconds=5.0):
    """Build a transcript where each topic runs for a number of utterances."""
    segments = []
    clock = 0.0
    for topic, count in topics_and_counts:
        for _ in range(count):
            segments.append(
                {"start": clock, "end": clock + seconds, "text": f"T{topic} 内容讨论"}
            )
            clock += seconds
    return Transcript.from_segments("vid", segments)


# ---- primitives --------------------------------------------------------------------


def test_cosine_distance_bounds():
    assert cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_distance([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)
    # A zero vector has no direction; treat it as maximally uninformative, not as a crash.
    assert cosine_distance([0.0, 0.0], [1.0, 0.0]) == 1.0


def test_percentile_interpolates():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)
    assert percentile([5.0], 90) == 5.0
    assert percentile([], 90) == 0.0


def test_chunking_groups_short_utterances():
    t = _topic_transcript([(0, 12)], seconds=5.0)
    chunks = chunk_utterances(t.utterances, target_seconds=15.0)
    assert len(chunks) == 4
    assert all(c.t1 - c.t0 >= 15.0 for c in chunks)
    # Chunking must not lose utterances.
    covered = {uid for c in chunks for uid in c.utterance_ids}
    assert covered == {u.id for u in t.utterances}


# ---- segmentation ------------------------------------------------------------------


def _dominant_topic(segment: Segment) -> str:
    tokens = [
        token
        for token in segment.text.split()
        if len(token) == 2 and token.startswith("T")
    ]
    return max(set(tokens), key=tokens.count)


def test_boundaries_land_where_the_topic_changes():
    """Boundary resolution is the chunk size, so a cut lands within one chunk of the truth.

    A chunk that straddles a topic change necessarily contains both topics; segmentation can
    only cut on chunk edges. What must hold is that each segment is dominated by one topic and
    that the cuts are near the real changes.
    """
    # Three topics, 20 utterances each at 5s = 100s per topic.
    t = _topic_transcript([(0, 20), (1, 20), (2, 20)])
    chunk_seconds = 15.0
    segments, info = segment_transcript(
        t,
        chapters=None,
        embedder=TopicEmbedder(),
        chunk_seconds=chunk_seconds,
        min_seconds=60.0,
    )

    assert info["method"] == "semantic"
    assert len(segments) == 3
    assert [_dominant_topic(s) for s in segments] == ["T0", "T1", "T2"]

    assert segments[0].t0 == 0.0
    assert segments[1].t0 == pytest.approx(100.0, abs=chunk_seconds)
    assert segments[2].t0 == pytest.approx(200.0, abs=chunk_seconds)


def test_segments_cover_the_whole_transcript():
    t = _topic_transcript([(0, 20), (1, 20), (2, 20)])
    segments, _ = segment_transcript(t, None, TopicEmbedder(), min_seconds=60.0)

    covered = {uid for s in segments for uid in s.utterance_ids}
    assert covered == {u.id for u in t.utterances}
    # And no utterance is claimed by two segments.
    total = sum(len(s.utterance_ids) for s in segments)
    assert total == len(t.utterances)


def test_short_topics_are_merged_to_respect_min_duration():
    """A 10-second aside is not a segment."""
    t = _topic_transcript([(0, 20), (1, 2), (2, 20)])
    segments, _ = segment_transcript(t, None, TopicEmbedder(), min_seconds=60.0)
    assert all(s.duration >= 60.0 or s is segments[-1] for s in segments)
    assert len(segments) <= 3


def test_long_uniform_topic_is_split_to_respect_max_duration():
    """One topic for 10 minutes must still be broken up for review."""
    t = _topic_transcript([(0, 120)])  # 600s, all one topic
    segments, _ = segment_transcript(
        t, None, TopicEmbedder(), min_seconds=60.0, max_seconds=300.0
    )
    assert len(segments) >= 2
    assert all(s.duration <= 320.0 for s in segments)


def test_segment_t0_is_a_real_utterance_start():
    t = _topic_transcript([(0, 20), (1, 20)])
    starts = {u.t0 for u in t.utterances}
    segments, _ = segment_transcript(t, None, TopicEmbedder(), min_seconds=60.0)
    assert all(s.t0 in starts for s in segments)


def test_no_embedder_refuses_to_invent_boundaries():
    """Without embeddings we cannot justify a boundary, so we must not emit one."""
    t = _topic_transcript([(0, 20), (1, 20)])
    segments, info = segment_transcript(t, None, None)

    assert info["method"] == "unsegmented"
    assert info["embedded"] is False
    assert len(segments) == 1
    assert segments[0].boundary_source == "unsegmented"
    assert "not segmented" in info["note"].lower()


def test_native_chapters_are_preferred():
    t = _topic_transcript([(0, 20), (1, 20)])
    chapters = [
        {"title": "开场", "start_time": 0.0, "end_time": 100.0},
        {"title": "正题", "start_time": 100.0, "end_time": 200.0},
    ]
    segments, info = segment_transcript(t, chapters, TopicEmbedder(), max_seconds=300.0)

    assert info["method"] == "native+semantic"
    assert [s.title for s in segments] == ["开场", "正题"]
    assert all(s.boundary_source == "native" for s in segments)


def test_long_native_chapter_is_subdivided():
    t = _topic_transcript([(0, 40), (1, 40)])  # 400s in one chapter
    chapters = [{"title": "全部", "start_time": 0.0, "end_time": 400.0}]
    segments, info = segment_transcript(
        t, chapters, TopicEmbedder(), min_seconds=60.0, max_seconds=200.0
    )

    assert len(segments) >= 2
    assert info["method"] == "native+semantic"
    assert segments[0].boundary_source == "native+semantic"
    # Only the first sub-segment may keep the author's title; the rest are ours to name.
    assert segments[0].title == "全部"
    assert all(s.title is None for s in segments[1:])


def test_empty_transcript_yields_nothing():
    segments, info = segment_transcript(
        Transcript.from_segments("vid", []), None, TopicEmbedder()
    )
    assert segments == []
    assert info["method"] == "empty"


def test_single_chunk_transcript_is_one_segment():
    t = _topic_transcript([(0, 1)], seconds=4.0)
    segments, info = segment_transcript(t, None, TopicEmbedder())
    assert len(segments) == 1
    assert info["method"] == "unsegmented"


def test_tuning_knobs_come_from_env_and_arguments(monkeypatch):
    from agentloom_media.distillation.segmentation import segmentation_settings

    for name in (
        "SEGMENT_CHUNK_SECONDS",
        "SEGMENT_PERCENTILE",
        "SEGMENT_MIN_SECONDS",
        "SEGMENT_MAX_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)

    assert segmentation_settings()["chunk_seconds"] == 15.0

    monkeypatch.setenv("SEGMENT_CHUNK_SECONDS", "30")
    monkeypatch.setenv("SEGMENT_PERCENTILE", "70")
    assert segmentation_settings()["chunk_seconds"] == 30.0
    assert segmentation_settings()["percentile_threshold"] == 70.0
    # An explicit argument still wins over the environment.
    assert segmentation_settings(chunk_seconds=5.0)["chunk_seconds"] == 5.0

    # Unusable values fall back rather than crashing a whole ingestion.
    monkeypatch.setenv("SEGMENT_CHUNK_SECONDS", "not-a-number")
    assert segmentation_settings()["chunk_seconds"] == 15.0
    monkeypatch.setenv("SEGMENT_CHUNK_SECONDS", "-3")
    assert segmentation_settings()["chunk_seconds"] == 15.0


def test_max_below_min_is_reconciled():
    from agentloom_media.distillation.segmentation import segmentation_settings

    settings = segmentation_settings(min_seconds=120.0, max_seconds=60.0)
    assert settings["max_seconds"] == 120.0


def test_env_knobs_change_the_segmentation(monkeypatch):
    monkeypatch.setenv("SEGMENT_MIN_SECONDS", "60")
    t = _topic_transcript([(0, 20), (1, 20), (2, 20)])
    segments, info = segment_transcript(t, None, TopicEmbedder())
    assert len(segments) == 3
    assert info["min_seconds"] == 60.0


def test_segment_dict_records_boundary_provenance():
    t = _topic_transcript([(0, 20), (1, 20)])
    segments, _ = segment_transcript(t, None, TopicEmbedder(), min_seconds=60.0)
    payload = segments[1].to_dict()
    assert payload["boundary_source"] == "semantic"
    assert payload["boundary_distance"] is not None
    assert payload["t0"] == segments[1].t0
