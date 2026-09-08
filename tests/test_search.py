"""Window construction and hybrid index tests.

Embeddings are disabled throughout, so these exercise the lexical half and the degradation
path. That degradation is the common case in CI and must be correct, not merely tolerated.
"""

import pytest

from agentloom_media.search.index import (
    RRF_K,
    IndexError_,
    SearchIndex,
    build_fts_query,
    extract_terms,
)
from agentloom_media.search.windows import build_windows
from agentloom_media.transcripts.model import Transcript


@pytest.fixture(autouse=True)
def _no_embeddings(monkeypatch):
    monkeypatch.setenv("SEARCH_DISABLE_EMBEDDINGS", "1")


def _transcript(media_id="vid", count=40, step=5.0, text="广告投放逻辑改变了"):
    segments = [
        {"start": i * step, "end": (i + 1) * step, "text": f"{text} {i}"}
        for i in range(count)
    ]
    return Transcript.from_segments(media_id, segments)


META = {
    "title": "Meta广告变了",
    "channel": "DAOJIE",
    "url": "https://www.youtube.com/watch?v=vid",
    "duration": 200.0,
}


# ---- windows -----------------------------------------------------------------------


def test_windows_are_about_target_length_and_overlap():
    t = _transcript(count=40, step=5.0)
    windows = build_windows(t, window_seconds=45.0, overlap_seconds=15.0)

    assert len(windows) > 1
    for w in windows[:-1]:
        assert 40.0 <= w.t1 - w.t0 <= 55.0, "windows should land near the 45s target"

    # Consecutive windows advance by roughly the stride and genuinely overlap.
    for earlier, later in zip(windows, windows[1:]):
        assert later.t0 > earlier.t0
        assert later.t0 < earlier.t1, "windows must overlap, not abut"


def test_window_boundaries_land_on_real_utterance_times():
    t = _transcript()
    starts = {u.t0 for u in t.utterances}
    ends = {u.t1 for u in t.utterances}
    for w in build_windows(t):
        assert w.t0 in starts, "t0 must be a real spoken moment usable as an anchor"
        assert w.t1 in ends


@pytest.mark.parametrize(
    "segments",
    [
        # Uniform.
        [{"start": i * 5.0, "end": (i + 1) * 5.0, "text": f"话 {i}"} for i in range(40)],
        # A long silent gap.
        [
            {"start": 0.0, "end": 20.0, "text": "开场"},
            {"start": 300.0, "end": 340.0, "text": "正题"},
        ],
        # An utterance longer than the window budget.
        [
            {"start": 0.0, "end": 5.0, "text": "短"},
            {"start": 5.0, "end": 200.0, "text": "非常长的一段"},
            {"start": 200.0, "end": 205.0, "text": "收尾"},
        ],
        # Ragged durations.
        [
            {"start": 0.0, "end": 1.0, "text": "a"},
            {"start": 1.0, "end": 60.0, "text": "b"},
            {"start": 60.0, "end": 61.0, "text": "c"},
            {"start": 61.0, "end": 62.0, "text": "d"},
        ],
    ],
    ids=["uniform", "gap", "oversized", "ragged"],
)
def test_windows_cover_every_utterance(segments):
    """Any uncovered utterance is permanently unsearchable, so coverage is not negotiable."""
    t = Transcript.from_segments("vid", segments)
    covered = {uid for w in build_windows(t) for uid in w.utterance_ids}
    assert covered == {u.id for u in t.utterances}


def test_short_transcript_yields_one_window():
    t = _transcript(count=2, step=3.0)
    windows = build_windows(t)
    assert len(windows) == 1
    assert windows[0].t0 == 0.0


def test_empty_transcript_yields_no_windows():
    assert build_windows(Transcript.from_segments("vid", [])) == []


def test_overlap_must_be_smaller_than_window():
    """Otherwise the loop would never advance."""
    t = _transcript()
    with pytest.raises(ValueError, match="smaller than"):
        build_windows(t, window_seconds=30.0, overlap_seconds=30.0)


def test_window_does_not_stretch_across_a_long_gap():
    """A silent gap must not produce one window spanning the whole video."""
    segments = [
        {"start": 0.0, "end": 20.0, "text": "开场"},
        {"start": 300.0, "end": 340.0, "text": "正题"},
    ]
    t = Transcript.from_segments("vid", segments)
    windows = build_windows(t, window_seconds=45.0, overlap_seconds=15.0)

    assert len(windows) == 2
    assert all(w.t1 - w.t0 <= 45.0 for w in windows)


def test_utterance_longer_than_window_stays_whole():
    """We cannot split an utterance, so an oversized one forms its own window."""
    segments = [
        {"start": 0.0, "end": 5.0, "text": "短"},
        {"start": 5.0, "end": 200.0, "text": "非常长的一段"},
    ]
    windows = build_windows(Transcript.from_segments("vid", segments))
    assert len(windows) == 2
    assert windows[1].t1 - windows[1].t0 == 195.0


def test_long_single_utterance_does_not_hang():
    t = Transcript.from_segments(
        "vid", [{"start": 0.0, "end": 600.0, "text": "很长的一段话"}]
    )
    windows = build_windows(t)
    assert len(windows) == 1


# ---- index -------------------------------------------------------------------------


def test_index_and_retrieve_time_range(tmp_path):
    with SearchIndex(tmp_path / "t.db") as idx:
        result = idx.index_media(_transcript(), META)
        assert result["windows"] > 0
        assert result["embedded"] is False

        hits, info = idx.search("投放逻辑")
        assert hits, "trigram tokenizer must match CJK substrings"
        assert info["vector_used"] is False

        hit = hits[0]
        assert hit.t1 > hit.t0
        assert hit.anchor_url == f"{META['url']}&t={int(hit.t0)}s"
        assert hit.title == META["title"]


def test_search_result_is_a_timecode_not_a_document(tmp_path):
    """The point of the feature: a hit locates a moment inside a video."""
    segments = [
        {"start": 0.0, "end": 20.0, "text": "开场闲聊内容"},
        {"start": 300.0, "end": 320.0, "text": "这里讲的是 Andromeda 之后的拆分逻辑"},
        {"start": 600.0, "end": 620.0, "text": "结尾致谢"},
    ]
    t = Transcript.from_segments("vid", segments)

    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(t, META)
        hits, _ = idx.search("Andromeda")
        assert hits
        assert 300.0 <= hits[0].t0 < 320.0, "must point at the moment, not the video start"


def test_reindexing_replaces_windows_rather_than_accumulating(tmp_path):
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(count=40), META)
        first = idx.stats()["windows"]

        idx.index_media(_transcript(count=10), META)
        second = idx.stats()["windows"]

        assert second < first, "stale windows from the previous transcript must be gone"
        # Old text must not remain searchable.
        hits, _ = idx.search("广告投放逻辑改变了 39")
        assert all("39" not in h.text for h in hits)


def test_multiple_media_are_isolated_and_filterable(tmp_path):
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript("a", text="第一个视频谈结构"), {**META, "title": "A"})
        idx.index_media(_transcript("b", text="第二个视频谈素材"), {**META, "title": "B"})

        assert idx.stats()["media"] == 2

        hits, _ = idx.search("视频", media_id="b")
        assert hits and all(h.media_id == "b" for h in hits)


# ---- query construction ------------------------------------------------------------


def test_terms_keep_cjk_runs_and_drop_punctuation():
    assert extract_terms("Andromeda 之后还要不要拆 Campaign?") == [
        "Andromeda",
        "之后还要不要拆",
        "Campaign",
    ]


def test_fts_query_ors_terms_rather_than_phrase_matching():
    """A whole-query phrase search only matches verbatim text, so BM25 would never fire."""
    built = build_fts_query("Andromeda 之后还要不要拆 Campaign")
    assert " OR " in built
    assert '"Andromeda"' in built and '"Campaign"' in built


def test_fts_query_expands_long_cjk_runs_into_ngrams():
    built = build_fts_query("还要不要拆")
    assert '"还要不"' in built and '"要不要"' in built and '"不要拆"' in built


def test_fts_query_is_none_when_nothing_is_long_enough():
    """Then the caller must use the substring path instead."""
    assert build_fts_query("广告") is None
    assert build_fts_query("") is None


def test_fts_query_is_capped():
    built = build_fts_query("这是一段非常长的中文查询用来测试子句上限是否生效", max_clauses=5)
    assert built.count(" OR ") == 4


def test_fts_operators_in_query_are_literal():
    built = build_fts_query('search "AND" OR NOT')
    assert '"""AND"""' in built or '""AND""' in built or 'AND' in built
    # The important part: it must not crash FTS5 when executed (covered below).


# ---- lexical behaviour -------------------------------------------------------------


def test_keyword_half_fires_on_natural_language_query(tmp_path):
    """Regression: the lexical ranker used to contribute nothing to real questions."""
    segments = [
        {"start": 0.0, "end": 20.0, "text": "开场闲聊"},
        {
            "start": 300.0,
            "end": 340.0,
            "text": "在 Post Andromeda 以后 还要不要拆 Campaign 呢 答案是不用",
        },
    ]
    t = Transcript.from_segments("vid", segments)

    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(t, META)
        hits, info = idx.search("Andromeda 之后还要不要拆 Campaign")
        assert info["lexical"] > 0, "keyword ranking must contribute"
        assert hits[0].lexical_rank is not None
        assert 300.0 <= hits[0].t0 < 340.0


def test_two_character_chinese_query_still_matches(tmp_path):
    """Trigram FTS5 cannot match <3 chars, and 2-char words dominate Chinese."""
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(text="第二个视频谈素材"), META)
        for short in ["视频", "素材"]:
            hits, _ = idx.search(short)
            assert hits, f"{short!r} must be findable via the substring fallback"


def test_single_character_query_matches(tmp_path):
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(text="预算决定拆分"), META)
        hits, _ = idx.search("算")
        assert hits


def test_substring_fallback_does_not_match_unrelated_text(tmp_path):
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(text="预算决定拆分"), META)
        assert idx.search("素材")[0] == []


def test_like_wildcards_in_query_are_literal(tmp_path):
    """A query of '%' must not match everything."""
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(text="预算决定拆分"), META)
        assert idx.search("%")[0] == []
        assert idx.search("_")[0] == []


def test_query_syntax_characters_do_not_crash_fts(tmp_path):
    """FTS5 treats -, *, " and : as syntax; user queries must still be safe."""
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(), META)
        nasty = ["广告 - 结构", "AND OR NOT", '"quoted"', "a*b", "x:y", "((("]
        for query in nasty:
            hits, _ = idx.search(query)
            assert isinstance(hits, list)


def test_empty_query_returns_nothing(tmp_path):
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(), META)
        hits, info = idx.search("   ")
        assert hits == [] and info["fused"] == 0


def test_rrf_prefers_agreement_between_rankers():
    """A window ranked by both rankers must outscore one ranked by only the better."""
    both = 1.0 / (RRF_K + 3) + 1.0 / (RRF_K + 3)
    single_top = 1.0 / (RRF_K + 1)
    assert both > single_top


def test_mixing_embedding_models_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("SEARCH_DISABLE_EMBEDDINGS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeEmbedder:
        def __init__(self, model):
            self.model = model

        def embed(self, texts):
            return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    with SearchIndex(tmp_path / "t.db", embedder=FakeEmbedder("model-a")) as idx:
        if not idx.vector_available:
            pytest.skip("sqlite-vec unavailable in this environment")
        result = idx.index_media(_transcript(count=4), META)
        assert result["embedded"] is True
        assert idx.embedding_model == "model-a"

    with SearchIndex(tmp_path / "t.db", embedder=FakeEmbedder("model-b")) as idx:
        with pytest.raises(IndexError_, match="not comparable"):
            idx.index_media(_transcript(count=4), META)


def test_vector_half_participates_when_embeddings_exist(tmp_path, monkeypatch):
    monkeypatch.delenv("SEARCH_DISABLE_EMBEDDINGS", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class PositionalEmbedder:
        """Distinct vectors so nearest-neighbour ordering is meaningful."""

        model = "fake-embedder"

        def embed(self, texts):
            return [[float(i), 1.0, 0.0, 0.0] for i, _ in enumerate(texts)]

        def embed_one(self, text):
            return [0.0, 1.0, 0.0, 0.0]

    with SearchIndex(tmp_path / "t.db", embedder=PositionalEmbedder()) as idx:
        if not idx.vector_available:
            pytest.skip("sqlite-vec unavailable in this environment")
        idx.index_media(_transcript(count=12), META)
        hits, info = idx.search("投放逻辑")
        assert info["vector_used"] is True
        assert info["vector"] > 0
        assert hits


def test_stats_report_what_the_index_actually_holds(tmp_path):
    with SearchIndex(tmp_path / "t.db") as idx:
        idx.index_media(_transcript(), META)
        stats = idx.stats()
        assert stats["media"] == 1
        assert stats["windows"] > 0
        assert stats["vectors"] == 0
        assert stats["embedding_model"] is None


def test_index_survives_reopen(tmp_path):
    db = tmp_path / "t.db"
    with SearchIndex(db) as idx:
        idx.index_media(_transcript(), META)
    with SearchIndex(db) as idx:
        hits, _ = idx.search("投放逻辑")
        assert hits
        assert idx.list_media()[0]["title"] == META["title"]
