"""ASR engine selection and glossary tests."""

import pytest

from agentloom_media.audio import engines
from agentloom_media.transcripts import normalize


def test_default_engine_is_whisper_api(monkeypatch):
    monkeypatch.delenv("ASR_ENGINE", raising=False)
    assert engines.resolve_asr_engine() == "whisper_api"


def test_env_and_explicit_override(monkeypatch):
    monkeypatch.setenv("ASR_ENGINE", "faster_whisper")
    assert engines.resolve_asr_engine() == "faster_whisper"
    assert engines.resolve_asr_engine("whisper_api") == "whisper_api"


def test_unknown_engine_names_the_alternatives(tmp_path):
    with pytest.raises(engines.AsrEngineUnavailable) as exc:
        engines.transcribe(tmp_path / "a.m4a", engine="nope")
    assert "whisper_api" in str(exc.value)


def test_deferred_engines_explain_themselves(tmp_path):
    for name in ("whisperx", "whisper_cpp"):
        with pytest.raises(engines.AsrEngineUnavailable) as exc:
            engines.transcribe(tmp_path / "a.m4a", engine=name)
        assert name in str(exc.value)


def test_whisper_api_reports_segment_granularity(tmp_path, monkeypatch):
    """The API returns segment timings; claiming word-level would be a lie."""
    import agentloom_media.audio.asr as asr

    monkeypatch.setattr(
        asr,
        "transcribe_audio_file",
        lambda path, language=None, prompt=None: {
            "text": "hi",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hi"}],
            "engine": "openai:whisper-1",
            "language": "en",
        },
    )

    result = engines.transcribe(tmp_path / "a.m4a", engine="whisper_api")
    assert result["timing_granularity"] == "segment"
    assert result["words"] == []
    assert result["model"] == "whisper-1"


def test_language_names_reduce_to_iso_codes():
    """Whisper reports 'chinese'; callers pass 'zh'. Manifests must not hold both."""
    assert engines.normalize_language("chinese") == "zh"
    assert engines.normalize_language("Chinese") == "zh"
    assert engines.normalize_language("zh") == "zh"
    assert engines.normalize_language("") is None
    assert engines.normalize_language(None) is None
    # Unknown values pass through rather than being dropped.
    assert engines.normalize_language("yue") == "yue"


def test_whisper_api_normalizes_reported_language(tmp_path, monkeypatch):
    import agentloom_media.audio.asr as asr

    monkeypatch.setattr(
        asr,
        "transcribe_audio_file",
        lambda path, language=None, prompt=None: {
            "segments": [],
            "engine": "openai:whisper-1",
            "language": "chinese",
        },
    )
    assert engines.transcribe(tmp_path / "a.m4a", engine="whisper_api")["language"] == "zh"


def test_whisper_api_receives_language_and_glossary_prompt(tmp_path, monkeypatch):
    import agentloom_media.audio.asr as asr

    captured = {}

    def fake(path, language=None, prompt=None):
        captured["language"] = language
        captured["prompt"] = prompt
        return {"segments": [], "engine": "openai:whisper-1"}

    monkeypatch.setattr(asr, "transcribe_audio_file", fake)
    engines.transcribe(
        tmp_path / "a.m4a", engine="whisper_api", language="zh", initial_prompt="Meta、CBO"
    )
    assert captured == {"language": "zh", "prompt": "Meta、CBO"}


def test_glossary_loads_from_repo_config():
    terms = normalize.load_glossary()
    assert "Andromeda" in terms and "CBO" in terms
    assert not any(t.startswith("#") for t in terms), "comments must be stripped"


def test_glossary_env_terms_take_precedence(monkeypatch):
    monkeypatch.setenv("ASR_GLOSSARY", "Foo, Bar")
    terms = normalize.load_glossary()
    assert terms[0] == "Foo" and terms[1] == "Bar"


def test_initial_prompt_is_a_term_list(monkeypatch):
    monkeypatch.setenv("ASR_GLOSSARY", "Meta, CBO")
    monkeypatch.setenv("ASR_GLOSSARY_FILE", "/nonexistent")
    assert normalize.build_initial_prompt() == "Meta, CBO"


def test_normalizer_can_be_disabled(monkeypatch):
    monkeypatch.setenv("TRANSCRIPT_NORMALIZE_SCRIPT", "off")
    fn, info = normalize.build_script_normalizer()
    assert fn is None
    assert info["applied"] is False


def test_normalizer_reports_whether_it_applied(monkeypatch):
    monkeypatch.delenv("TRANSCRIPT_NORMALIZE_SCRIPT", raising=False)
    fn, info = normalize.build_script_normalizer()
    # OpenCC may or may not be installed; either way the manifest must not overclaim.
    if fn is None:
        assert info["applied"] is False
    else:
        assert info["applied"] is True
        assert fn("廣告") == "广告"
