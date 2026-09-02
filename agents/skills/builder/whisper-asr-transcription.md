# skill:builder:whisper-asr-transcription: Whisper ASR Transcription

**Category**: Acoustic Transcription  
**Role**: role-builder (MediaLoom Engine)  
**Status**: Production  
**Last Updated**: 2026-09-02  

## Purpose

Dispatch prepared audio files to the ASR engine (OpenAI Whisper API or local `faster-whisper`), returning full transcribed text and segment-level start/end timestamps.

## Preconditions

- `OPENAI_API_KEY` set in environment with access to `whisper-1`.
- Audio prepared via `skill:builder:audio-chunking-compression`.

## Procedure

1. Invoke `client.audio.transcriptions.create`:
   - `model`: `"whisper-1"`
   - `response_format`: `"verbose_json"`
   - `timestamp_granularities`: `["segment"]`
2. For multi-chunk inputs:
   - Add chunk offset seconds to each segment's `start` and `end`.
   - Concatenate all segment lists into a single chronological array.
3. Cache raw transcription segments in `.cache/transcripts/<video_id>_segments.json` to prevent duplicate API spend.

## Verification

- Result contains chronological segments with `start`, `end`, and `text`.
- Final segment timestamp is within 5% of total audio duration.
