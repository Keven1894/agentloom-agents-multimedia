# skill:builder:probe-media-stream: Probe Media Stream & Metadata

**Category**: Audio Acquisition  
**Role**: role-builder (MediaLoom Engine)  
**Status**: Production  
**Last Updated**: 2026-09-02  

## Purpose

Inspect a target multimedia URL (YouTube, Vimeo, podcast, or local media file) to extract metadata, detect native chapter markers, and determine whether subtitles are available or if audio extraction is required.

## Preconditions

- `yt-dlp` installed and updated.
- Target URL is accessible.

## Procedure

1. Run `yt_dlp.YoutubeDL` with `simulate: True` and `download: False`.
2. Extract:
   - Video title, duration, author/channel, and description.
   - Native chapters (`chapters` list with `start_time`, `end_time`, and `title`).
   - Check `subtitles` and `automatic_captions` keys.
   - Inspect audio formats to locate format 140 (native AAC/m4a container).
3. If captions are present, route to Fast Track (`fetch_fast_transcript`).
4. If captions are disabled or unavailable, route to Heavy Path (`extract_audio_stream`).
5. If Heavy Path extraction fails, route to Slow Track (`browser_capture` at playbackRate 1.5, then rescale ASR timestamps).

## Verification

Probe output contains:
- Valid `id` and non-zero `duration`.
- List of native chapters (or empty list triggering auto-partitioning).
- Explicit boolean `has_subtitles`.
