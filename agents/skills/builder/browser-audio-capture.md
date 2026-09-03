# skill:builder:browser-audio-capture: Slow Track Browser Playback Capture

**Category**: Audio Acquisition  
**Role**: role-builder (MediaLoom Engine)  
**Status**: Planned  
**Last Updated**: 2026-09-02  

## Purpose

Last-resort audio acquisition when Fast Track captions and Heavy Path stream extraction both fail. Play an operator-accessible watch page in Chromium, record the **decoded media-element audio** (not the whole desktop), optionally at 1.5×–2.0×, then hand the file to ASR.

## Preconditions

- Fast Track returned no captions.
- Heavy Path `yt-dlp` extract failed or returned no usable audio object.
- The URL is a page the operator can already play.
- Playwright + Chromium available.

## Procedure

1. Launch Chromium via Playwright and navigate to the URL.
2. Wait for `HTMLMediaElement` (`video` or `audio`). Fail closed if none appears within timeout.
3. Mute the element (`muted = false` is required for captureStream audio in some browsers; keep tab volume isolated instead of speakers when possible).
4. Set `playbackRate` to **1.5** (default) or **2.0** (budget). Never above 2.0 for Chinese speech.
5. Call `media.captureStream()`, keep only audio tracks, start `MediaRecorder` (`audio/webm;codecs=opus`).
6. `play()` and wait for `ended` (or duration + slack).
7. Stop recorder, downsample to 16 kHz mono, dispatch to ASR.
8. **Rescale timestamps**: `t_original = t_asr × playbackRate` before chapter alignment.

## Fallbacks

1. Chromium tab-audio capture if `captureStream()` is blocked.
2. OS loopback (WASAPI) only if both element and tab capture fail. Loopback mixes other desktop audio.

## Verification

- Output audio duration ≈ `original_duration / playbackRate` (±5%).
- After rescale, last segment `end` is within 5% of original media duration.
- Chapter `?t=` anchors seek to the correct second on the source URL.
