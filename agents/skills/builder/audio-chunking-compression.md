# skill:builder:audio-chunking-compression: Audio Compression & Chunking

**Category**: Audio Preprocessing  
**Role**: role-builder (MediaLoom Engine)  
**Status**: Production  
**Last Updated**: 2026-09-02  

## Purpose

Prepare audio streams for cloud ASR (Whisper API) by enforcing the **25 MB upload limit**. Handles downsampling to 16kHz mono MP3 and temporal segment chunking for long-form recordings.

## Preconditions

- `imageio-ffmpeg` installed or system `ffmpeg` binary available in PATH.
- Input audio file downloaded (typically format 140 m4a).

## Heuristic & Rules

1. **Size Threshold (< 24 MB)**:
   - If raw audio is <= 24 MB, use directly without quality loss.
2. **Downsampling (16kHz Mono 32kbps)**:
   - For speech recognition, 16kHz mono at 32 kbps achieves equivalent transcription accuracy while reducing file size by 75%.
   - Reduces a 35-minute audio file from ~33 MB to ~8 MB.
3. **Temporal Chunking (for videos > 1.5 hours)**:
   - If compressed audio still exceeds 24 MB, chunk into 15-minute segments with a 1-second overlap.
   - Record offset seconds for each chunk to preserve global timestamps.

## Verification

Every audio segment emitted for ASR:
- Must have file size strictly `< 24.0 MB`.
- Must carry its start offset in seconds relative to original media.
