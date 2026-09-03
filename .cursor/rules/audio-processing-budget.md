# Audio Processing & ASR Budget Rule

**Category**: Audio Engine Governance  
**Audience**: MediaLoom Agent  

---

## 1. Fast Track vs. Heavy Path vs. Slow Track Priority

Always attempt paths in this order:
1. **Fast Track**: official or auto captions (`youtube-transcript-api` / platform CC APIs). Latency < 3 seconds, cost = $0.
2. **Heavy Path**: `yt-dlp` format 140 / native audio object, then 16 kHz preprocess + ASR.
3. **Slow Track** (last resort): Playwright opens the watch page, records `HTMLMediaElement.captureStream()` audio, optionally at `playbackRate` 1.5×–2.0×, then ASR. Rescale every timestamp: `t_original = t_asr × playbackRate`.

Do not start Slow Track when Fast or Heavy already succeeded. Do not use OS loopback capture unless element capture and tab capture both fail.

---

## 2. Audio Stream Sizing & Limits

When extracting audio via `yt-dlp`:
1. Use native audio format: `format 140` (AAC / m4a container, ~128 kbps).
2. Avoid downloading the full video stream.
3. **25 MB Hard Boundary**:
   - Cloud Whisper APIs reject files > 25 MB.
   - For audio files > 25 MB (typically > 25 minutes at 128 kbps):
     - Chunk the audio into ~15-minute segments using silence detection or native video chapter boundaries.
     - Or downsample the stream to 16kHz mono 32kbps MP3/OGG before API dispatch.

---

## 3. ASR Engine Routing

- Default: OpenAI Whisper API (`whisper-1`).
- Fallback: Local `faster-whisper` (medium or large-v3) when offline or processing confidential research recordings.
- Slow Track default `playbackRate`: **1.5**. Budget mode: **2.0**. Never exceed 2.0 for Chinese speech.
