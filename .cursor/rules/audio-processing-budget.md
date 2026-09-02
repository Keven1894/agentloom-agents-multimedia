# Audio Processing & ASR Budget Rule

**Category**: Audio Engine Governance  
**Audience**: MediaLoom Agent  

---

## 1. Fast Track vs. Heavy Path Priority

Always attempt the **Fast Track** first:
1. Probe video for official or automatically generated captions (`youtube-transcript-api`).
2. If captions are present, download them directly (latency < 3 seconds, bandwidth < 50 KB, cost = $0).
3. Only fall back to **Heavy Path** if captions are disabled or unavailable.

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
