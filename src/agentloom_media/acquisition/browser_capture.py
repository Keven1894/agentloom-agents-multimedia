"""browser_capture.py — Slow Track: Browser Playback Audio Capture (Authorized Personal Study Bridge).

Captures decoded audio directly from HTMLMediaElement via HTML5 captureStream() / MediaRecorder.
Used as an authorized personal study bridge when direct protocol stream extraction is blocked
by media player protections (MSE, Blob, DRM wrappers on purchased courses/tutorials).

Preserves timestamp provenance via playbackRate compensation:
    t_original = t_asr * playback_rate
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("agentloom_media.browser_capture")


@dataclass
class BrowserCaptureConfig:
    playback_rate: float = 1.5  # 1.5x (recommended) or 2.0x (aggressive)
    timeout_sec: int = 3600
    headless: bool = False      # Default False to allow operator interaction / login if needed
    user_data_dir: Optional[str] = None  # Persistent browser context to retain course login session
    max_duration_sec: Optional[int] = None


def is_browser_capture_available() -> bool:
    """Check if Playwright runtime is available in current environment."""
    try:
        import playwright
        return True
    except ImportError:
        return False


def rescale_timestamps(segments: List[Dict[str, Any]], playback_rate: float) -> List[Dict[str, Any]]:
    """Rescale accelerated ASR timestamps back onto the original media timeline.
    
    Invariant:
        t_original = t_asr * playback_rate
    """
    if abs(playback_rate - 1.0) < 1e-4:
        return segments

    rescaled = []
    for s in segments:
        orig_s = dict(s)
        orig_s["start"] = round(s.get("start", 0) * playback_rate, 2)
        orig_s["end"] = round(s.get("end", 0) * playback_rate, 2)
        if "duration" in orig_s:
            orig_s["duration"] = round(orig_s["duration"] * playback_rate, 2)
        rescaled.append(orig_s)
    return rescaled


async def capture_browser_audio_async(
    url: str,
    output_audio_path: Path,
    config: Optional[BrowserCaptureConfig] = None
) -> Path:
    """Capture in-page audio stream from HTMLMediaElement at configured playbackRate."""
    if not is_browser_capture_available():
        raise RuntimeError(
            "Playwright is required for Slow Track browser capture.\n"
            "Install with: pip install playwright && playwright install chromium"
        )

    from playwright.async_api import async_playwright

    cfg = config or BrowserCaptureConfig()
    output_audio_path = Path(output_audio_path)
    output_audio_path.parent.mkdir(parents=True, exist_ok=True)
    temp_webm = output_audio_path.with_suffix(".webm")

    logger.info(f"Initiating browser playback capture for {url} (playbackRate={cfg.playback_rate}x)")

    async with async_playwright() as p:
        # Launch browser with optional user profile to maintain session login
        launch_args = ["--autoplay-policy=no-user-gesture-required"]
        if cfg.user_data_dir:
            context = await p.chromium.launch_persistent_context(
                user_data_dir=cfg.user_data_dir,
                headless=cfg.headless,
                args=launch_args
            )
            page = context.pages[0] if context.pages else await context.new_page()
        else:
            browser = await p.chromium.launch(headless=cfg.headless, args=launch_args)
            context = await browser.new_context()
            page = await context.new_page()

        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # Wait for video/audio element to appear
            await page.wait_for_selector("video, audio", timeout=30000)

            # Inject MediaRecorder audio capture bridge into the page
            inject_script = f"""
            (() => {{
                window.__mediaLoomChunks = [];
                window.__mediaLoomEnded = false;
                window.__mediaLoomError = null;

                const media = document.querySelector('video, audio');
                if (!media) {{
                    window.__mediaLoomError = "No HTMLMediaElement found on page";
                    return;
                }}

                try {{
                    media.playbackRate = {cfg.playback_rate};
                    const stream = media.captureStream ? media.captureStream() : (media.mozCaptureStream ? media.mozCaptureStream() : null);
                    if (!stream) {{
                        window.__mediaLoomError = "HTMLMediaElement.captureStream() is not supported on this element";
                        return;
                    }}

                    const audioTracks = stream.getAudioTracks();
                    if (!audioTracks || audioTracks.length === 0) {{
                        window.__mediaLoomError = "No audio tracks present in captured media stream";
                        return;
                    }}

                    const audioStream = new MediaStream(audioTracks);
                    const recorder = new MediaRecorder(audioStream, {{ mimeType: 'audio/webm;codecs=opus' }});

                    recorder.ondataavailable = (e) => {{
                        if (e.data && e.data.size > 0) {{
                            const reader = new FileReader();
                            reader.onloadend = () => {{
                                window.__mediaLoomChunks.push(reader.result.split(',')[1]);
                            }};
                            reader.readAsDataURL(e.data);
                        }}
                    }};

                    recorder.onstop = () => {{
                        window.__mediaLoomEnded = true;
                    }};

                    media.onended = () => {{
                        recorder.stop();
                    }};

                    recorder.start(5000); // chunk every 5 seconds
                    media.play().catch(e => console.warn("Auto-play warning:", e));
                    window.__mediaLoomRecorder = recorder;
                }} catch (err) {{
                    window.__mediaLoomError = err.toString();
                }}
            }})();
            """
            await page.evaluate(inject_script)

            # Polling loop to check status and write chunks
            recorded_base64_chunks: List[str] = []
            start_time = asyncio.get_event_loop().time()

            while True:
                err = await page.evaluate("window.__mediaLoomError")
                if err:
                    raise RuntimeError(f"Browser capture script error: {err}")

                ended = await page.evaluate("window.__mediaLoomEnded")
                new_chunks = await page.evaluate("""
                    (() => {
                        const chunks = window.__mediaLoomChunks;
                        window.__mediaLoomChunks = [];
                        return chunks;
                    })()
                """)

                if new_chunks:
                    recorded_base64_chunks.extend(new_chunks)

                if ended:
                    break

                elapsed = asyncio.get_event_loop().time() - start_time
                if cfg.max_duration_sec and elapsed >= cfg.max_duration_sec:
                    await page.evaluate("if (window.__mediaLoomRecorder) window.__mediaLoomRecorder.stop();")
                    break

                if elapsed > cfg.timeout_sec:
                    raise TimeoutError(f"Browser capture timed out after {cfg.timeout_sec}s")

                await asyncio.sleep(2.0)

            # Assemble WebM
            import base64
            with open(temp_webm, "wb") as f:
                for b64 in recorded_base64_chunks:
                    f.write(base64.b64decode(b64))

            # Transcode WebM to 16kHz mono MP3 via imageio_ffmpeg
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            cmd = [
                ffmpeg_exe, "-y",
                "-i", str(temp_webm),
                "-ar", "16000",
                "-ac", "1",
                "-b:a", "32k",
                str(output_audio_path)
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if temp_webm.exists():
                temp_webm.unlink()

            return output_audio_path

        finally:
            if not cfg.user_data_dir:
                await browser.close()
            else:
                await context.close()


def capture_browser_audio(
    url: str,
    output_audio_path: Path,
    config: Optional[BrowserCaptureConfig] = None
) -> Path:
    """Synchronous entry point for browser audio capture."""
    return asyncio.run(capture_browser_audio_async(url, output_audio_path, config))
