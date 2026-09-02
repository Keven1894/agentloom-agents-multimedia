"""Command-line interface for AgentLoom Multimedia Knowledge Ingestion."""

import os
from pathlib import Path
import click
import dotenv
from rich.console import Console
from rich.panel import Panel

from agentloom_media.acquisition.probe import probe_media
from agentloom_media.acquisition.fast_transcript import fetch_fast_transcript
from agentloom_media.acquisition.audio_extractor import extract_audio_stream
from agentloom_media.audio.asr import transcribe_audio_file
from agentloom_media.distillation.chapter_aligner import align_transcript_with_chapters
from agentloom_media.distillation.distiller import distill_aligned_chapters
from agentloom_media.proposals.emitter import emit_distillation_artifacts

console = Console()
dotenv.load_dotenv()


@click.group()
def main():
    """AgentLoom Multimedia Knowledge Ingestion CLI."""
    pass


@main.command()
@click.argument("url")
@click.option("--output-dir", default=None, help="Repository root directory to save artifacts.")
@click.option("--model", default="gpt-4o-mini", help="LLM model for distillation.")
def ingest(url: str, output_dir: str, model: str):
    """Ingest a video/audio URL, transcribe, align with chapters, and distill into AgentLoom 3-track artifacts."""
    console.print(Panel(f"[bold cyan]MediaLoom Ingestion Pipeline[/bold cyan]\nTarget: {url}", expand=False))

    repo_root = Path(output_dir) if output_dir else Path.cwd()

    # Stage 1: Probe
    console.print("[yellow]Stage 1: Probing media metadata and streams...[/yellow]")
    try:
        meta = probe_media(url)
    except Exception as e:
        console.print(f"[red]Failed to probe media: {e}[/red]")
        return

    console.print(f"  [green]✓[/green] Title: [bold]{meta['title']}[/bold]")
    console.print(f"  [green]✓[/green] Channel: {meta['channel']}")
    console.print(f"  [green]✓[/green] Duration: {int(meta['duration'] // 60)} min ({meta['duration']}s)")
    console.print(f"  [green]✓[/green] Chapters found: {len(meta['chapters'])}")

    # Stage 2: Transcribe
    segments = None
    if meta.get("id"):
        console.print("[yellow]Stage 2A: Checking Fast Track (official/auto captions)...[/yellow]")
        raw_captions = fetch_fast_transcript(meta["id"])
        if raw_captions:
            console.print(f"  [green]✓ Fast Track succeeded![/green] Fetched {len(raw_captions)} caption entries.")
            segments = [
                {"start": c["start"], "end": c["start"] + c.get("duration", 0), "text": c["text"]}
                for c in raw_captions
            ]
        else:
            console.print("  [dim]Fast Track unavailable (subtitles disabled). Switching to Heavy Path...[/dim]")

    if not segments:
        console.print("[yellow]Stage 2B: Heavy Path (Extracting audio format 140)...[/yellow]")
        cache_dir = repo_root / ".cache" / "audio"
        audio_file = extract_audio_stream(url, str(cache_dir))
        console.print(f"  [green]✓[/green] Audio saved to: {audio_file} ({audio_file.stat().st_size / (1024*1024):.1f} MB)")

        console.print("[yellow]Stage 2C: Running ASR Engine (Whisper API)...[/yellow]")
        asr_result = transcribe_audio_file(audio_file)
        raw_segments = asr_result.get("segments", [])
        segments = [
            {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
            for s in raw_segments
        ]
        console.print(f"  [green]✓[/green] ASR transcription completed: {len(segments)} segments.")

    # Stage 3: Align with chapters
    console.print("[yellow]Stage 3: Aligning transcript segments with chapter timestamps...[/yellow]")
    aligned_chapters = align_transcript_with_chapters(segments, meta["chapters"], url)
    console.print(f"  [green]✓[/green] Aligned into {len(aligned_chapters)} chapter blocks with hyperlink anchors.")

    # Stage 4: Distill
    console.print(f"[yellow]Stage 4: LLM Distillation ({model})...[/yellow]")
    distilled = distill_aligned_chapters(meta["title"], meta["channel"], aligned_chapters, model=model)
    console.print("  [green]✓[/green] Multi-perspective distillation generated.")

    # Stage 5: Emit artifacts
    console.print("[yellow]Stage 5: Emitting AgentLoom 3-Track Artifacts...[/yellow]")
    emitted = emit_distillation_artifacts(meta, distilled, aligned_chapters, repo_root)

    console.print("\n[bold green]Ingestion Complete![/bold green] Generated files:")
    for k, v in emitted.items():
        console.print(f"  - [bold]{k}[/bold]: {v}")


if __name__ == "__main__":
    main()
