"""Command-line interface for AgentLoom Multimedia Knowledge Ingestion."""

import os
from pathlib import Path
import click
import dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentloom_media.acquisition.probe import probe_media
from agentloom_media.transcripts.acquire import acquire_transcript
from agentloom_media.transcripts.store import load_transcript, read_manifest
from agentloom_media.search.index import SearchIndex, default_db_path
from agentloom_media.search.embed import Embedder, embeddings_enabled
from agentloom_media.distillation.anchors import format_timestamp
from agentloom_media.distillation.distiller import resolve_distillation_model
from agentloom_media.distillation.passes import resolve_synthesis_model
from agentloom_media.distillation.pipeline import distill as distill_typed
from agentloom_media.proposals.emitter_v2 import emit_typed_distillation

console = Console()
# Look for .env in current working dir or repo root
_env_path = Path.cwd() / ".env"
if not _env_path.exists():
    _env_path = Path(__file__).resolve().parents[2] / ".env"
dotenv.load_dotenv(_env_path)


@click.group()
def main():
    """AgentLoom Multimedia Knowledge Ingestion CLI."""
    pass


@main.command()
@click.argument("url")
@click.option("--output-dir", default=None, help="Repository root directory to save artifacts.")
@click.option(
    "--model",
    default=None,
    help="LLM model for distillation. Defaults to DISTILLATION_MODEL or gpt-5.6-luna.",
)
@click.option(
    "--route",
    type=click.Choice(["auto", "fast", "heavy"]),
    default=None,
    help="Transcript route: fast=captions only, heavy=ASR only, auto=captions then ASR. "
    "Defaults to TRANSCRIPT_ROUTE or auto.",
)
@click.option(
    "--asr-engine",
    default=None,
    help="ASR engine: whisper_api | faster_whisper | whisperx | whisper_cpp. "
    "Defaults to ASR_ENGINE or whisper_api.",
)
@click.option("--language", default=None, help="Language hint for ASR, e.g. zh or en.")
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Re-acquire the transcript even if one is cached.",
)
@click.option(
    "--no-index",
    is_flag=True,
    default=False,
    help="Skip adding the transcript to the search index.",
)
@click.option(
    "--synthesis-model",
    default=None,
    help="Stronger model for document synthesis and skill extraction. "
    "Defaults to SYNTHESIS_MODEL or gpt-5.6-terra.",
)
def ingest(
    url: str,
    output_dir: str,
    model: str,
    route: str,
    asr_engine: str,
    language: str,
    force: bool,
    no_index: bool,
    synthesis_model: str,
):
    """Ingest a video/audio URL, transcribe, align with chapters, and distill into AgentLoom 3-track artifacts."""
    model = resolve_distillation_model(model)
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

    # Stage 2: Acquire a canonical transcript (cache → captions → ASR)
    console.print("[yellow]Stage 2: Acquiring canonical transcript...[/yellow]")
    try:
        acquisition = acquire_transcript(
            meta,
            url,
            repo_root,
            engine=asr_engine,
            language=language,
            route=route,
            force=force,
            log=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )
    except Exception as e:
        console.print(f"[red]Transcript acquisition failed: {e}[/red]")
        return

    segments = acquisition.segments
    console.print(f"  [green]✓[/green] {acquisition.describe()}")
    if acquisition.timing_granularity != "word":
        console.print(
            f"  [dim]Anchors are accurate to the {acquisition.timing_granularity} level; "
            "word-level seeking is unavailable with this engine.[/dim]"
        )

    # Stage 2b: Index for search. Cheap, and useful on its own even if distillation fails.
    if not no_index:
        console.print("[yellow]Stage 2b: Indexing transcript for search...[/yellow]")
        try:
            with SearchIndex(default_db_path(repo_root)) as index:
                result = index.index_media(
                    acquisition.transcript, meta, acquisition.provenance()
                )
            detail = (
                f"{result['vectors']} vectors via {result['embedding_model']}"
                if result["embedded"]
                else "lexical only (no embeddings)"
            )
            console.print(
                f"  [green]✓[/green] {result['windows']} windows indexed, {detail}."
            )
        except Exception as e:
            console.print(f"  [yellow]![/yellow] Indexing skipped: {e}")

    # Stage 3+4: Semantic segmentation and typed distillation passes.
    synthesis = resolve_synthesis_model(synthesis_model)
    console.print(
        f"[yellow]Stage 3: Segmentation and typed distillation "
        f"({model}, synthesis on {synthesis})...[/yellow]"
    )

    embedder = None
    if embeddings_enabled():
        embedder = Embedder()
    else:
        console.print(
            "  [yellow]![/yellow] No embeddings available, so the transcript cannot be "
            "segmented by topic. It will be treated as one span."
        )

    # The takeaway pass retrieves candidate spans for document-level claims, so it needs the
    # index built above. Held open across distillation rather than reopened per claim.
    with SearchIndex(default_db_path(repo_root)) as search_index:
        result = distill_typed(
            acquisition.transcript,
            meta,
            embedder=embedder,
            model=model,
            synthesis_model=synthesis,
            search_index=None if no_index else search_index,
            log=lambda msg: console.print(f"  [dim]{msg}[/dim]"),
        )

    if not result.segments:
        console.print("[red]Distillation produced no segments; nothing to emit.[/red]")
        return

    # Stage 5: Emit artifacts
    console.print("[yellow]Stage 4: Emitting AgentLoom 3-Track Artifacts...[/yellow]")
    emitted = emit_typed_distillation(
        meta,
        result,
        acquisition.transcript,
        repo_root,
        model=model,
        synthesis_model=synthesis,
        transcript_provenance=acquisition.provenance(),
    )

    report = emitted.pop("_anchor_report", None)

    console.print("\n[bold green]Ingestion Complete![/bold green] Generated files:")
    for k, v in emitted.items():
        console.print(f"  - [bold]{k}[/bold]: {v}")

    if report:
        seen = report["anchors_seen"]
        valid = report["anchors_valid"]
        console.print(
            f"\n[bold]Anchor validation:[/bold] {valid}/{seen} anchors resolved to real "
            "transcript times."
        )
        dropped = seen - valid
        if dropped:
            console.print(
                f"  [yellow]{dropped} dropped[/yellow] "
                f"(unparsable={report['anchors_rejected_unparsable']}, "
                f"out_of_range={report['anchors_rejected_out_of_range']}); "
                "those claims are rendered without evidence links."
            )
            for example in report["rejected_examples"]:
                console.print(f"    [dim]- {example}[/dim]")


@main.command()
@click.argument("query")
@click.option("--output-dir", default=None, help="Repository root holding data/medialoom.db.")
@click.option("--limit", default=10, type=int, help="Maximum number of time ranges to return.")
@click.option("--media-id", default=None, help="Restrict the search to one media item.")
def search(query: str, output_dir: str, limit: int, media_id: str):
    """Search indexed transcripts and return time ranges with deep links."""
    repo_root = Path(output_dir) if output_dir else Path.cwd()
    db_path = default_db_path(repo_root)

    if not db_path.exists():
        console.print(
            f"[red]No search index at {db_path}.[/red] Run "
            "[bold]agentloom-media index[/bold] or ingest a video first."
        )
        return

    with SearchIndex(db_path) as index:
        hits, info = index.search(query, limit=limit, media_id=media_id)
        stats = index.stats()

    if not hits:
        console.print(f"[yellow]No matches for[/yellow] {query!r}")
        console.print(
            f"  [dim]Index holds {stats['windows']} windows across "
            f"{stats['media']} media items.[/dim]"
        )
        return

    mode = "hybrid (BM25 + vector, RRF-fused)" if info["vector_used"] else "lexical only"
    console.print(
        f"\n[bold]{len(hits)}[/bold] matches for {query!r} "
        f"[dim]— {mode}, {info['fused']} candidates considered[/dim]\n"
    )

    # A per-hit block rather than a table: CJK quotes wrap badly in narrow columns, and the
    # deep link is the part the reader acts on.
    for position, hit in enumerate(hits, start=1):
        time_range = f"{format_timestamp(hit.t0)}–{format_timestamp(hit.t1)}"
        rankers = []
        if hit.lexical_rank:
            rankers.append(f"keyword #{hit.lexical_rank}")
        if hit.vector_rank:
            rankers.append(f"vector #{hit.vector_rank}")

        console.print(
            f"[bold cyan]{position}.[/bold cyan] [bold]{time_range}[/bold] "
            f"[dim]·[/dim] {hit.title}"
        )
        console.print(f"   [white]{hit.matched_quote(220)}[/white]")
        console.print(
            f"   [dim]{', '.join(rankers) or 'no ranker'}[/dim]"
            + (f"  [blue]{hit.anchor_url}[/blue]" if hit.anchor_url else "")
        )
        console.print()

    if not info["vector_used"]:
        console.print(
            "[dim]Vector ranking did not run (no embeddings in this index or no API key), "
            "so these results are keyword matches only.[/dim]"
        )


@main.command()
@click.option("--output-dir", default=None, help="Repository root holding data/.")
@click.option(
    "--media-id",
    default=None,
    help="Index only this media item. Defaults to every stored transcript.",
)
@click.option(
    "--window-seconds", default=None, type=float, help="Window duration. Default 45."
)
@click.option(
    "--overlap-seconds", default=None, type=float, help="Window overlap. Default 15."
)
@click.option(
    "--no-embed", is_flag=True, default=False, help="Build lexical index only."
)
def index(
    output_dir: str,
    media_id: str,
    window_seconds: float,
    overlap_seconds: float,
    no_embed: bool,
):
    """(Re)build the search index from stored canonical transcripts."""
    repo_root = Path(output_dir) if output_dir else Path.cwd()
    transcripts_root = repo_root / "data" / "transcripts"

    if not transcripts_root.exists():
        console.print(
            f"[red]No transcripts found at {transcripts_root}.[/red] Ingest a video first."
        )
        return

    media_ids = (
        [media_id]
        if media_id
        else sorted(p.name for p in transcripts_root.iterdir() if p.is_dir())
    )
    if not media_ids:
        console.print("[yellow]No stored transcripts to index.[/yellow]")
        return

    console.print(
        Panel(
            f"[bold cyan]Building search index[/bold cyan]\n"
            f"Items: {len(media_ids)} · Database: {default_db_path(repo_root)}",
            expand=False,
        )
    )

    total_windows = 0
    with SearchIndex(default_db_path(repo_root)) as idx:
        if not idx.vector_available:
            console.print(
                "  [yellow]![/yellow] sqlite-vec unavailable; building a lexical index only."
            )
        for mid in media_ids:
            loaded = load_transcript(repo_root, mid)
            if not loaded:
                console.print(f"  [yellow]![/yellow] {mid}: no transcript variant found.")
                continue

            transcript = loaded["transcript"]
            variant = loaded["variant"]
            manifest = read_manifest(repo_root, mid)
            item_meta = {
                "title": manifest.get("title") or mid,
                "channel": manifest.get("channel"),
                "url": manifest.get("media_url"),
                "duration": transcript.duration,
            }
            try:
                result = idx.index_media(
                    transcript,
                    item_meta,
                    variant,
                    window_seconds=window_seconds,
                    overlap_seconds=overlap_seconds,
                    embed=not no_embed,
                )
            except Exception as e:
                console.print(f"  [red]✗[/red] {mid}: {e}")
                continue

            total_windows += result["windows"]
            detail = (
                f"{result['vectors']} vectors ({result['embedding_model']})"
                if result["embedded"]
                else "lexical only"
            )
            console.print(
                f"  [green]✓[/green] {item_meta['title'][:48]}: "
                f"{result['windows']} windows, {detail}"
            )

        stats = idx.stats()

    console.print(
        f"\n[bold green]Index ready.[/bold green] {stats['media']} media items, "
        f"{stats['windows']} windows, {stats['vectors']} vectors."
    )
    if total_windows and not stats["vectors"]:
        console.print(
            "[dim]No vectors were stored, so searches will use keyword matching only.[/dim]"
        )


@main.command()
@click.option("--host", default="127.0.0.1", help="Host to bind the portal server to.")
@click.option("--port", default=8000, type=int, help="Port to listen on.")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload for development.")
def ui(host: str, port: int, reload: bool):
    """Launch the Agent-Native Built-in UI & HITL Review Portal."""
    import uvicorn
    console.print(Panel(
        f"[bold green]MediaLoom Built-in Portal & HITL Review Gateway[/bold green]\n"
        f"• Local URL: [bold cyan]http://{host}:{port}[/bold cyan]\n"
        f"• Paradigm:  [bold yellow]AgentLoom Human-in-the-Loop (HITL) Governance[/bold yellow]\n"
        f"• Features:  Profile Showcase, Data Digests, 3-Track KG, Proposal Review",
        expand=False
    ))
    uvicorn.run("agentloom_media.ui.server:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()

