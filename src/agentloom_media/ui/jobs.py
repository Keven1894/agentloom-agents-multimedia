"""jobs.py — Background Ingestion Job Manager & Real-Time Event Pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

from agentloom_media.acquisition.probe import probe_media
from agentloom_media.transcripts.acquire import acquire_transcript
from agentloom_media.distillation.distiller import resolve_distillation_model
from agentloom_media.distillation.passes import resolve_synthesis_model
from agentloom_media.distillation.pipeline import distill as distill_typed
from agentloom_media.proposals.emitter_v2 import emit_typed_distillation
from agentloom_media.search.embed import Embedder, embeddings_enabled

logger = logging.getLogger("agentloom_media.jobs")

JOB_SCHEMA = "medialoom/ingest-job/v1"
MAX_PERSISTED_LOGS = 400
LIVE_STATUSES = {"queued", "running"}


class IngestJob:
    def __init__(
        self,
        job_id: str,
        url: str,
        model: Optional[str] = None,
        output_dir: Optional[Path] = None,
        route: Optional[str] = None,
        asr_engine: Optional[str] = None,
        synthesis_model: Optional[str] = None,
    ):
        self.job_id = job_id
        self.url = url
        self.model = resolve_distillation_model(model)
        self.synthesis_model = resolve_synthesis_model(synthesis_model)
        self.route = route
        self.asr_engine = asr_engine
        self.output_dir = output_dir or Path.cwd()
        self.status = "queued"  # queued, running, completed, failed
        self.stage = "queued"   # queued, probing, transcribing, aligning, distilling, emitting, completed, failed
        self.progress = 0       # 0 - 100
        self.message = "Job queued and awaiting worker..."
        self.logs: List[Dict[str, str]] = []
        self.meta: Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}
        self.anchor_report: Optional[Dict[str, Any]] = None
        self.transcript_provenance: Optional[Dict[str, Any]] = None
        self.distillation_passes: Optional[Dict[str, str]] = None
        self.error: Optional[str] = None
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self._listeners: List[asyncio.Queue] = []
        self._on_change = None

    def log(self, text: str):
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "text": text
        }
        self.logs.append(entry)
        self._broadcast({
            "type": "log",
            "job_id": self.job_id,
            "entry": entry
        })

    def update_stage(self, stage: str, progress: int, message: str):
        self.stage = stage
        self.progress = progress
        self.message = message
        self.log(f"[{stage.upper()}] {message}")
        if self._on_change:
            self._on_change()
        self._broadcast({
            "type": "stage_update",
            "job_id": self.job_id,
            "stage": stage,
            "progress": progress,
            "message": message,
            "status": self.status
        })

    def add_listener(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._listeners.append(q)
        return q

    def remove_listener(self, q: asyncio.Queue):
        if q in self._listeners:
            self._listeners.remove(q)

    def _broadcast(self, event: Dict[str, Any]):
        for q in list(self._listeners):
            try:
                q.put_nowait(event)
            except Exception:
                pass

    def snapshot(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "url": self.url,
            "model": self.model,
            "synthesis_model": self.synthesis_model,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "meta": self.meta,
            "artifacts": self.artifacts,
            "route": self.route,
            "asr_engine": self.asr_engine,
            "anchor_report": self.anchor_report,
            "transcript_provenance": self.transcript_provenance,
            "distillation_passes": self.distillation_passes,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "log_count": len(self.logs),
        }

    def to_record(self) -> Dict[str, Any]:
        data = self.snapshot()
        data["schema"] = JOB_SCHEMA
        data["logs"] = self.logs[-MAX_PERSISTED_LOGS:]
        return data

    @classmethod
    def from_record(cls, data: Dict[str, Any], output_dir: Path) -> "IngestJob":
        job = cls(
            job_id=str(data.get("job_id") or "job_unknown"),
            url=str(data.get("url") or ""),
            model=data.get("model"),
            output_dir=output_dir,
            route=data.get("route"),
            asr_engine=data.get("asr_engine"),
            synthesis_model=data.get("synthesis_model"),
        )
        job.status = str(data.get("status") or "queued")
        job.stage = str(data.get("stage") or job.status)
        job.progress = int(data.get("progress") or 0)
        job.message = str(data.get("message") or "")
        job.meta = data.get("meta") or {}
        job.artifacts = data.get("artifacts") or {}
        job.anchor_report = data.get("anchor_report")
        job.transcript_provenance = data.get("transcript_provenance")
        job.distillation_passes = data.get("distillation_passes")
        job.error = data.get("error")
        job.created_at = str(data.get("created_at") or job.created_at)
        job.started_at = data.get("started_at")
        job.completed_at = data.get("completed_at")
        job.logs = list(data.get("logs") or [])
        return job


def job_from_proposal(path: Path, payload: Dict[str, Any], output_dir: Path) -> IngestJob:
    """Rebuild a completed job snapshot from a proposal written by CLI ingest."""
    source = payload.get("source") or {}
    models = payload.get("models") or {}
    provenance = payload.get("transcript_provenance") or {}
    stamp = str(payload.get("timestamp") or "")
    created = stamp.replace("T", " ")[:19] if stamp else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    digest_name = path.name
    if digest_name.startswith("proposal-") and digest_name.endswith(".json"):
        digest_name = digest_name[len("proposal-") : -len(".json")] + ".md"
    digest_path = output_dir / "docs" / "digests" / digest_name

    return IngestJob.from_record(
        {
            "job_id": f"job_imported_{path.stem}",
            "url": source.get("url") or "",
            "model": models.get("distillation") or payload.get("model"),
            "synthesis_model": models.get("synthesis"),
            "status": "completed",
            "stage": "completed",
            "progress": 100,
            "message": "Imported from an existing proposal (CLI or earlier ingest).",
            "meta": {
                "id": source.get("media_id"),
                "title": source.get("title"),
                "channel": source.get("channel"),
                "url": source.get("url"),
            },
            "artifacts": {
                "agentloom_proposal": str(path),
                **({"track2_digest": str(digest_path)} if digest_path.exists() else {}),
            },
            "route": provenance.get("route"),
            "asr_engine": provenance.get("engine"),
            "transcript_provenance": provenance or None,
            "distillation_passes": payload.get("passes"),
            "anchor_report": payload.get("anchor_validation"),
            "created_at": created,
            "completed_at": created,
            "logs": [
                {
                    "time": created[11:] if len(created) >= 19 else "",
                    "text": "Imported from proposal on disk. This ingest was not started from the portal.",
                }
            ],
        },
        output_dir,
    )


class JobManager:
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.jobs: Dict[str, IngestJob] = {}
        self._loaded = False
        if workspace_root is not None:
            self._load()

    def jobs_dir(self) -> Path:
        return Path(self.workspace_root) / "data" / "jobs"

    def attach(self, workspace_root: Path) -> None:
        """Point at the repo and load any jobs written by a previous portal process."""
        self.workspace_root = Path(workspace_root)
        self._load()

    def _job_path(self, job_id: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "._-" else "-" for c in job_id)
        return self.jobs_dir() / f"{safe}.json"

    def persist(self, job: IngestJob) -> None:
        path = self._job_path(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(job.to_record(), ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        folder = self.jobs_dir()
        if folder.exists():
            for path in folder.glob("*.json"):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(data, dict) or not data.get("job_id"):
                    continue
                job = IngestJob.from_record(data, self.workspace_root)
                if job.status in LIVE_STATUSES:
                    job.status = "interrupted"
                    job.stage = "interrupted"
                    job.error = (
                        "Portal restarted while this job was still running. "
                        "Re-submit the URL to finish it."
                    )
                    job.message = job.error
                    job.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    job.logs.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "text": job.error,
                    })
                    self.persist(job)
                job._on_change = lambda j=job: self.persist(j)
                self.jobs[job.job_id] = job
        self._import_proposals()

    def _known_proposal_names(self) -> set:
        names = set()
        for job in self.jobs.values():
            artifact = str((job.artifacts or {}).get("agentloom_proposal") or "")
            if artifact:
                names.add(Path(artifact).name)
        return names

    def _import_proposals(self) -> None:
        """CLI ingest writes a proposal but never a JobManager row. Surface those too."""
        proposals_dir = Path(self.workspace_root) / "proposals"
        if not proposals_dir.exists():
            return
        known = self._known_proposal_names()
        for path in sorted(proposals_dir.glob("*.json")):
            if path.name in known:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(payload, dict):
                continue
            job = job_from_proposal(path, payload, self.workspace_root)
            job._on_change = lambda j=job: self.persist(j)
            self.jobs[job.job_id] = job
            self.persist(job)

    def create_job(
        self,
        url: str,
        model: Optional[str] = None,
        output_dir: Optional[Path] = None,
        route: Optional[str] = None,
        asr_engine: Optional[str] = None,
        synthesis_model: Optional[str] = None,
    ) -> IngestJob:
        self._load()
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = IngestJob(
            job_id=job_id,
            url=url,
            model=model,
            output_dir=output_dir or self.workspace_root,
            route=route,
            asr_engine=asr_engine,
            synthesis_model=synthesis_model,
        )
        self.jobs[job_id] = job
        job._on_change = lambda: self.persist(job)
        self.persist(job)
        return job

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        self._load()
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
        self._load()
        return [j.snapshot() for j in sorted(self.jobs.values(), key=lambda x: x.created_at, reverse=True)]

    async def run_job(self, job: IngestJob):
        job.status = "running"
        job.started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        job.update_stage("probing", 5, f"Starting ingestion for {job.url}")

        loop = asyncio.get_running_loop()

        try:
            # Stage 1: Probing
            job.update_stage("probing", 10, "Probing media metadata and chapter scaffolding...")
            meta = await loop.run_in_executor(None, probe_media, job.url)
            job.meta = {
                "id": meta.get("id"),
                "title": meta.get("title"),
                "channel": meta.get("channel"),
                "duration": meta.get("duration"),
                "chapters_count": len(meta.get("chapters", [])),
                "url": meta.get("url")
            }
            job.log(f"Metadata verified: '{meta.get('title')}' by {meta.get('channel')} ({int(meta.get('duration', 0)//60)}m)")

            # Stage 2: Acquire a canonical transcript (cache -> captions -> ASR)
            job.update_stage("transcribing", 20, "Acquiring canonical transcript...")
            repo_root = job.output_dir

            acquisition = await loop.run_in_executor(
                None,
                partial(
                    acquire_transcript,
                    meta,
                    job.url,
                    repo_root,
                    engine=job.asr_engine,
                    route=job.route,
                    log=job.log,
                ),
            )
            segments = acquisition.segments
            job.transcript_provenance = acquisition.provenance()
            job.log(f"Transcript ready ({acquisition.describe()}).")
            if acquisition.timing_granularity != "word":
                job.log(
                    f"Anchors will be accurate to the {acquisition.timing_granularity} level; "
                    "word-level seeking is unavailable with this engine."
                )
            job.update_stage(
                "transcribing", 55, f"Transcript ready ({len(segments)} utterances)."
            )

            # Stage 3: Semantic segmentation and typed distillation passes
            synthesis_model = resolve_synthesis_model(job.synthesis_model)
            job.update_stage(
                "distilling",
                70,
                f"Segmenting and distilling ({job.model}, synthesis on {synthesis_model})...",
            )

            embedder = Embedder() if embeddings_enabled() else None
            if embedder is None:
                job.log(
                    "No embeddings available, so the transcript cannot be segmented by "
                    "topic. It will be treated as one span."
                )

            distill_fn = partial(
                distill_typed,
                acquisition.transcript,
                meta,
                embedder=embedder,
                model=job.model,
                synthesis_model=synthesis_model,
                log=job.log,
            )
            result = await loop.run_in_executor(None, distill_fn)
            job.distillation_passes = result.passes

            if not result.segments:
                raise RuntimeError(
                    "Distillation produced no segments; "
                    f"segmentation pass reported: {result.passes.get('segmentation')}"
                )

            # Stage 4: Emitting Artifacts & Proposal
            job.update_stage("emitting", 90, "Emitting AgentLoom 3-Track files and HITL review proposal...")
            emit_fn = partial(
                emit_typed_distillation,
                meta,
                result,
                acquisition.transcript,
                repo_root,
                model=job.model,
                synthesis_model=synthesis_model,
                transcript_provenance=job.transcript_provenance,
            )
            emitted = await loop.run_in_executor(None, emit_fn)
            anchor_report = emitted.pop("_anchor_report", None)
            job.artifacts = {k: str(v) for k, v in emitted.items()}
            job.log("Artifacts written: Track 2 Digest, Candidate Skills, and Review Proposal.")

            if anchor_report:
                job.anchor_report = anchor_report
                seen = anchor_report["anchors_seen"]
                valid = anchor_report["anchors_valid"]
                job.log(
                    f"Anchor validation: {valid}/{seen} anchors resolved to real transcript times."
                )
                if seen - valid:
                    job.log(
                        f"Dropped {seen - valid} unverifiable anchors "
                        f"(unparsable={anchor_report['anchors_rejected_unparsable']}, "
                        f"out_of_range={anchor_report['anchors_rejected_out_of_range']}); "
                        "affected claims have no evidence link."
                    )

            # Completion
            job.status = "completed"
            job.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.update_stage("completed", 100, "Ingestion and distillation completed. Ready for overview and evidence review.")

        except Exception as e:
            logger.exception(f"Job {job.job_id} failed: {e}")
            job.status = "failed"
            job.stage = "failed"
            job.error = str(e)
            job.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.log(f"ERROR: Pipeline terminated with failure: {e}")
            job._broadcast({
                "type": "error",
                "job_id": job.job_id,
                "error": str(e),
                "status": "failed"
            })
            if job._on_change:
                job._on_change()


# Global job manager singleton
global_job_manager = JobManager()
