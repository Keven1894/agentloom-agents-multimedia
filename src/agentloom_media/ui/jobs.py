"""jobs.py — Background Ingestion Job Manager & Real-Time Event Pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

from agentloom_media.acquisition.probe import probe_media
from agentloom_media.acquisition.fast_transcript import fetch_fast_transcript
from agentloom_media.acquisition.audio_extractor import extract_audio_stream
from agentloom_media.audio.asr import transcribe_audio_file
from agentloom_media.distillation.chapter_aligner import align_transcript_with_chapters
from agentloom_media.distillation.distiller import distill_aligned_chapters
from agentloom_media.proposals.emitter import emit_distillation_artifacts

logger = logging.getLogger("agentloom_media.jobs")


class IngestJob:
    def __init__(self, job_id: str, url: str, model: str = "gpt-4o", output_dir: Optional[Path] = None):
        self.job_id = job_id
        self.url = url
        self.model = model
        self.output_dir = output_dir or Path.cwd()
        self.status = "queued"  # queued, running, completed, failed
        self.stage = "queued"   # queued, probing, transcribing, aligning, distilling, emitting, completed, failed
        self.progress = 0       # 0 - 100
        self.message = "Job queued and awaiting worker..."
        self.logs: List[Dict[str, str]] = []
        self.meta: Dict[str, Any] = {}
        self.artifacts: Dict[str, str] = {}
        self.error: Optional[str] = None
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.started_at: Optional[str] = None
        self.completed_at: Optional[str] = None
        self._listeners: List[asyncio.Queue] = []

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
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "meta": self.meta,
            "artifacts": self.artifacts,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "log_count": len(self.logs),
        }


class JobManager:
    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace_root = workspace_root or Path.cwd()
        self.jobs: Dict[str, IngestJob] = {}

    def create_job(self, url: str, model: str = "gpt-4o", output_dir: Optional[Path] = None) -> IngestJob:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job = IngestJob(
            job_id=job_id,
            url=url,
            model=model,
            output_dir=output_dir or self.workspace_root
        )
        self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[IngestJob]:
        return self.jobs.get(job_id)

    def list_jobs(self) -> List[Dict[str, Any]]:
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

            # Stage 2: Transcribe (Check cache -> Fast Track -> Heavy Path)
            job.update_stage("transcribing", 20, "Checking transcript cache and caption tracks...")
            repo_root = job.output_dir
            cache_dir = repo_root / ".cache" / "transcripts"
            cache_dir.mkdir(parents=True, exist_ok=True)
            transcript_cache_file = cache_dir / f"{meta.get('id', 'media')}_segments.json"

            segments = None
            if transcript_cache_file.exists():
                job.log(f"Cache hit: Found existing transcript segments at {transcript_cache_file.name}")
                with open(transcript_cache_file, "r", encoding="utf-8") as f:
                    segments = json.load(f)
                job.update_stage("transcribing", 45, f"Loaded {len(segments)} segments from local cache.")
            elif meta.get("id"):
                job.log("Probing Fast Track (official & automatic subtitles)...")
                raw_captions = await loop.run_in_executor(None, fetch_fast_transcript, meta["id"])
                if raw_captions:
                    job.log(f"Fast Track succeeded! Extracted {len(raw_captions)} caption entries.")
                    segments = [
                        {"start": c["start"], "end": c["start"] + c.get("duration", 0), "text": c["text"]}
                        for c in raw_captions
                    ]
                    with open(transcript_cache_file, "w", encoding="utf-8") as f:
                        json.dump(segments, f, ensure_ascii=False, indent=2)
                    job.update_stage("transcribing", 50, f"Fast Track extracted {len(segments)} segments.")
                else:
                    job.log("Subtitles disabled or unavailable on Fast Track. Switching to Heavy Path...")

            if not segments:
                # Heavy Path: Extract audio stream format 140
                job.update_stage("transcribing", 25, "Heavy Path: Extracting lightweight audio stream (format 140)...")
                audio_cache_dir = repo_root / ".cache" / "audio"
                audio_cache_dir.mkdir(parents=True, exist_ok=True)
                audio_file = await loop.run_in_executor(None, extract_audio_stream, job.url, str(audio_cache_dir))
                size_mb = audio_file.stat().st_size / (1024 * 1024)
                job.log(f"Audio stream extracted: {audio_file.name} ({size_mb:.1f} MB)")

                job.update_stage("transcribing", 35, f"Transcribing audio with Whisper ASR ({size_mb:.1f} MB)...")
                asr_result = await loop.run_in_executor(None, transcribe_audio_file, audio_file)
                raw_segments = asr_result.get("segments", [])
                segments = [
                    {"start": s.get("start", 0), "end": s.get("end", 0), "text": s.get("text", "")}
                    for s in raw_segments
                ]
                job.log(f"Whisper ASR completed: Generated {len(segments)} segment timestamps.")
                with open(transcript_cache_file, "w", encoding="utf-8") as f:
                    json.dump(segments, f, ensure_ascii=False, indent=2)
                job.update_stage("transcribing", 55, f"ASR transcription completed ({len(segments)} segments).")

            # Stage 3: Align transcript with chapters
            job.update_stage("aligning", 65, "Aligning transcript segments with video chapter markers...")
            aligned_chapters = await loop.run_in_executor(
                None, align_transcript_with_chapters, segments, meta.get("chapters", []), job.url
            )
            job.log(f"Aligned into {len(aligned_chapters)} topical chapter blocks with hyperlink timestamp anchors.")

            # Stage 4: Multi-perspective LLM Distillation
            job.update_stage("distilling", 75, f"Synthesizing knowledge with LLM ({job.model})...")
            from functools import partial
            distill_fn = partial(
                distill_aligned_chapters,
                video_title=meta["title"],
                channel=meta.get("channel", "Unknown"),
                aligned_chapters=aligned_chapters,
                model=job.model
            )
            distilled = await loop.run_in_executor(None, distill_fn)
            cand_kg = distilled.get("candidate_kg_nodes", [])
            cand_skills = distilled.get("candidate_skills", [])
            job.log(f"Distillation completed: {len(cand_kg)} candidate KG nodes, {len(cand_skills)} candidate skills.")

            # Stage 5: Emitting Artifacts & Proposal
            job.update_stage("emitting", 90, "Emitting AgentLoom 3-Track files and HITL review proposal...")
            emitted = await loop.run_in_executor(
                None, emit_distillation_artifacts, meta, distilled, aligned_chapters, repo_root
            )
            job.artifacts = {k: str(v) for k, v in emitted.items()}
            job.log("Artifacts written: Track 2 Digest, Candidate Skills, and Review Proposal.")

            # Completion
            job.status = "completed"
            job.completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            job.update_stage("completed", 100, "Ingestion and distillation completed! Ready for Human Review.")

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


# Global job manager singleton
global_job_manager = JobManager()
