"""server.py — Agent-Native Built-in UI & HITL Review Portal Server.

Provides REST endpoints for:
1. Agent Profile & Capabilities Showcase
2. Data & Digest Explorer (processed media, transcripts, digests)
3. 3-Track Dual-Role Knowledge Graph Visualization
4. Human Review Portal (HITL proposal inspection, approve/reject gates)
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agentloom_media.ui.jobs import global_job_manager, IngestJob
from agentloom_media.distillation.distiller import resolve_distillation_model
from agentloom_media.review import (
    build_review_items,
    coverage,
    current_decisions,
    read_review,
    record_decision,
    summarize,
)
from agentloom_media.transcripts.store import load_transcript
from dotenv import load_dotenv

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
env_file = WORKSPACE_ROOT / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()
DOCS_DIR = WORKSPACE_ROOT / "docs"
DIGESTS_DIR = DOCS_DIR / "digests"
AGENTS_DIR = WORKSPACE_ROOT / "agents"
KG_DIR = AGENTS_DIR / "knowledge-graphs"
SKILLS_DIR = AGENTS_DIR / "skills"
PROPOSALS_DIR = WORKSPACE_ROOT / "proposals"
CACHE_DIR = WORKSPACE_ROOT / ".cache"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="MediaLoom Agent Portal",
    description="AgentLoom Autonomous Multimedia Ingestion & HITL Review Portal",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==========================================
# 1. Pillar 1: Agent Profile & Capabilities
# ==========================================
@app.get("/api/profile")
def get_agent_profile() -> Dict[str, Any]:
    api_key_set = bool(os.getenv("OPENAI_API_KEY") or os.getenv("Learning-agent-OpenAI-API-Key"))
    
    # Check ffmpeg
    ffmpeg_found = False
    try:
        import imageio_ffmpeg
        ffmpeg_found = bool(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        ffmpeg_found = shutil.which("ffmpeg") is not None

    # Disk usage / stats
    audio_files = list(CACHE_DIR.glob("audio/*")) if CACHE_DIR.exists() else []
    audio_bytes = sum(f.stat().st_size for f in audio_files if f.is_file())
    digests = list(DIGESTS_DIR.glob("*.md")) if DIGESTS_DIR.exists() else []
    proposals = list(PROPOSALS_DIR.glob("*.json")) if PROPOSALS_DIR.exists() else []
    accepted_skills = list(SKILLS_DIR.glob("domain/accepted/*.md")) if SKILLS_DIR.exists() else []
    candidate_skills = list(SKILLS_DIR.glob("domain/candidate/*.md")) if SKILLS_DIR.exists() else []

    return {
        "agent": {
            "id": "agentloom-multimedia",
            "name": "MediaLoom",
            "framework": "AgentLoom Core v3.0",
            "paradigm": "Human-in-the-Loop (HITL) Autonomous Agent",
            "version": "0.1.0",
            "roles": ["role-builder", "role-domain"],
            "description": "Autonomous multimedia knowledge acquisition, audio processing, acoustic transcription, and HITL conceptual distillation.",
        },
        "capabilities": [
            {
                "id": "fast-track-subtitles",
                "name": "Fast Track Subtitle Ingestion",
                "status": "active",
                "description": "Pulls official & auto-captions via YouTube Transcript API (0 cost, 2s latency).",
            },
            {
                "id": "heavy-path-audio",
                "name": "Heavy Path Audio Stream Extraction",
                "status": "active",
                "description": "Extracts format 140 m4a pure audio stream without downloading heavy video payloads.",
            },
            {
                "id": "slow-track-browser-capture",
                "name": "Slow Track Browser Playback Capture",
                "status": "planned",
                "description": "Last-resort path: play the watch page in Chromium, record element audio at 1.5x-2.0x via captureStream(), then rescale ASR timestamps to the original timeline.",
            },
            {
                "id": "budget-chunker",
                "name": "Audio Budget Chunker & 16kHz Compressor",
                "status": "active",
                "description": "Compresses audio to 16kHz mono 32kbps MP3 (reducing size by 75%) and chunks at 15-min intervals to enforce the 24MB Whisper limit.",
            },
            {
                "id": "whisper-asr",
                "name": "Acoustic Whisper ASR Engine",
                "status": "active",
                "description": "Produces word- and segment-level timestamps for fine-grained temporal provenance.",
            },
            {
                "id": "multi-perspective-distiller",
                "name": "Multi-Perspective Distillation",
                "status": "active",
                "description": "Concurrently emits Track 2 digests, Track 3 candidate KG nodes, and Track 3 executable skills.",
            },
            {
                "id": "hitl-review-gate",
                "name": "Human-in-the-Loop Review Gate",
                "status": "active",
                "description": "Strict gatekeeping mechanism ensuring candidate skills and KG nodes require human sign-off before canonization.",
            }
        ],
        "environment": {
            "openai_api_configured": api_key_set,
            "ffmpeg_ready": ffmpeg_found,
            "workspace_path": str(WORKSPACE_ROOT),
        },
        "metrics": {
            "cached_audio_count": len(audio_files),
            "cached_audio_mb": round(audio_bytes / (1024 * 1024), 2),
            "total_digests": len(digests),
            "pending_proposals": len(proposals),
            "accepted_skills": len(accepted_skills),
            "candidate_skills": len(candidate_skills),
        }
    }


# ==========================================
# 2. Pillar 2: Data & Digest Explorer
# ==========================================
@app.get("/api/data/digests")
def list_digests() -> List[Dict[str, Any]]:
    if not DIGESTS_DIR.exists():
        return []

    items = []
    for f in sorted(DIGESTS_DIR.glob("*.md"), reverse=True):
        try:
            content = f.read_text(encoding="utf-8")
            title = f.stem
            source_url = ""
            for line in content.splitlines()[:15]:
                if line.startswith("# "):
                    title = line.replace("# ", "").strip()
                if "Source" in line and "(" in line and ")" in line:
                    source_url = line.split("(")[-1].split(")")[0].strip()

            items.append({
                "filename": f.name,
                "title": title,
                "source_url": source_url,
                "modified_at": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
        except Exception:
            pass
    return items


@app.get("/api/data/digest/{filename}")
def get_digest_content(filename: str) -> Dict[str, Any]:
    safe_name = Path(filename).name
    target = DIGESTS_DIR / safe_name
    if not target.exists():
        raise HTTPException(status_code=404, detail="Digest not found")
    
    return {
        "filename": safe_name,
        "content": target.read_text(encoding="utf-8"),
    }


# ==========================================
# 3. Pillar 3: Interactive Knowledge Graphs
# ==========================================
@app.get("/api/kg/master")
def get_master_graph() -> Dict[str, Any]:
    master_path = KG_DIR / "master-graph.json"
    if not master_path.exists():
        raise HTTPException(status_code=404, detail="Master graph not found")
    with open(master_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/kg/graph/{role}/{track}")
def get_sub_graph(role: str, track: str) -> Dict[str, Any]:
    filename = f"{role}-{track}-graph.json"
    target = KG_DIR / filename
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Knowledge graph {filename} not found")
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================================
# 4. Pillar 4: Human Review Portal (HITL)
# ==========================================
class ReviewAction(BaseModel):
    action: str  # "approve" | "reject"
    reviewer: Optional[str] = "Human Operator"
    notes: Optional[str] = None


@app.get("/api/proposals")
def list_proposals() -> List[Dict[str, Any]]:
    if not PROPOSALS_DIR.exists():
        return []

    results = []
    for f in sorted(PROPOSALS_DIR.glob("*.json"), reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as pf:
                data = json.load(pf)
                data["_file_name"] = f.name
                results.append(data)
        except Exception:
            pass
    return results


# ==========================================
# 4b. Evidence-level review workbench (plan §6)
# ==========================================
class DecisionPayload(BaseModel):
    kind: str
    target_id: str
    verdict: str
    evidence_index: Optional[int] = None
    note: Optional[str] = None
    reviewer: str = "unknown"


def _proposal_path(proposal_file: str) -> Path:
    path = PROPOSALS_DIR / Path(proposal_file).name
    if not path.exists():
        raise HTTPException(status_code=404, detail="Proposal file not found")
    return path


@app.get("/review/{proposal_file}")
def review_page(proposal_file: str) -> FileResponse:
    _proposal_path(proposal_file)
    return FileResponse(STATIC_DIR / "review.html")


@app.get("/api/review/{proposal_file}")
def get_review_bundle(proposal_file: str) -> Dict[str, Any]:
    """Everything the workbench needs: proposal, targets, transcript, prior decisions."""
    with open(_proposal_path(proposal_file), "r", encoding="utf-8") as f:
        proposal = json.load(f)

    items = build_review_items(proposal)
    review = read_review(WORKSPACE_ROOT, proposal_file)

    source = proposal.get("source") or {}
    media_id = source.get("media_id") or source.get("id")

    # The transcript panel needs utterances and, crucially, the timing granularity: word-level
    # highlighting is a lie unless the ASR actually produced word timings.
    transcript: Dict[str, Any] = {"utterances": [], "timing_granularity": None}
    if media_id:
        cached = load_transcript(WORKSPACE_ROOT, media_id)
        if cached:
            loaded = cached["transcript"]
            transcript = {
                "media_id": media_id,
                "timing_granularity": loaded.timing_granularity,
                "utterances": [
                    {"id": u.id, "t0": u.t0, "t1": u.t1, "text": u.text}
                    for u in loaded.utterances
                ],
            }

    return {
        "proposal_file": Path(proposal_file).name,
        "source": source,
        "segmentation": proposal.get("segmentation"),
        "passes": proposal.get("passes"),
        "transcript_provenance": proposal.get("transcript_provenance"),
        "anchor_validation": proposal.get("anchor_validation"),
        "document": proposal.get("document"),
        "merge_candidates": (proposal.get("knowledge_graph") or {}).get(
            "merge_candidates"
        )
        or [],
        "items": items,
        "coverage": coverage(items),
        "transcript": transcript,
        "decisions": current_decisions(review),
        "summary": summarize(review),
    }


@app.post("/api/review/{proposal_file}/decision")
def post_review_decision(
    proposal_file: str, payload: DecisionPayload
) -> Dict[str, Any]:
    _proposal_path(proposal_file)
    try:
        entry = record_decision(
            WORKSPACE_ROOT,
            proposal_file,
            kind=payload.kind,
            target_id=payload.target_id,
            verdict=payload.verdict,
            evidence_index=payload.evidence_index,
            note=payload.note,
            reviewer=payload.reviewer,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    review = read_review(WORKSPACE_ROOT, proposal_file)
    return {"ok": True, "entry": entry, "summary": summarize(review)}


@app.post("/api/proposals/{proposal_file}/review")
def review_proposal(proposal_file: str, payload: ReviewAction) -> Dict[str, Any]:
    safe_name = Path(proposal_file).name
    p_path = PROPOSALS_DIR / safe_name
    if not p_path.exists():
        raise HTTPException(status_code=404, detail="Proposal file not found")

    with open(p_path, "r", encoding="utf-8") as f:
        proposal_data = json.load(f)

    action = payload.action.lower()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if action == "approve":
        # 1. Promote candidate skills to accepted/
        distilled = proposal_data.get("distilled", {})
        cand_skills = proposal_data.get("candidate_skills") or distilled.get("candidate_skills", [])
        cand_dir = SKILLS_DIR / "domain" / "candidate"
        acc_dir = SKILLS_DIR / "domain" / "accepted"
        acc_dir.mkdir(parents=True, exist_ok=True)

        promoted_skills = []
        for sk in cand_skills:
            from agentloom_media.proposals.emitter import slugify
            slug = slugify(sk.get("name") or sk.get("title", "skill"))
            src_file = cand_dir / f"{slug}.md"
            dst_file = acc_dir / f"{slug}.md"
            if src_file.exists():
                shutil.move(str(src_file), str(dst_file))
                # Update status inside markdown
                content = dst_file.read_text(encoding="utf-8")
                content = content.replace("Status**: Candidate (Propose-Review Pending)", f"Status**: Accepted (Approved by {payload.reviewer} at {timestamp})")
                dst_file.write_text(content, encoding="utf-8")
                promoted_skills.append(str(dst_file.name))

        # 2. Merge candidate KG nodes into domain-knowledge-graph.json
        cand_nodes = proposal_data.get("candidate_kg_nodes") or distilled.get("candidate_kg_nodes", [])
        domain_kg_path = KG_DIR / "domain-knowledge-graph.json"
        merged_nodes_count = 0
        if domain_kg_path.exists() and cand_nodes:
            from agentloom_media.proposals.emitter import slugify
            with open(domain_kg_path, "r", encoding="utf-8") as kf:
                domain_kg = json.load(kf)

            existing_ids = {n["id"] for n in domain_kg.get("nodes", [])}
            root_node = next((n for n in domain_kg.get("nodes", []) if n["id"] == "knowledge:domain:root"), None)

            source_meta = proposal_data.get("source") or proposal_data.get("metadata") or {}

            for cn in cand_nodes:
                raw_title = cn.get("title") or cn.get("name") or cn.get("id")
                clean_id = f"concept:{slugify(raw_title)}"
                if clean_id not in existing_ids:
                    node_entry = {
                        "id": clean_id,
                        "type": str(cn.get("type", "concept")).lower(),
                        "data": {
                            "title": raw_title,
                            "description": cn.get("description", ""),
                            "category": "domain-harvested",
                            "source_url": cn.get("timestamp_anchor") or source_meta.get("url", ""),
                            "tags": ["domain", "harvested", "hitl-approved"],
                        },
                        "relationships": {
                            "parent": "knowledge:domain:root",
                            "children": []
                        }
                    }
                    domain_kg["nodes"].append(node_entry)
                    existing_ids.add(clean_id)
                    if root_node and clean_id not in root_node["relationships"]["children"]:
                        root_node["relationships"]["children"].append(clean_id)
                    merged_nodes_count += 1

            domain_kg["version"] = f"1.0.{len(domain_kg['nodes'])}"
            with open(domain_kg_path, "w", encoding="utf-8") as kf:
                json.dump(domain_kg, kf, indent=2, ensure_ascii=False)

        # 3. Update proposal file state
        proposal_data["status"] = "approved"
        proposal_data["review_audit"] = {
            "decision": "approved",
            "reviewer": payload.reviewer,
            "timestamp": timestamp,
            "notes": payload.notes,
            "promoted_skills": promoted_skills,
            "merged_kg_nodes_count": merged_nodes_count,
        }
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(proposal_data, f, indent=2, ensure_ascii=False)

        return {
            "ok": True,
            "status": "approved",
            "promoted_skills": promoted_skills,
            "merged_kg_nodes": merged_nodes_count,
        }

    elif action == "reject":
        proposal_data["status"] = "rejected"
        proposal_data["review_audit"] = {
            "decision": "rejected",
            "reviewer": payload.reviewer,
            "timestamp": timestamp,
            "notes": payload.notes or "Rejected by reviewer",
        }
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(proposal_data, f, indent=2, ensure_ascii=False)

        return {"ok": True, "status": "rejected"}

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported review action '{action}'")


# ==========================================
# 5. One-Click Ingestion & SSE Live Pipeline
# ==========================================
class SubmitIngestPayload(BaseModel):
    url: str
    model: Optional[str] = None
    route: Optional[str] = None
    asr_engine: Optional[str] = None
    synthesis_model: Optional[str] = None


@app.post("/api/ingest/submit")
async def submit_ingest_job(payload: SubmitIngestPayload) -> Dict[str, Any]:
    url = payload.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="URL cannot be empty")

    global_job_manager.workspace_root = WORKSPACE_ROOT
    job = global_job_manager.create_job(
        url=url,
        model=payload.model or resolve_distillation_model(),
        output_dir=WORKSPACE_ROOT,
        route=payload.route,
        asr_engine=payload.asr_engine,
        synthesis_model=payload.synthesis_model,
    )
    import asyncio
    asyncio.create_task(global_job_manager.run_job(job))
    return {
        "ok": True,
        "job_id": job.job_id,
        "job": job.snapshot()
    }


@app.get("/api/ingest/jobs")
def list_ingest_jobs() -> Dict[str, Any]:
    return {"jobs": global_job_manager.list_jobs()}


@app.get("/api/ingest/jobs/{job_id}")
def get_ingest_job(job_id: str) -> Dict[str, Any]:
    job = global_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    data = job.snapshot()
    data["logs"] = job.logs
    return data


@app.get("/api/ingest/stream/{job_id}")
async def stream_ingest_job(job_id: str):
    job = global_job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    import asyncio

    async def event_generator():
        # First send the initial snapshot and existing logs
        init_payload = {
            "type": "init",
            "snapshot": job.snapshot(),
            "logs": job.logs
        }
        yield f"data: {json.dumps(init_payload)}\n\n"

        if job.status in ("completed", "failed"):
            return

        queue = job.add_listener()
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("type") in ("stage_update", "error"):
                        if event.get("status") in ("completed", "failed"):
                            break
                except asyncio.TimeoutError:
                    yield ": ping\n\n"
                    if job.status in ("completed", "failed"):
                        break
        finally:
            job.remove_listener(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# Mount static web assets
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
