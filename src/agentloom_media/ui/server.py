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
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
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
        cand_skills = distilled.get("candidate_skills", [])
        cand_dir = SKILLS_DIR / "domain" / "candidate"
        acc_dir = SKILLS_DIR / "domain" / "accepted"
        acc_dir.mkdir(parents=True, exist_ok=True)

        promoted_skills = []
        for sk in cand_skills:
            from agentloom_media.proposals.emitter import slugify
            slug = slugify(sk.get("name", "skill"))
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
        cand_nodes = distilled.get("candidate_kg_nodes", [])
        domain_kg_path = KG_DIR / "domain-knowledge-graph.json"
        merged_nodes_count = 0
        if domain_kg_path.exists() and cand_nodes:
            with open(domain_kg_path, "r", encoding="utf-8") as kf:
                domain_kg = json.load(kf)

            existing_ids = {n["id"] for n in domain_kg.get("nodes", [])}
            root_node = next((n for n in domain_kg.get("nodes", []) if n["id"] == "knowledge:domain:root"), None)

            for cn in cand_nodes:
                nid = cn.get("id")
                if nid and nid not in existing_ids:
                    node_entry = {
                        "id": nid,
                        "type": cn.get("type", "concept"),
                        "data": {
                            "title": cn.get("title", nid),
                            "description": cn.get("description", ""),
                            "category": "domain-harvested",
                            "source_url": cn.get("timestamp_anchor") or proposal_data.get("metadata", {}).get("url", ""),
                            "tags": ["domain", "harvested", "hitl-approved"],
                        },
                        "relationships": {
                            "parent": "knowledge:domain:root",
                            "children": []
                        }
                    }
                    domain_kg["nodes"].append(node_entry)
                    if root_node and nid not in root_node["relationships"]["children"]:
                        root_node["relationships"]["children"].append(nid)
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


# Mount static web assets
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
