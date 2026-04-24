"""
CRUD Router — Offres d'emploi personnalisées
Stockage SQLite + vectorisation SBERT à la volée
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import sqlite3
import numpy as np
import os
from pathlib import Path

router = APIRouter(prefix="/crud", tags=["CRUD Offres"])

DB_PATH = Path(__file__).parent.parent / "data" / "processed" / "custom_jobs.db"
# ══════════════════════════════════════════════════════════════
# MODÈLES PYDANTIC
# ══════════════════════════════════════════════════════════════

class JobCreate(BaseModel):
    job_title: str
    description: Optional[str] = ""
    skills_desc: Optional[str] = ""
    experience_level: Optional[str] = "mid"
    location: Optional[str] = ""
    salary_usd: Optional[float] = 0.0
    remote_ratio: Optional[int] = 0

class JobUpdate(BaseModel):
    job_title: Optional[str] = None
    description: Optional[str] = None
    skills_desc: Optional[str] = None
    experience_level: Optional[str] = None
    location: Optional[str] = None
    salary_usd: Optional[float] = None
    remote_ratio: Optional[int] = None

# ══════════════════════════════════════════════════════════════
# INITIALISATION SQLite
# ══════════════════════════════════════════════════════════════

def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS custom_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_title TEXT NOT NULL,
            description TEXT DEFAULT '',
            skills_desc TEXT DEFAULT '',
            experience_level TEXT DEFAULT 'mid',
            location TEXT DEFAULT '',
            salary_usd REAL DEFAULT 0.0,
            remote_ratio INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def get_conn():
    return sqlite3.connect(DB_PATH)

# ══════════════════════════════════════════════════════════════
# INJECTION DU MATCHING ENGINE (défini dans 08_api.py)
# ══════════════════════════════════════════════════════════════

_engine = None

def set_engine(engine):
    global _engine
    _engine = engine

def _vectorize_and_append(job: dict):
    """Vectorise une offre SBERT et l'ajoute à la matrice en mémoire"""
    if _engine is None:
        return

    text = f"{job['job_title']} {job['job_title']} {job['job_title']} "
    text += f"{job.get('description', '')[:1000]} "
    text += f"{job.get('skills_desc', '')} {job.get('skills_desc', '')} "
    text += f"{job.get('experience_level', '')} {job.get('location', '')}"

    vec = _engine.sbert_model.encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Ajouter au DataFrame et à la matrice SBERT
    import pandas as pd
    new_row = {**job, "has_salary": 1 if job.get("salary_usd", 0) > 0 else 0,
               "vector_idx": len(_engine.df)}
    _engine.df = pd.concat([_engine.df, pd.DataFrame([new_row])], ignore_index=True)
    _engine.sbert_matrix = np.vstack([_engine.sbert_matrix, vec])

def _remove_from_engine(job_id: int):
    """Supprime une offre custom du DataFrame et de la matrice SBERT"""
    if _engine is None:
        return
    mask = _engine.df["custom_id"] != job_id
    indices = _engine.df[~mask].index
    if len(indices) == 0:
        return
    _engine.df = _engine.df[mask].reset_index(drop=True)
    _engine.sbert_matrix = np.delete(_engine.sbert_matrix, indices, axis=0)

# ══════════════════════════════════════════════════════════════
# ENDPOINTS CRUD
# ══════════════════════════════════════════════════════════════

@router.get("/jobs")
def list_custom_jobs():
    conn = get_conn()
    rows = conn.execute("SELECT * FROM custom_jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    cols = ["id","job_title","description","skills_desc","experience_level",
            "location","salary_usd","remote_ratio","created_at"]
    return [dict(zip(cols, r)) for r in rows]


@router.post("/jobs", status_code=201)
def create_job(job: JobCreate):
    conn = get_conn()
    cur = conn.execute("""
        INSERT INTO custom_jobs (job_title, description, skills_desc, experience_level,
                                  location, salary_usd, remote_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (job.job_title, job.description, job.skills_desc, job.experience_level,
          job.location, job.salary_usd, job.remote_ratio))
    conn.commit()
    new_id = cur.lastrowid
    conn.close()

    # Vectoriser à la volée
    job_dict = job.dict()
    job_dict["custom_id"] = new_id
    _vectorize_and_append(job_dict)

    return {"id": new_id, "message": "Offre créée"}


@router.put("/jobs/{job_id}")
def update_job(job_id: int, job: JobUpdate):
    conn = get_conn()
    existing = conn.execute("SELECT * FROM custom_jobs WHERE id=?", (job_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Offre introuvable")

    fields = {k: v for k, v in job.dict().items() if v is not None}
    if not fields:
        conn.close()
        raise HTTPException(status_code=400, detail="Aucun champ à modifier")

    set_clause = ", ".join([f"{k}=?" for k in fields])
    conn.execute(f"UPDATE custom_jobs SET {set_clause} WHERE id=?",
                 (*fields.values(), job_id))
    conn.commit()
    conn.close()

    # Re-vectoriser : supprimer l'ancien puis ajouter le nouveau
    _remove_from_engine(job_id)
    cols = ["id","job_title","description","skills_desc","experience_level",
            "location","salary_usd","remote_ratio","created_at"]
    conn = get_conn()
    row = conn.execute("SELECT * FROM custom_jobs WHERE id=?", (job_id,)).fetchone()
    conn.close()
    job_dict = dict(zip(cols, row))
    job_dict["custom_id"] = job_id
    _vectorize_and_append(job_dict)

    return {"message": "Offre modifiée"}


@router.delete("/jobs/{job_id}")
def delete_job(job_id: int):
    conn = get_conn()
    existing = conn.execute("SELECT id FROM custom_jobs WHERE id=?", (job_id,)).fetchone()
    if not existing:
        conn.close()
        raise HTTPException(status_code=404, detail="Offre introuvable")
    conn.execute("DELETE FROM custom_jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()

    _remove_from_engine(job_id)
    return {"message": "Offre supprimée ✅"}