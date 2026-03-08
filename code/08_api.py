"""
API REST FastAPI — Job Matching Universel v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Endpoints :
  POST /match/text     → Texte libre → matching
  POST /match/file     → Upload CV (PDF/DOCX/TXT) → matching
  GET  /jobs           → Liste paginée des offres
  GET  /jobs/search    → Recherche par mot-clé
  GET  /health         → État du système
  GET  /stats          → Statistiques du corpus

Swagger UI : http://localhost:8000/docs

Lancer :
  python api.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess
import sys

def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["fastapi", "uvicorn[standard]", "python-multipart"]:
    try:
        __import__(pkg.split("[")[0].replace("-", "_"))
    except ImportError:
        print(f"📦 Installation {pkg}...")
        _install(pkg)

import os
import time
import tempfile
import importlib
import importlib.util
import glob
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))


# ── Chargement dynamique des modules locaux ───────────────────
def _load_module(name, patterns):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        for pat in patterns:
            files = glob.glob(str(Path(__file__).parent / pat))
            if files:
                spec = importlib.util.spec_from_file_location(name, files[0])
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        raise ImportError(f"Module '{name}' introuvable. "
                          f"Vérifie que cv_parser.py et matching_engine.py "
                          f"sont dans le même dossier que api.py.")

_cv = _load_module("cv_parser",       ["cv_parser*.py"])
_me = _load_module("matching_engine", ["matching_engine*.py", "*matching_engine*.py"])

CVParser         = _cv.CVParser
MatchingEngine   = _me.MatchingEngine
CandidateProfile = _me.CandidateProfile
parse_cv_to_profile = _me.parse_cv_to_profile


# ══════════════════════════════════════════════════════════════
# APPLICATION FASTAPI
# ══════════════════════════════════════════════════════════════

app = FastAPI(
    title="Job Matching API — Universel v3",
    description="""
## 🎯 Système de Matching CV ↔ Offres d'Emploi — Universel

API REST pour associer automatiquement **n'importe quel candidat** aux offres
les plus pertinentes grâce à **Sentence-BERT** et la similarité cosinus.

### ✅ Fonctionne pour tous les profils
- 👨‍💻 **Tech** : Développeurs, Data Scientists, DevOps, Architectes
- 💰 **Finance** : Comptables, Auditeurs, Analystes financiers, Contrôleurs
- 📣 **Marketing** : Chefs de projet digital, SEO, Community Managers
- 👥 **RH** : Recruteurs, Responsables RH, Talent Acquisition
- ⚖️ **Droit** : Juristes, Avocats, Compliance Officers
- 🏥 **Santé** : Médecins, Infirmiers, Pharmaciens
- 📊 **Management** : Chefs de projet, Product Owners, Managers

### 🌍 Multilingue
Détecte automatiquement les CV en **Français**, **Anglais** ou mixte.

### ⚡ Performance
- Résultats en **< 1 seconde** sur 111 000+ offres
- Modèle : Sentence-BERT `all-MiniLM-L6-v2` (384 dimensions)
    """,
    version="3.0.0",
    contact={"name": "Job Matching System v3"},
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── État global ───────────────────────────────────────────────
engine: MatchingEngine = None
parser: CVParser = None


@app.on_event("startup")
async def startup_event():
    global engine, parser
    print("🚀 Démarrage API v3 — chargement du Matching Engine...")
    try:
        engine = MatchingEngine()
        engine.load()
        parser = CVParser()
        print("✅ API v3 prête !")
    except Exception as e:
        print(f"❌ Erreur au démarrage : {e}")
        print("   L'API démarre quand même — certains endpoints seront indisponibles.")


def _check_engine():
    if not engine or not engine._loaded:
        raise HTTPException(
            status_code=503,
            detail="Moteur de matching non prêt. Réessaie dans quelques secondes."
        )


# ══════════════════════════════════════════════════════════════
# SCHÉMAS PYDANTIC
# ══════════════════════════════════════════════════════════════

class MatchRequest(BaseModel):
    """Corps de la requête pour /match/text"""
    name: str = Field(
        "Candidat",
        description="Nom du candidat"
    )
    cv_text: str = Field(
        ...,
        min_length=20,
        description="Texte du CV (résumé, compétences, expériences) — n'importe quel domaine"
    )
    experience_level: str = Field(
        "auto",
        description="Niveau : auto (détection auto) / entry / mid / senior / executive"
    )
    desired_location: str = Field(
        "",
        description="Ville ou pays souhaité (laisser vide = tous pays)"
    )
    min_salary: float = Field(0, ge=0, description="Salaire minimum en USD (0 = pas de filtre)")
    max_salary: float = Field(0, ge=0, description="Salaire maximum en USD (0 = pas de filtre)")
    remote_only: bool = Field(False, description="True = offres remote uniquement")
    employment_type: str = Field(
        "",
        description="Type de contrat : FT (plein temps) / PT (partiel) / CT (contrat) / vide = tous"
    )
    top_k: int = Field(10, ge=1, le=50, description="Nombre de résultats (1-50)")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "Amina Ben Salah",
                "cv_text": (
                    "Comptable confirmée, 5 ans d'expérience. "
                    "Comptabilité générale, Audit, IFRS, Sage, Excel, VBA, "
                    "Contrôle de gestion, Reporting financier, Budget. "
                    "Diplômée ISCAE Tunis."
                ),
                "experience_level": "auto",
                "desired_location": "",
                "min_salary": 0,
                "remote_only": False,
                "top_k": 10,
            }
        }


class JobResult(BaseModel):
    rank:             int
    job_title:        str
    location:         str
    salary_usd:       float
    has_salary:       bool
    experience_level: str
    employment_type:  str
    remote_ratio:     int
    company_size:     str
    source:           str
    sbert_score:      float
    filter_bonus:     float
    final_score:      float
    match_reasons:    List[str]


class MatchResponse(BaseModel):
    candidate_name:     str
    experience_level:   str
    domain_detected:    str
    skills_detected:    List[str]
    total_jobs_scanned: int
    eligible_jobs:      int
    results:            List[JobResult]
    top_score:          float
    avg_score:          float
    inference_time_ms:  float


class HealthResponse(BaseModel):
    status:       str
    engine_ready: bool
    corpus_size:  int
    model_name:   str
    version:      str


class StatsResponse(BaseModel):
    total_jobs:                int
    jobs_with_salary:          int
    jobs_remote:               int
    unique_locations:          int
    unique_titles:             int
    sources:                   dict
    experience_distribution:   dict
    employment_type_distribution: dict


# ══════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Système"],
    summary="État du système",
)
async def health_check():
    return HealthResponse(
        status       = "ok" if engine and engine._loaded else "loading",
        engine_ready = bool(engine and engine._loaded),
        corpus_size  = len(engine.df) if engine and engine._loaded else 0,
        model_name   = getattr(engine, "sbert_model_name", "N/A"),
        version      = "3.0.0",
    )


@app.get(
    "/stats",
    response_model=StatsResponse,
    tags=["Système"],
    summary="Statistiques du corpus d'offres",
)
async def get_stats():
    _check_engine()
    df = engine.df
    return StatsResponse(
        total_jobs                   = len(df),
        jobs_with_salary             = int((df["salary_usd"] > 1000).sum()),
        jobs_remote                  = int((df["remote_ratio"] == 100).sum()),
        unique_locations             = int(df["location"].nunique()),
        unique_titles                = int(df["job_title"].nunique()),
        sources                      = df["source"].value_counts().to_dict(),
        experience_distribution      = df["experience_level"].value_counts().to_dict(),
        employment_type_distribution = df["employment_type"].value_counts().to_dict(),
    )


@app.post(
    "/match/text",
    response_model=MatchResponse,
    tags=["Matching"],
    summary="Matching depuis texte libre",
    description="""
Lance le matching à partir d'un **texte libre** (résumé, profil, extrait de CV).

**Fonctionne pour tous les domaines :**
tech, finance, marketing, RH, droit, santé, management, vente...

**Étapes internes :**
1. Extraction automatique des compétences (tous domaines)
2. Détection du domaine (tech / finance / RH / marketing / ...)
3. Détection du niveau (entry / mid / senior / executive)
4. Encodage Sentence-BERT + matching sur toutes les offres
5. Application des filtres et bonus
6. Retour des Top-K offres
    """,
)
async def match_from_text(request: MatchRequest):
    _check_engine()
    t0 = time.time()

    # Parser le texte pour extraire automatiquement skills + niveau + domaine
    cv = parser.parse_text(request.cv_text)

    # Si le niveau est fourni (pas "auto"), on l'utilise ; sinon on garde celui détecté
    exp_level = (
        cv.experience_level
        if request.experience_level in ("auto", "unknown", "")
        else request.experience_level
    )

    candidate = CandidateProfile(
        name             = request.name,
        summary          = request.cv_text,
        skills           = cv.skills,
        experience_level = exp_level,
        desired_location = request.desired_location or cv.location,
        min_salary       = request.min_salary,
        max_salary       = request.max_salary,
        remote_only      = request.remote_only,
        employment_type  = request.employment_type,
        domain           = cv.domain,
    )

    results = engine.match(candidate, top_k=request.top_k, min_score=0.20)
    elapsed = (time.time() - t0) * 1000

    try:
        eligible = int(engine._apply_filters(candidate).sum())
    except Exception:
        eligible = len(results)

    return MatchResponse(
        candidate_name     = request.name,
        experience_level   = candidate.experience_level,
        domain_detected    = cv.domain,
        skills_detected    = cv.skills[:25],
        total_jobs_scanned = len(engine.df),
        eligible_jobs      = eligible,
        results            = [JobResult(**r.__dict__) for r in results],
        top_score          = round(results[0].final_score, 4) if results else 0.0,
        avg_score          = round(float(np.mean([r.final_score for r in results])), 4)
                             if results else 0.0,
        inference_time_ms  = round(elapsed, 2),
    )


@app.post(
    "/match/file",
    response_model=MatchResponse,
    tags=["Matching"],
    summary="Matching depuis fichier CV (PDF / DOCX / TXT)",
    description="""
Lance le matching à partir d'un **fichier CV uploadé**.

**Formats acceptés :** `.pdf`, `.docx`, `.doc`, `.txt`

**Étapes internes :**
1. Extraction du texte (pdfplumber / python-docx)
2. Parsing universel : nom, email, compétences, niveau, domaine, localisation
3. Encodage SBERT + matching
4. Retour des Top-K offres les plus pertinentes
    """,
)
async def match_from_file(
    file: UploadFile = File(..., description="Fichier CV — PDF, DOCX ou TXT"),
    experience_level: str   = Query("auto",  description="auto / entry / mid / senior / executive"),
    desired_location: str   = Query("",      description="Ville ou pays souhaité"),
    min_salary:       float = Query(0,       description="Salaire minimum USD"),
    remote_only:      bool  = Query(False,   description="Remote uniquement"),
    employment_type:  str   = Query("",      description="FT / PT / CT / vide = tous"),
    top_k:            int   = Query(10, ge=1, le=50, description="Nombre de résultats"),
):
    _check_engine()

    # Vérifier le format
    allowed_ext = {".pdf", ".docx", ".doc", ".txt"}
    file_ext    = Path(file.filename or "file.txt").suffix.lower()
    if file_ext not in allowed_ext:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Format non supporté : '{file_ext}'. "
                f"Utilise PDF, DOCX, DOC ou TXT."
            )
        )

    t0 = time.time()

    # Sauvegarder temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        content = await file.read()
        if len(content) == 0:
            raise HTTPException(status_code=400, detail="Fichier vide.")
        if len(content) > 10 * 1024 * 1024:  # 10 Mo max
            raise HTTPException(status_code=413, detail="Fichier trop lourd (max 10 Mo).")
        tmp.write(content)
        tmp_path = tmp.name

    try:
        cv = parser.parse_file(tmp_path)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Erreur lecture CV : {str(e)}")
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    if not cv.is_valid():
        raise HTTPException(
            status_code=422,
            detail=(
                "CV illisible ou trop court. "
                "Vérifie que le fichier contient du texte (pas une image scannée)."
            )
        )

    exp_level = (
        cv.experience_level
        if experience_level in ("auto", "unknown", "")
        else experience_level
    )

    candidate = CandidateProfile(
        name             = cv.name or "Candidat",
        summary          = cv.raw_text,
        skills           = cv.skills,
        experience_level = exp_level,
        desired_location = desired_location or cv.location,
        min_salary       = min_salary,
        remote_only      = remote_only,
        employment_type  = employment_type,
        domain           = cv.domain,
    )

    results = engine.match(candidate, top_k=top_k, min_score=0.20)
    elapsed = (time.time() - t0) * 1000

    try:
        eligible = int(engine._apply_filters(candidate).sum())
    except Exception:
        eligible = len(results)

    return MatchResponse(
        candidate_name     = cv.name or "Candidat",
        experience_level   = candidate.experience_level,
        domain_detected    = cv.domain,
        skills_detected    = cv.skills[:25],
        total_jobs_scanned = len(engine.df),
        eligible_jobs      = eligible,
        results            = [JobResult(**r.__dict__) for r in results],
        top_score          = round(results[0].final_score, 4) if results else 0.0,
        avg_score          = round(float(np.mean([r.final_score for r in results])), 4)
                             if results else 0.0,
        inference_time_ms  = round(elapsed, 2),
    )


@app.get(
    "/jobs",
    tags=["Offres"],
    summary="Liste paginée des offres",
    description="Retourne les offres du corpus avec pagination.",
)
async def list_jobs(
    page:     int = Query(1,   ge=1,          description="Page (commence à 1)"),
    per_page: int = Query(20,  ge=1,  le=100, description="Offres par page (max 100)"),
    source:   str = Query("",                 description="Filtrer par source"),
    level:    str = Query("",                 description="Filtrer par niveau : entry / mid / senior"),
):
    _check_engine()

    df = engine.df.copy()
    if source:
        df = df[df["source"] == source]
    if level:
        df = df[df["experience_level"].str.lower() == level.lower()]

    total = len(df)
    start = (page - 1) * per_page
    end   = start + per_page

    cols = ["job_title", "location", "salary_usd", "experience_level",
            "employment_type", "remote_ratio", "company_size", "source"]
    jobs = df.iloc[start:end][cols].fillna("").to_dict(orient="records")

    return {
        "total":    total,
        "page":     page,
        "per_page": per_page,
        "pages":    max(1, (total + per_page - 1) // per_page),
        "jobs":     jobs,
    }


@app.get(
    "/jobs/search",
    tags=["Offres"],
    summary="Recherche d'offres par mot-clé",
    description="Recherche par mot-clé dans les titres de postes.",
)
async def search_jobs(
    q:   str = Query(..., min_length=2, description="Mot-clé dans le titre"),
    top: int = Query(20,  ge=1, le=100, description="Nombre max de résultats"),
):
    _check_engine()

    mask    = engine.df["job_title"].str.lower().str.contains(q.lower(), na=False)
    results = engine.df[mask].head(top)

    return {
        "query":   q,
        "count":   int(mask.sum()),
        "results": (
            results[["job_title", "location", "salary_usd",
                      "experience_level", "remote_ratio", "source"]]
            .fillna("")
            .to_dict(orient="records")
        ),
    }


# ══════════════════════════════════════════════════════════════
# LANCEMENT
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("\n" + "="*60)
    print("  🚀 JOB MATCHING API v3 — UNIVERSEL")
    print("="*60)
    print("  📖 Swagger UI  : http://localhost:8000/docs")
    print("  📖 ReDoc       : http://localhost:8000/redoc")
    print("  🔍 Health      : http://localhost:8000/health")
    print("  📊 Stats       : http://localhost:8000/stats")
    print("="*60 + "\n")

    uvicorn.run(
        app,
        host  = "0.0.0.0",
        port  = 8000,
        reload= False,
    )