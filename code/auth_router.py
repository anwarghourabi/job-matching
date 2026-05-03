"""
Auth Router — Signup / Login / Profil utilisateur
JWT + SQLite + Recommandations personnalisées
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, List
import sqlite3
import hashlib
import jwt
import datetime
from pathlib import Path
import random
import smtplib 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

router = APIRouter(prefix="/auth", tags=["Authentification"])
recommendations_router = APIRouter(prefix="/recommendations", tags=["Recommandations"])
# Config email — mets tes vraies infos
SMTP_EMAIL = os.environ.get("SMTP_EMAIL", "anwarghourabi8@gmail.com")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "lias fkvm jwkt imvj")
DB_PATH = Path(__file__).parent.parent / "data" / "processed" / "custom_jobs.db"
SECRET_KEY = "job_matching_secret_key_2024"
ALGORITHM = "HS256"
security = HTTPBearer(auto_error=False)

# Stockage temporaire des codes (en mémoire)
_verification_codes: dict = {}  # email -> {code, expires_at, user_data}
# ══════════════════════════════════════════════════════════════
# MODÈLES
# ══════════════════════════════════════════════════════════════

class SignupRequest(BaseModel):
    email: str
    password: str
    full_name: str

class LoginRequest(BaseModel):
    email: str
    password: str

class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    cv_text: Optional[str] = None
    experience_level: Optional[str] = None
    desired_location: Optional[str] = None
    skills: Optional[str] = None

class HistoryEntry(BaseModel):
    job_title: str
    job_source: Optional[str] = ""
    score: Optional[float] = 0.0
    
class SendCodeRequest(BaseModel):
    email: str
    full_name: str
    password: str
    confirm_password: str

class VerifyCodeRequest(BaseModel):
    email: str
    code: str
# ══════════════════════════════════════════════════════════════
# DB INIT
# ══════════════════════════════════════════════════════════════

def init_auth_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT DEFAULT '',
            cv_text TEXT DEFAULT '',
            experience_level TEXT DEFAULT 'mid',
            desired_location TEXT DEFAULT '',
            skills TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS match_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_title TEXT NOT NULL,
            job_source TEXT DEFAULT '',
            score REAL DEFAULT 0.0,
            viewed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    conn.commit()
    conn.close()

def get_conn():
    return sqlite3.connect(DB_PATH)

# ══════════════════════════════════════════════════════════════
# HELPERS JWT
# ══════════════════════════════════════════════════════════════

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_token(user_id: int, email: str) -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expiré")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Token invalide")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Token manquant")
    return decode_token(credentials.credentials)

def get_optional_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if not credentials:
        return None
    try:
        return decode_token(credentials.credentials)
    except:
        return None

# ══════════════════════════════════════════════════════════════
# ENGINE REFERENCE
# ══════════════════════════════════════════════════════════════

_engine = None

def set_auth_engine(engine):
    global _engine
    _engine = engine

# ══════════════════════════════════════════════════════════════
# AUTH ENDPOINTS
# ══════════════════════════════════════════════════════════════

@router.post("/signup")
def signup(req: SignupRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Mot de passe trop court (min 6 caractères)")
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Email invalide")

    conn = get_conn()
    existing = conn.execute("SELECT id FROM users WHERE email=?", (req.email,)).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="Email déjà utilisé")

    pwd_hash = hash_password(req.password)
    cur = conn.execute(
        "INSERT INTO users (email, password_hash, full_name) VALUES (?,?,?)",
        (req.email, pwd_hash, req.full_name)
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()

    token = create_token(user_id, req.email)
    return {"token": token, "user": {"id": user_id, "email": req.email, "full_name": req.full_name}}

@router.post("/send-code")
def send_code(req: SendCodeRequest):
    """Étape 1 du signup — envoyer le code de vérification"""
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Mot de passe trop court (min 6 caractères)")
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Les mots de passe ne correspondent pas")
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Email invalide")

    conn = get_conn()
    existing = conn.execute("SELECT id FROM users WHERE email=?", (req.email,)).fetchone()
    conn.close()
    if existing:
        raise HTTPException(status_code=409, detail="Email déjà utilisé")

    # Générer code 6 chiffres
    code = str(random.randint(100000, 999999))
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)

    _verification_codes[req.email] = {
        "code": code,
        "expires_at": expires_at,
        "full_name": req.full_name,
        "password": req.password
    }

    send_verification_email(req.email, code, req.full_name)
    return {"message": f"Code envoyé à {req.email}"}


@router.post("/verify-code")
def verify_code(req: VerifyCodeRequest):
    """Étape 2 du signup — vérifier le code et créer le compte"""
    entry = _verification_codes.get(req.email)

    if not entry:
        raise HTTPException(status_code=400, detail="Aucun code envoyé pour cet email")
    if datetime.datetime.utcnow() > entry["expires_at"]:
        del _verification_codes[req.email]
        raise HTTPException(status_code=400, detail="Code expiré — recommence l'inscription")
    if entry["code"] != req.code:
        raise HTTPException(status_code=400, detail="Code incorrect")

    # Créer le compte
    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, full_name) VALUES (?,?,?)",
            (req.email, hash_password(entry["password"]), entry["full_name"])
        )
        conn.commit()
        user_id = cur.lastrowid
    except Exception:
        conn.close()
        raise HTTPException(status_code=409, detail="Email déjà utilisé")
    conn.close()

    del _verification_codes[req.email]
    token = create_token(user_id, req.email)
    return {"token": token, "user": {"id": user_id, "email": req.email, "full_name": entry["full_name"]}}

@router.post("/login")
def login(req: LoginRequest):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, email, full_name, password_hash FROM users WHERE email=?",
        (req.email,)
    ).fetchone()
    conn.close()

    if not row or row[3] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    token = create_token(row[0], row[1])
    return {"token": token, "user": {"id": row[0], "email": row[1], "full_name": row[2]}}


@router.get("/me")
def get_me(user=Depends(get_current_user)):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, email, full_name, cv_text, experience_level, desired_location, skills, created_at FROM users WHERE id=?",
        (user["user_id"],)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return {
        "id": row[0], "email": row[1], "full_name": row[2],
        "cv_text": row[3], "experience_level": row[4],
        "desired_location": row[5], "skills": row[6], "created_at": row[7]
    }


@router.put("/profile")
def update_profile(req: ProfileUpdate, user=Depends(get_current_user)):
    fields = {k: v for k, v in req.dict().items() if v is not None}
    if not fields:
        raise HTTPException(status_code=400, detail="Aucun champ à modifier")

    set_clause = ", ".join([f"{k}=?" for k in fields])
    conn = get_conn()
    conn.execute(f"UPDATE users SET {set_clause} WHERE id=?", (*fields.values(), user["user_id"]))
    conn.commit()
    conn.close()
    return {"message": "Profil mis à jour ✅"}


@router.post("/history")
def add_to_history(entry: HistoryEntry, user=Depends(get_current_user)):
    conn = get_conn()
    conn.execute(
        "INSERT INTO match_history (user_id, job_title, job_source, score) VALUES (?,?,?,?)",
        (user["user_id"], entry.job_title, entry.job_source, entry.score)
    )
    conn.commit()
    conn.close()
    return {"message": "Ajouté à l'historique"}


@router.get("/history")
def get_history(user=Depends(get_current_user)):
    conn = get_conn()
    rows = conn.execute(
        "SELECT job_title, job_source, score, viewed_at FROM match_history WHERE user_id=? ORDER BY viewed_at DESC LIMIT 50",
        (user["user_id"],)
    ).fetchall()
    conn.close()
    return [{"job_title": r[0], "job_source": r[1], "score": r[2], "viewed_at": r[3]} for r in rows]
def send_verification_email(email: str, code: str, full_name: str):
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print(f"   ⚠️  Email non configuré — code: {code}")
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "🔐 Code de vérification — Job Matching"
    msg["From"] = SMTP_EMAIL
    msg["To"] = email

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:480px;margin:0 auto;background:#13132a;padding:2rem;border-radius:12px;border:1px solid rgba(99,255,180,0.2)">
      <h2 style="color:#63ffb4;margin:0 0 1rem">Job Matching Platform</h2>
      <p style="color:#fff">Bonjour <strong>{full_name}</strong>,</p>
      <p style="color:rgba(255,255,255,0.7)">Voici ton code de vérification :</p>
      <div style="background:rgba(99,255,180,0.1);border:1px solid rgba(99,255,180,0.3);border-radius:10px;padding:1.5rem;text-align:center;margin:1.5rem 0">
        <span style="font-size:2.5rem;font-weight:700;color:#63ffb4;letter-spacing:0.3em">{code}</span>
      </div>
      <p style="color:rgba(255,255,255,0.4);font-size:0.85rem">Ce code expire dans <strong style="color:#fff">10 minutes</strong>.</p>
    </div>
    """
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, email, msg.as_string())
        print(f"   ✅ Email envoyé à {email}")
    except Exception as e:
        print(f"   ❌ Erreur email : {e}")

# ══════════════════════════════════════════════════════════════
# RECOMMANDATIONS
# ══════════════════════════════════════════════════════════════

@recommendations_router.get("")
def get_recommendations(user=Depends(get_current_user)):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine non prêt")

    conn = get_conn()
    row = conn.execute(
        "SELECT cv_text, experience_level, desired_location, skills FROM users WHERE id=?",
        (user["user_id"],)
    ).fetchone()
    history = conn.execute(
        "SELECT job_title FROM match_history WHERE user_id=? ORDER BY viewed_at DESC LIMIT 20",
        (user["user_id"],)
    ).fetchall()
    conn.close()

    cv_text = row[0] or ""
    experience_level = row[1] or "mid"
    desired_location = row[2] or ""
    skills = row[3] or ""
    history_titles = [h[0] for h in history]

    # Construire le texte de recommandation
    profile_text = f"{cv_text} {skills} "
    if history_titles:
        profile_text += " ".join(history_titles)

    if not profile_text.strip():
        return {"recommendations": [], "message": "Complète ton profil pour obtenir des recommandations"}

    # Vectoriser et matcher
    import numpy as np
    vec = _engine.sbert_model.encode(
        [profile_text],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    scores = _engine.sbert_matrix @ vec
    top_indices = np.argsort(scores)[::-1][:10]

    results = []
    for idx in top_indices:
        row = _engine.df.iloc[idx]
        results.append({
            "job_title": str(row.get("job_title", "")),
            "location": str(row.get("location", "")),
            "experience_level": str(row.get("experience_level", "")),
            "salary_usd": float(row.get("salary_usd", 0)),
            "remote_ratio": int(row.get("remote_ratio", 0)),
            "source": str(row.get("source", "")),
            "score": round(float(scores[idx]), 4)
        })

    return {"recommendations": results, "based_on": {
        "cv": bool(cv_text),
        "history": len(history_titles)
    }}