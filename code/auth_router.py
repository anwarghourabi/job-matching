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
from fastapi import UploadFile, File, Query

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
    role: str = "candidate"
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
    role: str = "candidate"

class VerifyCodeRequest(BaseModel):
    email: str
    code: str

class ApplicationRequest(BaseModel):
    job_title: str
    job_source: str = ""
    job_location: str = ""
    salary_usd: float = 0
    cover_letter: str = ""

class StatusUpdate(BaseModel):
    status: str  # "acceptée" | "rejetée"

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
            cv_filename TEXT    DEFAULT '',   
            cv_file     BLOB    DEFAULT NULL,
            experience_level TEXT DEFAULT 'mid',
            desired_location TEXT DEFAULT '',
            skills TEXT DEFAULT '',
            role TEXT DEFAULT 'candidate',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP

        )
    """)
    try:
        conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'candidate'")
        conn.commit()
        print("✅ Migration : colonne 'role' ajoutée")
    except Exception:
        pass 
    try:
        conn.execute("ALTER TABLE users ADD COLUMN cv_filename TEXT DEFAULT ''")
        conn.commit()
    except: pass
    try:
        conn.execute("ALTER TABLE users ADD COLUMN cv_file BLOB DEFAULT NULL")
        conn.commit()
    except: pass

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

    conn.execute("""
        CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            job_title TEXT NOT NULL,
            job_source TEXT DEFAULT '',
            job_location TEXT DEFAULT '',
            salary_usd REAL DEFAULT 0,
            cover_letter TEXT DEFAULT '',
            status TEXT DEFAULT 'envoyée',
            cv_file      BLOB DEFAULT NULL,    
            cv_filename  TEXT DEFAULT '',   
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    for col in ["cv_file BLOB DEFAULT NULL", "cv_filename TEXT DEFAULT ''"]:
        try:
            conn.execute(f"ALTER TABLE applications ADD COLUMN {col}")
            conn.commit()
        except:
            pass
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
        "INSERT INTO users (email, password_hash, full_name, role) VALUES (?,?,?,?)",
        (req.email, pwd_hash, req.full_name, req.role)
    )
    conn.commit()
    user_id = cur.lastrowid
    conn.close()

    token = create_token(user_id, req.email)
    return {"token": token, "user": {"id": user_id, "email": req.email, "full_name": req.full_name}}

@router.post("/send-code")
def send_code(req: SendCodeRequest):
    if len(req.password) < 6:
        raise HTTPException(status_code=400, detail="Mot de passe trop court (min 6 caractères)")
    if req.password != req.confirm_password:
        raise HTTPException(status_code=400, detail="Les mots de passe ne correspondent pas")
    if "@" not in req.email:
        raise HTTPException(status_code=400, detail="Email invalide")

    # Normaliser le rôle
    role_map = {
        "recruiter": "recruteur", "recruteur": "recruteur",
        "candidate": "candidat",  "candidat":  "candidat",
    }
    role = role_map.get(req.role.lower(), "candidat")   # ← normalisation

    conn = get_conn()
    existing = conn.execute("SELECT id FROM users WHERE email=?", (req.email,)).fetchone()
    conn.close()
    if existing:
        raise HTTPException(status_code=409, detail="Email déjà utilisé")

    code = str(random.randint(100000, 999999))
    expires_at = datetime.datetime.utcnow() + datetime.timedelta(minutes=10)

    _verification_codes[req.email] = {
        "code": code,
        "expires_at": expires_at,
        "full_name": req.full_name,
        "password": req.password,
        "role": role             # ← était manquant !
    }

    send_verification_email(req.email, code, req.full_name)
    return {"message": f"Code envoyé à {req.email}"}


@router.post("/verify-code")
def verify_code(req: VerifyCodeRequest):
    entry = _verification_codes.get(req.email)

    if not entry:
        raise HTTPException(status_code=400, detail="Aucun code envoyé pour cet email")
    if datetime.datetime.utcnow() > entry["expires_at"]:
        del _verification_codes[req.email]
        raise HTTPException(status_code=400, detail="Code expiré — recommence l'inscription")
    if entry["code"] != req.code:
        raise HTTPException(status_code=400, detail="Code incorrect")

    # Fallback si role absent (rétrocompatibilité)
    role = entry.get("role", "candidate")

    conn = get_conn()
    try:
        cur = conn.execute(
            "INSERT INTO users (email, password_hash, full_name, role) VALUES (?,?,?,?)",
            (req.email, hash_password(entry["password"]), entry["full_name"], role)
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        # Seule vraie raison : UNIQUE constraint sur email
        conn.close()
        raise HTTPException(status_code=409, detail="Email déjà utilisé")
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Erreur base de données : {str(e)}")
    finally:
        # Garantit la fermeture même si conn.close() a déjà été appelé
        try:
            conn.close()
        except Exception:
            pass

    del _verification_codes[req.email]
    token = create_token(user_id, req.email)
    return {
        "token": token,
        "user": {
            "id": user_id,
            "email": req.email,
            "full_name": entry["full_name"],
            "role": role        # ← indispensable pour le frontend
        }
    }

@router.post("/login")
def login(req: LoginRequest):
    conn = get_conn()
    row = conn.execute(
        "SELECT id, email, full_name, password_hash, role FROM users WHERE email=?",
        (req.email,)
    ).fetchone()
    conn.close()

    if not row or row[3] != hash_password(req.password):
        raise HTTPException(status_code=401, detail="Email ou mot de passe incorrect")

    token = create_token(row[0], row[1])
    return {
        "token": token,
        "user": {
            "id":        row[0],
            "email":     row[1],
            "full_name": row[2],
            "role":      row[4]   # ← ajouter
        }
    }


@router.get("/me")
def get_me(user=Depends(get_current_user)):
    conn = get_conn()
    row = conn.execute(
        """SELECT id, email, full_name, cv_text, experience_level, 
                  desired_location, skills, role, created_at 
           FROM users WHERE id=?""",
        (user["user_id"],)
    ).fetchone()
    conn.close()
    if not row:
        raise HTTPException(status_code=404, detail="Utilisateur introuvable")
    return {
        "id":               row[0],
        "email":            row[1],
        "full_name":        row[2],
        "cv_text":          row[3],
        "experience_level": row[4],
        "desired_location": row[5],
        "skills":           row[6],
        "role":             row[7],   # ← ajouter
        "created_at":       row[8],
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

@router.post("/apply")
def apply_to_job(req: ApplicationRequest, user=Depends(get_current_user)):
    conn = get_conn()

    existing = conn.execute(
        "SELECT id FROM applications WHERE user_id=? AND job_title=?",
        (user["user_id"], req.job_title)
    ).fetchone()
    if existing:
        conn.close()
        raise HTTPException(status_code=409, detail="Vous avez déjà postulé à cette offre")

    # Récupérer le CV actuel du candidat au moment de la postulation
    cv_row = conn.execute(
        "SELECT cv_file, cv_filename FROM users WHERE id=?",
        (user["user_id"],)
    ).fetchone()
    cv_file     = cv_row[0] if cv_row else None
    cv_filename = cv_row[1] if cv_row else ""

    conn.execute(
        """INSERT INTO applications 
           (user_id, job_title, job_source, job_location, salary_usd, cover_letter, cv_file, cv_filename)
           VALUES (?,?,?,?,?,?,?,?)""",
        (user["user_id"], req.job_title, req.job_source,
         req.job_location, req.salary_usd, req.cover_letter,
         cv_file, cv_filename)   # ← snapshot du CV
    )
    conn.commit()
    conn.close()
    return {"message": "Candidature envoyée ✅"}    

@router.get("/applications")
def get_applications(user=Depends(get_current_user)):
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, job_title, job_source, job_location, salary_usd, status, applied_at FROM applications WHERE user_id=? ORDER BY applied_at DESC",
        (user["user_id"],)
    ).fetchall()
    conn.close()
    return [{"id": r[0], "job_title": r[1], "job_source": r[2], "job_location": r[3],
            "salary_usd": r[4], "status": r[5], "applied_at": r[6]} for r in rows]


@router.get("/recruiter/applications")
def get_all_applications(user=Depends(get_current_user)):
    conn = get_conn()

    row = conn.execute("SELECT role FROM users WHERE id=?", (user["user_id"],)).fetchone()
    if not row or row[0] != "recruteur":
        conn.close()
        raise HTTPException(status_code=403, detail="Accès réservé aux recruteurs")

    rows = conn.execute("""
        SELECT 
            a.id,               -- 0
            a.job_title,        -- 1
            a.job_location,     -- 2
            a.salary_usd,       -- 3
            a.cover_letter,     -- 4
            a.status,           -- 5
            a.applied_at,       -- 6
            u.id,               -- 7
            u.full_name,        -- 8
            u.email,            -- 9
            u.experience_level, -- 10
            u.skills,           -- 11
            u.cv_text,          -- 12
            a.cv_filename,      -- 13 ← depuis applications
            a.id                -- 14 ← app_id pour le téléchargement
        FROM applications a
        JOIN users u ON a.user_id = u.id
        JOIN custom_jobs cj ON cj.job_title = a.job_title
                            AND cj.user_id = ?
        ORDER BY a.applied_at DESC
    """, (user["user_id"],)).fetchall()

    return [
        {
            "id":               r[0],
            "job_title":        r[1],
            "job_location":     r[2],
            "salary_usd":       r[3],
            "cover_letter":     r[4],
            "status":           r[5],
            "applied_at":       r[6],
            "candidate_id":     r[7],
            "candidate_name":   r[8],
            "candidate_email":  r[9],
            "experience_level": r[10],
            "skills":           r[11],
            "cv_text":          r[12],
            "cv_filename":      r[13] if len(r) > 13 else "",
            "application_id":   r[14],  # ← pour télécharger le bon CV
        }
        for r in rows
    ]
 
 
@router.put("/recruiter/applications/{app_id}")
def update_application_status(app_id: int, req: StatusUpdate, user=Depends(get_current_user)):
    """Accepter ou rejeter une candidature"""
    if req.status not in ("acceptée", "rejetée"):
        raise HTTPException(status_code=400, detail="Statut invalide — utilisez 'acceptée' ou 'rejetée'")
 
    # Vérifier rôle recruteur
    conn = get_conn()
    row = conn.execute("SELECT role FROM users WHERE id=?", (user["user_id"],)).fetchone()
    if not row or row[0] != "recruteur":
        conn.close()
        raise HTTPException(status_code=403, detail="Accès réservé aux recruteurs")
 
    # Récupérer la candidature + infos candidat
    app_row = conn.execute("""
        SELECT a.id, a.job_title, a.status, u.email, u.full_name
        FROM applications a
        JOIN users u ON a.user_id = u.id
        WHERE a.id = ?
    """, (app_id,)).fetchone()
 
    if not app_row:
        conn.close()
        raise HTTPException(status_code=404, detail="Candidature introuvable")
 
    _, job_title, current_status, candidate_email, candidate_name = app_row
 
    # Mettre à jour le statut
    conn.execute(
        "UPDATE applications SET status=? WHERE id=?",
        (req.status, app_id)
    )
    conn.commit()
    conn.close()
 
    # Envoyer email uniquement si acceptée
    if req.status == "acceptée":
        _send_acceptance_email(candidate_email, candidate_name, job_title)
 
    return {"message": f"Candidature {req.status}", "status": req.status}
 
 
def _send_acceptance_email(email: str, candidate_name: str, job_title: str):
    """Envoyer un email d'acceptation au candidat"""
    if not SMTP_EMAIL or not SMTP_PASSWORD:
        print(f"   ⚠️  Email non configuré — acceptation pour {email}")
        return
 
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🎉 Votre candidature a été acceptée — {job_title}"
    msg["From"]    = SMTP_EMAIL
    msg["To"]      = email
 
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:520px;margin:0 auto;
                background:#13132a;padding:2rem;border-radius:12px;
                border:1px solid rgba(99,255,180,0.2)">
 
      <h2 style="color:#63ffb4;margin:0 0 1rem">🎉 Félicitations !</h2>
 
      <p style="color:#fff;margin:0 0 0.75rem">
        Bonjour <strong>{candidate_name}</strong>,
      </p>
      <p style="color:rgba(255,255,255,0.75);margin:0 0 1.5rem;line-height:1.6">
        Nous avons le plaisir de vous informer que votre candidature pour le poste de
        <strong style="color:#fff">{job_title}</strong> a été <strong style="color:#63ffb4">acceptée</strong>.
      </p>
 
      <div style="background:rgba(99,255,180,0.08);border:1px solid rgba(99,255,180,0.25);
                  border-radius:10px;padding:1.25rem;margin-bottom:1.5rem">
        <p style="color:#63ffb4;font-weight:700;margin:0 0 0.5rem;font-size:0.9rem">
          Prochaines étapes
        </p>
        <p style="color:rgba(255,255,255,0.65);font-size:0.85rem;margin:0;line-height:1.6">
          Le recruteur va prendre contact avec vous prochainement pour convenir
          d'un entretien. Pensez à vérifier votre boîte mail régulièrement.
        </p>
      </div>
 
      <p style="color:rgba(255,255,255,0.35);font-size:0.8rem;margin:0">
        Cet email a été envoyé automatiquement par <strong style="color:rgba(255,255,255,0.5)">JobMatch AI</strong>.
      </p>
    </div>
    """
 
    msg.attach(MIMEText(html, "html"))
 
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SMTP_EMAIL, SMTP_PASSWORD)
            server.sendmail(SMTP_EMAIL, email, msg.as_string())
        print(f"   ✅ Email d'acceptation envoyé à {email}")
    except Exception as e:
        print(f"   ❌ Erreur email acceptation : {e}")

@router.get("/profile/cv/application/{app_id}")
def download_application_cv(
    app_id: int,
    token: str = Query(None),
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    raw_token = credentials.credentials if credentials else token
    if not raw_token:
        raise HTTPException(status_code=401, detail="Token manquant")

    payload = decode_token(raw_token)

    conn = get_conn()
    # Vérifier recruteur
    recruiter = conn.execute(
        "SELECT role FROM users WHERE id=?", (payload["user_id"],)
    ).fetchone()
    if not recruiter or recruiter[0] != "recruteur":
        conn.close()
        raise HTTPException(status_code=403, detail="Accès réservé aux recruteurs")

    # Lire CV depuis applications
    row = conn.execute(
        "SELECT cv_file, cv_filename FROM applications WHERE id=?", (app_id,)
    ).fetchone()
    conn.close()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Aucun CV pour cette candidature")

    from fastapi.responses import Response
    ext = Path(row[1] or "cv.pdf").suffix.lower()
    media_types = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc":  "application/msword",
        ".txt":  "text/plain",
    }
    return Response(
        content    = row[0],
        media_type = media_types.get(ext, "application/octet-stream"),
        headers    = {"Content-Disposition": f'attachment; filename="{row[1]}"'}
    )

@router.post("/profile/cv")
async def upload_cv(
    file: UploadFile = File(...),
    user = Depends(get_current_user)
):
    allowed = {".pdf", ".docx", ".doc", ".txt"}
    ext = Path(file.filename or "file.txt").suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Format non supporté : {ext}")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Fichier trop lourd (max 10 Mo)")

    conn = get_conn()
    conn.execute(
        "UPDATE users SET cv_filename=?, cv_file=? WHERE id=?",
        (file.filename, content, user["user_id"])
    )
    conn.commit()
    conn.close()

    return {"message": "CV enregistré ✅", "filename": file.filename}


from fastapi import Query
from fastapi.security import HTTPAuthorizationCredentials

@router.get("/profile/cv/{user_id}")
def download_cv(
    user_id: int,
    token: str = Query(None),                          # ← token depuis URL
    credentials: HTTPAuthorizationCredentials = Depends(security)  # ← token depuis header
):
    # Résoudre le token depuis l'une ou l'autre source
    raw_token = None
    if credentials:
        raw_token = credentials.credentials
    elif token:
        raw_token = token
    else:
        raise HTTPException(status_code=401, detail="Token manquant")

    # Décoder manuellement
    try:
        payload = decode_token(raw_token)
    except:
        raise HTTPException(status_code=401, detail="Token invalide")

    # Vérifier que c'est un recruteur
    conn = get_conn()
    recruiter = conn.execute(
        "SELECT role FROM users WHERE id=?", (payload["user_id"],)
    ).fetchone()
    if not recruiter or recruiter[0] != "recruteur":
        conn.close()
        raise HTTPException(status_code=403, detail="Accès réservé aux recruteurs")

    row = conn.execute(
        "SELECT cv_file, cv_filename FROM users WHERE id=?", (user_id,)
    ).fetchone()
    conn.close()

    if not row or not row[0]:
        raise HTTPException(status_code=404, detail="Aucun CV enregistré")

    from fastapi.responses import Response
    ext = Path(row[1] or "cv.pdf").suffix.lower()
    media_types = {
        ".pdf":  "application/pdf",
        ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ".doc":  "application/msword",
        ".txt":  "text/plain",
    }
    return Response(
        content    = row[0],
        media_type = media_types.get(ext, "application/octet-stream"),
        headers    = {"Content-Disposition": f'attachment; filename="{row[1]}"'}
    )
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
# ══════════════════════════════════════════════════════════════
# ESPACE ADMIN 
# ══════════════════════════════════════════════════════════════
admin_router = APIRouter(prefix="/admin", tags=["Admin"])
 
# ── Guard admin ───────────────────────────────────────────────
def get_admin_user(user=Depends(get_current_user)):
    conn = get_conn()
    row = conn.execute("SELECT role FROM users WHERE id=?", (user["user_id"],)).fetchone()
    conn.close()
    if not row or row[0] != "admin":
        raise HTTPException(status_code=403, detail="Accès réservé aux administrateurs")
    return user
 
# ── Dashboard stats ───────────────────────────────────────────
@admin_router.get("/dashboard")
def admin_dashboard(user=Depends(get_admin_user)):
    conn = get_conn()
 
    total_users      = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    total_candidates = conn.execute("SELECT COUNT(*) FROM users WHERE role='candidat'").fetchone()[0]
    total_recruiters = conn.execute("SELECT COUNT(*) FROM users WHERE role='recruteur'").fetchone()[0]
    total_jobs       = conn.execute("SELECT COUNT(*) FROM custom_jobs").fetchone()[0]
    total_apps       = conn.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
    pending_apps     = conn.execute("SELECT COUNT(*) FROM applications WHERE status='envoyée'").fetchone()[0]
    accepted_apps    = conn.execute("SELECT COUNT(*) FROM applications WHERE status='acceptée'").fetchone()[0]
    rejected_apps    = conn.execute("SELECT COUNT(*) FROM applications WHERE status='rejetée'").fetchone()[0]
 
    # Inscriptions par jour (7 derniers jours)
    registrations = conn.execute("""
        SELECT DATE(created_at) as day, COUNT(*) as count
        FROM users
        WHERE created_at >= DATE('now', '-7 days')
        GROUP BY DATE(created_at)
        ORDER BY day
    """).fetchall()
 
    # Candidatures par jour (7 derniers jours)
    applications_by_day = conn.execute("""
        SELECT DATE(applied_at) as day, COUNT(*) as count
        FROM applications
        WHERE applied_at >= DATE('now', '-7 days')
        GROUP BY DATE(applied_at)
        ORDER BY day
    """).fetchall()
 
    conn.close()
    return {
        "users": {
            "total":      total_users,
            "candidates": total_candidates,
            "recruiters": total_recruiters,
        },
        "jobs":         total_jobs,
        "applications": {
            "total":    total_apps,
            "pending":  pending_apps,
            "accepted": accepted_apps,
            "rejected": rejected_apps,
        },
        "registrations_7d":   [{"day": r[0], "count": r[1]} for r in registrations],
        "applications_7d":    [{"day": r[0], "count": r[1]} for r in applications_by_day],
    }
 
# ── Liste users ────────────────────────────────────────────────
@admin_router.get("/users")
def admin_get_users(
    search: str = Query(""),
    role:   str = Query(""),
    page:   int = Query(1, ge=1),
    limit:  int = Query(20, ge=1, le=100),
    user=Depends(get_admin_user)
):
    conn = get_conn()
    conditions = ["role != 'admin'"]
    params = []
 
    if search:
        conditions.append("(full_name LIKE ? OR email LIKE ?)")
        params += [f"%{search}%", f"%{search}%"]
    if role:
        conditions.append("role = ?")
        params.append(role)
 
    where = "WHERE " + " AND ".join(conditions)
    total = conn.execute(f"SELECT COUNT(*) FROM users {where}", params).fetchone()[0]
 
    offset = (page - 1) * limit
    rows = conn.execute(f"""
        SELECT id, email, full_name, role, experience_level, skills, created_at
        FROM users {where}
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()
    conn.close()
 
    return {
        "total": total,
        "pages": max(1, (total + limit - 1) // limit),
        "users": [
            {
                "id":               r[0],
                "email":            r[1],
                "full_name":        r[2],
                "role":             r[3],
                "experience_level": r[4],
                "skills":           r[5],
                "created_at":       r[6],
            }
            for r in rows
        ]
    }
 
# ── Modifier le rôle d'un user ─────────────────────────────────
@admin_router.put("/users/{user_id}/role")
def admin_update_role(user_id: int, body: dict, user=Depends(get_admin_user)):
    new_role = body.get("role", "")
    if new_role not in ("candidat", "recruteur"):
        raise HTTPException(status_code=400, detail="Rôle invalide")
    conn = get_conn()
    conn.execute("UPDATE users SET role=? WHERE id=?", (new_role, user_id))
    conn.commit()
    conn.close()
    return {"message": f"Rôle mis à jour : {new_role}"}
 
# ── Supprimer un user ──────────────────────────────────────────
@admin_router.delete("/users/{user_id}")
def admin_delete_user(user_id: int, user=Depends(get_admin_user)):
    conn = get_conn()
    # Supprimer les données liées
    conn.execute("DELETE FROM applications  WHERE user_id=?",  (user_id,))
    conn.execute("DELETE FROM match_history WHERE user_id=?",  (user_id,))
    conn.execute("DELETE FROM users         WHERE id=?",       (user_id,))
    conn.commit()
    conn.close()
    return {"message": "Utilisateur supprimé"}
 
# ── Liste toutes les offres ────────────────────────────────────
@admin_router.get("/jobs")
def admin_get_jobs(
    search: str = Query(""),
    page:   int = Query(1, ge=1),
    limit:  int = Query(20, ge=1, le=100),
    user=Depends(get_admin_user)
):
    conn = get_conn()
    conditions, params = [], []
    if search:
        conditions.append("(cj.job_title LIKE ? OR cj.location LIKE ?)")
        params += [f"%{search}%", f"%{search}%"]
 
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    total = conn.execute(f"SELECT COUNT(*) FROM custom_jobs cj {where}", params).fetchone()[0]
 
    offset = (page - 1) * limit
    rows = conn.execute(f"""
        SELECT cj.id, cj.job_title, cj.location, cj.experience_level,
               cj.salary_usd, cj.remote_ratio, cj.created_at,
               u.full_name, u.email
        FROM custom_jobs cj
        LEFT JOIN users u ON u.id = cj.user_id
        {where}
        ORDER BY cj.created_at DESC
        LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()
    conn.close()
 
    return {
        "total": total,
        "pages": max(1, (total + limit - 1) // limit),
        "jobs": [
            {
                "id":               r[0],
                "job_title":        r[1],
                "location":         r[2],
                "experience_level": r[3],
                "salary_usd":       r[4],
                "remote_ratio":     r[5],
                "created_at":       r[6],
                "recruiter_name":   r[7] or "—",
                "recruiter_email":  r[8] or "—",
            }
            for r in rows
        ]
    }
 
# ── Supprimer une offre ────────────────────────────────────────
@admin_router.delete("/jobs/{job_id}")
def admin_delete_job(job_id: int, user=Depends(get_admin_user)):
    conn = get_conn()
    conn.execute("DELETE FROM custom_jobs WHERE id=?", (job_id,))
    conn.commit()
    conn.close()
    return {"message": "Offre supprimée"}
 
# ── Liste toutes les candidatures ─────────────────────────────
@admin_router.get("/applications")
def admin_get_applications(
    status: str = Query(""),
    page:   int = Query(1, ge=1),
    limit:  int = Query(20, ge=1, le=100),
    user=Depends(get_admin_user)
):
    conn = get_conn()
    conditions, params = [], []
    if status:
        conditions.append("a.status = ?")
        params.append(status)
 
    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""
    total = conn.execute(f"SELECT COUNT(*) FROM applications a {where}", params).fetchone()[0]
 
    offset = (page - 1) * limit
    rows = conn.execute(f"""
        SELECT a.id, a.job_title, a.status, a.applied_at,
               u.full_name, u.email
        FROM applications a
        JOIN users u ON a.user_id = u.id
        {where}
        ORDER BY a.applied_at DESC
        LIMIT ? OFFSET ?
    """, params + [limit, offset]).fetchall()
    conn.close()
 
    return {
        "total": total,
        "pages": max(1, (total + limit - 1) // limit),
        "applications": [
            {
                "id":             r[0],
                "job_title":      r[1],
                "status":         r[2],
                "applied_at":     r[3],
                "candidate_name": r[4],
                "candidate_email":r[5],
            }
            for r in rows
        ]
    }
 