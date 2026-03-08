"""
Matching Engine — Version Universelle v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Améliorations v3 :
  ✅ Aucune dépendance à candidate_text_builder externe
  ✅ to_text() intégré directement dans CandidateProfile
  ✅ Enrichissement sémantique multilingue (FR + EN)
  ✅ Normalisation du texte avant encoding SBERT
  ✅ Domaine du candidat intégré dans le texte enrichi
  ✅ Pénalité renforcée pour niveaux incompatibles
  ✅ Bonus compétences : score skills overlap candidat ↔ offre
  ✅ Support universel : tous domaines (tech, finance, RH, marketing...)
  ✅ parse_cv_to_profile() robuste via CVParser v3
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import numpy as np
import json
import re
import time
import warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional

warnings.filterwarnings("ignore")

DATA_PROCESSED = Path("data/processed")
VECTORS_DIR    = DATA_PROCESSED / "vectors"
RESULTS_DIR    = DATA_PROCESSED / "matching_results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# TRADUCTIONS FR → EN (termes métier courants dans les CV FR)
# ══════════════════════════════════════════════════════════════

FR_TO_EN_TERMS = {
    # Niveaux / titres
    "ingénieur":         "engineer",
    "ingenieur":         "engineer",
    "développeur":       "developer",
    "developpeur":       "developer",
    "analyste":          "analyst",
    "chef de projet":    "project manager",
    "responsable":       "manager",
    "directeur":         "director",
    "stagiaire":         "intern",
    "stage":             "internship",
    "alternant":         "apprentice",
    # Domaines tech
    "données":           "data",
    "donnees":           "data",
    "apprentissage automatique": "machine learning",
    "intelligence artificielle": "artificial intelligence",
    "traitement du langage naturel": "natural language processing",
    "vision par ordinateur": "computer vision",
    "entrepôt de données": "data warehouse",
    "entrepot de donnees": "data warehouse",
    "base de données":   "database",
    "base de donnees":   "database",
    "architecture logicielle": "software architecture",
    "développement web": "web development",
    "developpement web": "web development",
    # Finance
    "comptabilité":      "accounting",
    "comptabilite":      "accounting",
    "trésorerie":        "treasury",
    "tresorerie":        "treasury",
    "gestion budgétaire": "budget management",
    "contrôle de gestion": "controlling",
    "controle de gestion": "controlling",
    "analyse financière": "financial analysis",
    # RH
    "ressources humaines": "human resources",
    "recrutement":       "recruitment",
    "formation":         "training",
    "paie":              "payroll",
    # Marketing
    "marketing numérique": "digital marketing",
    "marketing numerique": "digital marketing",
    "réseaux sociaux":   "social media",
    "reseaux sociaux":   "social media",
    # Général
    "expérience":        "experience",
    "experience":        "experience",
    "compétences":       "skills",
    "competences":       "skills",
    "projets":           "projects",
    "formation":         "education",
    "entreprise":        "company",
    "poste":             "position",
    "emploi":            "job",
    "mission":           "mission",
    "tableau de bord":   "dashboard",
    "visualisation":     "visualization",
    "rapport":           "report",
    "optimisation":      "optimization",
    "gestion":           "management",
    "système":           "system",
    "systeme":           "system",
    "logiciel":          "software",
    "application":       "application",
    "réseau":            "network",
    "reseau":            "network",
    "sécurité":          "security",
    "securite":          "security",
    "communication":     "communication",
    "travail en équipe": "teamwork",
    "travail en equipe": "teamwork",
    "méthodes agiles":   "agile methods",
    "methodes agiles":   "agile methods",
}

# Expansion sémantique : skills → termes anglais associés
# Permet au SBERT de trouver des offres utilisant des termes proches
SKILL_SEMANTIC_EXPANSION = {
    # Tech / Data
    "Machine Learning":   "machine learning ML data modeling predictive analytics",
    "Deep Learning":      "deep learning neural networks AI artificial intelligence",
    "NLP":                "natural language processing text mining language model",
    "Computer Vision":    "computer vision image recognition object detection",
    "Power BI":           "power bi business intelligence data visualization reporting dashboard BI analyst",
    "Power Bi":           "power bi business intelligence data visualization reporting dashboard BI analyst",
    "IBM Cognos":         "cognos business intelligence reporting analytics",
    "IBM cognos analytics": "ibm cognos business intelligence reporting analytics dashboard",
    "Tableau":            "tableau data visualization analytics dashboard",
    "Talend":             "talend etl data integration pipeline data engineer",
    "SQL":                "sql database query data extraction data analyst",
    "PL-SQL":             "pl-sql oracle database stored procedures",
    "MDX":                "mdx olap cube business intelligence multidimensional",
    "Oracle":             "oracle database sql data management",
    "Python":             "python data science programming scripting analytics",
    "Django":             "django python web development backend",
    "Apache Spark":       "spark big data distributed computing etl",
    "Airflow":            "airflow orchestration data pipeline workflow",
    "AWS":                "aws amazon cloud computing infrastructure",
    "Docker":             "docker containerization devops deployment",
    "Odoo":               "odoo erp enterprise resource planning business management",
    "Scrum":              "scrum agile project management sprint",
    "Salesforce":         "salesforce crm customer relationship sales",
    "SAP":                "sap erp enterprise resource planning",
    "Excel":              "excel spreadsheet data analysis microsoft office",
    "Figma":              "figma ui ux design prototyping",
    # Groupes de skills retournés par Groq (phrases longues → expansion)
    "Analyses de données et visualisation": "data analysis visualization business intelligence power bi tableau dashboard reporting",
    "Gestion des bases de données":         "database management sql oracle mysql postgresql data",
    "Développement web":                    "web development html css javascript php frontend backend",
    "Data mining":                          "data mining machine learning pattern recognition analytics",
    "Machine learning":                     "machine learning ML predictive modeling data science",
    "ERP":                                  "erp enterprise resource planning odoo sap business",
    "Conception":                           "uml design modeling software architecture",
    # Finance
    "Comptabilité":       "accounting finance bookkeeping financial statements",
    "Audit":              "audit financial review compliance risk",
    "Trésorerie":         "treasury cash management finance",
    # RH
    "Recrutement":        "recruitment talent acquisition hiring staffing",
    "Paie":               "payroll compensation hr administration",
    # Marketing
    "SEO":                "seo search engine optimization digital marketing",
    "Photoshop":          "photoshop graphic design adobe creative",
    # Droit
    "Droit":              "law legal compliance regulation contracts",
}

# Normalisation : certains skills retournés par Groq sont des phrases longues
# On les mappe vers les termes courts que SBERT reconnaît mieux
SKILL_NORMALIZE = {
    "analyses de données et visualisation": "data analysis visualization power bi tableau",
    "gestion des bases de données":         "SQL database Oracle PostgreSQL",
    "développement web":                    "HTML CSS JavaScript PHP web development",
    "machine learning":                     "Machine Learning",
    "data mining":                          "Data Mining",
    "erp":                                  "ERP Odoo SAP",
    "conception":                           "UML Star UML design",
    "programmation":                        "programming Python Java C",
    "ibm cognos analytics":                 "IBM Cognos business intelligence",
    "java script":                          "JavaScript",
    "sql-server":                           "SQL Server",
    "pl-sql":                               "PL-SQL Oracle",
}

# Mapping niveau d'expérience FR → EN pour le texte enrichi
EXP_LEVEL_CONTEXT = {
    "entry":     "entry level junior fresh graduate intern internship 0-2 years experience",
    "mid":       "mid level experienced professional 3-6 years experience",
    "senior":    "senior expert lead specialist 7+ years experience",
    "executive": "executive director vice president C-level leadership strategic",
    "unknown":   "",
}


def _translate_fr_to_en(text: str) -> str:
    """Traduit les termes FR métier courants vers l'anglais"""
    t = text.lower()
    for fr, en in sorted(FR_TO_EN_TERMS.items(), key=lambda x: len(x[0]), reverse=True):
        t = t.replace(fr, en)
    return t


def _normalize_skills(skills: List[str]) -> List[str]:
    """
    Normalise les skills retournés par Groq.
    Groq retourne parfois des phrases longues comme "Analyses de données et visualisation"
    au lieu de "Power BI", "Tableau", etc.
    On les remplace par les termes courts que SBERT reconnaît mieux.
    """
    normalized = []
    for skill in skills:
        key = skill.lower().strip()
        if key in SKILL_NORMALIZE:
            # Remplacer la phrase longue par les termes courts
            normalized.extend(SKILL_NORMALIZE[key].split())
        else:
            normalized.append(skill)
    return normalized


def build_candidate_text(
    summary: str,
    skills: List[str],
    experience_level: str,
    desired_location: str = "",
    domain: str = "",
) -> str:
    """
    Construit le texte enrichi du candidat à encoder par SBERT.

    Pipeline :
      1. Traduction FR→EN du résumé
      2. Normalisation des skills (phrases longues → termes courts)
      3. Skills répétés 3× + expansion sémantique complète
      4. Contexte de niveau d'expérience
      5. Résumé brut original (ancrage sémantique)
    """
    parts = []

    # 1. Résumé traduit FR→EN
    translated_summary = _translate_fr_to_en(summary)
    parts.append(translated_summary)

    # 2. Skills normalisés et enrichis
    if skills:
        # Normaliser d'abord les phrases longues de Groq
        skills_normalized = _normalize_skills(skills)
        skills_raw = " ".join(skills_normalized)

        # Répéter 3× pour donner plus de poids aux skills
        parts.append(skills_raw)
        parts.append(skills_raw)
        parts.append(skills_raw)

        # Expansion sémantique sur skills originaux ET normalisés
        for skill in skills + skills_normalized:
            expansion = SKILL_SEMANTIC_EXPANSION.get(skill, "")
            if not expansion:
                expansion = SKILL_SEMANTIC_EXPANSION.get(skill.lower(), "")
            if expansion:
                parts.append(expansion)

    # 3. Contexte niveau
    level_ctx = EXP_LEVEL_CONTEXT.get(experience_level, "")
    if level_ctx:
        parts.append(level_ctx)

    # 4. Domaine
    if domain and domain != "unknown":
        domain_context = {
            "tech":       "software engineer developer data science technology IT",
            "finance":    "finance accounting audit controlling financial analyst",
            "marketing":  "marketing digital communication brand content",
            "rh":         "human resources hr recruitment talent people",
            "sante":      "health medical clinical hospital nursing",
            "droit":      "law legal compliance juridical attorney",
            "management": "project management leadership strategy operations",
            "vente":      "sales commercial business development account",
        }
        parts.append(domain_context.get(domain, f"{domain} professional"))

    # 5. Localisation
    if desired_location:
        parts.append(f"location {desired_location}")

    # 6. Résumé brut original (ancrage sémantique)
    parts.append(summary)

    return " ".join(filter(None, parts))


# ══════════════════════════════════════════════════════════════
# STRUCTURES DE DONNÉES
# ══════════════════════════════════════════════════════════════

@dataclass
class CandidateProfile:
    """
    Profil candidat universel — fonctionne pour tous domaines.
    Construit manuellement ou via parse_cv_to_profile().
    """
    name:               str
    summary:            str               # texte libre : résumé ou CV complet
    skills:             List[str] = field(default_factory=list)
    experience_level:   str = "unknown"   # entry / mid / senior / executive
    desired_location:   str = ""
    min_salary:         float = 0.0
    max_salary:         float = 0.0
    employment_type:    str = ""          # FT / PT / CT / "" = tous
    remote_only:        bool = False
    domain:             str = "unknown"   # tech / finance / rh / marketing / ...

    def to_text(self) -> str:
        """Construit le texte enrichi pour SBERT (aucune dépendance externe)"""
        return build_candidate_text(
            summary          = self.summary,
            skills           = self.skills,
            experience_level = self.experience_level,
            desired_location = self.desired_location,
            domain           = self.domain,
        )


@dataclass
class MatchResult:
    """Un résultat de matching — une offre avec son score"""
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


# ══════════════════════════════════════════════════════════════
# PARSING CV → PROFIL CANDIDAT
# ══════════════════════════════════════════════════════════════

def parse_cv_to_profile(
    name: str,
    cv_text: str,
    desired_location: str = "",
    min_salary: float = 0.0,
    max_salary: float = 0.0,
    employment_type: str = "",
    remote_only: bool = False,
) -> CandidateProfile:
    """
    Parse automatiquement un texte de CV brut et construit un CandidateProfile.
    Utilise le CVParser v3 (universel, tous domaines, multilingue).

    Args:
        name            : Nom du candidat
        cv_text         : Texte brut extrait du CV
        desired_location: Localisation souhaitée (optionnel, détectée auto sinon)
        min_salary      : Salaire minimum USD (0 = pas de filtre)
        max_salary      : Salaire maximum USD (0 = pas de filtre)
        employment_type : "FT" / "PT" / "CT" / "" = tous
        remote_only     : True = remote uniquement

    Returns:
        CandidateProfile prêt pour le matching
    """
    # Import local pour éviter les imports circulaires
    try:
        from cv_parser import CVParser
    except ImportError:
        # Fallback : extraction minimale si cv_parser non disponible
        from matching_engine import _minimal_parse
        return _minimal_parse(name, cv_text, desired_location,
                              min_salary, max_salary, employment_type, remote_only)

    parser = CVParser()
    cv = parser.parse_text(cv_text)

    return CandidateProfile(
        name             = name or cv.name,
        summary          = cv_text,   # texte brut complet pour SBERT
        skills           = cv.skills,
        experience_level = cv.experience_level,
        desired_location = desired_location or cv.location,
        min_salary       = min_salary,
        max_salary       = max_salary,
        employment_type  = employment_type,
        remote_only      = remote_only,
        domain           = cv.domain,
    )


def _minimal_parse(name, cv_text, desired_location="",
                   min_salary=0, max_salary=0, employment_type="", remote_only=False):
    """Fallback si cv_parser.py n'est pas disponible"""
    from matching_engine import _detect_level_minimal, _extract_skills_minimal
    return CandidateProfile(
        name             = name,
        summary          = cv_text,
        skills           = _extract_skills_minimal(cv_text),
        experience_level = _detect_level_minimal(cv_text),
        desired_location = desired_location,
        min_salary       = min_salary,
        max_salary       = max_salary,
        employment_type  = employment_type,
        remote_only      = remote_only,
    )


def _detect_level_minimal(text: str) -> str:
    """Détection minimale de niveau si cv_parser non dispo"""
    t = text.lower()
    student_kw = ["student", "étudiant", "stage", "pfe", "intern", "bachelor",
                  "licence", "master", "university", "school"]
    senior_kw  = ["senior", "lead", "architect", "director", "10 years", "8 years"]
    if any(k in t for k in student_kw):
        return "entry"
    if any(k in t for k in senior_kw):
        return "senior"
    m = re.search(r"(\d+)\s*(?:years?|ans?)\s*(?:of\s*)?(?:experience|expérience)?", t)
    if m:
        y = int(m.group(1))
        return "senior" if y >= 8 else "mid" if y >= 4 else "entry"
    return "unknown"


def _extract_skills_minimal(text: str) -> List[str]:
    """Extraction minimale de skills si cv_parser non dispo"""
    common = ["Python", "Java", "SQL", "AWS", "React", "Docker",
              "TensorFlow", "Spark", "Excel", "Power BI", "Scrum"]
    t = text.lower()
    return [s for s in common if s.lower() in t]


# ══════════════════════════════════════════════════════════════
# MATCHING ENGINE v3
# ══════════════════════════════════════════════════════════════

class MatchingEngine:
    """
    Moteur de matching CV ↔ offres d'emploi — v3 Universel
    Fonctionne pour tous types de profils (tech, finance, RH, marketing...)
    """

    EXP_HIERARCHY = {
        "entry":     1,
        "mid":       2,
        "senior":    3,
        "executive": 4,
        "unknown":   0,
    }

    def __init__(self):
        self.df           = None
        self.sbert_matrix = None
        self.sbert_model  = None
        self.sbert_model_name = "unknown"
        self._loaded      = False

    # ── Chargement ────────────────────────────────────────────

    def load(self, hf_token: str = None):
        print("📂 Chargement du Matching Engine v3...")
        t0 = time.time()

        # Chercher le fichier de données
        pkl_path = DATA_PROCESSED / "jobs_vectorized.pkl"
        csv_path = DATA_PROCESSED / "jobs_cleaned.csv"

        if pkl_path.exists():
            self.df = pd.read_pickle(str(pkl_path))
        elif csv_path.exists():
            self.df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(
                "❌ jobs_vectorized.pkl introuvable.\n"
                "   Lance d'abord : python code/04_nlp_vectorization.py"
            )

        # S'assurer que les colonnes nécessaires existent
        required_cols = ["job_title", "location", "salary_usd",
                         "experience_level", "employment_type", "remote_ratio"]
        for col in required_cols:
            if col not in self.df.columns:
                if col == "salary_usd":
                    self.df[col] = 0.0
                elif col == "remote_ratio":
                    self.df[col] = 0
                else:
                    self.df[col] = "unknown"

        # Remplir les NaN
        self.df["salary_usd"]       = pd.to_numeric(self.df["salary_usd"], errors="coerce").fillna(0)
        self.df["remote_ratio"]     = pd.to_numeric(self.df["remote_ratio"], errors="coerce").fillna(0).astype(int)
        self.df["experience_level"] = self.df["experience_level"].fillna("unknown")
        self.df["location"]         = self.df["location"].fillna("")
        self.df["job_title"]        = self.df["job_title"].fillna("Unknown")
        self.df["employment_type"]  = self.df["employment_type"].fillna("FT")
        self.df["company_size"]     = self.df.get("company_size", pd.Series(["M"]*len(self.df))).fillna("M")
        self.df["source"]           = self.df.get("source", pd.Series([""]*len(self.df))).fillna("")
        self.df["has_salary"]       = (self.df["salary_usd"] > 1000).astype(int)

        print(f"   ✅ Offres chargées        : {len(self.df):,}")

        # Charger la matrice SBERT
        sbert_path = VECTORS_DIR / "sbert_matrix.npy"
        if not sbert_path.exists():
            raise FileNotFoundError(
                f"❌ sbert_matrix.npy introuvable dans {VECTORS_DIR}\n"
                "   Lance d'abord : python code/04_nlp_vectorization.py"
            )
        self.sbert_matrix = np.load(str(sbert_path))
        print(f"   ✅ Vecteurs SBERT chargés  : {self.sbert_matrix.shape}")

        self._load_sbert_model(hf_token)

        elapsed = time.time() - t0
        print(f"   ✅ Engine prêt en {elapsed:.1f}s\n")
        self._loaded = True
        return self

    def _load_sbert_model(self, hf_token=None):
        from sentence_transformers import SentenceTransformer
        import os
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token

        # Ordre de préférence des modèles (du meilleur au plus léger)
        models = [
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L3-v2",
            "all-MiniLM-L12-v2",
        ]
        for model_name in models:
            try:
                kwargs = {"token": hf_token} if hf_token else {}
                self.sbert_model      = SentenceTransformer(model_name, **kwargs)
                self.sbert_model_name = model_name
                print(f"   ✅ Modèle SBERT chargé    : {model_name}")
                return
            except Exception as e:
                print(f"   ⚠️  {model_name} : {str(e)[:60]}")
        raise RuntimeError("❌ Impossible de charger un modèle SBERT.")

    # ── Encoding ──────────────────────────────────────────────

    def _encode_candidate(self, text: str) -> np.ndarray:
        vec = self.sbert_model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vec[0]

    # ── Filtres ───────────────────────────────────────────────

    def _apply_filters(self, candidate: CandidateProfile) -> np.ndarray:
        mask = np.ones(len(self.df), dtype=bool)

        if candidate.min_salary > 0:
            mask &= (
                (self.df["salary_usd"] >= candidate.min_salary) |
                (self.df["salary_usd"] == 0)
            ).values

        if candidate.max_salary > 0:
            mask &= (
                (self.df["salary_usd"] <= candidate.max_salary) |
                (self.df["salary_usd"] == 0)
            ).values

        if candidate.remote_only:
            mask &= (self.df["remote_ratio"] == 100).values

        if candidate.employment_type:
            mask &= (
                self.df["employment_type"].str.upper() ==
                candidate.employment_type.upper()
            ).values

        if candidate.desired_location and not candidate.remote_only:
            loc_lower = candidate.desired_location.lower()
            loc_mask  = self.df["location"].str.lower().str.contains(
                loc_lower, na=False, regex=False
            )
            # Appliquer filtre localisation seulement si assez de résultats
            if loc_mask.sum() >= 10:
                mask &= loc_mask.values

        return mask

    def _compute_filter_bonus(
        self,
        candidate: CandidateProfile,
        row: pd.Series,
    ) -> tuple:
        """
        v3 : bonus/pénalité enrichis.

        Règles :
          - Niveau exact         → +0.12
          - Niveau adjacent ±1   → +0.05
          - Incompatible > ±1    → −0.07 (pénalité renforcée v3)
          - Niveau inconnu       → +0.02
          - Salaire renseigné    → +0.02
          - Remote match         → +0.05
          - Localisation match   → +0.03
        """
        bonus   = 0.0
        reasons = []

        # ── Niveau d'expérience ───────────────────────────────
        if candidate.experience_level not in ("unknown", ""):
            c_lvl = self.EXP_HIERARCHY.get(candidate.experience_level, 0)
            j_lvl = self.EXP_HIERARCHY.get(
                str(row.get("experience_level", "unknown")).lower(), 0
            )
            if j_lvl == 0:
                bonus   += 0.02
                reasons.append("niveau non spécifié (ouvert)")
            elif c_lvl == j_lvl:
                bonus   += 0.12
                reasons.append(f"✅ niveau exact ({row.get('experience_level')})")
            elif abs(c_lvl - j_lvl) == 1:
                bonus   += 0.05
                reasons.append(f"niveau proche ({row.get('experience_level')})")
            else:
                bonus   -= 0.07   # pénalité renforcée v3
                reasons.append(f"⚠️ niveau incompatible ({row.get('experience_level')})")

        # ── Salaire ───────────────────────────────────────────
        if row.get("has_salary", 0) == 1 and row.get("salary_usd", 0) > 1000:
            bonus   += 0.02
            reasons.append(f"salaire renseigné (${row['salary_usd']:,.0f})")

        # ── Remote ────────────────────────────────────────────
        if candidate.remote_only and row.get("remote_ratio", 0) == 100:
            bonus   += 0.05
            reasons.append("100% remote ✓")
        elif row.get("remote_ratio", 0) == 100:
            bonus   += 0.01
            reasons.append("remote disponible")

        # ── Localisation ──────────────────────────────────────
        if candidate.desired_location:
            loc_l = candidate.desired_location.lower()
            if loc_l in str(row.get("location", "")).lower():
                bonus   += 0.03
                reasons.append(f"📍 {row.get('location')}")

        # Clamp final : [-0.10 ; +0.20]
        return min(max(bonus, -0.10), 0.20), reasons

    # ── Matching principal ────────────────────────────────────

    def match(
        self,
        candidate: CandidateProfile,
        top_k: int = 10,
        min_score: float = 0.25,
    ) -> List[MatchResult]:
        """
        Lance le matching du candidat contre toutes les offres.

        Args:
            candidate  : Profil du candidat
            top_k      : Nombre de résultats à retourner
            min_score  : Score SBERT minimum (0.25 recommandé)

        Returns:
            Liste de MatchResult triée par score décroissant
        """
        assert self._loaded, "Appelle d'abord engine.load()"

        t0 = time.time()

        # Encoder le candidat
        candidate_text = candidate.to_text()
        candidate_vec  = self._encode_candidate(candidate_text)

        # Scores cosinus avec toutes les offres
        sbert_scores  = self.sbert_matrix @ candidate_vec
        eligible_mask = self._apply_filters(candidate)
        nb_eligible   = eligible_mask.sum()

        # Masquer les offres non éligibles
        masked_scores = np.where(eligible_mask, sbert_scores, -1.0)

        # Adapter min_score si pas assez de résultats
        if nb_eligible > 0:
            above_min = masked_scores >= min_score
            if above_min.sum() < top_k:
                eligible_sorted = np.sort(masked_scores[eligible_mask])[::-1]
                idx_cutoff = min(top_k - 1, nb_eligible - 1)
                min_score = float(eligible_sorted[idx_cutoff])

        # Sélectionner les candidats (3× top_k pour avoir de la marge après bonus)
        top_indices = np.argsort(masked_scores)[::-1][:top_k * 3]
        top_indices = [
            i for i in top_indices
            if masked_scores[i] >= min_score
        ][:top_k * 2]

        # Construire les résultats
        results = []
        for idx in top_indices:
            row         = self.df.iloc[idx]
            sbert_score = float(sbert_scores[idx])
            bonus, reasons = self._compute_filter_bonus(candidate, row)
            final_score = sbert_score + bonus

            results.append(MatchResult(
                rank             = 0,
                job_title        = str(row.get("job_title",        "Unknown")),
                location         = str(row.get("location",         "Unknown")),
                salary_usd       = float(row.get("salary_usd",     0)),
                has_salary       = bool(row.get("has_salary",       0)),
                experience_level = str(row.get("experience_level", "unknown")),
                employment_type  = str(row.get("employment_type",  "FT")),
                remote_ratio     = int(row.get("remote_ratio",     0)),
                company_size     = str(row.get("company_size",     "M")),
                source           = str(row.get("source",           "")),
                sbert_score      = round(sbert_score, 4),
                filter_bonus     = round(bonus, 4),
                final_score      = round(final_score, 4),
                match_reasons    = reasons,
            ))

        # Trier et numéroter
        results.sort(key=lambda x: x.final_score, reverse=True)
        results = results[:top_k]
        for i, r in enumerate(results):
            r.rank = i + 1

        elapsed = time.time() - t0
        print(f"   ⏱️  Matching terminé en {elapsed:.3f}s "
              f"({nb_eligible:,} offres éligibles / {len(self.df):,})")

        return results

    # ── Affichage ─────────────────────────────────────────────

    def display_results(self, candidate: CandidateProfile, results: List[MatchResult]):
        print(f"\n{'═'*65}")
        print(f"🎯 MATCHING RESULTS v3 — {candidate.name}")
        print(f"{'═'*65}")
        print(f"📝 Profil  : {candidate.summary[:80]}...")
        if candidate.skills:
            print(f"🔧 Skills  : {', '.join(candidate.skills[:10])}")
        print(f"🎯 Domaine : {candidate.domain}")
        print(f"👤 Niveau  : {candidate.experience_level}")
        if candidate.desired_location:
            print(f"📍 Lieu    : {candidate.desired_location}")
        if candidate.min_salary > 0:
            print(f"💰 Salaire : ≥ ${candidate.min_salary:,.0f}")
        if candidate.remote_only:
            print(f"🏠 Remote  : uniquement")
        print(f"{'─'*65}")

        if not results:
            print("   ⚠️  Aucun résultat — essaie de réduire les filtres")
            return

        for r in results:
            sal_str = f"${r.salary_usd:>10,.0f}" if r.has_salary else "       N/A  "
            rem_str = f"🏠{r.remote_ratio}%" if r.remote_ratio > 0 else "🏢 on-site"
            reasons = " | ".join(r.match_reasons) if r.match_reasons else "similarité sémantique"

            print(f"\n  #{r.rank:>2}  [{r.final_score:.3f}]  {r.job_title[:50]}")
            print(f"       📍 {r.location:<28} {rem_str:<12} {sal_str}")
            print(f"       👤 {r.experience_level:<12} 🏢 {r.employment_type}  "
                  f"📊 SBERT={r.sbert_score:.3f} +bonus={r.filter_bonus:.3f}")
            print(f"       ✅ {reasons}")

        print(f"\n{'─'*65}")
        avg_score = np.mean([r.final_score for r in results])
        print(f"   📊 Score moyen Top-{len(results)} : {avg_score:.3f}")

    # ── Export JSON ───────────────────────────────────────────

    def save_results(self, candidate: CandidateProfile, results: List[MatchResult]) -> Path:
        safe_name = re.sub(r"[^\w]", "_", candidate.name.lower()).strip("_")
        output = {
            "candidate_name":     candidate.name,
            "experience_level":   candidate.experience_level,
            "domain":             candidate.domain,
            "skills_detected":    candidate.skills,
            "total_jobs_scanned": len(self.df),
            "results":            [asdict(r) for r in results],
            "top_score":          round(results[0].final_score, 4) if results else 0,
            "avg_score":          round(float(np.mean([r.final_score for r in results])), 4)
                                  if results else 0,
        }
        path = RESULTS_DIR / f"{safe_name}_v3.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"   💾 Résultats sauvegardés : {path}")
        return path


# ══════════════════════════════════════════════════════════════
# MAIN — Tests multi-domaines
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("MATCHING ENGINE v3 — UNIVERSEL")
    print("="*65)

    HF_TOKEN = None  # "hf_XXXXXXXXXXXXXXXX"

    engine = MatchingEngine()
    engine.load(hf_token=HF_TOKEN)

    # Candidats de test — domaines variés
    TEST_CVS = [
        # ── Tech : Étudiante BI ────────────────────────────
        (
            "Lajnef Ghada",
            """
            3ème année licence Business Intelligence à l'ISG-Tunis.
            Cherche stage PFE. Power BI, Talend, IBM Cognos, SQL, PL-SQL,
            Python, Machine Learning, Data Mining, MDX, Odoo.
            Stage SOFRECOM : tableaux de bord Power BI, gestion budgétaire.
            """,
        ),
        # ── Tech : Senior Data Scientist ──────────────────
        (
            "Alice — Senior Data Scientist",
            """
            Senior Data Scientist — 8 years of experience.
            Python, TensorFlow, PyTorch, Scikit-learn, SQL, Apache Spark,
            Deep Learning, NLP, MLOps, AWS, Docker.
            Led ML teams, deployed models to 5M users.
            """,
        ),
        # ── Finance : Comptable confirmé ──────────────────
        (
            "Pierre — Comptable Confirmé",
            """
            Comptable confirmé, 5 ans d'expérience.
            Paris, France. Comptabilité générale, Audit, IFRS, Sage,
            Excel, VBA, Contrôle de gestion, Reporting financier, Budget.
            """,
        ),
        # ── RH : Responsable recrutement ──────────────────
        (
            "Amina — Responsable RH",
            """
            Responsable Ressources Humaines — 6 ans d'expérience.
            Casablanca, Maroc.
            Recrutement, Talent Acquisition, Paie, Formation, GPEC,
            Droit du travail, SIRH Workday, Onboarding, RGPD.
            """,
        ),
    ]

    for name, cv_text in TEST_CVS:
        print(f"\n{'─'*65}")
        print(f"🔍 Matching : {name}")
        candidate = parse_cv_to_profile(name=name, cv_text=cv_text)
        results   = engine.match(candidate, top_k=10, min_score=0.25)
        engine.display_results(candidate, results)
        engine.save_results(candidate, results)

    print(f"\n✅ Résultats sauvegardés dans : {RESULTS_DIR}")


if __name__ == "__main__":
    main()