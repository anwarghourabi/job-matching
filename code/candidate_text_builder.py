"""
Patch — CandidateProfile.to_text() enrichi FR→EN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Stratégie :
  1. Dictionnaire FR→EN des termes métier les plus fréquents dans les CVs
  2. Expansion sémantique : chaque skill → synonymes anglais courants dans les offres
  3. Phrases de contexte naturelles en anglais pour mieux matcher les job titles
  4. Aucune dépendance externe (pas de googletrans, pas d'API)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ══════════════════════════════════════════════════════════════
# DICTIONNAIRE DE TRADUCTION FR → EN (termes CV / métier)
# ══════════════════════════════════════════════════════════════

FR_TO_EN = {
    # Statuts / niveaux
    "étudiant":               "student",
    "étudiante":              "student",
    "licence":                "bachelor degree",
    "master":                 "master degree",
    "stage":                  "position analyst",
    "stage pfe":              "final year project business intelligence analyst position",
    "pfe":                    "final year project business intelligence position",
    "alternance":             "work-study position",
    "en quête d'un stage":   "looking for analyst position business intelligence",
    "baccalauréat":           "high school diploma",
    "mention bien":           "with honors",

    # Domaines techniques — BI fortement boosté
    "analyse de données":     "data analysis data analytics business intelligence reporting",
    "visualisation":          "data visualization dashboards reporting business intelligence",
    "tableaux de bord":       "dashboards reports business intelligence Power BI data visualization",
    "gestion budgétaire":     "budget management financial reporting financial analysis BI",
    "suivi des missions":     "project tracking reporting monitoring analytics",
    "base de données":        "database SQL Oracle data management",
    "développement web":      "web development frontend backend PHP JavaScript",
    "site web":               "website web application development",
    "gestion":                "management analysis reporting",
    "conception":             "design architecture modeling UML",
    "système de recommandation": "recommendation system machine learning Python analytics",
    "filtrage collaboratif":  "collaborative filtering recommendation system machine learning",
    "agence de location":     "management system Java application development",
    "boite de messagerie":    "messaging application C programming development",
    "e-commerce":             "e-commerce web development online store",
    "business intelligence":  "business intelligence BI analytics dashboards reporting data warehouse",

    # Titres de sections
    "expérience professionnelle": "professional experience work experience",
    "compétences":            "skills technical skills",
    "parcours académique":    "education academic background",
    "projets":                "projects",
    "langues":                "languages",
    "profil personnel":       "professional profile summary",

    # Soft skills
    "esprit d'équipe":        "team player teamwork",
    "autonomie":              "autonomous self-motivated",
    "capacité d'adaptation":  "adaptability flexible",
    "gestion du stress":      "stress management",
    "communication":          "communication skills",
    "négociation":            "negotiation",
    "dynamique":              "dynamic motivated",
    "rigoureuse":             "rigorous detail-oriented",

    # Lieux
    "tunis":                  "Tunis Tunisia",
    "tunisie":                "Tunisia",

    # Mots courants
    "et":                     "",
    "de":                     "",
    "à":                      "",
    "les":                    "",
    "des":                    "",
    "pour":                   "for",
    "avec":                   "with",
    "dans":                   "in",
    "sur":                    "on",
    "par":                    "by",
    "une":                    "a",
    "un":                     "a",
}

# ══════════════════════════════════════════════════════════════
# EXPANSION SÉMANTIQUE DES SKILLS
# Chaque skill → termes anglais utilisés dans les offres d'emploi
# ══════════════════════════════════════════════════════════════

SKILL_EXPANSION = {
    # BI & Data
    "Power BI":        "Power BI business intelligence dashboards reporting data visualization Microsoft BI",
    "Talend":          "Talend ETL data integration data pipeline extract transform load",
    "IBM Cognos":      "IBM Cognos business intelligence analytics reporting OLAP",
    "MDX":             "MDX multidimensional expressions OLAP cube queries data warehouse",
    "Tableau":         "Tableau data visualization business intelligence dashboards",
    "Data Mining":     "data mining pattern recognition machine learning analytics",
    "Machine Learning":"machine learning ML artificial intelligence predictive modeling",
    "Collaborative Filtering": "collaborative filtering recommendation system user-based item-based",

    # Bases de données
    "SQL":             "SQL database queries relational database data management",
    "PL-SQL":          "PL-SQL Oracle stored procedures triggers database programming",
    "Oracle":          "Oracle database enterprise SQL PL-SQL",
    "SQL Server":      "SQL Server Microsoft database T-SQL",

    # Web
    "HTML":            "HTML web development frontend markup",
    "CSS":             "CSS styling web design frontend",
    "JavaScript":      "JavaScript JS frontend web development",
    "PHP":             "PHP backend web development server-side",
    "Django":          "Django Python web framework backend REST API",
    "WordPress":       "WordPress CMS content management web development",

    # Langages
    "Python":          "Python data science machine learning scripting automation",
    "Java":            "Java object-oriented programming backend development",
    "C":               "C programming systems development",

    # Outils
    "Odoo":            "Odoo ERP enterprise resource planning business management",
    "Star UML":        "UML modeling software design architecture diagrams",
    "Git":             "Git version control collaboration",
}

# ══════════════════════════════════════════════════════════════
# PHRASES CONTEXTUELLES PAR NIVEAU
# ══════════════════════════════════════════════════════════════

LEVEL_CONTEXT = {
    "entry": (
        "entry level junior analyst developer business intelligence "
        "recent graduate data analyst BI analyst junior developer "
        "eager to learn motivated first job"
    ),
    "mid": (
        "mid level experienced professional confirmed engineer developer "
        "3 to 5 years experience team collaboration"
    ),
    "senior": (
        "senior lead expert principal architect "
        "8 plus years experience technical leadership"
    ),
    "executive": (
        "director executive manager head of department "
        "strategic leadership C-level"
    ),
}

# ══════════════════════════════════════════════════════════════
# FONCTION DE TRADUCTION LÉGÈRE
# ══════════════════════════════════════════════════════════════

import re


def translate_fr_to_en(text: str) -> str:
    """
    Traduit les termes français courants en anglais dans un texte de CV.
    Approche : substitution par dictionnaire, insensible à la casse.
    Retourne le texte enrichi (FR + EN côte à côte pour ne rien perdre).
    """
    text_lower = text.lower()
    translations = []

    # Trier par longueur décroissante pour matcher les expressions multi-mots d'abord
    for fr_term, en_term in sorted(FR_TO_EN.items(), key=lambda x: len(x[0]), reverse=True):
        if fr_term in text_lower and en_term:
            translations.append(en_term)

    return " ".join(translations)


def expand_skills(skills: list) -> str:
    """
    Étend chaque skill avec ses synonymes anglais courants dans les offres d'emploi.
    Ex: "Power BI" → "Power BI business intelligence dashboards reporting..."
    """
    expanded = []
    for skill in skills:
        expansion = SKILL_EXPANSION.get(skill, skill)
        expanded.append(expansion)
    return " ".join(expanded)


# ══════════════════════════════════════════════════════════════
# NOUVELLE MÉTHODE to_text() — À REMPLACER DANS CandidateProfile
# ══════════════════════════════════════════════════════════════

def build_candidate_text(summary: str, skills: list, experience_level: str,
                          desired_location: str = "") -> str:
    """
    Construit le texte enrichi à encoder par SBERT.

    Structure :
      1. Traduction FR→EN du résumé (termes métier)
      2. Skills répétés 2× + expansion sémantique
      3. Phrases contextuelles de niveau en anglais
      4. Résumé brut original (garde le contexte complet)

    Args:
        summary          : texte brut du CV
        skills           : liste des skills extraits
        experience_level : entry / mid / senior / executive
        desired_location : localisation souhaitée (optionnel)

    Returns:
        Texte enrichi prêt pour SBERT
    """
    parts = []

    # ── 1. Traduction des termes FR → EN ──────────────────────
    fr_translated = translate_fr_to_en(summary)
    if fr_translated:
        parts.append(fr_translated)

    # ── 2. Skills bruts répétés 2× ────────────────────────────
    if skills:
        skills_str = " ".join(skills)
        parts.append(skills_str)
        parts.append(skills_str)  # répétition pour booster le poids

    # ── 3. Expansion sémantique des skills ────────────────────
    if skills:
        expanded = expand_skills(skills)
        parts.append(expanded)

    # ── 4. Contexte de niveau en anglais ──────────────────────
    if experience_level in LEVEL_CONTEXT:
        parts.append(LEVEL_CONTEXT[experience_level])

    # ── 5. Résumé brut original (tronqué à 1500 chars) ────────
    parts.append(summary[:1500])

    # ── 6. Localisation ───────────────────────────────────────
    if desired_location:
        parts.append(f"location {desired_location}")

    return " ".join(parts)


# ══════════════════════════════════════════════════════════════
# PATCH À APPLIQUER DANS 05_matching_engine.py
# Remplace la méthode to_text() de CandidateProfile
# ══════════════════════════════════════════════════════════════

PATCH_CODE = '''
# ── Ajoute cet import en haut de 05_matching_engine.py ──────
from candidate_text_builder import build_candidate_text

# ── Remplace to_text() dans CandidateProfile par : ──────────
def to_text(self) -> str:
    return build_candidate_text(
        summary          = self.summary,
        skills           = self.skills,
        experience_level = self.experience_level,
        desired_location = self.desired_location,
    )
'''


# ══════════════════════════════════════════════════════════════
# TEST STANDALONE
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*65)
    print("TEST — Enrichissement texte candidat FR→EN")
    print("="*65)

    ghada_summary = """
    Actuellement en 3ème année licence spécialisée en Business Intelligence
    à l'institut supérieur de gestion de Tunis (ISG-Tunis),
    je suis en quête d'un stage PFE.
    Stage en Analyse de Données et Gestion Budgétaire, SOFRECOM Tunisie.
    Conception de tableaux de bord interactifs à l'aide de Power BI.
    Système de recommandation basé sur le filtrage collaboratif (Python).
    """

    ghada_skills = [
        "Power BI", "Talend", "IBM Cognos", "MDX", "SQL", "PL-SQL",
        "Oracle", "SQL Server", "Python", "Java", "PHP", "JavaScript",
        "HTML", "CSS", "Django", "Odoo", "Star UML",
        "Machine Learning", "Data Mining", "Collaborative Filtering",
    ]

    text = build_candidate_text(
        summary          = ghada_summary,
        skills           = ghada_skills,
        experience_level = "entry",
        desired_location = "",
    )

    print(f"\n📝 Texte enrichi ({len(text)} chars) :\n")
    print(text[:800] + "...\n")

    print("✅ Termes FR traduits :")
    translated = translate_fr_to_en(ghada_summary)
    print(f"   {translated}\n")

    print("✅ Expansion skills (extrait) :")
    for skill in ["Power BI", "Talend", "Machine Learning"]:
        print(f"   {skill} → {SKILL_EXPANSION.get(skill, skill)}")

    print("\n" + "="*65)
    print("✅ candidate_text_builder opérationnel")
    print("="*65)
    print(PATCH_CODE)
