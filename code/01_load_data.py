"""
Week 1 - Étape 1: Charger les données
VERSION CORRIGÉE + SOURCE SYNTHÉTIQUE INTÉGRÉE

Sources :
  1. hugginglearners/data-science-job-salaries  (token HF requis)
  2. datastax/linkedin_job_listings             (token HF requis)
  3. RemoteOK API                               (public, sans clé)
  4. ✅ Offres synthétiques                     (local, toujours disponible)
     → Comble les lacunes du corpus : BI/Data entry, ML, Finance, RH, Marketing...
"""

import pandas as pd
import numpy as np
import random
from pathlib import Path

# Configuration
DATA_RAW = Path('data/raw')
DATA_RAW.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════
# 🔑 TON TOKEN HUGGINGFACE  (READ token, gratuit)
#    https://huggingface.co/settings/tokens
# ══════════════════════════════════════════════════════
HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXX"   # ← Coller ton token ici
# ══════════════════════════════════════════════════════


def safe_save_csv(df, path, label="fichier"):
    """Sauvegarde robuste : gère les PermissionError (fichier ouvert dans Excel)"""
    path = Path(path)
    for attempt in range(3):
        try:
            df.to_csv(path, index=False, encoding='utf-8-sig')
            print(f"   ✅ Sauvegardé: {path}")
            return path
        except PermissionError:
            alt_path = path.with_stem(path.stem + f"_v{attempt+2}")
            print(f"   ⚠️  {path.name} verrouillé (Excel ouvert?), essai: {alt_path.name}")
            path = alt_path
    print(f"   ❌ Impossible de sauvegarder {label}")
    return None


# ══════════════════════════════════════════════════════════════
# SOURCE 4 : OFFRES SYNTHÉTIQUES
# Comble les lacunes du corpus (BI entry, ML, Finance, RH...)
# ══════════════════════════════════════════════════════════════

SYNTHETIC_JOBS = {

    "bi_entry": {
        "level": "entry", "salary_range": (35000, 65000), "remote_ratios": [0, 50, 100],
        "titles": [
            "Business Intelligence Analyst", "Junior BI Analyst", "BI Analyst Intern",
            "Data Analyst Intern", "Junior Data Analyst", "Power BI Developer Junior",
            "Power BI Analyst", "Reporting Analyst Junior", "BI Developer Entry Level",
            "Data Visualization Analyst", "Junior SQL Analyst", "Analytics Intern",
            "Business Intelligence Intern", "Data Reporting Analyst", "Data Analyst Graduate",
        ],
        "descriptions": [
            "Design and develop interactive dashboards using Power BI and Tableau. "
            "Write SQL queries to extract and analyze data from Oracle and SQL Server. "
            "Support business users with reporting and data visualization. Entry level.",

            "Build and maintain BI reports using Power BI, IBM Cognos, and Tableau. "
            "Work with SQL, PL-SQL, and MDX queries. Analyze financial and operational data. "
            "Entry level position ideal for Business Intelligence graduates.",

            "Junior BI Analyst role focused on Power BI dashboard development, "
            "SQL queries, ETL processes with Talend. Oracle and SQL Server databases.",

            "Data Analyst intern. Python and SQL to analyze datasets. "
            "Create visualizations using Power BI and Tableau. "
            "Support data mining and machine learning projects.",

            "Develop BI solutions using IBM Cognos and Power BI. "
            "Complex SQL and MDX queries. ETL pipelines with Talend and SSIS.",

            "Entry-level BI position. Design star schema data models. "
            "Build Power BI reports. Work with SQL Server, Oracle, PostgreSQL.",
        ],
    },

    "ml_entry": {
        "level": "entry", "salary_range": (45000, 80000), "remote_ratios": [0, 50, 100],
        "titles": [
            "Junior Data Scientist", "Machine Learning Engineer Intern",
            "Data Science Intern", "Junior ML Engineer", "AI Engineer Entry Level",
            "Data Science Graduate", "Junior NLP Engineer", "Machine Learning Analyst",
            "AI Research Intern", "Junior Computer Vision Engineer",
        ],
        "descriptions": [
            "Entry-level Data Scientist. Build predictive models using Python, "
            "scikit-learn, XGBoost. SQL databases. Machine learning and data mining.",

            "Junior ML Engineer. Implement machine learning models in Python. "
            "PyTorch or TensorFlow. NLP and computer vision projects. Docker, AWS.",

            "Data Science intern. Python, pandas, numpy. "
            "Machine learning models: classification, regression, clustering. "
            "Visualize with matplotlib and Power BI.",
        ],
    },

    "ml_senior": {
        "level": "senior", "salary_range": (90000, 200000), "remote_ratios": [0, 50, 100],
        "titles": [
            "Senior Data Scientist", "Senior Machine Learning Engineer",
            "Lead ML Engineer", "Principal Data Scientist", "Lead AI Engineer",
            "Senior NLP Engineer", "Staff Machine Learning Engineer",
            "Senior Computer Vision Engineer", "ML Platform Engineer Senior",
        ],
        "descriptions": [
            "Senior Data Scientist 7+ years. Lead ML projects. "
            "Python, TensorFlow, PyTorch, scikit-learn. MLOps, Kubernetes, AWS SageMaker.",

            "Lead ML Engineer. LLM fine-tuning (LoRA, PEFT), RAG, prompt engineering. "
            "Mentor junior data scientists. Python, PyTorch, HuggingFace.",

            "Senior AI Engineer 8+ years. Deep learning, NLP, computer vision. "
            "AWS, GCP, Docker, Kubernetes, MLflow, Airflow.",
        ],
    },

    "data_eng": {
        "level": "mid", "salary_range": (70000, 130000), "remote_ratios": [0, 50, 100],
        "titles": [
            "Data Engineer", "Senior Data Engineer", "ETL Developer",
            "Data Pipeline Engineer", "Analytics Engineer", "Cloud Data Engineer",
        ],
        "descriptions": [
            "Data Engineer. ETL pipelines using Apache Spark, Airflow, dbt. "
            "AWS (S3, Redshift, Glue). SQL, Python, Kafka.",

            "Senior Data Engineer. Data warehouse on Snowflake. "
            "Spark, Airflow, dbt, Docker, Kubernetes.",
        ],
    },

    "web_entry": {
        "level": "entry", "salary_range": (30000, 60000), "remote_ratios": [0, 50, 100],
        "titles": [
            "Junior Web Developer", "Frontend Developer Junior",
            "Junior Full Stack Developer", "Web Developer Intern",
            "Junior PHP Developer", "Junior Python Developer", "Junior Django Developer",
        ],
        "descriptions": [
            "Junior Web Developer. HTML, CSS, JavaScript, PHP. "
            "Django backend. MySQL, PostgreSQL. Git. Entry level.",

            "Junior Backend Developer. Python, Django, REST API. "
            "PostgreSQL, MySQL. Docker basics. Entry level for recent graduates.",
        ],
    },

    "finance_entry": {
        "level": "entry", "salary_range": (30000, 55000), "remote_ratios": [0, 0, 50],
        "titles": [
            "Junior Accountant", "Accounting Intern", "Junior Financial Analyst",
            "Finance Graduate Trainee", "Audit Assistant", "Junior Controller",
        ],
        "descriptions": [
            "Junior Accountant entry level. General accounting, IFRS. "
            "Excel, Sage, SAP. Financial reporting and budget management.",

            "Junior Financial Analyst. Financial modeling, Excel, VBA. "
            "Budget analysis, reporting, KPI dashboards.",
        ],
    },

    "finance_senior": {
        "level": "senior", "salary_range": (80000, 160000), "remote_ratios": [0, 0, 50],
        "titles": [
            "Senior Financial Analyst", "Finance Manager", "Senior Controller",
            "Head of Finance", "Senior Audit Manager", "Director of Finance",
        ],
        "descriptions": [
            "Senior Financial Analyst 7+ years. IFRS, consolidation, SAP, Power BI.",
            "Finance Manager. Financial reporting, IFRS, GAAP, audit, SAP.",
        ],
    },

    "rh_all": {
        "level": "entry", "salary_range": (28000, 50000), "remote_ratios": [0, 50],
        "titles": [
            "HR Assistant", "Recruitment Intern", "Junior HR Generalist",
            "Talent Acquisition Intern", "HR Coordinator Entry Level",
        ],
        "descriptions": [
            "HR Assistant entry level. Recruitment, onboarding, payroll. Workday, BambooHR.",
            "Recruitment Intern. Source candidates, screen CVs. LinkedIn Recruiter, ATS.",
        ],
    },

    "marketing_entry": {
        "level": "entry", "salary_range": (28000, 52000), "remote_ratios": [0, 50, 100],
        "titles": [
            "Digital Marketing Intern", "Junior SEO Analyst",
            "Social Media Manager Junior", "Content Marketing Intern",
            "Junior Marketing Analyst", "Community Manager Junior",
        ],
        "descriptions": [
            "Digital Marketing Intern. SEO, Google Analytics, Google Ads. HubSpot, Mailchimp.",
            "Junior SEO Analyst. On-page SEO, Google Analytics, keyword research.",
        ],
    },

    "erp_entry": {
        "level": "entry", "salary_range": (32000, 58000), "remote_ratios": [0, 50],
        "titles": [
            "Odoo Developer Junior", "ERP Analyst Junior",
            "Junior SAP Consultant", "Odoo Implementation Consultant", "ERP Support Analyst",
        ],
        "descriptions": [
            "Junior Odoo Developer. Python, Odoo ERP, PostgreSQL, XML, JavaScript.",
            "ERP Analyst Junior. SAP or Odoo. Business process analysis, SQL, Excel.",
        ],
    },
}

LOCATIONS_SYNTHETIC = [
    "Paris, France", "Lyon, France", "London, UK", "Berlin, Germany",
    "Amsterdam, Netherlands", "Brussels, Belgium", "Madrid, Spain",
    "New York, NY", "San Francisco, CA", "Austin, TX", "Chicago, IL",
    "Seattle, WA", "Boston, MA", "Toronto, Canada", "Montreal, Canada",
    "Dubai, UAE", "Tunis, Tunisia", "Casablanca, Morocco",
    "Remote", "Remote - Europe", "Remote - US", "Remote - Worldwide",
]


def load_synthetic_jobs(n_per_category: int = 200) -> pd.DataFrame:
    """
    Génère des offres synthétiques réalistes pour enrichir le corpus.
    Cible les domaines sous-représentés : BI/Data entry, ML, Finance, RH, Marketing...
    """
    print("\n4️⃣ 🏭 Génération des offres synthétiques...")
    random.seed(42)
    all_jobs = []

    for category, config in SYNTHETIC_JOBS.items():
        for _ in range(n_per_category):
            title   = random.choice(config["titles"])
            desc    = random.choice(config["descriptions"])
            loc     = random.choice(LOCATIONS_SYNTHETIC)
            remote  = random.choice(config["remote_ratios"])
            sal_min, sal_max = config["salary_range"]

            has_salary = random.random() > 0.35
            salary = round(random.uniform(sal_min, sal_max), 2) if has_salary else 0.0

            all_jobs.append({
                "job_title":        title,
                "salary_usd":       salary,
                "experience_level": config["level"],
                "employment_type":  "FT",
                "location":         loc,
                "remote_ratio":     remote,
                "company_size":     random.choice(["S", "M", "L"]),
                "source":           "synthetic",
            })

    df = pd.DataFrame(all_jobs).sample(frac=1, random_state=42).reset_index(drop=True)

    safe_save_csv(df, DATA_RAW / 'synthetic_jobs.csv', "Offres synthétiques")

    print(f"   ✅ {len(df):,} offres synthétiques générées")
    print(f"   Répartition niveau :")
    print(df["experience_level"].value_counts().to_string())
    print(f"\n   Top titres générés :")
    print(df["job_title"].value_counts().head(10).to_string())

    return df


# ══════════════════════════════════════════════════════════════
# SOURCES EXISTANTES (inchangées)
# ══════════════════════════════════════════════════════════════

def load_huggingface_ds_salaries():
    print("\n1️⃣ 📥 Chargement → hugginglearners/data-science-job-salaries")
    try:
        from datasets import load_dataset
        ds = load_dataset("hugginglearners/data-science-job-salaries", token=HF_TOKEN)
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        print(f"   ✅ Chargé : {len(df):,} lignes")
        df_std = pd.DataFrame({
            'job_title':        df['job_title'].astype(str),
            'salary_usd':       pd.to_numeric(df['salary_in_usd'], errors='coerce').fillna(0),
            'experience_level': df['experience_level'].astype(str).str.lower(),
            'employment_type':  df['employment_type'].astype(str),
            'location':         df['company_location'].astype(str),
            'remote_ratio':     pd.to_numeric(df['remote_ratio'], errors='coerce').fillna(0),
            'company_size':     df['company_size'].astype(str),
            'source':           'hf_ds_salaries'
        })
        df_std = df_std[df_std['job_title'].notna() & (df_std['job_title'] != 'nan')]
        safe_save_csv(df_std, DATA_RAW / 'huggingface_ds_salaries.csv')
        return df_std
    except Exception as e:
        print(f"   ❌ Erreur : {e}")
        return None


def load_huggingface_linkedin():
    print("\n2️⃣ 📥 Chargement → datastax/linkedin_job_listings")
    try:
        from datasets import load_dataset
        ds = load_dataset("datastax/linkedin_job_listings", token=HF_TOKEN)
        split = "train" if "train" in ds else list(ds.keys())[0]
        df = ds[split].to_pandas()
        print(f"   ✅ Chargé : {len(df):,} lignes")

        def find_col(df, *keywords):
            for kw in keywords:
                for col in df.columns:
                    if kw.lower() in col.lower():
                        return col
            return None

        title_col    = find_col(df, 'title', 'job_title', 'position')
        salary_col   = find_col(df, 'salary', 'pay')
        exp_col      = find_col(df, 'experience', 'level', 'seniority')
        remote_col   = find_col(df, 'remote', 'work_type')
        location_col = find_col(df, 'location', 'city')

        salary_values  = pd.to_numeric(df[salary_col], errors='coerce').fillna(0) if salary_col else pd.Series([0]*len(df))
        exp_values     = df[exp_col].astype(str).str.lower() if exp_col else pd.Series(['unknown']*len(df))
        location_values= df[location_col].astype(str) if location_col else pd.Series(['Unknown']*len(df))
        remote_values  = pd.Series([50]*len(df))
        if remote_col:
            remote_values = df[remote_col].astype(str).str.lower().apply(
                lambda x: 100 if any(w in x for w in ('remote','fully','virtual'))
                         else 50 if any(w in x for w in ('hybrid','flexible'))
                         else 0
            )

        df_std = pd.DataFrame({
            'job_title':        df[title_col].astype(str) if title_col else 'Unknown',
            'salary_usd':       salary_values,
            'experience_level': exp_values,
            'employment_type':  'FT',
            'location':         location_values,
            'remote_ratio':     remote_values,
            'company_size':     'M',
            'source':           'hf_linkedin'
        })
        df_std = df_std[df_std['job_title'].notna() & (df_std['job_title'] != 'Unknown')]
        safe_save_csv(df_std, DATA_RAW / 'huggingface_linkedin.csv')
        print(f"   ✅ {len(df_std):,} offres LinkedIn")
        return df_std
    except Exception as e:
        print(f"   ❌ Erreur LinkedIn : {e}")
        return None


def load_remoteok():
    print("\n3️⃣ 📥 Chargement RemoteOK (API publique)...")
    try:
        import requests
        resp = requests.get("https://remoteok.com/api", headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        jobs = []
        for job in data:
            if isinstance(job, dict) and job.get("company") and job.get("position"):
                salary = 0
                try:
                    if job.get("salary_min") and job.get("salary_max"):
                        salary = (float(job["salary_min"]) + float(job["salary_max"])) / 2
                except (ValueError, TypeError):
                    salary = 0
                jobs.append({
                    'job_title':        str(job.get("position", "Unknown")),
                    'salary_usd':       salary,
                    'experience_level': 'unknown',
                    'employment_type':  'FT',
                    'location':         str(job.get("location", "Remote")),
                    'remote_ratio':     100,
                    'company_size':     'M',
                    'source':           'remoteok_api'
                })
        df_std = pd.DataFrame(jobs)
        safe_save_csv(df_std, DATA_RAW / 'remoteok_api.csv')
        print(f"   ✅ {len(df_std):,} offres RemoteOK")
        return df_std
    except Exception as e:
        print(f"   ❌ Erreur RemoteOK : {e}")
        return None


# ══════════════════════════════════════════════════════════════
# COMBINAISON
# ══════════════════════════════════════════════════════════════

def combine_all_sources():
    print("\n" + "="*60)
    print("COMBINAISON DE TOUS LES DATASETS")
    print("="*60)

    if HF_TOKEN == "hf_XXXXXXXXXXXXXXXXXXXX" or not HF_TOKEN.startswith("hf_"):
        print("\n⚠️  HF_TOKEN non configuré — datasets HuggingFace ignorés.")
        print("   Les offres synthétiques seront toujours chargées.\n")

    dfs = []

    # Sources HuggingFace + RemoteOK
    df1 = load_huggingface_ds_salaries()
    if df1 is not None and len(df1) > 0:
        dfs.append(df1)

    df2 = load_huggingface_linkedin()
    if df2 is not None and len(df2) > 0:
        dfs.append(df2)

    df3 = load_remoteok()
    if df3 is not None and len(df3) > 0:
        dfs.append(df3)

    # ✅ Source synthétique — TOUJOURS ajoutée
    df4 = load_synthetic_jobs(n_per_category=200)  # 200 × 9 = 1800 offres
    dfs.append(df4)
    print(f"\n   ✅ Offres synthétiques ajoutées : {len(df4):,}")

    print("\n5️⃣ 🔗 Fusion des datasets...")
    df_combined = pd.concat(dfs, ignore_index=True)
    print(f"   Avant déduplication : {len(df_combined):,} offres")

    df_combined = df_combined.drop_duplicates(subset=['job_title', 'location'], keep='first')
    print(f"   Après déduplication : {len(df_combined):,} offres")

    print(f"\n   📊 Distribution par source :")
    print(df_combined['source'].value_counts().to_string())

    print(f"\n   📊 Distribution par niveau :")
    print(df_combined['experience_level'].value_counts().to_string())

    salary_count = (df_combined['salary_usd'] > 1000).sum()
    pct = salary_count / len(df_combined) * 100
    print(f"\n   💰 Salaires disponibles : {salary_count:,} ({pct:.1f}%)")
    print(f"   🌍 Localisations uniques : {df_combined['location'].nunique()}")
    print(f"   💼 Titres uniques        : {df_combined['job_title'].nunique()}")

    print(f"\n6️⃣ 💾 Sauvegarde...")
    safe_save_csv(df_combined, DATA_RAW / 'jobs_merged.csv', "dataset fusionné")

    return df_combined


def main():
    print("\n" + "="*60)
    print("ÉTAPE 1: CHARGER LES DONNÉES")
    print("="*60)

    df_combined = combine_all_sources()

    if df_combined is not None and len(df_combined) > 0:
        print("\n" + "="*60)
        print("✅ ÉTAPE 1 COMPLÉTÉE!")
        print("="*60)
        print(f"\n📊 {len(df_combined):,} offres chargées au total")
        print(f"💰 {(df_combined['salary_usd'] > 1000).sum():,} avec salaire renseigné")
        print(f"\n🚀 Prochaine étape : python code/02_clean_data.py")
        return df_combined
    else:
        print("\n❌ ERREUR LORS DU CHARGEMENT")
        return None


if __name__ == '__main__':
    df = main()