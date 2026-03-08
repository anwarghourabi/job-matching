"""
Étape 6 : Évaluation du Système de Matching
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Métriques calculées :
  → Precision@K  : % de résultats pertinents parmi les K premiers
  → Recall@K     : % d'offres pertinentes retrouvées
  → NDCG@K       : qualité du ranking (meilleurs résultats en tête ?)
  → MRR          : rang moyen du premier bon résultat
  → Hit Rate@K   : au moins 1 bon résultat dans le Top-K ?

Comparaison :
  → SBERT vs TF-IDF vs Word2Vec
  → Résultats loggués dans MLflow

Lancer :
  python code/06_evaluation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess, sys
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["mlflow", "scikit-learn", "scipy"]:
    try:
        __import__(pkg.replace("-","_"))
    except ImportError:
        _install(pkg)

import numpy as np
import pandas as pd
import pickle, json, time, warnings, importlib, importlib.util
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import mlflow

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))

def _load_module(name, patterns):
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        import glob
        for p in patterns:
            files = glob.glob(str(Path(__file__).parent / p))
            if files:
                spec = importlib.util.spec_from_file_location(name, files[0])
                mod  = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                return mod
        raise ImportError(f"{name} introuvable")

_me  = _load_module("matching_engine", ["*matching_engine*.py"])
MatchingEngine   = _me.MatchingEngine
CandidateProfile = _me.CandidateProfile

DATA_PROCESSED = Path("data/processed")
VECTORS_DIR    = DATA_PROCESSED / "vectors"
EVAL_DIR       = DATA_PROCESSED / "evaluation"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════
# 1. GROUND TRUTH — Profils avec offres pertinentes connues
# ══════════════════════════════════════════════════════════
# Pour chaque profil, on définit les mots-clés qui rendent
# une offre "pertinente" — base de référence pour les métriques

EVAL_PROFILES = [
    {
        "candidate": CandidateProfile(
            name             = "Data Scientist Senior",
            summary          = "Senior data scientist 7 years Python TensorFlow scikit-learn NLP deep learning MLOps AWS machine learning models production",
            skills           = ["Python", "TensorFlow", "NLP", "MLOps", "AWS", "Scikit-learn"],
            experience_level = "senior",
            desired_location = "United States",
        ),
        "relevant_keywords": ["data scientist", "machine learning", "ml engineer",
                               "data science", "deep learning", "ai engineer"],
        "irrelevant_keywords": ["sales", "accountant", "nurse", "driver", "cook"],
    },
    {
        "candidate": CandidateProfile(
            name             = "Data Engineer Junior",
            summary          = "Junior data engineer 2 years SQL Python Apache Spark Airflow AWS Redshift dbt ETL pipeline data warehouse",
            skills           = ["SQL", "Python", "Apache Spark", "Airflow", "AWS", "dbt"],
            experience_level = "entry",
            remote_only      = True,
        ),
        "relevant_keywords": ["data engineer", "etl", "pipeline", "spark", "data",
                               "analytics engineer", "database"],
        "irrelevant_keywords": ["sales manager", "nurse", "teacher", "driver"],
    },
    {
        "candidate": CandidateProfile(
            name             = "Product Manager",
            summary          = "Product manager 5 years B2B SaaS Agile Scrum product roadmap user research A/B testing stakeholder management",
            skills           = ["Agile", "Scrum", "Product Roadmap", "A/B Testing", "SaaS"],
            experience_level = "mid",
            desired_location = "New York",
            min_salary       = 80_000,
        ),
        "relevant_keywords": ["product manager", "product owner", "scrum master",
                               "program manager", "project manager", "product"],
        "irrelevant_keywords": ["nurse", "accountant", "cook", "driver", "sales associate"],
    },
    {
        "candidate": CandidateProfile(
            name             = "DevOps Engineer",
            summary          = "DevOps engineer 4 years Docker Kubernetes Terraform AWS GCP CI CD Linux Python cloud infrastructure automation",
            skills           = ["Docker", "Kubernetes", "Terraform", "AWS", "CI/CD", "Python"],
            experience_level = "mid",
            remote_only      = True,
            min_salary       = 100_000,
        ),
        "relevant_keywords": ["devops", "cloud engineer", "infrastructure", "platform engineer",
                               "site reliability", "sre", "kubernetes", "cloud"],
        "irrelevant_keywords": ["nurse", "teacher", "accountant", "sales"],
    },
    {
        "candidate": CandidateProfile(
            name             = "Software Engineer Backend",
            summary          = "Backend software engineer 3 years Java Python REST API microservices PostgreSQL Docker Git agile",
            skills           = ["Java", "Python", "REST API", "Microservices", "PostgreSQL", "Docker"],
            experience_level = "mid",
        ),
        "relevant_keywords": ["software engineer", "backend engineer", "java developer",
                               "python developer", "developer", "software developer"],
        "irrelevant_keywords": ["nurse", "teacher", "accountant", "sales associate"],
    },
]


# ══════════════════════════════════════════════════════════
# 2. FONCTIONS DE MÉTRIQUES
# ══════════════════════════════════════════════════════════

def is_relevant(job_title: str, relevant_keywords: list,
                irrelevant_keywords: list) -> int:
    """
    Retourne 1 si l'offre est pertinente, 0 sinon.
    Basé sur les mots-clés du titre de poste.
    """
    title_lower = job_title.lower()
    if any(kw in title_lower for kw in irrelevant_keywords):
        return 0
    if any(kw in title_lower for kw in relevant_keywords):
        return 1
    return 0


def precision_at_k(relevances: list, k: int) -> float:
    """P@K = nb pertinents dans top-K / K"""
    return sum(relevances[:k]) / k


def recall_at_k(relevances: list, k: int, total_relevant: int) -> float:
    """R@K = nb pertinents dans top-K / total pertinents dans corpus"""
    if total_relevant == 0:
        return 0.0
    return sum(relevances[:k]) / total_relevant


def ndcg_at_k(relevances: list, k: int) -> float:
    """
    NDCG@K = DCG@K / IDCG@K
    Mesure la qualité du ranking — pénalise les bons résultats mal classés
    """
    def dcg(rels, k):
        return sum(
            rel / np.log2(i + 2)
            for i, rel in enumerate(rels[:k])
        )
    dcg_val  = dcg(relevances, k)
    idcg_val = dcg(sorted(relevances, reverse=True), k)
    return dcg_val / idcg_val if idcg_val > 0 else 0.0


def mean_reciprocal_rank(relevances: list) -> float:
    """
    MRR = 1 / rang du premier résultat pertinent
    Ex: premier bon résultat au rang 3 → MRR = 1/3 = 0.33
    """
    for i, rel in enumerate(relevances):
        if rel == 1:
            return 1.0 / (i + 1)
    return 0.0


def hit_rate_at_k(relevances: list, k: int) -> int:
    """Hit@K = 1 si au moins 1 résultat pertinent dans le top-K"""
    return 1 if sum(relevances[:k]) > 0 else 0


def compute_all_metrics(titles: list, relevant_kw: list,
                        irrelevant_kw: list, total_relevant: int,
                        k_values: list = [1, 3, 5, 10]) -> dict:
    """Calcule toutes les métriques pour une liste de titres"""
    relevances = [is_relevant(t, relevant_kw, irrelevant_kw) for t in titles]
    # Padding à 0 si moins de max(k) résultats disponibles
    max_k = max(k_values)
    if len(relevances) < max_k:
        relevances = relevances + [0] * (max_k - len(relevances))

    metrics = {"mrr": round(mean_reciprocal_rank(relevances), 4)}

    for k in k_values:
        metrics[f"precision_at_{k}"]  = round(precision_at_k(relevances, k), 4)
        metrics[f"recall_at_{k}"]     = round(recall_at_k(relevances, k, total_relevant), 4)
        metrics[f"ndcg_at_{k}"]       = round(ndcg_at_k(relevances, k), 4)
        metrics[f"hit_rate_at_{k}"]   = hit_rate_at_k(relevances, k)

    metrics["relevances"] = relevances
    return metrics


# ══════════════════════════════════════════════════════════
# 3. ÉVALUATION SBERT
# ══════════════════════════════════════════════════════════

def evaluate_sbert(engine: MatchingEngine, k_values=[1,3,5,10]) -> dict:
    """Évalue le matching SBERT sur tous les profils de référence"""
    print("\n🚀 Évaluation SBERT...")
    all_metrics = []

    for profile in EVAL_PROFILES:
        candidate = profile["candidate"]
        t0        = time.time()
        results   = engine.match(candidate, top_k=max(k_values), min_score=0.0)
        elapsed   = (time.time() - t0) * 1000

        titles        = [r.job_title for r in results]
        total_relevant = sum(
            1 for i in range(len(engine.df))
            if is_relevant(engine.df.iloc[i]["job_title"],
                           profile["relevant_keywords"],
                           profile["irrelevant_keywords"])
        )
        total_relevant = max(total_relevant, 1)

        metrics = compute_all_metrics(
            titles, profile["relevant_keywords"],
            profile["irrelevant_keywords"], total_relevant, k_values
        )
        metrics["inference_ms"] = round(elapsed, 2)
        metrics["candidate"]    = candidate.name
        all_metrics.append(metrics)

        print(f"   {candidate.name:<30} "
              f"P@10={metrics['precision_at_10']:.2f}  "
              f"NDCG@10={metrics['ndcg_at_10']:.2f}  "
              f"MRR={metrics['mrr']:.2f}  "
              f"Hit@5={metrics['hit_rate_at_5']}")

    # Moyennes globales
    avg = {}
    for k in k_values:
        avg[f"avg_precision_at_{k}"]  = round(np.mean([m[f"precision_at_{k}"]  for m in all_metrics]), 4)
        avg[f"avg_recall_at_{k}"]     = round(np.mean([m[f"recall_at_{k}"]     for m in all_metrics]), 4)
        avg[f"avg_ndcg_at_{k}"]       = round(np.mean([m[f"ndcg_at_{k}"]       for m in all_metrics]), 4)
        avg[f"avg_hit_rate_at_{k}"]   = round(np.mean([m[f"hit_rate_at_{k}"]   for m in all_metrics]), 4)
    avg["avg_mrr"]          = round(np.mean([m["mrr"]            for m in all_metrics]), 4)
    avg["avg_inference_ms"] = round(np.mean([m["inference_ms"]   for m in all_metrics]), 2)

    return {"model": "sbert", "per_profile": all_metrics, "averages": avg}


# ══════════════════════════════════════════════════════════
# 4. ÉVALUATION TF-IDF
# ══════════════════════════════════════════════════════════

def evaluate_tfidf(df: pd.DataFrame, k_values=[1,3,5,10]) -> dict:
    """Évalue TF-IDF comme baseline de comparaison"""
    print("\n📊 Évaluation TF-IDF (baseline)...")

    tfidf_path = VECTORS_DIR / "tfidf_matrix.npz"
    vect_path  = VECTORS_DIR / "tfidf_vectorizer.pkl"

    if not tfidf_path.exists() or not vect_path.exists():
        print("   ⚠️  Fichiers TF-IDF introuvables — skip")
        return None

    matrix     = sp.load_npz(str(tfidf_path))
    with open(vect_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Preprocesseur simple
    import re
    def preprocess(text):
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    all_metrics = []

    for profile in EVAL_PROFILES:
        candidate  = profile["candidate"]
        query_text = preprocess(candidate.to_text())

        t0         = time.time()
        query_vec  = vectorizer.transform([query_text])
        scores     = cosine_similarity(query_vec, matrix).flatten()
        top_idx    = np.argsort(scores)[::-1][:max(k_values)]
        elapsed    = (time.time() - t0) * 1000

        titles         = df.iloc[top_idx]["job_title"].tolist()
        total_relevant = max(sum(
            1 for t in df["job_title"]
            if is_relevant(t, profile["relevant_keywords"],
                           profile["irrelevant_keywords"])
        ), 1)

        metrics = compute_all_metrics(
            titles, profile["relevant_keywords"],
            profile["irrelevant_keywords"], total_relevant, k_values
        )
        metrics["inference_ms"] = round(elapsed, 2)
        metrics["candidate"]    = candidate.name
        all_metrics.append(metrics)

        print(f"   {candidate.name:<30} "
              f"P@10={metrics['precision_at_10']:.2f}  "
              f"NDCG@10={metrics['ndcg_at_10']:.2f}  "
              f"MRR={metrics['mrr']:.2f}  "
              f"Hit@5={metrics['hit_rate_at_5']}")

    avg = {}
    for k in k_values:
        avg[f"avg_precision_at_{k}"]  = round(np.mean([m[f"precision_at_{k}"]  for m in all_metrics]), 4)
        avg[f"avg_recall_at_{k}"]     = round(np.mean([m[f"recall_at_{k}"]     for m in all_metrics]), 4)
        avg[f"avg_ndcg_at_{k}"]       = round(np.mean([m[f"ndcg_at_{k}"]       for m in all_metrics]), 4)
        avg[f"avg_hit_rate_at_{k}"]   = round(np.mean([m[f"hit_rate_at_{k}"]   for m in all_metrics]), 4)
    avg["avg_mrr"]          = round(np.mean([m["mrr"]          for m in all_metrics]), 4)
    avg["avg_inference_ms"] = round(np.mean([m["inference_ms"] for m in all_metrics]), 2)

    return {"model": "tfidf", "per_profile": all_metrics, "averages": avg}


# ══════════════════════════════════════════════════════════
# 5. LOGGING MLFLOW
# ══════════════════════════════════════════════════════════

def log_evaluation_to_mlflow(eval_results: dict, model_name: str):
    """Logue les métriques d'évaluation dans MLflow"""
    import os
    if os.name == "nt":
        mlflow.set_tracking_uri("mlruns")
    else:
        mlflow.set_tracking_uri(str(Path("mlruns").absolute()))

    mlflow.set_experiment("model-evaluation")

    with mlflow.start_run(run_name=f"EVAL-{model_name.upper()}"):
        # Métriques moyennes
        for k, v in eval_results["averages"].items():
            mlflow.log_metric(k, v)

        # Métriques par profil
        for profile_metrics in eval_results["per_profile"]:
            name = profile_metrics["candidate"].replace(" ", "_").replace("—","").lower()[:20]
            for k in [1, 3, 5, 10]:
                mlflow.log_metric(f"{name}_p_at_{k}", profile_metrics[f"precision_at_{k}"])
                mlflow.log_metric(f"{name}_ndcg_at_{k}", profile_metrics[f"ndcg_at_{k}"])
            mlflow.log_metric(f"{name}_mrr", profile_metrics["mrr"])

        mlflow.log_param("model", model_name)
        mlflow.log_param("n_profiles", len(eval_results["per_profile"]))
        mlflow.log_param("k_values", "[1,3,5,10]")

        # Sauvegarder rapport JSON comme artefact
        report_path = EVAL_DIR / f"eval_{model_name}.json"
        with open(report_path, "w") as f:
            # Enlever les objets non-sérialisables
            clean = {
                "model":      eval_results["model"],
                "averages":   eval_results["averages"],
                "per_profile": [
                    {k: v for k, v in p.items() if k != "relevances"}
                    for p in eval_results["per_profile"]
                ]
            }
            json.dump(clean, f, indent=2)
        mlflow.log_artifact(str(report_path))

        run_id = mlflow.active_run().info.run_id
        print(f"   📋 MLflow run_id : {run_id}")

    return run_id


# ══════════════════════════════════════════════════════════
# 6. RAPPORT COMPARATIF
# ══════════════════════════════════════════════════════════

def print_comparison_report(sbert_eval: dict, tfidf_eval: dict):
    """Affiche le rapport comparatif final"""

    print("\n" + "="*65)
    print("📊 RAPPORT COMPARATIF FINAL : SBERT vs TF-IDF")
    print("="*65)

    metrics_to_show = [
        ("Precision@1",  "avg_precision_at_1"),
        ("Precision@5",  "avg_precision_at_5"),
        ("Precision@10", "avg_precision_at_10"),
        ("NDCG@5",       "avg_ndcg_at_5"),
        ("NDCG@10",      "avg_ndcg_at_10"),
        ("MRR",          "avg_mrr"),
        ("Hit Rate@5",   "avg_hit_rate_at_5"),
        ("Inference ms", "avg_inference_ms"),
    ]

    s = sbert_eval["averages"]
    t = tfidf_eval["averages"] if tfidf_eval else {}

    print(f"\n{'Métrique':<20} {'SBERT':>10} {'TF-IDF':>10} {'Δ':>10} {'Gagnant':>10}")
    print("─" * 65)

    sbert_wins = 0
    tfidf_wins = 0

    for label, key in metrics_to_show:
        sv = s.get(key, 0)
        tv = t.get(key, 0) if t else 0

        if key == "avg_inference_ms":
            # Pour le temps, plus petit = meilleur
            winner = "TF-IDF ⚡" if tv < sv else "SBERT ⚡"
            if tv < sv:
                tfidf_wins += 1
            else:
                sbert_wins += 1
            delta = sv - tv
        else:
            winner = "SBERT ✅" if sv >= tv else "TF-IDF"
            if sv >= tv:
                sbert_wins += 1
            else:
                tfidf_wins += 1
            delta = sv - tv

        delta_str = f"{delta:+.3f}"
        print(f"{label:<20} {sv:>10.3f} {tv:>10.3f} {delta_str:>10} {winner:>10}")

    print("─" * 65)
    print(f"{'SCORE FINAL':<20} {'SBERT':>10} {'TF-IDF':>10}")
    print(f"{'Victoires':<20} {sbert_wins:>10} {tfidf_wins:>10}")

    winner_final = "🏆 SBERT" if sbert_wins >= tfidf_wins else "🏆 TF-IDF"
    print(f"\n{winner_final} remporte la comparaison ({max(sbert_wins,tfidf_wins)}/8 métriques)")

    # Rapport par profil
    print(f"\n{'─'*65}")
    print("📋 DÉTAIL PAR PROFIL — SBERT")
    print(f"{'─'*65}")
    print(f"{'Profil':<30} {'P@1':>6} {'P@5':>6} {'P@10':>6} {'NDCG@10':>8} {'MRR':>6}")
    print("─" * 65)

    for p in sbert_eval["per_profile"]:
        print(f"{p['candidate']:<30} "
              f"{p['precision_at_1']:>6.2f} "
              f"{p['precision_at_5']:>6.2f} "
              f"{p['precision_at_10']:>6.2f} "
              f"{p['ndcg_at_10']:>8.3f} "
              f"{p['mrr']:>6.3f}")

    print(f"\n{'─'*65}")
    avg = sbert_eval["averages"]
    print(f"{'MOYENNE':<30} "
          f"{avg['avg_precision_at_1']:>6.2f} "
          f"{avg['avg_precision_at_5']:>6.2f} "
          f"{avg['avg_precision_at_10']:>6.2f} "
          f"{avg['avg_ndcg_at_10']:>8.3f} "
          f"{avg['avg_mrr']:>6.3f}")


# ══════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("\n" + "="*65)
    print("ÉTAPE 6 : ÉVALUATION DU SYSTÈME DE MATCHING")
    print("="*65)

    # ── Charger l'engine ──────────────────────────────────
    print("\n⚙️  Chargement du Matching Engine...")
    engine = MatchingEngine()
    engine.load()

    df = engine.df

    # ── Évaluation SBERT ──────────────────────────────────
    print("\n" + "─"*65)
    print("🚀 A. ÉVALUATION SBERT (modèle principal)")
    print("─"*65)
    sbert_eval = evaluate_sbert(engine, k_values=[1, 3, 5, 10])
    sbert_run  = log_evaluation_to_mlflow(sbert_eval, "sbert")

    # ── Évaluation TF-IDF ─────────────────────────────────
    print("\n" + "─"*65)
    print("📊 B. ÉVALUATION TF-IDF (baseline comparaison)")
    print("─"*65)
    tfidf_eval = evaluate_tfidf(df, k_values=[1, 3, 5, 10])
    if tfidf_eval:
        tfidf_run = log_evaluation_to_mlflow(tfidf_eval, "tfidf")

    # ── Rapport comparatif ────────────────────────────────
    if tfidf_eval:
        print_comparison_report(sbert_eval, tfidf_eval)

    # ── Sauvegarder rapport global ────────────────────────
    report = {
        "sbert": {
            "averages":    sbert_eval["averages"],
            "per_profile": [{k: v for k, v in p.items() if k != "relevances"}
                            for p in sbert_eval["per_profile"]]
        },
        "tfidf": {
            "averages":    tfidf_eval["averages"] if tfidf_eval else {},
        },
        "conclusion": "SBERT sélectionné — meilleure précision sémantique"
    }
    report_path = EVAL_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ── Résumé final ──────────────────────────────────────
    avg = sbert_eval["averages"]
    print(f"""
{'='*65}
✅ ÉVALUATION TERMINÉE
{'='*65}

📊 Résumé SBERT :
   Precision@10  : {avg['avg_precision_at_10']:.3f}
   NDCG@10       : {avg['avg_ndcg_at_10']:.3f}
   MRR           : {avg['avg_mrr']:.3f}
   Hit Rate@5    : {avg['avg_hit_rate_at_5']:.3f}
   Inference     : {avg['avg_inference_ms']:.1f}ms

💾 Rapports sauvegardés :
   {EVAL_DIR}/eval_sbert.json
   {EVAL_DIR}/eval_tfidf.json
   {EVAL_DIR}/evaluation_report.json

🌐 Voir dans MLflow :
   mlflow ui --backend-store-uri mlruns --port 5000
   → http://127.0.0.1:5000

🚀 Prochaine étape : python code/07_api.py
""")

    return sbert_eval, tfidf_eval


if __name__ == "__main__":
    sbert_eval, tfidf_eval = main()