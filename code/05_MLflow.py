"""
Étape 3 : Sélection du modèle avec MLflow
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  CE FICHIER TOURNE AVANT LA VECTORISATION FINALE

Objectif : comparer TF-IDF vs Word2Vec vs SBERT sur un
échantillon du corpus, loguer les métriques dans MLflow,
et choisir automatiquement le meilleur modèle.

Pipeline :
  1. Charger un échantillon des données nettoyées
  2. Construire des profils candidats de référence
  3. Tester chaque modèle sur l'échantillon
  4. Loguer métriques + paramètres dans MLflow
  5. Comparer et sélectionner le meilleur modèle
  6. Sauvegarder le choix → utilisé par 04_nlp_vectorization.py

Lancer :
  python code/03_model_selection_mlflow.py
  mlflow ui --port 5000  →  http://localhost:5000
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import subprocess, sys
def _install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["mlflow", "nltk", "scikit-learn", "gensim", "sentence-transformers", "tqdm"]:
    try:
        __import__(pkg.replace("-","_"))
    except ImportError:
        print(f"📦 Installation {pkg}...")
        _install(pkg)

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import re, pickle, time, json, warnings
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

DATA_PROCESSED = Path("data/processed")
VECTORS_DIR    = DATA_PROCESSED / "vectors"
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

# ── Fichier qui stocke le modèle choisi ──────────────────
MODEL_CHOICE_FILE = DATA_PROCESSED / "selected_model.json"

# ══════════════════════════════════════════════════════════
# PROFILS DE RÉFÉRENCE POUR L'ÉVALUATION
# ══════════════════════════════════════════════════════════
# Ces profils ont des "offres attendues" connues
# pour calculer une précision approximative

REFERENCE_PROFILES = [
    {
        "name":     "Data Scientist",
        "text":     "data scientist python machine learning tensorflow scikit-learn nlp deep learning aws",
        "keywords": ["data scientist", "machine learning", "data science", "ml engineer"],
    },
    {
        "name":     "Data Engineer",
        "text":     "data engineer python sql apache spark airflow aws redshift etl pipeline dbt",
        "keywords": ["data engineer", "etl", "pipeline", "spark", "data"],
    },
    {
        "name":     "DevOps Engineer",
        "text":     "devops engineer docker kubernetes terraform aws gcp ci cd linux python infrastructure",
        "keywords": ["devops", "cloud", "infrastructure", "engineer", "kubernetes"],
    },
    {
        "name":     "Product Manager",
        "text":     "product manager agile scrum roadmap user research b2b saas stakeholder",
        "keywords": ["product manager", "product", "manager", "scrum", "agile"],
    },
    {
        "name":     "Software Engineer",
        "text":     "software engineer java python javascript react nodejs microservices rest api backend",
        "keywords": ["software engineer", "developer", "engineer", "backend", "software"],
    },
]


def precision_at_k(results_titles: list, expected_keywords: list, k: int = 10) -> float:
    """
    Calcule Precision@K :
    proportion des K premiers résultats qui contiennent
    au moins un mot-clé attendu.
    """
    relevant = 0
    for title in results_titles[:k]:
        title_lower = title.lower()
        if any(kw in title_lower for kw in expected_keywords):
            relevant += 1
    return relevant / min(k, len(results_titles))


# ══════════════════════════════════════════════════════════
# PRÉPROCESSEUR NLP (inline pour éviter dépendance)
# ══════════════════════════════════════════════════════════

def preprocess_text(text: str) -> str:
    import nltk
    for r in ["stopwords", "wordnet", "punkt", "punkt_tab"]:
        try:
            nltk.data.find(f"corpora/{r}" if r not in ["punkt","punkt_tab"] else f"tokenizers/{r}")
        except LookupError:
            nltk.download(r, quiet=True)

    from nltk.corpus import stopwords
    from nltk.stem   import WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text   = text.lower()
    text   = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in stop_words and len(t) > 2]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════
# MODÈLE 1 — TF-IDF
# ══════════════════════════════════════════════════════════

def evaluate_tfidf(df_sample: pd.DataFrame) -> dict:
    """Entraîne et évalue TF-IDF sur l'échantillon"""
    from sklearn.feature_extraction.text import TfidfVectorizer

    print("   ⚙️  TF-IDF — preprocessing...")
    texts_nlp = [preprocess_text(t) for t in df_sample["text_raw"].tolist()]

    params = dict(max_features=10_000, ngram_range=(1,2),
                  min_df=2, max_df=0.85, sublinear_tf=True)

    t0         = time.time()
    vectorizer = TfidfVectorizer(**params)
    matrix     = vectorizer.fit_transform(texts_nlp)
    train_time = time.time() - t0
    print(f"      Entraîné en {train_time:.1f}s | shape: {matrix.shape}")

    # Évaluer sur les profils de référence
    precisions = []
    inf_times  = []

    for profile in REFERENCE_PROFILES:
        query_nlp = preprocess_text(profile["text"])
        t0        = time.time()
        query_vec = vectorizer.transform([query_nlp])
        scores    = cosine_similarity(query_vec, matrix).flatten()
        top_idx   = np.argsort(scores)[::-1][:10]
        inf_times.append(time.time() - t0)

        top_titles = df_sample.iloc[top_idx]["job_title"].tolist()
        p_at_k     = precision_at_k(top_titles, profile["keywords"], k=10)
        precisions.append(p_at_k)
        print(f"      {profile['name']:<20} P@10={p_at_k:.2f}  "
              f"top1_score={scores[top_idx[0]]:.3f}  "
              f"top1={top_titles[0][:35]}")

    return {
        "model_type":       "tfidf",
        "train_time_s":     round(train_time, 3),
        "avg_precision_at_10": round(np.mean(precisions), 4),
        "avg_inference_ms": round(np.mean(inf_times) * 1000, 2),
        "vocab_size":       len(vectorizer.vocabulary_),
        "matrix_shape":     str(matrix.shape),
        "params":           params,
        "vectorizer":       vectorizer,
        "matrix":           matrix,
        "texts_nlp":        texts_nlp,
    }


# ══════════════════════════════════════════════════════════
# MODÈLE 2 — WORD2VEC
# ══════════════════════════════════════════════════════════

def evaluate_word2vec(df_sample: pd.DataFrame, texts_nlp: list) -> dict:
    """Entraîne et évalue Word2Vec sur l'échantillon"""
    from gensim.models import Word2Vec

    tokenized = [t.split() for t in texts_nlp]
    params    = dict(vector_size=150, window=8, min_count=5,
                     workers=4, epochs=20, sg=0)

    t0    = time.time()
    model = Word2Vec(sentences=tokenized, **params)
    train_time = time.time() - t0
    print(f"      Entraîné en {train_time:.1f}s | vocab: {len(model.wv):,}")

    # Construire la matrice de vecteurs
    def text_to_vec(tokens):
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(params["vector_size"])

    t0     = time.time()
    matrix = np.array([text_to_vec(t) for t in tokenized])
    # Normaliser
    norms  = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms  = np.where(norms == 0, 1, norms)
    matrix = matrix / norms
    build_time = time.time() - t0

    precisions = []
    inf_times  = []

    for profile in REFERENCE_PROFILES:
        query_tokens = preprocess_text(profile["text"]).split()
        t0           = time.time()
        query_vec    = text_to_vec(query_tokens)
        query_norm   = np.linalg.norm(query_vec)
        if query_norm > 0:
            query_vec = query_vec / query_norm
        scores   = matrix @ query_vec
        top_idx  = np.argsort(scores)[::-1][:10]
        inf_times.append(time.time() - t0)

        top_titles = df_sample.iloc[top_idx]["job_title"].tolist()
        p_at_k     = precision_at_k(top_titles, profile["keywords"], k=10)
        precisions.append(p_at_k)
        print(f"      {profile['name']:<20} P@10={p_at_k:.2f}  "
              f"top1_score={scores[top_idx[0]]:.3f}  "
              f"top1={top_titles[0][:35]}")

    return {
        "model_type":          "word2vec",
        "train_time_s":        round(train_time + build_time, 3),
        "avg_precision_at_10": round(np.mean(precisions), 4),
        "avg_inference_ms":    round(np.mean(inf_times) * 1000, 2),
        "vocab_size":          len(model.wv),
        "vector_dim":          params["vector_size"],
        "params":              params,
        "model":               model,
        "matrix":              matrix,
    }


# ══════════════════════════════════════════════════════════
# MODÈLE 3 — SENTENCE-BERT
# ══════════════════════════════════════════════════════════

def evaluate_sbert(df_sample: pd.DataFrame, hf_token: str = None,
                   sample_size_sbert: int = 10_000) -> dict:
    """
    Charge et évalue Sentence-BERT.
    Utilise un échantillon plus grand que TF-IDF/W2V car SBERT
    a besoin de diversité sémantique pour être bien évalué.
    """
    from sentence_transformers import SentenceTransformer
    import os

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token

    # Essayer plusieurs modèles
    models_to_try = ["all-MiniLM-L6-v2", "paraphrase-MiniLM-L3-v2"]
    sbert_model   = None
    model_name    = None

    for name in models_to_try:
        try:
            kwargs     = {"token": hf_token} if hf_token else {}
            sbert_model = SentenceTransformer(name, **kwargs)
            model_name  = name
            print(f"      Modèle chargé : {name}")
            break
        except Exception as e:
            print(f"      ⚠️  {name} : {str(e)[:50]}")

    if not sbert_model:
        raise RuntimeError("Impossible de charger SBERT")

    # Encoder le corpus échantillon
    texts  = df_sample["text_raw"].tolist()
    t0     = time.time()
    matrix = sbert_model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    train_time = time.time() - t0
    print(f"      Encodé en {train_time:.1f}s | shape: {matrix.shape}")

    precisions = []
    inf_times  = []

    for profile in REFERENCE_PROFILES:
        t0        = time.time()
        query_vec = sbert_model.encode(
            [profile["text"]],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        scores    = matrix @ query_vec
        top_idx   = np.argsort(scores)[::-1][:10]
        inf_times.append(time.time() - t0)

        top_titles = df_sample.iloc[top_idx]["job_title"].tolist()
        p_at_k     = precision_at_k(top_titles, profile["keywords"], k=10)
        precisions.append(p_at_k)
        print(f"      {profile['name']:<20} P@10={p_at_k:.2f}  "
              f"top1_score={scores[top_idx[0]]:.3f}  "
              f"top1={top_titles[0][:35]}")

    return {
        "model_type":          "sbert",
        "model_name":          model_name,
        "train_time_s":        round(train_time, 3),
        "avg_precision_at_10": round(np.mean(precisions), 4),
        "avg_inference_ms":    round(np.mean(inf_times) * 1000, 2),
        "embedding_dim":       matrix.shape[1],
        "params":              {"model": model_name, "normalize": True, "batch_size": 64},
        "sbert_model":         sbert_model,
        "matrix":              matrix,
    }


# ══════════════════════════════════════════════════════════
# LOGGING MLFLOW
# ══════════════════════════════════════════════════════════

def log_experiment(eval_results: dict, experiment_name: str = "model-selection"):
    """Logue une expérience dans MLflow"""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=eval_results["model_type"].upper()):

        # Paramètres
        mlflow.log_params({
            "model_type":   eval_results["model_type"],
            "sample_size":  eval_results.get("sample_size", 0),
            **{k: str(v) for k, v in eval_results.get("params", {}).items()
               if k not in ["vectorizer", "model", "matrix"]},
        })

        # Métriques principales
        mlflow.log_metrics({
            "avg_precision_at_10": eval_results["avg_precision_at_10"],
            "avg_inference_ms":    eval_results["avg_inference_ms"],
            "train_time_s":        eval_results["train_time_s"],
        })

        # Métriques spécifiques
        if "vocab_size" in eval_results:
            mlflow.log_metric("vocab_size", eval_results["vocab_size"])
        if "embedding_dim" in eval_results:
            mlflow.log_metric("embedding_dim", eval_results["embedding_dim"])

        run_id = mlflow.active_run().info.run_id

    return run_id


# ══════════════════════════════════════════════════════════
# SÉLECTION AUTOMATIQUE DU MEILLEUR MODÈLE
# ══════════════════════════════════════════════════════════

def select_best_model(results: list) -> dict:
    """
    Sélectionne le meilleur modèle selon un score composite :
    Score = 0.80 × Precision@10 + 0.20 × (1 / log(inference_ms + 1))
    Priorité : qualité (80%) > vitesse (20%)
    """
    scored = []
    for r in results:
        quality_score = r["avg_precision_at_10"]
        speed_score   = 1.0 / np.log(r["avg_inference_ms"] + 2)
        composite     = 0.80 * quality_score + 0.20 * speed_score
        scored.append({**r, "composite_score": round(composite, 4)})

    best = max(scored, key=lambda x: x["composite_score"])
    return best, scored


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("ÉTAPE 3 : SÉLECTION DU MODÈLE AVEC MLFLOW")
    print("(tourne AVANT la vectorisation finale)")
    print("="*60)

    # ── Token HF ─────────────────────────────────────────
    HF_TOKEN = None  # "hf_XXXXXXXX" si besoin

    # ── MLflow URI ────────────────────────────────────────
    import os
    _mlruns = Path("mlruns").absolute()
    _mlruns.mkdir(parents=True, exist_ok=True)
    # Windows : utiliser chemin relatif "mlruns" (pas file://)
    # Linux/Mac : utiliser chemin absolu
    if os.name == "nt":
        mlflow.set_tracking_uri("mlruns")
    else:
        mlflow.set_tracking_uri(str(_mlruns))
    print(f"\n📂 MLflow URI : {mlflow.get_tracking_uri()}")

    # ── Charger les données ───────────────────────────────
    print("\n📥 Chargement des données nettoyées...")
    pkl_path = DATA_PROCESSED / "jobs_cleaned.pkl"
    csv_path = DATA_PROCESSED / "jobs_cleaned.csv"

    if pkl_path.exists():
        df = pd.read_pickle(str(pkl_path))
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError("Lance d'abord 02_clean_data.py")

    print(f"   Corpus complet : {len(df):,} offres")

    # ── Échantillon STRATIFIÉ ────────────────────────────
    # Stratifié par job_title pour garder la diversité
    # TF-IDF et Word2Vec : 5k offres (rapide)
    # SBERT              : 10k offres stratifiées (plus représentatif)
    SAMPLE_SIZE = 5_000

    # Extraire la catégorie principale du titre (premier mot)
    df["title_cat"] = df["job_title"].str.lower().str.split().str[0].fillna("other")

    # Échantillon stratifié : garder les proportions par catégorie
    try:
        df_sample = df.groupby("title_cat", group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(SAMPLE_SIZE * len(x) / len(df))),
                random_state=42
            )
        ).reset_index(drop=True)
        # Ajuster à SAMPLE_SIZE exact
        if len(df_sample) > SAMPLE_SIZE:
            df_sample = df_sample.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
    except Exception:
        df_sample = df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)

    print(f"   Échantillon    : {len(df_sample):,} offres (stratifié par titre de poste)")
    print(f"   Catégories     : {df_sample['title_cat'].nunique()} titres distincts")

    # Construire le champ texte
    def build_text(row):
        title = str(row.get("job_title", ""))
        loc   = str(row.get("location",  ""))
        exp   = str(row.get("experience_level", ""))
        return f"{title} {title} {title} {loc} {exp}".strip()

    df_sample["text_raw"] = df_sample.apply(build_text, axis=1)

    # ══════════════════════════════════════════════════════
    # ÉVALUATION DES 3 MODÈLES
    # ══════════════════════════════════════════════════════

    all_results = []

    # ── Modèle 1 : TF-IDF ────────────────────────────────
    print("\n" + "─"*60)
    print("📊 MODÈLE 1 : TF-IDF (baseline)")
    print("─"*60)
    tfidf_res = evaluate_tfidf(df_sample)
    tfidf_res["sample_size"] = len(df_sample)
    tfidf_run = log_experiment(tfidf_res)
    print(f"   📋 MLflow run_id : {tfidf_run}")
    all_results.append(tfidf_res)

    # ── Modèle 2 : Word2Vec ───────────────────────────────
    print("\n" + "─"*60)
    print("🧠 MODÈLE 2 : Word2Vec")
    print("─"*60)
    w2v_res = evaluate_word2vec(df_sample, tfidf_res["texts_nlp"])
    w2v_res["sample_size"] = len(df_sample)
    w2v_run = log_experiment(w2v_res)
    print(f"   📋 MLflow run_id : {w2v_run}")
    all_results.append(w2v_res)

    # ── Modèle 3 : SBERT ─────────────────────────────────
    print("\n" + "─"*60)
    print("🚀 MODÈLE 3 : Sentence-BERT")
    print("─"*60)
    # SBERT évalué sur 10k offres stratifiées pour être juste
    SBERT_SAMPLE = 10_000
    try:
        df_sbert = df.groupby("title_cat", group_keys=False).apply(
            lambda x: x.sample(
                n=max(1, int(SBERT_SAMPLE * len(x) / len(df))),
                random_state=42
            )
        ).reset_index(drop=True)
        if len(df_sbert) > SBERT_SAMPLE:
            df_sbert = df_sbert.sample(n=SBERT_SAMPLE, random_state=42).reset_index(drop=True)
    except Exception:
        df_sbert = df.sample(n=SBERT_SAMPLE, random_state=42).reset_index(drop=True)
    df_sbert["text_raw"] = df_sbert.apply(build_text, axis=1)
    print(f"   Échantillon SBERT : {len(df_sbert):,} offres (stratifié)")
    sbert_res = evaluate_sbert(df_sbert, hf_token=HF_TOKEN)
    sbert_res["sample_size"] = len(df_sample)
    sbert_run = log_experiment(sbert_res)
    print(f"   📋 MLflow run_id : {sbert_run}")
    all_results.append(sbert_res)

    # ══════════════════════════════════════════════════════
    # COMPARAISON & SÉLECTION
    # ══════════════════════════════════════════════════════

    print("\n" + "="*60)
    print("📊 COMPARAISON DES MODÈLES")
    print("="*60)

    best_model, scored = select_best_model(all_results)

    print(f"\n{'Modèle':<15} {'P@10':>8} {'Inference':>12} {'Score':>10}")
    print("─" * 50)
    for r in sorted(scored, key=lambda x: x["composite_score"], reverse=True):
        marker = " ✅ CHOISI" if r["model_type"] == best_model["model_type"] else ""
        print(f"{r['model_type']:<15} "
              f"{r['avg_precision_at_10']:>8.3f} "
              f"{r['avg_inference_ms']:>10.1f}ms "
              f"{r['composite_score']:>10.4f}"
              f"{marker}")

    # ── Sauvegarder le choix ──────────────────────────────
    model_choice = {
        "selected_model":     best_model["model_type"],
        "model_name":         best_model.get("model_name", best_model["model_type"]),
        "avg_precision_at_10": best_model["avg_precision_at_10"],
        "avg_inference_ms":   best_model["avg_inference_ms"],
        "composite_score":    best_model["composite_score"],
        "sample_size":        len(df_sample),
        "all_scores":         [
            {"model": r["model_type"], "precision": r["avg_precision_at_10"],
             "inference_ms": r["avg_inference_ms"], "score": r["composite_score"]}
            for r in scored
        ],
    }

    with open(MODEL_CHOICE_FILE, "w") as f:
        json.dump(model_choice, f, indent=2)

    print(f"\n💾 Choix sauvegardé : {MODEL_CHOICE_FILE}")

    # ── Loguer le choix final dans MLflow ─────────────────
    mlflow.set_experiment("model-selection")
    with mlflow.start_run(run_name="BEST-MODEL-SELECTED"):
        mlflow.log_params({
            "selected_model": best_model["model_type"],
            "selection_criteria": "0.80*precision@10 + 0.20*speed — echantillon stratifie",
            "sample_size": len(df_sample),
        })
        mlflow.log_metrics({
            "best_precision_at_10": best_model["avg_precision_at_10"],
            "best_inference_ms":    best_model["avg_inference_ms"],
            "best_composite_score": best_model["composite_score"],
        })
        mlflow.log_artifact(str(MODEL_CHOICE_FILE))

    # ── Résumé final ──────────────────────────────────────
    print("\n" + "="*60)
    print("✅ SÉLECTION TERMINÉE")
    print("="*60)
    print(f"""
🏆 Modèle sélectionné : {best_model['model_type'].upper()}
   Precision@10       : {best_model['avg_precision_at_10']:.3f}
   Inference          : {best_model['avg_inference_ms']:.1f}ms
   Score composite    : {best_model['composite_score']:.4f}

🌐 Voir le dashboard MLflow :
   mlflow ui --port 5000
   → http://localhost:5000

🚀 Prochaine étape : python code/04_nlp_vectorization.py
   (utilisera automatiquement {best_model['model_type'].upper()})
""")

    return model_choice


if __name__ == "__main__":
    choice = main()