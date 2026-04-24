"""
Étape 4 : NLP & Vectorisation
VERSION CORRIGÉE — Fix token HuggingFace expiré pour SBERT

PROBLÈME : User Access Token "llama2" is expired
SOLUTION  : 3 stratégies dans l'ordre :
  1. Utiliser HF_TOKEN si fourni
  2. Supprimer le token expiré du cache et retenter sans auth
  3. Fallback sur paraphrase-MiniLM-L3-v2 (modèle alternatif léger)
"""

import pandas as pd
import numpy as np
import re
import pickle
import time
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DATA_PROCESSED = Path("data/processed")
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
VECTORS_DIR = DATA_PROCESSED / "vectors"
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 🔑 TON TOKEN HUGGINGFACE (optionnel mais recommandé)
#    https://huggingface.co/settings/tokens  → New token → Read
# ══════════════════════════════════════════════════════════════
HF_TOKEN = (os.getenv("HF_TOKEN"  )) # Ex: "hf_abcXXXXXXXXXXXXXXXXXXXXXXXXXX"
# ══════════════════════════════════════════════════════════════


def fix_hf_token_cache():
    """
    Supprime les tokens expirés du cache HuggingFace pour éviter
    l'erreur 401 avec un token périmé comme 'llama2'
    """
    import subprocess, sys

    # Méthode 1 : via huggingface_hub
    try:
        from huggingface_hub import logout
        logout()
        print("   ✅ Token HF expiré supprimé du cache")
    except Exception:
        pass

    # Méthode 2 : supprimer le fichier token manuellement
    token_paths = [
        Path.home() / ".huggingface" / "token",
        Path.home() / ".cache" / "huggingface" / "token",
        Path(os.environ.get("HF_HOME", "")) / "token" if os.environ.get("HF_HOME") else None,
    ]
    for p in token_paths:
        if p and p.exists():
            try:
                p.unlink()
                print(f"   ✅ Token supprimé : {p}")
            except Exception:
                pass

    # Méthode 3 : variable d'environnement
    os.environ.pop("HUGGING_FACE_HUB_TOKEN", None)
    os.environ.pop("HUGGINGFACE_HUB_TOKEN", None)
    os.environ.pop("HF_TOKEN", None)

    # Appliquer le nouveau token si fourni
    if HF_TOKEN and HF_TOKEN.startswith("hf_"):
        os.environ["HF_TOKEN"] = HF_TOKEN
        try:
            from huggingface_hub import login
            login(token=HF_TOKEN, add_to_git_credential=False)
            print(f"   ✅ Nouveau token appliqué")
        except Exception as e:
            print(f"   ⚠️  Token login: {e}")


def check_and_install():
    import subprocess, sys
    packages = {
        "nltk":                  "nltk",
        "sklearn":               "scikit-learn",
        "gensim":                "gensim",
        "sentence_transformers": "sentence-transformers",
        "tqdm":                  "tqdm",
    }
    for import_name, pip_name in packages.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"   📦 Installation de {pip_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name, "-q"])

    import nltk
    for resource in ["punkt", "stopwords", "wordnet", "omw-1.4", "punkt_tab"]:
        try:
            nltk.data.find(
                f"tokenizers/{resource}" if "punkt" in resource else f"corpora/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


# ══════════════════════════════════════════════════════════════
# NLP PREPROCESSOR
# ══════════════════════════════════════════════════════════════

class NLPPreprocessor:
    def __init__(self, language="english"):
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words(language))
        self.stop_words.update([
            "job", "work", "company", "team", "experience", "year", "years",
            "skill", "skills", "ability", "looking", "seeking", "candidate",
            "position", "role", "opportunity", "responsibilities", "requirements",
            "qualifications", "preferred", "plus", "including", "etc",
            "must", "will", "also", "may", "well", "within", "across"
        ])

    def clean(self, text):
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        text = re.sub(r"\b\d+\b", " ", text)
        text = re.sub(r"[^a-z0-9\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize_and_lemmatize(self, text):
        from nltk.tokenize import word_tokenize
        cleaned = self.clean(text)
        if not cleaned:
            return []
        tokens = word_tokenize(cleaned)
        return [
            self.lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in self.stop_words and len(tok) > 2 and not tok.isdigit()
        ]

    def process(self, text):
        return " ".join(self.tokenize_and_lemmatize(text))

    def process_batch(self, texts, show_progress=True):
        from tqdm import tqdm
        if show_progress:
            return [self.process(t) for t in tqdm(texts, desc="   NLP preprocessing", ncols=80)]
        return [self.process(t) for t in texts]


# ══════════════════════════════════════════════════════════════
# TEXT FIELD BUILDER
# ══════════════════════════════════════════════════════════════

def build_text_field(df):
    def row_to_text(row):
        parts = []
        title = str(row.get("job_title", "")).strip()
        if title and title.lower() not in ("nan", "unknown", ""):
            parts.extend([title] * 3)
        desc = str(row.get("description", "")).strip()
        if desc and desc.lower() not in ("nan", "none", ""):
            parts.append(desc[:1000])
        skills = str(row.get("skills_desc", "")).strip()
        if skills and skills.lower() not in ("nan", "none", ""):
            parts.extend([skills] * 2)
        exp = str(row.get("experience_level", "")).strip()
        if exp and exp not in ("unknown", "nan"):
            parts.append(exp)
        loc = str(row.get("location", "")).strip()
        if loc and loc.lower() not in ("unknown", "nan", ""):
            parts.append(loc)
        return " ".join(parts) if parts else title
    return df.apply(row_to_text, axis=1)


# ══════════════════════════════════════════════════════════════
# TF-IDF
# ══════════════════════════════════════════════════════════════

class TFIDFVectorizer:
    def __init__(self, max_features=10_000, ngram_range=(1, 2)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
        )

    def fit_transform(self, texts):
        print("   ⚙️  Entraînement TF-IDF...")
        matrix = self.vectorizer.fit_transform(texts)
        print(f"   ✅ Matrice TF-IDF : {matrix.shape}")
        return matrix

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"   💾 TF-IDF vectorizer sauvegardé : {path}")

    def get_top_terms(self, n=10):
        names = self.vectorizer.get_feature_names_out()
        idfs  = self.vectorizer.idf_
        top   = np.argsort(idfs)[:n]
        return [(names[i], round(idfs[i], 3)) for i in top]


# ══════════════════════════════════════════════════════════════
# WORD2VEC — VERSION AMÉLIORÉE
# ══════════════════════════════════════════════════════════════
#
# CAUSES DU BRUIT PRÉCÉDENT :
#   1. Textes trop courts (médiane 12 tokens) → contexte insuffisant
#      FIX : augmenter la fenêtre de contexte (window=8) ET
#            enrichir chaque document en répétant les tokens importants
#   2. min_count=2 trop bas → inclut les noms propres/typos rares
#      FIX : min_count=5 pour éliminer les termes aberrants
#   3. Skip-gram avec peu de données → instable
#      FIX : CBOW (sg=0) + epochs=20 + négatif sampling plus fort
#   4. Moyenne simple des vecteurs → noie les mots importants
#      FIX : moyenne pondérée par TF-IDF (mots rares = plus de poids)
# ══════════════════════════════════════════════════════════════

class Word2VecVectorizer:
    def __init__(self, vector_size=150, window=8, min_count=5, workers=4, epochs=20):
        self.params = dict(
            vector_size=vector_size,
            window=window,          # fenêtre plus large pour textes courts
            min_count=min_count,    # ignorer les termes trop rares (typos, noms propres)
            workers=workers,
            epochs=epochs,          # plus d'itérations = meilleure convergence
            sg=0,                   # CBOW : plus stable sur petits corpus
            hs=0,                   # negative sampling
            negative=10,            # 10 échantillons négatifs
            ns_exponent=0.75,       # distribution standard negative sampling
            alpha=0.025,            # learning rate initial
            min_alpha=0.0001,       # learning rate final
        )
        self.model        = None
        self.word_weights = {}      # poids TF-IDF par mot (pour moyenne pondérée)

    def _build_tfidf_weights(self, tokenized_texts):
        """
        Calcule les poids IDF de chaque mot pour la moyenne pondérée.
        Les mots rares (discriminants) auront plus de poids que les mots communs.
        """
        from math import log
        doc_count    = len(tokenized_texts)
        word_doc_cnt = {}

        for tokens in tokenized_texts:
            for word in set(tokens):     # set() → compter 1 fois par doc
                word_doc_cnt[word] = word_doc_cnt.get(word, 0) + 1

        # IDF = log(N / df) — plus le mot est rare, plus il a de poids
        self.word_weights = {
            word: log(doc_count / (cnt + 1))
            for word, cnt in word_doc_cnt.items()
        }
        print(f"   📊 Poids TF-IDF calculés pour {len(self.word_weights):,} mots")

    def _augment_tokens(self, tokens):
        """
        Enrichit les tokens en répétant les mots importants (IDF élevé).
        Compense la faible longueur des textes de postes.
        """
        if not tokens:
            return tokens

        avg_weight = np.mean(list(self.word_weights.values())) if self.word_weights else 1.0
        augmented  = list(tokens)  # copie

        for tok in tokens:
            weight = self.word_weights.get(tok, avg_weight)
            # Si le mot est très discriminant (IDF élevé), le répéter
            if weight > avg_weight * 1.5:
                augmented.append(tok)   # répéter 1 fois supplémentaire

        return augmented

    def fit(self, tokenized_texts):
        from gensim.models import Word2Vec

        print(f"   ⚙️  Calcul des poids TF-IDF pour pondération...")
        self._build_tfidf_weights(tokenized_texts)

        # Enrichir les textes courts avant entraînement
        print(f"   ⚙️  Enrichissement des tokens courts...")
        augmented_texts = [self._augment_tokens(t) for t in tokenized_texts]

        avg_len_before = np.mean([len(t) for t in tokenized_texts])
        avg_len_after  = np.mean([len(t) for t in augmented_texts])
        print(f"      Longueur moyenne : {avg_len_before:.1f} → {avg_len_after:.1f} tokens")

        print(f"   ⚙️  Entraînement Word2Vec (CBOW, window={self.params['window']}, "
              f"min_count={self.params['min_count']}, epochs={self.params['epochs']})...")
        self.model = Word2Vec(sentences=augmented_texts, **self.params)

        vocab_size = len(self.model.wv)
        print(f"   ✅ Vocabulaire : {vocab_size:,} mots "
              f"(vs 16 623 avant — moins de bruit grâce à min_count={self.params['min_count']})")
        return self

    def transform(self, tokenized_texts):
        """
        Moyenne pondérée par IDF : les mots rares/discriminants ont plus de poids
        que les mots communs comme 'manager' ou 'senior'
        """
        from tqdm import tqdm

        avg_weight = np.mean(list(self.word_weights.values())) if self.word_weights else 1.0
        vectors    = []

        for tokens in tqdm(tokenized_texts, desc="   Word2Vec transform (IDF-weighted)", ncols=80):
            valid_tokens = [t for t in tokens if t in self.model.wv]

            if not valid_tokens:
                vectors.append(np.zeros(self.params["vector_size"]))
                continue

            # Récupérer vecteurs et poids IDF
            vecs    = np.array([self.model.wv[t] for t in valid_tokens])
            weights = np.array([self.word_weights.get(t, avg_weight) for t in valid_tokens])
            weights = np.maximum(weights, 1e-8)   # éviter division par zéro

            # Moyenne pondérée
            weighted_mean = np.average(vecs, axis=0, weights=weights)
            vectors.append(weighted_mean)

        result = np.array(vectors)
        print(f"   ✅ Matrice Word2Vec : {result.shape}")
        return result

    def save(self, path):
        self.model.save(str(path))
        # Sauvegarder aussi les poids IDF
        weights_path = str(path).replace(".bin", "_idf_weights.pkl")
        with open(weights_path, "wb") as f:
            pickle.dump(self.word_weights, f)
        print(f"   💾 Word2Vec sauvegardé : {path}")
        print(f"   💾 Poids IDF sauvegardés : {weights_path}")

    def most_similar(self, word, topn=5):
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []

    def analogy_test(self):
        """
        Test d'analogie sémantique :
        'senior' - 'junior' + 'engineer' ≈ ? (devrait donner 'lead engineer')
        """
        tests = [
            ("senior", "junior", "engineer"),
            ("python", "java",   "developer"),
            ("manager", "analyst", "data"),
        ]
        print(f"\n   🔬 Tests d'analogie (A - B + C ≈ ?) :")
        for a, b, c in tests:
            if all(w in self.model.wv for w in [a, b, c]):
                try:
                    result = self.model.wv.most_similar(
                        positive=[a, c], negative=[b], topn=3
                    )
                    res_str = ", ".join([f"{w}({s:.2f})" for w, s in result])
                    print(f"      '{a}' - '{b}' + '{c}' → {res_str}")
                except Exception:
                    pass


# ══════════════════════════════════════════════════════════════
# SENTENCE-BERT — VERSION ROBUSTE AVEC FALLBACK
# ══════════════════════════════════════════════════════════════

# Modèles SBERT par ordre de préférence
# Tous sont publics, légers et efficaces pour le matching
SBERT_MODELS = [
    "all-MiniLM-L6-v2",           # 80MB — le meilleur rapport qualité/vitesse
    "paraphrase-MiniLM-L3-v2",    # 60MB — plus léger, presque aussi bon
    "all-MiniLM-L12-v2",          # 120MB — légèrement meilleur que L6
    "paraphrase-albert-small-v2", # 45MB — très léger
]


class SentenceBERTVectorizer:
    def __init__(self, batch_size=64):
        self.batch_size  = batch_size
        self.model       = None
        self.model_name  = None

    def load_model(self):
        """
        Tente de charger SBERT avec plusieurs stratégies :
        1. Nettoyer le token expiré du cache
        2. Essayer chaque modèle de la liste
        3. Utiliser HF_TOKEN si disponible
        """
        from sentence_transformers import SentenceTransformer

        # ── Étape 1 : nettoyer le token expiré ─────────────
        print("   🔧 Nettoyage du token HF expiré...")
        fix_hf_token_cache()

        # ── Étape 2 : essayer chaque modèle ─────────────────
        for model_name in SBERT_MODELS:
            for use_token in ([HF_TOKEN, None] if HF_TOKEN else [None]):
                try:
                    print(f"   ⏳ Chargement : {model_name}"
                          + (" (avec token)" if use_token else " (sans token)") + "...")

                    kwargs = {}
                    if use_token:
                        kwargs["token"] = use_token

                    self.model      = SentenceTransformer(model_name, **kwargs)
                    self.model_name = model_name
                    print(f"   ✅ Modèle chargé : {model_name}")
                    return self

                except Exception as e:
                    err = str(e)
                    if "401" in err or "expired" in err.lower() or "token" in err.lower():
                        print(f"   ⚠️  Erreur token pour {model_name} — on nettoie et on retente")
                        fix_hf_token_cache()
                    elif "404" in err or "not found" in err.lower():
                        print(f"   ⚠️  Modèle {model_name} introuvable, essai suivant...")
                        break
                    else:
                        print(f"   ⚠️  Erreur {model_name}: {err[:80]}...")

        raise RuntimeError(
            "❌ Impossible de charger un modèle SBERT.\n"
            "   Solutions :\n"
            "   1. Crée un token sur https://huggingface.co/settings/tokens\n"
            "      et colle-le dans HF_TOKEN en haut du fichier\n"
            "   2. Désactiver l'ancien token : huggingface-cli logout\n"
            "   3. Lancer : pip install -U sentence-transformers"
        )

    def transform(self, texts):
        if self.model is None:
            self.load_model()

        print(f"   ⚙️  Encodage SBERT ({self.model_name}) — {len(texts):,} textes...")
        t0 = time.time()

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        elapsed = time.time() - t0
        print(f"   ✅ Matrice SBERT : {embeddings.shape} | Temps : {elapsed:.1f}s")
        return embeddings

    def top_k_similar(self, query_vec, corpus_vecs, k=10):
        scores = corpus_vecs @ query_vec
        top_k  = np.argsort(scores)[::-1][:k]
        return [(int(i), float(scores[i])) for i in top_k]


# ══════════════════════════════════════════════════════════════
# DÉMONSTRATION MATCHING
# ══════════════════════════════════════════════════════════════

def demo_matching(df, sbert_vecs, sbert_model):
    candidates = [
        {
            "name":    "Alice — Data Scientist senior",
            "profile": (
                "Senior data scientist with 7 years in Python, machine learning, "
                "deep learning, TensorFlow, scikit-learn. NLP, MLOps, A/B testing."
            )
        },
        {
            "name":    "Bob — Data Engineer junior",
            "profile": (
                "Junior data engineer, 2 years experience. SQL, Apache Spark, "
                "Airflow, AWS S3, Redshift, Python, dbt, ETL pipelines."
            )
        },
        {
            "name":    "Carol — Product Manager Tech",
            "profile": (
                "Product manager 5 years. Agile, Scrum, roadmap, user research, "
                "A/B testing. SaaS, engineering collaboration."
            )
        },
    ]

    print("\n" + "─"*60)
    print("🎯 DÉMONSTRATION MATCHING CANDIDAT ↔ OFFRES (SBERT)")
    print("─"*60)

    for cand in candidates:
        print(f"\n👤 {cand['name']}")

        query_vec = sbert_model.transform([cand["profile"]])[0]
        scores    = sbert_vecs @ query_vec
        top5      = np.argsort(scores)[::-1][:5]

        print(f"   🏆 Top 5 offres :")
        for rank, idx in enumerate(top5, 1):
            row   = df.iloc[idx]
            score = scores[idx]
            sal   = f"${row['salary_usd']:,.0f}" if row.get("has_salary", 0) == 1 else "N/A"
            title = str(row["job_title"])[:40]
            loc   = str(row["location"])[:18]
            print(f"   {rank}. [{score:.3f}] {title:<42} {loc:<20} {sal}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*60)
    print("ÉTAPE 4 : NLP & VECTORISATION")
    print("="*60)

    print("\n📦 Vérification des dépendances...")
    check_and_install()
    print("   ✅ Toutes les librairies disponibles")

    # ── Charger les données ───────────────────────────
    pkl_path = DATA_PROCESSED / "jobs_cleaned.pkl"
    csv_path = DATA_PROCESSED / "jobs_cleaned.csv"
    if pkl_path.exists():
        df = pd.read_pickle(pkl_path)
        print(f"\n📥 Données chargées (PKL): {df.shape}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"\n📥 Données chargées (CSV): {df.shape}")
    else:
        print("❌ jobs_cleaned introuvable. Lance d'abord 02_clean_data.py")
        return None

    # ── Champ texte ───────────────────────────────────
    print("\n📝 Construction du champ texte combiné...")
    df["text_raw"] = build_text_field(df)
    df = df[df["text_raw"].str.strip() != ""].reset_index(drop=True)
    lengths = df["text_raw"].str.len()
    print(f"   {len(df):,} offres | longueur médiane : {lengths.median():.0f} chars")

    # ══════════════════════════════════════════════════
    # A — PREPROCESSING NLP
    # ══════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("🧹 A. PREPROCESSING NLP")
    print("─"*60)

    preprocessor   = NLPPreprocessor()
    df["text_nlp"] = preprocessor.process_batch(df["text_raw"].tolist())

    tokens_count = df["text_nlp"].str.split().str.len()
    print(f"   Tokens — min:{tokens_count.min()} max:{tokens_count.max()} "
          f"median:{tokens_count.median():.0f}")

    tokenized_texts = [t.split() for t in df["text_nlp"].tolist()]

    # ══════════════════════════════════════════════════
    # B — TF-IDF
    # ══════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("📊 B. TF-IDF VECTORISATION")
    print("─"*60)

    tfidf_model  = TFIDFVectorizer()
    tfidf_matrix = tfidf_model.fit_transform(df["text_nlp"].tolist())

    print(f"   🔑 Top 10 termes communs :")
    for term, score in tfidf_model.get_top_terms(10):
        print(f"      {term:<30} IDF={score}")

    import scipy.sparse as sp
    sp.save_npz(str(VECTORS_DIR / "tfidf_matrix.npz"), tfidf_matrix)
    tfidf_model.save(VECTORS_DIR / "tfidf_vectorizer.pkl")

    # ══════════════════════════════════════════════════
    # C — WORD2VEC
    # ══════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("🧠 C. WORD2VEC EMBEDDINGS")
    print("─"*60)

    w2v_model  = Word2VecVectorizer(vector_size=150, window=8, min_count=5, epochs=20)
    w2v_model.fit(tokenized_texts)
    w2v_matrix = w2v_model.transform(tokenized_texts)

    print(f"\n   🔍 Test sémantique (résultats propres sans noms propres/typos) :")
    for word in ["python", "data", "engineer", "manager"]:
        sim = w2v_model.most_similar(word, topn=5)
        if sim:
            s = ", ".join([f"{w}({v:.2f})" for w, v in sim])
            print(f"      '{word}' → {s}")

    w2v_model.analogy_test()

    np.save(str(VECTORS_DIR / "w2v_matrix.npy"), w2v_matrix)
    w2v_model.save(VECTORS_DIR / "w2v_model.bin")

    # ══════════════════════════════════════════════════
    # D — SENTENCE-BERT (stratégie robuste)
    # ══════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("🚀 D. SENTENCE-BERT EMBEDDINGS")
    print("─"*60)

    sbert_model  = SentenceBERTVectorizer(batch_size=64)
    sbert_matrix = sbert_model.transform(df["text_raw"].tolist())

    np.save(str(VECTORS_DIR / "sbert_matrix.npy"), sbert_matrix)
    print(f"   💾 SBERT sauvegardé : {VECTORS_DIR / 'sbert_matrix.npy'}")

    # ══════════════════════════════════════════════════
    # E — SAUVEGARDER LE DATASET ENRICHI
    # ══════════════════════════════════════════════════
    print("\n" + "─"*60)
    print("💾 E. SAUVEGARDE")
    print("─"*60)

    df["vector_idx"] = range(len(df))
    df.to_pickle(str(DATA_PROCESSED / "jobs_vectorized.pkl"))
    df.drop(columns=["text_raw", "text_nlp"], errors="ignore").to_csv(
        DATA_PROCESSED / "jobs_vectorized.csv", index=False, encoding="utf-8-sig"
    )
    print(f"   ✅ jobs_vectorized.pkl + .csv sauvegardés")

    # ══════════════════════════════════════════════════
    # F — DÉMO MATCHING
    # ══════════════════════════════════════════════════
    demo_matching(df, sbert_matrix, sbert_model)

    # ══════════════════════════════════════════════════
    # RÉSUMÉ
    # ══════════════════════════════════════════════════
    print("\n" + "="*60)
    print("✅ ÉTAPE 4 COMPLÉTÉE!")
    print("="*60)
    print(f"""
📊 Vecteurs produits :
   TF-IDF       : {tfidf_matrix.shape}  → tfidf_matrix.npz
   Word2Vec     : {w2v_matrix.shape}  → w2v_matrix.npy
   SBERT ✅     : {sbert_matrix.shape}  → sbert_matrix.npy
   Modèle SBERT : {sbert_model.model_name}

🚀 Prochaine étape : python code/05_matching_engine.py
""")

    return {
        "df": df, "tfidf_matrix": tfidf_matrix,
        "w2v_matrix": w2v_matrix, "sbert_matrix": sbert_matrix,
        "tfidf_model": tfidf_model, "w2v_model": w2v_model,
        "sbert_model": sbert_model,
    }


if __name__ == "__main__":
    results = main()