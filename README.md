# ⚡ Job Candidate Matching System

> Système intelligent de matching CV ↔ Offres d'emploi basé sur **Sentence-BERT** — Universel, Multilingue, Ultra-rapide.

---

## 📋 Table des matières

- [Description](#-description)
- [Architecture globale](#-architecture-globale)
- [Pipeline ML — Étapes](#-pipeline-ml--étapes)
- [Technologies utilisées](#-technologies-utilisées)
- [Structure des fichiers](#-structure-des-fichiers)
- [Installation & Lancement](#-installation--lancement)
- [Déploiement Docker](#-déploiement-docker)
- [API Endpoints](#-api-endpoints)
- [Interface Angular](#-interface-angular)
- [Évaluation des modèles](#-évaluation-des-modèles)

---

## 📌 Description

**Job Candidate Matching System** est un pipeline complet de Data Science et ML qui associe automatiquement un candidat aux offres d'emploi les plus pertinentes grâce à la similarité sémantique.

### Fonctionnalités principales

- 🎯 **Matching universel** — Tech, Finance, Marketing, RH, Droit, Santé, Management...
- 🌍 **Multilingue** — Détection automatique Français / Anglais / Arabe / mixte
- ⚡ **Ultra-rapide** — Résultats en < 1 seconde sur **112 796 offres**
- 📄 **Upload CV** — Parsing automatique PDF, DOCX, TXT via Groq LLM (`llama-3.3-70b-versatile`)
- ✍️ **Texte libre** — Matching depuis un résumé de profil
- 📊 **MLflow** — Tracking des expériences et comparaison des modèles
- 🐳 **Docker** — Déploiement containerisé complet

---

## 🏗 Architecture globale

```
Données brutes (HuggingFace + RemoteOK + Synthétiques)
        │
        ▼
   01_load_data.py     → Chargement & fusion des sources
        │
        ▼
   02_clean_data.py    → Nettoyage & normalisation
        │
        ▼
   03_explore_data.py  → EDA + visualisations + rapport PDF
        │
        ▼
   03_model_selection_mlflow.py  → Comparaison TF-IDF / Word2Vec / SBERT
        │
        ▼
   04_nlp_vectorization.py  → Preprocessing NLP + vectorisation
        │
        ▼
   matching_engine.py  → Moteur SBERT (similarité cosinus + filtres)
        │
        ▼
   cv_parser.py        → Parsing CV via Groq API
        │
        ▼
   08_api.py           → API REST FastAPI
        │
        ▼
   Frontend Angular 20 → Interface utilisateur
```

---

## 🔬 Pipeline ML — Étapes

### Étape 1 — Chargement des données (`01_load_data.py`)

Fusionne **4 sources** de données :

| Source | Description | Taille estimée |
|---|---|---|
| `hugginglearners/data-science-job-salaries` | Offres Data Science avec salaires | ~600 offres |
| `datastax/linkedin_job_listings` | Offres LinkedIn multi-domaines | Variable |
| `RemoteOK API` | Offres 100% remote (API publique) | Variable |
| **Synthétiques** | Offres générées localement (BI, ML, Finance, RH...) | **1 800 offres** |

> Les offres synthétiques compensent les lacunes du corpus sur les profils sous-représentés : BI junior, Finance, RH, Marketing, Odoo/ERP.

```bash
python code/01_load_data.py
```

**Sortie :** `data/raw/jobs_merged.csv`

---

### Étape 2 — Nettoyage des données (`02_clean_data.py`)

Normalisation complète du corpus :

- Standardisation des **niveaux d'expérience** : `EN/MI/SE/EX` → `entry/mid/senior/executive`
- Correction des **salaires aberrants** (valeurs > $500k → tentative de conversion mensuel→annuel)
- Nettoyage des **titres de postes** (caractères spéciaux, espaces multiples)
- Normalisation du **remote ratio** : continu → discret (0 / 50 / 100%)
- Flag `has_salary` : 1 si salaire > $1 000

```bash
python code/02_clean_data.py
```

**Sortie :** `data/processed/jobs_cleaned.csv` + `.pkl`

---

### Étape 3 — Analyse Exploratoire (`03_explore_data.py`)

Génère automatiquement :

- **4 visualisations PNG** : distribution salaires (histogramme + violin + KDE + boxplot), niveaux d'expérience, top 15 localisations, remote ratio
- **Rapport TXT** : statistiques détaillées du corpus
- **Rapport PDF** avec tables ReportLab et graphiques intégrés

```bash
python code/03_explore_data.py
```

**Sortie :** `data/visualizations/*.png` + `data/processed/EDA_REPORT.pdf`

---

### Étape 4 — Sélection du modèle avec MLflow (`05_MLflow.py`)

Compare **3 modèles** de vectorisation sur un échantillon stratifié de 5 000 offres :

| Modèle | Avantages | Limites |
|---|---|---|
| **TF-IDF** | Très rapide, baseline solide | Pas de compréhension sémantique |
| **Word2Vec** (CBOW, window=8) | Contexte local, moyenne pondérée IDF | Sensible aux textes courts |
| **Sentence-BERT** ✅ | Compréhension sémantique complète | Plus lent (~50ms/requête) |

**Score composite de sélection :**
```
Score = 0.80 × Precision@10 + 0.20 × (1 / log(inference_ms + 1))
```
Priorité qualité (80%) > vitesse (20%). **SBERT est sélectionné automatiquement.**

```bash
python code/05_MLflow.py
mlflow ui --port 5000   # → http://localhost:5000
```

**Sortie :** `data/processed/selected_model.json` + expériences MLflow loggées

---

### Étape 5 — Vectorisation NLP (`04_nlp_vectorization.py`)

Pipeline complet sur le corpus entier :

**A. Preprocessing NLP**
- Nettoyage HTML, URLs, emails, chiffres isolés
- Tokenisation + Lemmatisation (NLTK WordNet)
- Suppression stopwords enrichi (termes métier génériques ajoutés)

**B. TF-IDF** (baseline de comparaison)
- `max_features=10 000`, `ngram_range=(1,2)`, `sublinear_tf=True`
- Sortie : `vectors/tfidf_matrix.npz`

**C. Word2Vec amélioré**
- CBOW, `window=8`, `min_count=5`, `epochs=20`
- Moyenne pondérée par IDF (mots rares = plus de poids)
- Tests d'analogie sémantique : `'senior' - 'junior' + 'engineer' ≈ 'lead engineer'`
- Sortie : `vectors/w2v_matrix.npy`

**D. Sentence-BERT** (modèle final retenu)
- Modèle : `all-MiniLM-L6-v2` (80 MB, 384 dimensions)
- Encodage par batch de 64, `normalize_embeddings=True`
- Stratégie robuste : fallback automatique vers `paraphrase-MiniLM-L3-v2` si nécessaire
- Sortie : `vectors/sbert_matrix.npy`

```bash
python code/04_nlp_vectorization.py
```

---

### Étape 6 — Moteur de matching (`matching_engine.py`)

Le cœur du système. Pour chaque candidat :

**1. Construction du texte enrichi (FR→EN)**
- Traduction FR → EN des termes métier (~80 termes : `tableaux de bord` → `dashboards`, `gestion budgétaire` → `budget management`...)
- Skills répétés **3×** + expansion sémantique (ex: `Power BI` → `power bi business intelligence dashboards reporting data visualization Microsoft BI`)
- Contexte de niveau d'expérience en anglais naturel
- Contexte de domaine (`tech`, `finance`, `marketing`...)

**2. Encodage + Scoring**
- Encodage SBERT du texte enrichi → vecteur 384 dimensions
- Similarité cosinus candidat ↔ 112 796 offres (< 1 seconde)

**3. Filtres appliqués**
- Salaire minimum/maximum
- Remote only
- Type de contrat (FT/PT/CT)
- Localisation (si assez de résultats ≥ 10)

**4. Bonus/Pénalités sur le score final**

| Condition | Effet |
|---|---|
| Niveau exact | +0.12 |
| Niveau adjacent ±1 | +0.05 |
| Niveau incompatible >±1 | **−0.07** |
| Salaire renseigné | +0.02 |
| Remote 100% match | +0.05 |
| Localisation match | +0.03 |

Score final clampé entre `[-0.10 ; +0.20]`

---

### Étape 7 — Parser CV avec Groq (`cv_parser.py`)

Parse automatiquement tout CV en PDF, DOCX ou TXT via **Groq API gratuite** :

- Modèle LLM : `llama-3.3-70b-versatile`
- Temps moyen : **~1 seconde par CV**
- Extrait : nom, email, téléphone, localisation, résumé, skills (max 25), niveau, domaine, années d'expérience

**Garde-fous automatiques (corrections des erreurs du LLM) :**

| Condition | Correction |
|---|---|
| ≥ 15 ans d'expérience | → `executive` |
| ≥ 10 ans + classifié `entry/mid` | → `senior` |
| ≥ 7 ans + classifié `entry` | → `senior` |
| ≥ 3 ans + pas de signal étudiant | → `mid` |

**Limites du compte Groq gratuit :** 14 400 requêtes/jour, 30 req/minute.

---

### Étape 8 — Évaluation (`07_evaluation.py`)

Métriques calculées sur **5 profils de référence** (Data Scientist Senior, Data Engineer Junior, Product Manager, DevOps Engineer, Software Engineer Backend) :

| Métrique | Description |
|---|---|
| **Precision@K** | % de résultats pertinents parmi les K premiers |
| **Recall@K** | % d'offres pertinentes retrouvées dans le Top-K |
| **NDCG@K** | Qualité du ranking — pénalise les bons résultats mal classés |
| **MRR** | 1 / rang du premier bon résultat (Mean Reciprocal Rank) |
| **Hit Rate@K** | Au moins 1 bon résultat dans le Top-K ? |

```bash
python code/07_evaluation.py
mlflow ui --port 5000
```

---

## 🛠 Technologies utilisées

### Backend

| Technologie | Usage |
|---|---|
| **Python 3.11** | Langage principal |
| **FastAPI** | API REST avec documentation Swagger automatique |
| **Sentence-BERT** `all-MiniLM-L6-v2` | Encodage sémantique 384 dimensions |
| **Groq API** `llama-3.3-70b-versatile` | Parsing intelligent des CV (gratuit) |
| **MLflow** | Tracking expériences, comparaison modèles |
| **Gensim Word2Vec** | Modèle de comparaison |
| **scikit-learn TF-IDF** | Baseline de comparaison |
| **pdfplumber** | Extraction texte PDF |
| **python-docx** | Extraction texte DOCX |
| **pandas / numpy** | Traitement et vectorisation des données |
| **matplotlib / seaborn** | Visualisations EDA |
| **ReportLab** | Génération rapport PDF |
| **Uvicorn** | Serveur ASGI Python |

### Frontend

| Technologie | Usage |
|---|---|
| **Angular 20** (Standalone Components) | Framework frontend |
| **TypeScript** | Langage |
| **RxJS** | Gestion flux HTTP réactifs |
| **Nginx** | Serveur de production (Multi-stage Docker build) |

### Déploiement

| Technologie | Usage |
|---|---|
| **Docker** | Containerisation des 2 services |
| **Docker Compose** | Orchestration backend + frontend |

---

## 📁 Structure des fichiers

```
job-candidate-matching/
├── code/
│   ├── 01_load_data.py                  # Chargement & fusion des sources
│   ├── 02_clean_data.py                 # Nettoyage & normalisation
│   ├── 03_explore_data.py               # EDA + rapport PDF
│   ├── 04_visualize.py               # EDA + rapport PDF
│   ├── 05_MLflow.py                     # Sélection modèle via MLflow
│   ├── 06_nlp_vectorization.py          # TF-IDF + Word2Vec + SBERT
│   ├── candidate_text_builder.py        # Enrichissement texte FR→EN
│   ├── matching_engine.py               # Moteur SBERT + filtres + bonus
│   ├── cv_parser.py                     # Parser CV via Groq LLM
│   ├── 07_evaluation.py                 # Métriques Precision/NDCG/MRR
│   └── 08_api.py                        # API REST FastAPI
│
├── data/
│   ├── raw/
│   │   ├── jobs_merged.csv              # Données brutes fusionnées
│   │   └── synthetic_jobs.csv           # 1 800 offres synthétiques
│   ├── processed/
│   │   ├── jobs_cleaned.csv             # Corpus nettoyé
│   │   ├── jobs_vectorized.pkl          # Corpus + métadonnées
│   │   ├── selected_model.json          # Modèle sélectionné par MLflow
│   │   ├── EDA_REPORT.txt               # Rapport statistiques texte
│   │   ├── EDA_REPORT.pdf               # Rapport PDF avec graphiques
│   │   ├── vectors/
│   │   │   ├── sbert_matrix.npy         # Embeddings SBERT (112k × 384)
│   │   │   ├── tfidf_matrix.npz         # Matrice TF-IDF sparse
│   │   │   └── w2v_matrix.npy           # Embeddings Word2Vec (150 dims)
│   │   ├── matching_results/            # Résultats JSON par candidat
│   │   └── evaluation/                  # Rapports métriques MLflow
│   └── visualizations/
│       ├── 01_salary_distribution.png
│       ├── 02_experience_distribution.png
│       ├── 03_top_locations.png
│       └── 04_remote_ratio.png
│
├── frontend/
│   └── src/app/
│       ├── core/
│       │   ├── models/api.models.ts     # Interfaces TypeScript
│       │   └── services/api.service.ts  # Appels HTTP vers FastAPI
│       ├── pages/
│       │   ├── home/                    # Dashboard + santé système
│       │   ├── match-text/              # Matching par texte libre
│       │   ├── match-file/              # Upload CV drag & drop
│       │   ├── jobs/                    # Liste offres + recherche
│       │   └── stats/                   # Statistiques corpus
│       └── shared/components/
│           ├── header/                  # Navigation
│           └── job-card/                # Carte résultat avec scores
│
├── mlruns/                              # Expériences MLflow
├── Dockerfile.backend                   # Image Python 3.11 + ML
├── Dockerfile.frontend                  # Multi-stage Node→Nginx
├── nginx.conf                           # Config Nginx pour Angular Router
├── docker-compose.yml                   # Orchestration 2 services
└── requirements.txt                     # Dépendances Python
```

---

## ✅ Prérequis

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installé et lancé
- Clé API **Groq** gratuite → [console.groq.com](https://console.groq.com) *(pour l'upload de fichiers CV)*
- Au minimum **6 Go d'espace disque** disponible

---

## 🚀 Installation & Lancement

### Option A — Avec Docker (recommandé)

**1. Configurer la clé Groq dans `docker-compose.yml` :**
```yaml
services:
  backend:
    environment:
      - GROQ_API_KEY=gsk_XXXXXXXXXXXXXXXX
```

**2. Lancer :**
```bash
docker-compose up --build -d
```

**3. Vérifier :**
```bash
docker-compose ps
```
Les 2 services doivent être `Up` :
```
job-candidate-matching-backend-1    Up   0.0.0.0:8000->8000/tcp
job-candidate-matching-frontend-1   Up   0.0.0.0:4200->80/tcp
```


### Option B — Lancement local (développement)

```bash
# 1. Installer les dépendances
pip install -r requirements.txt

# 2. Lancer le pipeline ML (une seule fois)
python code/01_load_data.py
python code/02_clean_data.py
python code/03_explore_data.py
python code/03_model_selection_mlflow.py
python code/04_nlp_vectorization.py

# 3. Lancer l'API
python code/08_api.py

# 4. Dans un second terminal, lancer le frontend
cd frontend
ng serve
```

---

## 🐳 Déploiement Docker

### Commandes essentielles

```bash
docker-compose up -d                    # Démarrer
docker-compose down                     # Arrêter
docker-compose logs -f backend          # Logs backend en temps réel
docker-compose logs -f frontend         # Logs frontend
docker-compose restart backend          # Redémarrer le backend seul
docker-compose up -d --force-recreate backend  # Appliquer une modif .yml
docker system prune -a                  # Nettoyer le cache Docker
```

### ⚠️ Attendre que SBERT soit prêt

Au démarrage, le modèle Sentence-BERT charge 112 796 vecteurs en mémoire (30-60 secondes).
Surveille avec `docker-compose logs -f backend` jusqu'à voir :

```
✅ Vecteurs SBERT chargés  : (112796, 384)
✅ Modèle SBERT chargé    : all-MiniLM-L6-v2
✅ Engine prêt en 24.8s
INFO: Application startup complete.
```

---

## 🌐 Accès à l'application

| Service | URL |
|---|---|
| 🌐 Interface Angular | http://localhost:4200 |
| ⚡ API FastAPI | http://localhost:8000 |
| 📖 Swagger UI | http://localhost:8000/docs |
| 📖 ReDoc | http://localhost:8000/redoc |
| ❤️ Health Check | http://localhost:8000/health |
| 📊 Statistiques | http://localhost:8000/stats |

---

## 🔌 API Endpoints

| Méthode | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | État du système (`engine_ready`, taille corpus, modèle) |
| `GET` | `/stats` | Distribution sources, niveaux, remote, types de contrat |
| `POST` | `/match/text` | Matching depuis texte libre |
| `POST` | `/match/file` | Matching depuis fichier CV (PDF/DOCX/TXT) |
| `GET` | `/jobs` | Liste paginée (filtre source/niveau, max 100/page) |
| `GET` | `/jobs/search?q=...` | Recherche par mot-clé dans les titres |

### Exemple — Matching par texte

```bash
curl -X POST http://localhost:8000/match/text \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Ghada Lajnef",
    "cv_text": "3ème année BI à ISG-Tunis. Power BI, Talend, IBM Cognos, SQL, Python.",
    "experience_level": "auto",
    "top_k": 10
  }'
```

---

## 🖥 Interface Angular

### Pages disponibles

| Route | Description |
|---|---|
| `/home` | Dashboard avec état du système en temps réel |
| `/match-text` | Formulaire + presets exemples + résultats avec scores |
| `/match-file` | Drag & drop upload CV (PDF/DOCX/TXT) + résultats |
| `/jobs` | Liste paginée avec recherche et filtres par niveau |
| `/stats` | KPI cards + graphiques bar charts du corpus |

### Carte résultat (JobCard)
Chaque résultat affiche un **code couleur de pertinence** :
- 🟢 **Excellent** ≥ 75% — offre très bien adaptée
- 🔵 **Bon** ≥ 55% — bonne correspondance
- 🟡 **Correct** ≥ 40% — correspondance partielle
- 🔴 **Faible** < 40% — peu pertinent

Ainsi que le détail : SBERT score + filter_bonus + raisons du matching.

---

## 📊 Évaluation des modèles

Résultats sur 5 profils de référence (Data Scientist, Data Engineer, Product Manager, DevOps, Software Engineer) :

| Métrique | SBERT | TF-IDF | Avantage SBERT |
|---|---|---|---|
| Precision@10 | **~0.75** | ~0.45 | +67% |
| NDCG@10 | **~0.80** | ~0.55 | +45% |
| MRR | **~0.85** | ~0.60 | +42% |
| Inference | ~50ms | ~5ms | TF-IDF 10× plus rapide |

**Conclusion :** SBERT offre une précision sémantique nettement supérieure. La latence de ~50ms est acceptable pour une application interactive.

---

## 👨‍🏫 Contexte académique

**Module :** ING-4 SDIA — Data Science & Intelligence Artificielle
**Tuteur :** Haythem Ghazouani