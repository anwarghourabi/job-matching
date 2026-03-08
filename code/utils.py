"""
Utilitaires pour le projet Job-Candidate Matching
"""

import pandas as pd
from pathlib import Path

def load_cleaned_data():
    """Charger les données nettoyées"""
    csv_path = Path('data/processed/jobs_cleaned.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Données nettoyées non trouvées: {csv_path}")
    return pd.read_csv(csv_path)

def load_raw_data():
    """Charger les données brutes fusionnées"""
    csv_path = Path('data/raw/jobs_merged.csv')
    if not csv_path.exists():
        raise FileNotFoundError(f"Données fusionnées non trouvées: {csv_path}")
    return pd.read_csv(csv_path)

def get_statistics(df):
    """Obtenir les statistiques principales"""
    stats = {
        'total_jobs': len(df),
        'total_locations': df['location'].nunique(),
        'salary_min': df['salary_usd'].min(),
        'salary_max': df['salary_usd'].max(),
        'salary_mean': df['salary_usd'].mean(),
        'salary_median': df['salary_usd'].median(),
        'remote_100_pct': (len(df[df['remote_ratio'] == 100]) / len(df)) * 100,
    }
    return stats

def get_top_locations(df, n=10):
    """Obtenir les top N localisations"""
    return df['location'].value_counts().head(n)

def get_salary_by_experience(df):
    """Obtenir les salaires moyen par niveau d'expérience"""
    return df.groupby('experience_level')['salary_usd'].agg([
        ('count', 'count'),
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max')
    ]).round(2)

def ensure_directories():
    """Créer les répertoires nécessaires"""
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/visualizations').mkdir(parents=True, exist_ok=True)  