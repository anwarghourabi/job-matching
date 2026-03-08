"""
Week 1 - Étape 2: Nettoyer les données
VERSION CORRIGÉE — Fix ValueError shapes broadcast (masques booléens après reset_index)
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

DATA_RAW       = Path('data/raw')
DATA_PROCESSED = Path('data/processed')
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


def safe_save(df, stem):
    for ext, method in [('.csv', lambda p: df.to_csv(p, index=False, encoding='utf-8-sig')),
                        ('.pkl', lambda p: df.to_pickle(p))]:
        path = DATA_PROCESSED / (stem + ext)
        for attempt in range(3):
            try:
                method(path)
                print(f"   ✅ Sauvegardé: {path}")
                break
            except PermissionError:
                path = DATA_PROCESSED / (stem + f"_v{attempt+2}" + ext)


def normalize_experience(val):
    val = str(val).lower().strip()
    if val in ('en',):
        return 'entry'
    if val in ('mi',):
        return 'mid'
    if val in ('se',):
        return 'senior'
    if val in ('ex',):
        return 'executive'
    if any(w in val for w in ['entry', 'internship', 'intern', 'junior', 'graduate', 'associate']):
        return 'entry'
    if any(w in val for w in ['mid-senior', 'mid senior']):
        return 'mid'
    if any(w in val for w in ['mid', 'middle', 'intermediate']):
        return 'mid'
    if any(w in val for w in ['senior', 'lead', 'staff', 'principal', 'expert']):
        return 'senior'
    if any(w in val for w in ['executive', 'director', 'vp', 'vice president',
                               'chief', 'president', 'head', 'c-level']):
        return 'executive'
    return 'unknown'


def normalize_employment_type(val):
    val = str(val).lower().strip()
    if any(w in val for w in ['ft', 'full']):
        return 'FT'
    if any(w in val for w in ['pt', 'part']):
        return 'PT'
    if any(w in val for w in ['ct', 'contract', 'freelance']):
        return 'CT'
    return 'FT'


def normalize_company_size(val):
    val = str(val).lower().strip()
    if val in ['s', 'small']:
        return 'S'
    if val in ['l', 'large']:
        return 'L'
    return 'M'


def clean_job_title(title):
    title = str(title).strip()
    title = re.sub(r'^[\W_]+|[\W_]+$', '', title)
    title = re.sub(r'\s+', ' ', title)
    return title if len(title) >= 2 else None


def clean_location(loc):
    loc = str(loc).strip()
    if loc.lower() in ('nan', 'none', 'unknown', '', 'null'):
        return 'Unknown'
    loc = re.sub(r'\s+\d{5}(-\d{4})?$', '', loc)
    return loc.strip()


def fix_aberrant_salaries(salary_series):
    """
    Corrige les salaires aberrants sur une Series — sans problème d'index.
    Retourne une nouvelle Series avec les valeurs corrigées.
    """
    result = salary_series.copy()

    for idx in salary_series[salary_series > 500_000].index:
        val = salary_series.loc[idx]
        corrected = val / 12
        if 20_000 <= corrected <= 500_000:
            result.loc[idx] = corrected
        else:
            result.loc[idx] = 0

    return result


def clean_data(df):
    print(f"\n🧹 Nettoyage du dataset ({len(df):,} lignes)...")

    # ── 1. Copie + reset_index propre ────────────────
    df = df.copy().reset_index(drop=True)

    # ── 2. Colonnes ──────────────────────────────────
    print("  1. Standardisation colonnes...")
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')

    # ── 3. Doublons ──────────────────────────────────
    print("  2. Suppression doublons stricts...")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)   # ← reset_index après drop
    print(f"     Supprimés: {before - len(df):,}")

    # ── 4. Titres ─────────────────────────────────────
    print("  3. Nettoyage titres de postes...")
    df['job_title'] = df['job_title'].apply(clean_job_title)
    before = len(df)
    df = df[df['job_title'].notna()].reset_index(drop=True)
    print(f"     Titres invalides supprimés: {before - len(df):,}")

    # ── 5. Localisations ─────────────────────────────
    print("  4. Nettoyage localisations...")
    df['location'] = df['location'].apply(clean_location)

    # ── 6. Expérience ─────────────────────────────────
    print("  5. Normalisation expérience...")
    df['experience_level'] = df['experience_level'].apply(normalize_experience)
    print(f"     Distribution:\n{df['experience_level'].value_counts().to_string()}")

    # ── 7. Employment type ────────────────────────────
    print("  6. Normalisation type d'emploi...")
    df['employment_type'] = df['employment_type'].apply(normalize_employment_type)

    # ── 8. Company size ───────────────────────────────
    print("  7. Normalisation taille entreprise...")
    df['company_size'] = df['company_size'].apply(normalize_company_size)

    # ── 9. Salaires ───────────────────────────────────
    print("  8. Traitement des salaires...")
    df['salary_usd'] = pd.to_numeric(df['salary_usd'], errors='coerce').fillna(0)

    # ✅ Correction salaires aberrants — sans masque booléen externe
    nb_aberrant = (df['salary_usd'] > 500_000).sum()
    if nb_aberrant > 0:
        print(f"     Salaires > $500k détectés: {nb_aberrant:,} → correction...")
        df['salary_usd'] = fix_aberrant_salaries(df['salary_usd'])
        nb_still = (df['salary_usd'] > 500_000).sum()
        print(f"     Après correction, encore aberrants: {nb_still:,}")

    # Flag has_salary
    df['has_salary'] = (df['salary_usd'] > 1_000).astype(int)
    print(f"     Offres avec salaire    : {df['has_salary'].sum():,}")
    print(f"     Offres sans salaire    : {(df['has_salary']==0).sum():,} (conservées)")

    # ── 10. Remote ratio ──────────────────────────────
    print("  9. Normalisation remote_ratio...")
    df['remote_ratio'] = pd.to_numeric(df['remote_ratio'], errors='coerce').fillna(0)
    df['remote_ratio'] = df['remote_ratio'].apply(
        lambda x: 100 if x >= 75 else (50 if x >= 25 else 0)
    )

    print(f"\n  📊 Shape finale: {df.shape}")
    return df


def main():
    print("\n" + "="*60)
    print("ÉTAPE 2: NETTOYER LES DONNÉES")
    print("="*60)

    merged_path = DATA_RAW / 'jobs_merged.csv'
    if not merged_path.exists():
        print(f"❌ Fichier introuvable : {merged_path}")
        print("   Lance d'abord : python code/01_load_data.py")
        return None

    df = pd.read_csv(merged_path)
    print(f"\n📥 Données brutes: {df.shape}")
    print(f"   Colonnes: {list(df.columns)}")
    print(f"   Distribution par source:\n{df['source'].value_counts().to_string()}")

    df_clean = clean_data(df)

    # ── Statistiques finales ──────────────────────────
    print("\n" + "="*60)
    print("📊 STATISTIQUES APRÈS NETTOYAGE")
    print("="*60)
    print(f"\n   Shape: {df_clean.shape}")

    df_sal = df_clean[df_clean['has_salary'] == 1]
    if len(df_sal) > 0:
        print(f"\n💰 SALAIRE ({len(df_sal):,} offres avec salaire):")
        print(f"   Min    : ${df_sal['salary_usd'].min():>12,.0f}")
        print(f"   Max    : ${df_sal['salary_usd'].max():>12,.0f}")
        print(f"   Mean   : ${df_sal['salary_usd'].mean():>12,.0f}")
        print(f"   Median : ${df_sal['salary_usd'].median():>12,.0f}")
        print(f"\n   Par source:")
        for src in df_clean['source'].unique():
            sub = df_clean[df_clean['source'] == src]
            n   = sub['has_salary'].sum()
            print(f"      {src}: {n:,}/{len(sub):,} ({n/len(sub)*100:.1f}%)")

    print(f"\n👤 EXPÉRIENCE:")
    print(df_clean['experience_level'].value_counts().to_string())

    print(f"\n🌍 TOP 15 LOCALISATIONS:")
    print(df_clean['location'].value_counts().head(15).to_string())

    print(f"\n📱 REMOTE RATIO:")
    print(df_clean['remote_ratio'].value_counts().sort_index().to_string())

    print(f"\n📊 SOURCES:")
    print(df_clean['source'].value_counts().to_string())

    print(f"\n💼 COMPANY SIZE:")
    print(df_clean['company_size'].value_counts().to_string())

    print(f"\n💾 Sauvegarde...")
    safe_save(df_clean, 'jobs_cleaned')

    print("\n" + "="*60)
    print("✅ ÉTAPE 2 COMPLÉTÉE!")
    print("="*60)
    print(f"\n📊 {len(df_clean):,} offres nettoyées")
    print(f"   dont {df_clean['has_salary'].sum():,} avec salaire ({df_clean['has_salary'].mean()*100:.1f}%)")
    print(f"   et   {(df_clean['has_salary']==0).sum():,} sans salaire (conservées)")
    print("\n🚀 Prochaine étape: python code/03_explore_data.py")

    return df_clean


if __name__ == '__main__':
    df = main()