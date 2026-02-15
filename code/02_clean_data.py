"""
Week 1 - Étape 2: Nettoyer les données
VERSION CORRIGÉE POUR DATASET FUSIONNÉ (HF + RemoteOK)
"""

import pandas as pd
import re
from unicodedata import normalize
from pathlib import Path

class DataCleaner:
    """Nettoyer et normaliser les données"""
    
    # ADAPTÉ AUX COLONNES DU DATASET FUSIONNÉ
    # Le dataset fusionné a: job_title, salary_usd, experience_level, 
    #                        employment_type, location, remote_ratio, company_size, source
    COLUMN_MAPPING = {
        'job_title': 'job_title',
        'salary_usd': 'salary_usd',
        'experience_level': 'experience_level',
        'employment_type': 'employment_type',
        'location': 'location',  # Dataset fusionné a 'location' pas 'company_location'
        'remote_ratio': 'remote_ratio',
        'company_size': 'company_size',
        'source': 'source',
    }
    
    # Mapping des niveaux d'expérience
    EXPERIENCE_MAPPING = {
        'EN': 'entry-level',
        'MI': 'mid-level',
        'SE': 'senior',
        'EX': 'executive',
    }
    
    @staticmethod
    def clean_text(text):
        """Nettoyer le texte"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()
        text = normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
        text = re.sub(r'[^\w\s\-+#]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @classmethod
    def clean_dataset(cls, df):
        """Pipeline complet de nettoyage"""
        print("\n🧹 Nettoyage du dataset...")
        
        # 1. Standardiser colonnes
        print("  1. Standardisation colonnes...")
        df = df.rename(columns=cls.COLUMN_MAPPING)
        
        # 2. Supprimer les doublons
        print("  2. Suppression doublons...")
        before = len(df)
        # Utiliser 'location' (colonne du dataset fusionné)
        df = df.drop_duplicates(
            subset=['job_title', 'location'], 
            keep='first'
        )
        print(f"     Supprimés: {before - len(df)}")
        
        # 3. Gestion valeurs manquantes
        print("  3. Gestion valeurs manquantes...")
        df['salary_usd'].fillna(df['salary_usd'].median(), inplace=True)
        
        # 4. Nettoyer texte
        print("  4. Nettoyage texte...")
        df['job_title'] = df['job_title'].apply(cls.clean_text)
        
        # 5. Normaliser expérience
        print("  5. Normalisation expérience...")
        df['experience_level'] = df['experience_level'].map(
            cls.EXPERIENCE_MAPPING
        ).fillna('unknown')
        
        # 6. Supprimer colonnes inutiles
        print("  6. Suppression colonnes inutiles...")
        cols_to_keep = [
            'job_title', 'salary_usd', 'experience_level',
            'employment_type', 'location', 'remote_ratio',
            'company_size', 'source'
        ]
        df = df[[col for col in cols_to_keep if col in df.columns]]
        
        print(f"✅ Nettoyage complet: {df.shape}")
        return df

def main():
    """Fonction principale"""
    print("\n" + "="*60)
    print("ÉTAPE 2: NETTOYER LES DONNÉES")
    print("="*60)
    
    # Charger les données brutes
    csv_path = Path('data/raw/jobs_merged.csv')
    if not csv_path.exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("   Exécutez d'abord: python code/06_merge_remoteok.py")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\n📥 Données brutes: {df.shape}")
    print(f"   Colonnes: {df.columns.tolist()}")
    
    # Nettoyer
    df_clean = DataCleaner.clean_dataset(df)
    
    # Afficher statistiques
    print("\n📊 STATISTIQUES APRÈS NETTOYAGE:")
    print(f"   Shape: {df_clean.shape}")
    print(f"   Colonnes: {df_clean.columns.tolist()}")
    
    print("\n💰 SALAIRE:")
    print(f"   Min: ${df_clean['salary_usd'].min():,.0f}")
    print(f"   Max: ${df_clean['salary_usd'].max():,.0f}")
    print(f"   Mean: ${df_clean['salary_usd'].mean():,.0f}")
    print(f"   Median: ${df_clean['salary_usd'].median():,.0f}")
    
    print("\n👤 EXPÉRIENCE:")
    print(df_clean['experience_level'].value_counts())
    
    print("\n🌍 TOP LOCALISATIONS:")
    print(df_clean['location'].value_counts().head(10))
    
    print("\n📱 REMOTE RATIO:")
    print(df_clean['remote_ratio'].value_counts())
    
    print("\n📊 SOURCES:")
    print(df_clean['source'].value_counts())
    
    # Sauvegarder
    output_path = Path('data/processed/jobs_cleaned.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    # Backup pickle
    pkl_path = Path('data/processed/jobs_cleaned.pkl')
    df_clean.to_pickle(pkl_path)
    
    print(f"\n✅ Sauvegardé:")
    print(f"   CSV: {output_path}")
    print(f"   PKL: {pkl_path}")

if __name__ == '__main__':
    main()