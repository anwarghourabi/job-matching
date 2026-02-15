import pandas as pd
from pathlib import Path

def merge_datasets():
    """Fusionner HuggingFace + RemoteOK"""
    
    print("\n" + "="*60)
    print("FUSION DES DATASETS")
    print("="*60)
    
    # ============================================================
    # 1. CHARGER HUGGINGFACE
    # ============================================================
    print("\n📥 Chargement HuggingFace...")
    hf_path = Path('data/raw/huggingface_salaries.csv')
    
    if not hf_path.exists():
        print(f"❌ Fichier non trouvé: {hf_path}")
        return None
    
    df_hf = pd.read_csv(hf_path)
    print(f"   ✅ Chargé: {len(df_hf)} offres")
    print(f"   Colonnes: {df_hf.columns.tolist()}")
    
    # ============================================================
    # 2. CHARGER REMOTEOK
    # ============================================================
    print("\n📥 Chargement RemoteOK...")
    remote_path = Path('data/raw/jobs_scraped_remoteok.csv')
    
    if not remote_path.exists():
        print(f"⚠️  Fichier non trouvé: {remote_path}")
        print("   Exécutez d'abord: python code/05_scrape_remoteok.py")
        df_remote = None
    else:
        df_remote = pd.read_csv(remote_path)
        print(f"   ✅ Chargé: {len(df_remote)} offres")
        print(f"   Colonnes: {df_remote.columns.tolist()}")
    
    # ============================================================
    # 3. STANDARDISER HUGGINGFACE
    # ============================================================
    print("\n🔄 Standardisation HuggingFace...")
    
    # Sélectionner les colonnes pertinentes
    df_hf_std = df_hf[[
        'job_title', 'salary_in_usd', 'experience_level',
        'employment_type', 'company_location', 'remote_ratio', 'company_size'
    ]].copy()
    
    # Renommer pour uniformité
    df_hf_std.columns = [
        'job_title', 'salary_usd', 'experience_level',
        'employment_type', 'location', 'remote_ratio', 'company_size'
    ]
    
    # Ajouter source
    df_hf_std['source'] = 'huggingface'
    
    print(f"   ✅ Standardisé: {df_hf_std.shape}")
    print(f"   Colonnes: {df_hf_std.columns.tolist()}")
    
    # ============================================================
    # 4. STANDARDISER REMOTEOK (si disponible)
    # ============================================================
    if df_remote is not None:
        print("\n🔄 Standardisation RemoteOK...")
        
        # RemoteOK a ces colonnes: job_title, company, location, tags, salary, source
        # Nous devons les mapper aux colonnes HF
        
        df_remote_std = pd.DataFrame({
            'job_title': df_remote['job_title'],
            'salary_usd': None,  # RemoteOK n'a pas de salaire en USD standardisé
            'experience_level': None,  # RemoteOK n'a pas ce champ
            'employment_type': None,  # RemoteOK n'a pas ce champ
            'location': df_remote['location'],
            'remote_ratio': 100,  # RemoteOK = toujours 100% remote (par nature)
            'company_size': None,  # RemoteOK n'a pas ce champ
            'source': 'remoteok',
            # Colonnes bonus de RemoteOK
            'company': df_remote['company'],
            'tags': df_remote['tags'],
            'salary_original': df_remote['salary']  # Garder pour info
        })
        
        print(f"   ✅ Standardisé: {df_remote_std.shape}")
        
        # ============================================================
        # 5. FUSIONNER LES DATASETS
        # ============================================================
        print("\n🔗 Fusion des datasets...")
        
        # Sélectionner colonnes communes
        cols_to_merge = [
            'job_title', 'salary_usd', 'experience_level',
            'employment_type', 'location', 'remote_ratio', 'company_size', 'source'
        ]
        
        df_hf_to_merge = df_hf_std[cols_to_merge].copy()
        df_remote_to_merge = df_remote_std[cols_to_merge].copy()
        
        # Concat
        df_combined = pd.concat(
            [df_hf_to_merge, df_remote_to_merge],
            ignore_index=True
        )
        
        print(f"   ✅ Fusionné: {df_combined.shape}")
        print(f"   Total offres: {len(df_combined):,}")
        
    else:
        print("\n⚠️  RemoteOK non disponible")
        print("   Utilisation HuggingFace seul")
        df_combined = df_hf_std.copy()
    
    # ============================================================
    # 6. AFFICHER STATISTIQUES
    # ============================================================
    print("\n📊 STATISTIQUES FUSIONNÉES:")
    print(f"   Total offres: {len(df_combined):,}")
    
    print(f"\n   Distribution par source:")
    print(df_combined['source'].value_counts())
    
    print(f"\n   Distribution remote_ratio:")
    print(df_combined['remote_ratio'].value_counts().sort_index())
    
    print(f"\n   Salaires disponibles:")
    salary_count = df_combined['salary_usd'].notna().sum()
    print(f"   {salary_count} offres avec salaire / {len(df_combined)} totales ({salary_count/len(df_combined)*100:.1f}%)")
    
    print(f"\n   Top localisations:")
    print(df_combined['location'].value_counts().head(10))
    
    # ============================================================
    # 7. SAUVEGARDER
    # ============================================================
    print("\n💾 Sauvegarde...")
    
    output_path = Path('data/raw/jobs_merged.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    
    print(f"   ✅ Sauvegardé: {output_path}")
    print(f"   Fichier size: {len(df_combined)} lignes × {len(df_combined.columns)} colonnes")
    
    # ============================================================
    # 8. AFFICHER APERÇU
    # ============================================================
    print(f"\n📋 Aperçu des données:")
    print(df_combined.head(10))
    
    print(f"\n   Info colonnes:")
    print(df_combined.info())
    
    return df_combined

if __name__ == '__main__':
    df = merge_datasets()
    
    if df is not None:
        print("\n" + "="*60)
        print("✅ FUSION COMPLÉTÉE AVEC SUCCÈS!")
        print("="*60)
        print("\n📝 Prochaines étapes:")
        print("   1. Modifier code/02_clean_data.py")
        print("      Changer: csv_path = Path('data/raw/huggingface_salaries.csv')")
        print("      En:      csv_path = Path('data/raw/jobs_merged.csv')")
        print("   2. Exécuter: python code/02_clean_data.py")
        print("   3. Le dataset fusionné sera nettoyé")
    else:
        print("\n❌ ERREUR LORS DE LA FUSION")