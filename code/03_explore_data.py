import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_dataset():
    """Analyser le dataset nettoyé"""
    
    csv_path = Path('data/processed/jobs_cleaned.csv')
    if not csv_path.exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("   Exécutez d'abord: python code/02_clean_data.py")
        return None
    
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*60)
    print("ÉTAPE 3: ANALYSE EXPLORATOIRE")
    print("="*60)
    
    print(f"\n📊 DATASET:")
    print(f"   Size: {len(df):,} offres")
    print(f"   Colonnes: {df.shape[1]}")
    
    print(f"\n💰 SALAIRE (USD):")
    print(f"   Min: ${df['salary_usd'].min():,.0f}")
    print(f"   Max: ${df['salary_usd'].max():,.0f}")
    print(f"   Mean: ${df['salary_usd'].mean():,.0f}")
    print(f"   Median: ${df['salary_usd'].median():,.0f}")
    print(f"   Std: ${df['salary_usd'].std():,.0f}")
    
    print(f"\n👤 EXPÉRIENCE:")
    for level, count in df['experience_level'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"   {level}: {count} ({pct:.1f}%)")
    
    print(f"\n🌍 TOP 10 LOCALISATIONS:")
    for loc, count in df['company_location'].value_counts().head(10).items():
        pct = (count / len(df)) * 100
        print(f"   {loc}: {count} ({pct:.1f}%)")
    
    print(f"\n📱 REMOTE RATIO:")
    for ratio, count in df['remote_ratio'].value_counts().items():
        pct = (count / len(df)) * 100
        print(f"   {ratio}%: {count} ({pct:.1f}%)")
    
    print(f"\n💼 COMPANY SIZE:")
    for size, count in df['company_size'].value_counts().items():
        pct = (count / len(df)) * 100
        label_map = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
        label = label_map.get(size, size)
        print(f"   {label}: {count} ({pct:.1f}%)")
    
    print(f"\n✅ Analyse complète!")
    
    return df