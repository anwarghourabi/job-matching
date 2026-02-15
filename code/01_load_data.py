
import pandas as pd
from pathlib import Path

# Configuration
DATA_RAW = Path('data/raw')
DATA_RAW.mkdir(parents=True, exist_ok=True)

def load_huggingface():
    """Charger dataset Hugging Face"""
    print("📥 Chargement HuggingFace...")
    
    try:
        from datasets import load_dataset
        ds = load_dataset("hugginglearners/data-science-job-salaries")
        df = ds['train'].to_pandas()
        
        # Sauvegarder
        csv_path = DATA_RAW / 'huggingface_salaries.csv'
        df.to_csv(csv_path, index=False)
        
        print(f"✅ Sauvegardé: {csv_path}")
        print(f"   Shape: {df.shape}")
        print(f"   Colonnes: {df.columns.tolist()}")
        
        return df
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

def load_kaggle(filepath):
    """Charger dataset Kaggle"""
    print(f"📥 Chargement Kaggle depuis {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Chargé: {filepath}")
        print(f"   Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

if __name__ == '__main__':
    print("\n" + "="*60)
    print("ÉTAPE 1: CHARGER LES DONNÉES")
    print("="*60 + "\n")
    
    df_hf = load_huggingface()
    
    # Optionnel: charger Kaggle si fichier existe
    # df_kaggle = load_kaggle('data/raw/kaggle_jobs.csv')