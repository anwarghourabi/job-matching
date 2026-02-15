"""
Week 1 - Étape bonus: Scraper RemoteOK
UTILISE VOTRE CODE - TRÈS BON!
"""

import requests
import pandas as pd
from pathlib import Path

def scrape_remoteok():
    """Scraper RemoteOK API - VOTRE CODE OPTIMISÉ"""
    
    print("\n" + "="*60)
    print("SCRAPING REMOTEOK API")
    print("="*60)
    
    try:
        print("\n🌐 Requête API RemoteOK...")
        
        url = "https://remoteok.com/api"  # API publique
        headers = {"User-Agent": "Mozilla/5.0"}
        
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Réponse reçue: {len(data)} offres totales")
        
        # Filtrer et extraire les offres
        jobs = []
        for job in data:
            if job.get("company") and job.get("position"):  # filtrer les objets inutiles
                jobs.append({
                    "job_title": job["position"],
                    "company": job["company"],
                    "location": job.get("location", "Remote"),
                    "tags": ", ".join(job.get("tags", [])),
                    "salary": job.get("salary", None),
                    "source": "RemoteOK"
                })
        
        print(f"✅ {len(jobs)} offres valides extraites")
        
        # Créer le DataFrame
        df = pd.DataFrame(jobs)
        
        # Afficher les stats
        print(f"\n📊 STATISTIQUES REMOTEOK:")
        print(f"   Total offres: {len(df)}")
        
        if len(df) > 0:
            print(f"\n   Top compagnies:")
            print(df['company'].value_counts().head(10))
            print(f"\n   Top locations:")
            print(df['location'].value_counts().head(10))
            print(f"\n   Top tags:")
            print(df['tags'].value_counts().head(10))
        
        # Sauvegarder
        Path('data/raw').mkdir(parents=True, exist_ok=True)
        output_path = 'data/raw/jobs_scraped_remoteok.csv'
        df.to_csv(output_path, index=False)
        
        print(f"\n✅ Sauvegardé: {output_path}")
        print(f"   Nombre d'offres: {len(df)}")
        print(f"   Colonnes: {df.columns.tolist()}")
        
        # Afficher les premières lignes
        print(f"\n📋 Aperçu des données:")
        print(df.head(10))
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erreur requête: {e}")
        return None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return None

if __name__ == '__main__':
    df = scrape_remoteok()
    
    if df is not None:
        print("\n" + "="*60)
        print("✅ SCRAPING REMOTEOK RÉUSSI!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("❌ ERREUR LORS DU SCRAPING")
        print("="*60)