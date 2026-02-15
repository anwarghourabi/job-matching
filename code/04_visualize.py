"""
Week 1 - Étape 4: Créer visualisations et rapport
VERSION FINALE - Génère EDA_REPORT.txt ET EDA_REPORT_PDF.pdf
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

# ============================================================
# IMPORTS POUR LE PDF
# ============================================================
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

def create_visualizations(df):
    """Créer les visualisations"""
    
    print("\n" + "="*60)
    print("ÉTAPE 4: CRÉER VISUALISATIONS")
    print("="*60)
    
    colors_pal = sns.color_palette("husl", 8)
    output_dir = Path('data/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Distribution des salaires
    print("\n📊 Création: 01_salary_distribution.png...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Histogramme
    axes[0, 0].hist(df['salary_usd'].dropna(), bins=50, color=colors_pal[0], edgecolor='black')
    axes[0, 0].set_title('Histogramme des Salaires', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Salaire (USD)')
    axes[0, 0].set_ylabel('Fréquence')
    
    # Box plot par expérience
    df.boxplot(column='salary_usd', by='experience_level', ax=axes[0, 1])
    axes[0, 1].set_title('Salaire par Niveau d\'Expérience', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Niveau d\'Expérience')
    axes[0, 1].set_ylabel('Salaire (USD)')
    
    # Violin plot
    sns.violinplot(data=df, x='experience_level', y='salary_usd', ax=axes[1, 0], palette=colors_pal)
    axes[1, 0].set_title('Distribution des Salaires (Violin)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Niveau d\'Expérience')
    axes[1, 0].set_ylabel('Salaire (USD)')
    
    # KDE
    df['salary_usd'].plot(kind='kde', ax=axes[1, 1], color=colors_pal[2], linewidth=2)
    axes[1, 1].set_title('Densité des Salaires (KDE)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Salaire (USD)')
    
    plt.tight_layout()
    plt.savefig(output_dir / '01_salary_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Sauvegardé")
    
    # 2. Distribution expérience
    print("\n📊 Création: 02_experience_distribution.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    exp_counts = df['experience_level'].value_counts()
    axes[0].pie(exp_counts, labels=exp_counts.index, autopct='%1.1f%%', colors=colors_pal, startangle=90)
    axes[0].set_title('Distribution Expérience (Pie)', fontsize=12, fontweight='bold')
    
    exp_counts.plot(kind='bar', ax=axes[1], color=colors_pal)
    axes[1].set_title('Distribution Expérience (Bar)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Niveau d\'Expérience')
    axes[1].set_ylabel('Nombre d\'Offres')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / '02_experience_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Sauvegardé")
    
    # 3. Top localisations
    print("\n📊 Création: 03_top_locations.png...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    top_locs = df['location'].value_counts().head(15)
    top_locs.plot(kind='barh', ax=ax, color=colors_pal[0])
    ax.set_title('Top 15 Localisations', fontsize=12, fontweight='bold')
    ax.set_xlabel('Nombre d\'Offres')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(output_dir / '03_top_locations.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Sauvegardé")
    
    # 4. Remote ratio
    print("\n📊 Création: 04_remote_ratio.png...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    remote_counts = df['remote_ratio'].value_counts().sort_index()
    axes[0].bar(remote_counts.index, remote_counts.values, color=colors_pal[2])
    axes[0].set_title('Remote Ratio Distribution', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Remote Ratio (%)')
    axes[0].set_ylabel('Nombre d\'Offres')
    
    # Salaire par remote ratio
    df.boxplot(column='salary_usd', by='remote_ratio', ax=axes[1])
    axes[1].set_title('Salaire par Remote Ratio', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Remote Ratio (%)')
    axes[1].set_ylabel('Salaire (USD)')
    
    plt.tight_layout()
    plt.savefig(output_dir / '04_remote_ratio.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Sauvegardé")
    
    print(f"\n✅ Visualisations créées: {output_dir}/")

def generate_report(df):
    """Générer un rapport EDA en TXT"""
    
    print("\n" + "="*60)
    print("GÉNÉRATION DU RAPPORT TXT")
    print("="*60)
    
    # Calculs pour le rapport
    senior_salary = df[df['experience_level'] == 'senior']['salary_usd'].mean()
    entry_salary = df[df['experience_level'] == 'entry-level']['salary_usd'].mean()
    remote_100 = len(df[df['remote_ratio'] == 100])
    remote_0 = len(df[df['remote_ratio'] == 0])
    top_location = df['location'].value_counts().head(1).index[0]
    top_location_count = df['location'].value_counts().head(1).values[0]
    
    report = f"""
{'='*70}
RAPPORT EDA - JOB-CANDIDATE MATCHING SYSTEM
Dataset fusionné: HuggingFace + RemoteOK
{'='*70}

📊 DONNÉES GÉNÉRALES
─────────────────────────────────────────────────────────────────────
- Nombre d'offres: {len(df):,}
- Nombre de colonnes: {df.shape[1]}
- Localisations: {df['location'].nunique()} pays/régions
- Sources: HuggingFace + RemoteOK

💰 ANALYSE DES SALAIRES (USD)
─────────────────────────────────────────────────────────────────────
- Salaire minimum: ${df['salary_usd'].min():,.0f}
- Salaire maximum: ${df['salary_usd'].max():,.0f}
- Salaire moyen: ${df['salary_usd'].mean():,.0f}
- Salaire médian: ${df['salary_usd'].median():,.0f}
- Écart-type: ${df['salary_usd'].std():,.0f}
- Q1 (25%): ${df['salary_usd'].quantile(0.25):,.0f}
- Q3 (75%): ${df['salary_usd'].quantile(0.75):,.0f}

👤 DISTRIBUTION PAR EXPÉRIENCE
─────────────────────────────────────────────────────────────────────
"""
    
    for level, count in df['experience_level'].value_counts().items():
        pct = (count / len(df)) * 100
        salary_mean = df[df['experience_level'] == level]['salary_usd'].mean()
        report += f"\n{level.upper()}:\n"
        report += f"  - Nombre: {count} ({pct:.1f}%)\n"
        report += f"  - Salaire moyen: ${salary_mean:,.0f}\n"
    
    report += f"""

🌍 TOP 10 LOCALISATIONS
─────────────────────────────────────────────────────────────────────
"""
    
    for loc, count in df['location'].value_counts().head(10).items():
        pct = (count / len(df)) * 100
        salary_mean = df[df['location'] == loc]['salary_usd'].mean()
        report += f"{loc}: {count} offres ({pct:.1f}%) - Salaire moyen: ${salary_mean:,.0f}\n"
    
    report += f"""

📱 REMOTE RATIO
─────────────────────────────────────────────────────────────────────
"""
    
    for ratio, count in df['remote_ratio'].value_counts().sort_index().items():
        pct = (count / len(df)) * 100
        label = "100% Remote" if ratio == 100 else f"{ratio}% Remote"
        report += f"{label}: {count} ({pct:.1f}%)\n"
    
    report += f"""

💼 COMPANY SIZE
─────────────────────────────────────────────────────────────────────
"""
    
    size_mapping = {'S': 'Small', 'M': 'Medium', 'L': 'Large'}
    for size, count in df['company_size'].value_counts().items():
        pct = (count / len(df)) * 100
        label = size_mapping.get(size, size)
        report += f"{label}: {count} ({pct:.1f}%)\n"
    
    report += f"""

📊 DISTRIBUTION PAR SOURCE
─────────────────────────────────────────────────────────────────────
"""
    
    for source, count in df['source'].value_counts().items():
        pct = (count / len(df)) * 100
        report += f"{source}: {count} offres ({pct:.1f}%)\n"
    
    report += f"""

🔍 OBSERVATIONS CLÉS
─────────────────────────────────────────────────────────────────────
1. Distribution des salaires:
   → Forte concentration entre ${df['salary_usd'].quantile(0.25):,.0f} et ${df['salary_usd'].quantile(0.75):,.0f}

2. Tendance expérience vs salaire:
   → Senior gagnent en moyenne ${senior_salary:,.0f}
   → Entry-level gagnent en moyenne ${entry_salary:,.0f}

3. Télétravail:
   → {remote_100} offres 100% remote
   → {remote_0} offres on-site

4. Localisations principales:
   → {top_location}: {top_location_count} offres

5. Sources de données:
   → HuggingFace: offres avec salaires structurées
   → RemoteOK: offres 100% remote, sans salaires standardisées

📈 RECOMMANDATIONS POUR WEEK 2
─────────────────────────────────────────────────────────────────────
1. Feature Engineering:
   - Créer des catégories de salaires
   - Encoder les niveaux d'expérience
   - Normaliser les données
   - Gérer les valeurs manquantes

2. Vectorisation:
   - Encoder job_title avec one-hot ou label encoding
   - Normaliser salary_usd
   - Gérer les NaN pour RemoteOK

3. Clustering:
   - Préparer pour K-Means
   - Déterminer nombre optimal de clusters
   - Analyser les clusters par source

{'='*70}
"""
    
    # Sauvegarder
    report_path = Path('data/processed/EDA_REPORT.txt')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Rapport TXT sauvegardé: {report_path}")

def generate_pdf_report(df):
    """
    ============================================================
    NOUVELLE FONCTION: Générer le rapport PDF avec images
    ============================================================
    """
    
    print("\n" + "="*60)
    print("GÉNÉRATION DU RAPPORT PDF")
    print("="*60)
    
    # Chemin PDF de sortie
    pdf_path = Path('data/processed/EDA_REPORT.pdf')
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Créer le document PDF
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                            rightMargin=0.4*inch, leftMargin=0.4*inch,
                            topMargin=0.5*inch, bottomMargin=0.5*inch)
    
    # Définir les styles
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=12,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=13,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=10,
        spaceBefore=10,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    
    # Construire le contenu (story)
    story = []
    
    # ==================== PAGE 1: TITRE ====================
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("📊 RAPPORT D'ANALYSE EXPLORATOIRE", title_style))
    story.append(Paragraph("Job-Candidate Matching System", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"<i>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>", styles['Normal']))
    story.append(Spacer(1, 0.5*inch))
    
    # Tableau d'information
    info_data = [
        ['Métrique', 'Valeur'],
        ['Total Offres', f'{len(df):,}'],
        ['Sources', 'HuggingFace + RemoteOK'],
        ['Localisations', f'{df["location"].nunique()}'],
        ['Colonnes', f'{df.shape[1]}'],
    ]
    info_table = Table(info_data, colWidths=[2.5*inch, 2.5*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(info_table)
    
    story.append(PageBreak())
    
    # ==================== PAGE 2: SALAIRES ====================
    story.append(Paragraph("1. ANALYSE DES SALAIRES", heading_style))
    
    salary_data = [
        ['Métrique', 'Valeur'],
        ['Minimum', f"${df['salary_usd'].min():,.0f}"],
        ['Maximum', f"${df['salary_usd'].max():,.0f}"],
        ['Moyen', f"${df['salary_usd'].mean():,.0f}"],
        ['Médian', f"${df['salary_usd'].median():,.0f}"],
        ['Q1-Q3', f"${df['salary_usd'].quantile(0.25):,.0f} - ${df['salary_usd'].quantile(0.75):,.0f}"],
    ]
    salary_table = Table(salary_data, colWidths=[2.5*inch, 2.5*inch])
    salary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightgreen),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(salary_table)
    
    story.append(PageBreak())
    
    # ==================== PAGE 3: IMAGE 1 ====================
    story.append(Paragraph("2. DISTRIBUTION DES SALAIRES", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    try:
        img1 = Image('data/visualizations/01_salary_distribution.png', width=7*inch, height=5*inch)
        story.append(img1)
    except Exception as e:
        story.append(Paragraph(f"<i>[Image non disponible: {e}]</i>", body_style))
    
    story.append(PageBreak())
    
    # ==================== PAGE 4: EXPÉRIENCE ====================
    story.append(Paragraph("3. DISTRIBUTION PAR EXPÉRIENCE", heading_style))
    
    exp_data = [['Niveau', 'Count', '%']]
    for level, count in df['experience_level'].value_counts().items():
        pct = (count / len(df)) * 100
        exp_data.append([level, str(count), f'{pct:.1f}%'])
    
    exp_table = Table(exp_data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    exp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightyellow),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    story.append(exp_table)
    
    story.append(PageBreak())
    
    # ==================== PAGE 5: IMAGE 2 ====================
    story.append(Paragraph("4. VISUALISATION EXPÉRIENCE", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    try:
        img2 = Image('data/visualizations/02_experience_distribution.png', width=7*inch, height=3*inch)
        story.append(img2)
    except Exception as e:
        story.append(Paragraph(f"<i>[Image non disponible: {e}]</i>", body_style))
    
    story.append(PageBreak())
    
    # ==================== PAGE 6: IMAGE 3 ====================
    story.append(Paragraph("5. TOP 15 LOCALISATIONS", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    try:
        img3 = Image('data/visualizations/03_top_locations.png', width=7*inch, height=4.5*inch)
        story.append(img3)
    except Exception as e:
        story.append(Paragraph(f"<i>[Image non disponible: {e}]</i>", body_style))
    
    story.append(PageBreak())
    
    # ==================== PAGE 7: IMAGE 4 ====================
    story.append(Paragraph("6. TÉLÉTRAVAIL", heading_style))
    story.append(Spacer(1, 0.1*inch))
    
    try:
        img4 = Image('data/visualizations/04_remote_ratio.png', width=7*inch, height=3.5*inch)
        story.append(img4)
    except Exception as e:
        story.append(Paragraph(f"<i>[Image non disponible: {e}]</i>", body_style))
    
    story.append(PageBreak())
    
    # ==================== PAGE 8: INSIGHTS ====================
    story.append(Paragraph("7. INSIGHTS CLÉS", heading_style))
    
    insights = [
        f"✓ Salaires: Min ${df['salary_usd'].min():,.0f}, Max ${df['salary_usd'].max():,.0f}, Moyen ${df['salary_usd'].mean():,.0f}",
        f"✓ {len(df[df['remote_ratio'] == 100])} offres 100% remote ({len(df[df['remote_ratio'] == 100])/len(df)*100:.1f}%)",
        f"✓ {df['location'].nunique()} pays/régions représentés",
        f"✓ Top localisation: {df['location'].value_counts().head(1).index[0]} ({df['location'].value_counts().head(1).values[0]} offres)",
    ]
    
    for insight in insights:
        story.append(Paragraph(f"• {insight}", body_style))
        story.append(Spacer(1, 0.08*inch))
    
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("=" * 70, body_style))
    story.append(Paragraph(
        f"<b>✓ WEEK 1 COMPLÉTÉE</b><br/>"
        f"Status: Ready for Week 2<br/>"
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        body_style
    ))
    
    # Construire le PDF
    doc.build(story)
    
    print(f"✅ Rapport PDF sauvegardé: {pdf_path}")

if __name__ == '__main__':
    # Charger les données nettoyées
    csv_path = Path('data/processed/jobs_cleaned.csv')
    if not csv_path.exists():
        print(f"❌ Fichier non trouvé: {csv_path}")
        print("   Exécutez d'abord: python code/02_clean_data.py")
        exit()
    
    df = pd.read_csv(csv_path)
    
    if df is not None:
        # 1. Créer visualisations
        create_visualizations(df)
        
        # 2. Générer rapport TXT
        generate_report(df)
        
        # 3. NOUVEAU: Générer rapport PDF
        generate_pdf_report(df)
        
        print("\n" + "="*60)
        print("✅ ÉTAPES 3-4 COMPLÉTÉES!")
        print("   - 4 visualisations PNG créées")
        print("   - Rapport EDA.txt généré")
        print("   - Rapport EDA.pdf généré")
        print("="*60)