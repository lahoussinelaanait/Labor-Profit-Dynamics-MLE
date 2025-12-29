# ============================================
# MONTAGE GOOGLE DRIVE
# ============================================
from google.colab import drive
drive.mount('/content/drive')

# ============================================
# IMPORTS ET CONFIGURATION
# ============================================
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
import warnings
warnings.filterwarnings('ignore')

# ============================================
# FONCTIONS UTILITAIRES
# ============================================
def calculate_kendall_details(x, y):
    """Calcule tau de Kendall et compte les paires concordantes/discordantes"""
    n = len(x)
    concordant = 0
    discordant = 0
    
    for i in range(n):
        for j in range(i+1, n):
            xi_xj = x[i] - x[j]
            yi_yj = y[i] - y[j]
            prod = xi_xj * yi_yj
            
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
    
    total_pairs = n * (n-1) / 2
    tau = (concordant - discordant) / total_pairs if total_pairs > 0 else 0
    
    return tau, concordant, discordant, int(total_pairs)

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
print("📁 CHARGEMENT DES DONNÉES TECHNOLOGIQUES...")
print("=" * 70)

# Charger les fichiers MLE
try:
    gamma_tech = pd.read_csv('/content/drive/MyDrive/GAMMA_EXPECTATIONS_TECHNOLOGY_2005_2023.csv')
    print(f"✅ Gamma expectations technologie: {gamma_tech.shape}")
    print("\nContenu de gamma_tech:")
    print(gamma_tech)
except Exception as e:
    print(f"❌ Erreur chargement gamma: {e}")
    gamma_tech = None

try:
    std_tech = pd.read_csv('/content/drive/MyDrive/MLE_STANDARD_DEVIATIONS_TECHNOLOGY_2005_2023.csv')
    print(f"\n✅ MLE std technologie: {std_tech.shape}")
    print("\nContenu de std_tech:")
    print(std_tech)
except Exception as e:
    print(f"❌ Erreur chargement std: {e}")
    std_tech = None

try:
    # Charger avec point-virgule comme séparateur
    tech_data = pd.read_csv('/content/drive/MyDrive/tech_intensive_panel2005-2023.csv', sep=';')
    print(f"\n✅ Données technologiques: {tech_data.shape}")
    
    print("\n📊 Colonnes disponibles:")
    for i, col in enumerate(tech_data.columns):
        print(f"{i+1:2d}. {col}: {tech_data[col].dtype}")
        
except Exception as e1:
    print(f"❌ Premier essai avec ';' a échoué: {e1}")
    try:
        # Essayer avec la virgule
        tech_data = pd.read_csv('/content/drive/MyDrive/tech_intensive_panel2005-2023.csv')
        print(f"✅ Données technologiques (avec virgule): {tech_data.shape}")
        
        print("\n📊 Colonnes disponibles:")
        for i, col in enumerate(tech_data.columns):
            print(f"{i+1:2d}. {col}: {tech_data[col].dtype}")
            
    except Exception as e2:
        print(f"❌ Deuxième essai avec virgule a échoué: {e2}")
        try:
            # Essayer avec détection automatique
            tech_data = pd.read_csv('/content/drive/MyDrive/tech_intensive_panel2005-2023.csv', sep=None, engine='python')
            print(f"✅ Données technologiques (auto-détection): {tech_data.shape}")
            
            print("\n📊 Colonnes disponibles:")
            for i, col in enumerate(tech_data.columns):
                print(f"{i+1:2d}. {col}: {tech_data[col].dtype}")
                
        except Exception as e3:
            print(f"❌ Tous les essais ont échoué: {e3}")
            tech_data = None

# ============================================
# EXTRACTION DES VALEURS MLE
# ============================================
print("\n🔍 EXTRACTION DES VALEURS MLE...")
print("=" * 70)

# Extraire les valeurs avec les noms exacts
mle_values = {}

if gamma_tech is not None:
    print("\n📊 Extraction des espérances MLE:")
    
    # Chercher chaque variable
    gamma_mapping = {
        'EBIT': ['EBIT (positive only)', 'EBIT'],
        'R&D': ['R&D'],
        'PP&E': ['PP&E'],
        'Personnel': ['Personnel'],
        'Z1': ['Z1 = EBIT × Personnel', 'Z1'],
        'Z2': ['Z2 = EBIT × PP&E', 'Z2'],
        'Z3': ['Z3 = EBIT × R&D', 'Z3'],
        'Z4': ['Z4 = R&D × Personnel', 'Z4']
    }
    
    for key, patterns in gamma_mapping.items():
        found = False
        for pattern in patterns:
            matches = gamma_tech[gamma_tech['Variable'].str.contains(pattern, na=False, case=False)]
            if not matches.empty:
                value = matches.iloc[0]['Gamma_Expectation']
                var_name = matches.iloc[0]['Variable']
                mle_values[key] = float(value)
                print(f"  E[{key}] = {value:.6f} (trouvé comme '{var_name}')")
                found = True
                break
        if not found:
            print(f"  ❌ {key}: NON TROUVÉ")
            mle_values[key] = None

if std_tech is not None:
    print("\n📊 Extraction des écarts-types MLE:")
    
    std_mapping = {
        'EBIT': ['EBIT'],
        'R&D': ['R&D'],
        'PP&E': ['PP&E'],
        'Personnel': ['Personnel']
    }
    
    for key, patterns in std_mapping.items():
        found = False
        for pattern in patterns:
            matches = std_tech[std_tech['Variable'].str.contains(pattern, na=False, case=False)]
            if not matches.empty:
                value = matches.iloc[0]['MLE_Std_Deviation']
                var_name = matches.iloc[0]['Variable']
                mle_values[f'std_{key}'] = float(value)
                print(f"  σ[{key}] = {value:.6f} (trouvé comme '{var_name}')")
                found = True
                break
        if not found:
            print(f"  ❌ σ[{key}]: NON TROUVÉ")
            mle_values[f'std_{key}'] = None

# ============================================
# EXTRACTION DES DONNÉES ORIGINALES
# ============================================
print("\n🔍 EXTRACTION DES DONNÉES ORIGINALES...")
print("=" * 70)

if tech_data is not None:
    # Afficher les premières lignes pour comprendre la structure
    print("\n📊 Aperçu des données (5 premières lignes):")
    print(tech_data.head())
    
    # Chercher les colonnes par noms exacts ou partiels
    column_mapping = {}
    
    # Noms possibles pour chaque variable
    possible_names = {
        'EBIT': ['EBIT', 'ebit', 'EBIT (Billion USD)'],
        'Personnel': ['Employees', 'Personnel', 'employees', 'Employees (10k units)'],
        'PP&E': ['PP&E', 'PPE', 'PP&E (Billion USD)', 'property', 'plant'],
        'R&D': ['R&D', 'R&D (Billion USD)', 'research', 'development']
    }
    
    print("\n🔍 Recherche des colonnes:")
    for var_name, possible_list in possible_names.items():
        found_cols = []
        for col in tech_data.columns:
            col_clean = str(col).strip()
            for possible in possible_list:
                if possible in col_clean:
                    found_cols.append(col)
                    break
        
        if found_cols:
            column_mapping[var_name] = found_cols[0]
            print(f"  {var_name}: '{found_cols[0]}'")
        else:
            # Essayer une recherche insensible à la casse
            for col in tech_data.columns:
                col_lower = str(col).lower()
                var_lower = var_name.lower()
                if var_lower in col_lower:
                    column_mapping[var_name] = col
                    print(f"  {var_name}: '{col}' (recherche insensible)")
                    break
            else:
                column_mapping[var_name] = None
                print(f"  {var_name}: NON TROUVÉ")
    
    # Vérifier si toutes les colonnes sont trouvées
    missing_vars = [var for var, col in column_mapping.items() if col is None]
    
    if not missing_vars:
        # Préparer les données
        data_columns = [column_mapping['EBIT'], column_mapping['Personnel'], 
                       column_mapping['PP&E'], column_mapping['R&D']]
        
        print(f"\n📊 Colonnes sélectionnées pour l'analyse:")
        for var, col in column_mapping.items():
            print(f"  {var}: {col}")
        
        # Créer un DataFrame nettoyé
        data_clean = tech_data[data_columns].copy()
        
        # Convertir toutes les colonnes en numérique
        for col in data_columns:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        
        # Supprimer les valeurs manquantes
        initial_count = len(data_clean)
        data_clean = data_clean.dropna()
        final_count = len(data_clean)
        
        # Extraire les arrays numpy
        ebit_data = data_clean[column_mapping['EBIT']].values.astype(float)
        personnel_data = data_clean[column_mapping['Personnel']].values.astype(float)
        ppe_data = data_clean[column_mapping['PP&E']].values.astype(float)
        rd_data = data_clean[column_mapping['R&D']].values.astype(float)
        
        print(f"\n✅ Données extraites avec succès:")
        print(f"  Observations initiales: {initial_count}")
        print(f"  Observations après nettoyage: {final_count}")
        print(f"  Valeurs supprimées: {initial_count - final_count}")
        
        # Statistiques descriptives
        if final_count > 0:
            print(f"\n📈 Statistiques descriptives (après nettoyage):")
            stats_data = {
                'EBIT': ebit_data,
                'Personnel': personnel_data,
                'PP&E': ppe_data,
                'R&D': rd_data
            }
            
            for name, data in stats_data.items():
                print(f"  {name}: n={len(data)}, mean={data.mean():.2f}, std={data.std():.2f}, min={data.min():.2f}, max={data.max():.2f}")
    else:
        print(f"\n❌ Variables manquantes: {missing_vars}")
        
        # Si EBIT n'est pas trouvé, chercher manuellement
        print("\n🔍 Recherche manuelle des colonnes:")
        print("Colonnes disponibles:")
        for i, col in enumerate(tech_data.columns):
            print(f"{i+1:2d}. {col}")
        
        # Proposer une sélection par index
        print("\n⚠️  Sélection manuelle requise.")
        print("Veuillez indiquer les indices des colonnes pour:")
        
        var_indices = {}
        for var in missing_vars:
            idx = int(input(f"  {var}: Entrez le numéro de colonne (1-{len(tech_data.columns)}): "))
            var_indices[var] = tech_data.columns[idx-1]
            print(f"    → Sélectionné: {var_indices[var]}")
        
        # Mettre à jour column_mapping
        for var, col in var_indices.items():
            column_mapping[var] = col
        
        # Réessayer l'extraction
        data_columns = [column_mapping['EBIT'], column_mapping['Personnel'], 
                       column_mapping['PP&E'], column_mapping['R&D']]
        
        data_clean = tech_data[data_columns].copy()
        for col in data_columns:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        
        data_clean = data_clean.dropna()
        
        ebit_data = data_clean[column_mapping['EBIT']].values.astype(float)
        personnel_data = data_clean[column_mapping['Personnel']].values.astype(float)
        ppe_data = data_clean[column_mapping['PP&E']].values.astype(float)
        rd_data = data_clean[column_mapping['R&D']].values.astype(float)
        
        print(f"\n✅ Données extraites après sélection manuelle:")
        print(f"  Observations: {len(data_clean)}")
        
else:
    print("❌ Données technologiques non disponibles")
    ebit_data = personnel_data = ppe_data = rd_data = None

# ============================================
# CALCUL DES CORRÉLATIONS MLE
# ============================================
print("\n" + "="*80)
print("🧮 CALCUL DES CORRÉLATIONS MLE - TECHNOLOGIE")
print("="*80)

def calculate_mle_pair(var1, var2, z_var, pair_name):
    """Calcule la corrélation MLE pour une paire de variables"""
    print(f"\n{pair_name}:")
    print("-" * 40)
    
    # Vérifier que nous avons toutes les valeurs nécessaires
    required_values = [var1, var2, z_var, f'std_{var1}', f'std_{var2}']
    missing = [k for k in required_values if k not in mle_values or mle_values[k] is None]
    
    if missing:
        print(f"  ❌ Données MLE manquantes: {missing}")
        return None, None
    
    E1 = mle_values[var1]
    E2 = mle_values[var2]
    E_z = mle_values[z_var]
    std1 = mle_values[f'std_{var1}']
    std2 = mle_values[f'std_{var2}']
    
    # Calcul covariance
    covariance = E_z - (E1 * E2)
    print(f"  E[{var1}] = {E1:.6f}")
    print(f"  E[{var2}] = {E2:.6f}")
    print(f"  E[{z_var}] = {E_z:.6f}")
    print(f"  Covariance MLE = E[{z_var}] - E[{var1}] × E[{var2}]")
    print(f"                 = {E_z:.6f} - ({E1:.6f} × {E2:.6f})")
    print(f"                 = {covariance:.6f}")
    
    # Vérifier que les écarts-types ne sont pas nuls
    if std1 == 0 or std2 == 0:
        print(f"  ❌ Écart-type nul (σ[{var1}]={std1}, σ[{var2}]={std2})")
        return covariance, None
    
    # Calcul corrélation
    correlation = covariance / (std1 * std2)
    print(f"\n  σ[{var1}] = {std1:.6f}")
    print(f"  σ[{var2}] = {std2:.6f}")
    print(f"  Corrélation MLE = Covariance / (σ[{var1}] × σ[{var2}])")
    print(f"                   = {covariance:.6f} / ({std1:.6f} × {std2:.6f})")
    print(f"                   = {correlation:.6f}")
    
    return covariance, correlation

# Calculer les 4 paires
print("📊 CALCUL SELON VOS FORMULES:")
print("="*60)

mle_results = {}

# 1. (EBIT, Personnel)
cov1, corr1 = calculate_mle_pair('EBIT', 'Personnel', 'Z1', "1. (EBIT, Personnel)")
mle_results['cov_ebit_personnel'] = cov1
mle_results['corr_ebit_personnel'] = corr1

# 2. (EBIT, PP&E)
cov2, corr2 = calculate_mle_pair('EBIT', 'PP&E', 'Z2', "2. (EBIT, PP&E)")
mle_results['cov_ebit_ppe'] = cov2
mle_results['corr_ebit_ppe'] = corr2

# 3. (EBIT, R&D)
cov3, corr3 = calculate_mle_pair('EBIT', 'R&D', 'Z3', "3. (EBIT, R&D)")
mle_results['cov_ebit_rd'] = cov3
mle_results['corr_ebit_rd'] = corr3

# 4. (Personnel, R&D)
cov4, corr4 = calculate_mle_pair('Personnel', 'R&D', 'Z4', "4. (Personnel, R&D)")
mle_results['cov_personnel_rd'] = cov4
mle_results['corr_personnel_rd'] = corr4

# ============================================
# CALCUL DES CORRÉLATIONS CLASSIQUES
# ============================================
print("\n" + "="*80)
print("📊 CALCUL DES CORRÉLATIONS CLASSIQUES - TECHNOLOGIE")
print("="*80)

if all(data is not None for data in [ebit_data, personnel_data, ppe_data, rd_data]) and len(ebit_data) > 1:
    print(f"📈 Données disponibles: {len(ebit_data)} observations")
    
    # Calculer les corrélations pour chaque paire
    results_classical = {}
    
    # Fonction pour calculer et afficher
    def calculate_and_display(x, y, pair_name):
        print(f"\n{pair_name}:")
        print("-" * 30)
        
        try:
            # Pearson
            pearson_corr, pearson_p = pearsonr(x, y)
            
            # Spearman
            spearman_corr, spearman_p = spearmanr(x, y)
            
            # Kendall
            kendall_tau, concordant, discordant, total_pairs = calculate_kendall_details(x, y)
            
            print(f"  Pearson:  {pearson_corr:.6f} (p-value: {pearson_p:.6e})")
            print(f"  Spearman: {spearman_corr:.6f} (p-value: {spearman_p:.6e})")
            print(f"  Kendall:  {kendall_tau:.6f}")
            print(f"    Paires concordantes: {concordant:,}")
            print(f"    Paires discordantes: {discordant:,}")
            print(f"    Total paires: {total_pairs:,}")
            
            return {
                'pearson': pearson_corr,
                'pearson_p': pearson_p,
                'spearman': spearman_corr,
                'spearman_p': spearman_p,
                'kendall': kendall_tau,
                'concordant': concordant,
                'discordant': discordant,
                'total_pairs': total_pairs
            }
            
        except Exception as e:
            print(f"  ❌ Erreur dans les calculs: {e}")
            return None
    
    # Calculer les 4 paires
    classical_pairs = [
        ("1. CORRÉLATIONS (EBIT, Personnel)", ebit_data, personnel_data, 'ebit_personnel'),
        ("2. CORRÉLATIONS (EBIT, PP&E)", ebit_data, ppe_data, 'ebit_ppe'),
        ("3. CORRÉLATIONS (EBIT, R&D)", ebit_data, rd_data, 'ebit_rd'),
        ("4. CORRÉLATIONS (Personnel, R&D)", personnel_data, rd_data, 'personnel_rd')
    ]
    
    for display_name, x_data, y_data, key in classical_pairs:
        result = calculate_and_display(x_data, y_data, display_name)
        if result is not None:
            results_classical[key] = result
    
else:
    print("❌ Données insuffisantes pour calculer les corrélations classiques")
    results_classical = None

# ============================================
# TABLEAU COMPARATIF
# ============================================
print("\n" + "="*80)
print("📋 TABLEAU COMPARATIF DES CORRÉLATIONS - TECHNOLOGIE")
print("="*80)

# Créer le tableau
table_data = []

pairs_info = [
    ('(EBIT, Personnel)', 'cov_ebit_personnel', 'corr_ebit_personnel', 'ebit_personnel'),
    ('(EBIT, PP&E)', 'cov_ebit_ppe', 'corr_ebit_ppe', 'ebit_ppe'),
    ('(EBIT, R&D)', 'cov_ebit_rd', 'corr_ebit_rd', 'ebit_rd'),
    ('(Personnel, R&D)', 'cov_personnel_rd', 'corr_personnel_rd', 'personnel_rd')
]

for pair_name, cov_key, corr_key, classical_key in pairs_info:
    row = {'Paire': pair_name}
    
    # Ajouter MLE
    cov_val = mle_results.get(cov_key)
    corr_val = mle_results.get(corr_key)
    
    row['Covariance MLE'] = f"{cov_val:.6f}" if cov_val is not None else "N/A"
    row['Corrélation MLE'] = f"{corr_val:.6f}" if corr_val is not None else "N/A"
    
    # Ajouter classiques
    if results_classical and classical_key in results_classical:
        classical = results_classical[classical_key]
        row['Pearson'] = f"{classical['pearson']:.6f}"
        row['p-value Pearson'] = f"{classical['pearson_p']:.6e}"
        row['Spearman'] = f"{classical['spearman']:.6f}"
        row['p-value Spearman'] = f"{classical['spearman_p']:.6e}"
        row['Kendall tau'] = f"{classical['kendall']:.6f}"
        row['Paires concordantes'] = f"{classical['concordant']:,}"
        row['Paires discordantes'] = f"{classical['discordant']:,}"
        row['Total paires'] = f"{classical['total_pairs']:,}"
    else:
        for col in ['Pearson', 'p-value Pearson', 'Spearman', 'p-value Spearman', 
                   'Kendall tau', 'Paires concordantes', 'Paires discordantes', 'Total paires']:
            row[col] = "N/A"
    
    table_data.append(row)

# Créer DataFrame
results_df = pd.DataFrame(table_data)

# Afficher le tableau
print("\n" + results_df.to_string(index=False))

# ============================================
# ANALYSE COMPARATIVE
# ============================================
print("\n" + "="*80)
print("📊 ANALYSE COMPARATIVE - SECTEUR TECHNOLOGIQUE")
print("="*80)

print("\n🔍 ÉCHELLE DE CORRÉLATION (valeur absolue):")
print("  0.00 - 0.19 : Très faible")
print("  0.20 - 0.39 : Faible")
print("  0.40 - 0.59 : Modérée")
print("  0.60 - 0.79 : Forte")
print("  0.80 - 1.00 : Très forte")

print("\n📈 INTERPRÉTATION DES CORRÉLATIONS MLE:")

for _, row in results_df.iterrows():
    pair = row['Paire']
    corr_mle = row['Corrélation MLE']
    
    if corr_mle != "N/A":
        corr_val = float(corr_mle)
        abs_corr = abs(corr_val)
        
        if abs_corr >= 0.8:
            force = "très forte"
        elif abs_corr >= 0.6:
            force = "forte"
        elif abs_corr >= 0.4:
            force = "modérée"
        elif abs_corr >= 0.2:
            force = "faible"
        else:
            force = "très faible"
        
        direction = "positive" if corr_val > 0 else "négative"
        
        print(f"\n  {pair}:")
        print(f"    • Corrélation MLE: {force} et {direction} ({corr_val:.3f})")
        
        pearson = row['Pearson']
        if pearson != "N/A":
            pearson_val = float(pearson)
            diff = abs(corr_val - pearson_val)
            if diff > 0.1:
                print(f"    • Note: Différence notable avec Pearson ({pearson_val:.3f})")
                print(f"      → Écart de {diff:.3f} entre les méthodes")

# ============================================
# SAUVEGARDE DES RÉSULTATS
# ============================================
print("\n" + "="*80)
print("💾 SAUVEGARDE DES RÉSULTATS")
print("="*80)

try:
    # Sauvegarder le tableau principal
    output_path = '/content/drive/MyDrive/RESULTATS_CORRELATIONS_TECHNOLOGIE.csv'
    results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"✅ Tableau principal sauvegardé: {output_path}")
    
    # Sauvegarder les détails MLE
    details_path = '/content/drive/MyDrive/DETAILS_CALCULS_MLE_TECHNOLOGIE.txt'
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("DÉTAILS DES CALCULS MLE - SECTEUR TECHNOLOGIQUE\n")
        f.write("="*70 + "\n\n")
        
        f.write("Valeurs extraites:\n")
        f.write("-"*40 + "\n")
        for key, value in mle_values.items():
            if value is not None:
                f.write(f"{key}: {value}\n")
        
        f.write("\n\nCalculs des covariances MLE:\n")
        f.write("="*40 + "\n\n")
        
        if mle_results['cov_ebit_personnel'] is not None:
            f.write("1. Covariance(EBIT, Personnel) = E[Z1] - E[EBIT] × E[Personnel]\n")
            f.write(f"   = {mle_values.get('Z1', 'N/A')} - ({mle_values.get('EBIT', 'N/A')} × {mle_values.get('Personnel', 'N/A')})\n")
            f.write(f"   = {mle_results['cov_ebit_personnel']:.6f}\n\n")
        
        if mle_results['cov_ebit_ppe'] is not None:
            f.write("2. Covariance(EBIT, PP&E) = E[Z2] - E[EBIT] × E[PP&E]\n")
            f.write(f"   = {mle_values.get('Z2', 'N/A')} - ({mle_values.get('EBIT', 'N/A')} × {mle_values.get('PP&E', 'N/A')})\n")
            f.write(f"   = {mle_results['cov_ebit_ppe']:.6f}\n\n")
        
        if mle_results['cov_ebit_rd'] is not None:
            f.write("3. Covariance(EBIT, R&D) = E[Z3] - E[EBIT] × E[R&D]\n")
            f.write(f"   = {mle_values.get('Z3', 'N/A')} - ({mle_values.get('EBIT', 'N/A')} × {mle_values.get('R&D', 'N/A')})\n")
            f.write(f"   = {mle_results['cov_ebit_rd']:.6f}\n\n")
        
        if mle_results['cov_personnel_rd'] is not None:
            f.write("4. Covariance(Personnel, R&D) = E[Z4] - E[Personnel] × E[R&D]\n")
            f.write(f"   = {mle_values.get('Z4', 'N/A')} - ({mle_values.get('Personnel', 'N/A')} × {mle_values.get('R&D', 'N/A')})\n")
            f.write(f"   = {mle_results['cov_personnel_rd']:.6f}\n")
    
    print(f"✅ Détails MLE sauvegardés: {details_path}")
    
except Exception as e:
    print(f"⚠️  Erreur lors de la sauvegarde: {e}")

print("\n" + "="*80)
print("✅ ANALYSE DU SECTEUR TECHNOLOGIQUE TERMINÉE !")
print("="*80)

print("\n🎯 RÉSUMÉ:")
print("  1. Fichiers MLE analysés avec succès")
print("  2. 4 paires de corrélations calculées")
print("  3. Tous les résultats sauvegardés dans votre Google Drive")
print("  4. Chemins des fichiers:")
print(f"     - Tableau: /content/drive/MyDrive/RESULTATS_CORRELATIONS_TECHNOLOGIE.csv")
print(f"     - Détails MLE: /content/drive/MyDrive/DETAILS_CALCULS_MLE_TECHNOLOGIE.txt")