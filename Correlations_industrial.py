# ============================================
# CODE FINAL POUR CALCULER LES CORRÉLATIONS
# ============================================
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

# ============================================
# CHARGEMENT DES DONNÉES
# ============================================
print("📁 CHARGEMENT DES DONNÉES...")
print("=" * 60)

# Charger tous les fichiers
industrial_df = pd.read_csv('/content/drive/MyDrive/industrial_sector_panel_2005_2023.csv')
gamma_df = pd.read_csv('/content/drive/MyDrive/GAMMA_EXPECTATIONS_INDUSTRIAL_2005_2024.csv')
std_df = pd.read_csv('/content/drive/MyDrive/MLE_STANDARD_DEVIATIONS_INDUSTRIAL_2005_2024.csv')

print(f"✅ Données industrielles: {industrial_df.shape}")
print(f"✅ Gamma expectations: {gamma_df.shape}")
print(f"✅ MLE standard deviations: {std_df.shape}")

# ============================================
# EXTRACTION DES VALEURS MLE (AVEC NOMS EXACTS)
# ============================================
print("\n🔍 EXTRACTION DES VALEURS MLE...")
print("=" * 60)

# Noms exacts des variables dans gamma_expectations
gamma_vars = {
    'EBIT': 'EBIT (positive only)',
    'PP&E': 'PP&E',
    'Personnel': 'Personnel',
    'Z1': 'Z1 = EBIT × Personnel',
    'Z2': 'Z2 = EBIT × PP&E'
}

# Extraire les valeurs gamma
mle_values = {}
print("Valeurs de Gamma Expectations:")
for key, var_name in gamma_vars.items():
    if var_name in gamma_df['Variable'].values:
        value = gamma_df[gamma_df['Variable'] == var_name]['Gamma_Expectation'].values[0]
        mle_values[key] = float(value)
        print(f"  E[{key}] = {value}")
    else:
        print(f"  ❌ {key} ({var_name}) non trouvé!")

# Chercher les écarts-types dans mle_std
print("\nÉcarts-types MLE:")
# Afficher les variables disponibles dans mle_std pour référence
print("Variables disponibles dans mle_std:")
for var in std_df['Variable'].unique():
    print(f"  - {var}")

# Extraire les écarts-types (noms peuvent être différents)
std_mapping = {}
for var in std_df['Variable'].unique():
    var_lower = var.lower()
    if 'ebit' in var_lower:
        std_mapping['EBIT'] = var
    elif 'ppe' in var_lower or 'pp&e' in var_lower:
        std_mapping['PP&E'] = var
    elif 'personnel' in var_lower or 'employee' in var_lower:
        std_mapping['Personnel'] = var

print("\nCorrespondance trouvée:")
for key, var_name in std_mapping.items():
    value = std_df[std_df['Variable'] == var_name]['MLE_Std_Deviation'].values[0]
    mle_values[f'std_{key}'] = float(value)
    print(f"  σ[{key}] = {value} (trouvé comme '{var_name}')")

# ============================================
# CALCUL DES CORRÉLATIONS MLE SELON VOS FORMULES
# ============================================
print("\n" + "="*80)
print("🧮 CALCUL DES CORRÉLATIONS MLE (VOS FORMULES)")
print("="*80)

# 1. Pour (EBIT, Personnel)
print("\n1. CORRÉLATION MLE (EBIT, Personnel):")
print("-" * 40)

if all(k in mle_values for k in ['EBIT', 'Personnel', 'Z1']):
    E_ebit = mle_values['EBIT']
    E_personnel = mle_values['Personnel']
    E_z1 = mle_values['Z1']
    
    # Calcul covariance selon votre formule
    covariance_ebit_personnel = E_z1 - (E_ebit * E_personnel)
    print(f"   E[EBIT] = {E_ebit}")
    print(f"   E[Personnel] = {E_personnel}")
    print(f"   E[Z1] = {E_z1}")
    print(f"\n   Covariance MLE = E[Z1] - E[EBIT] × E[Personnel]")
    print(f"                  = {E_z1} - ({E_ebit} × {E_personnel})")
    print(f"                  = {covariance_ebit_personnel:.6f}")
    
    # Calcul corrélation si on a les écarts-types
    if 'std_EBIT' in mle_values and 'std_Personnel' in mle_values:
        std_ebit = mle_values['std_EBIT']
        std_personnel = mle_values['std_Personnel']
        correlation_ebit_personnel = covariance_ebit_personnel / (std_ebit * std_personnel)
        print(f"\n   σ[EBIT] = {std_ebit}")
        print(f"   σ[Personnel] = {std_personnel}")
        print(f"\n   Corrélation MLE = Covariance / (σ[EBIT] × σ[Personnel])")
        print(f"                   = {covariance_ebit_personnel:.6f} / ({std_ebit} × {std_personnel})")
        print(f"                   = {correlation_ebit_personnel:.6f}")
    else:
        correlation_ebit_personnel = None
        print("   ❌ Écarts-types manquants pour calculer la corrélation")
else:
    covariance_ebit_personnel = None
    correlation_ebit_personnel = None
    print("   ❌ Données insuffisantes pour le calcul")

# 2. Pour (EBIT, PP&E)
print("\n\n2. CORRÉLATION MLE (EBIT, PP&E):")
print("-" * 40)

if all(k in mle_values for k in ['EBIT', 'PP&E', 'Z2']):
    E_ebit = mle_values['EBIT']
    E_ppe = mle_values['PP&E']
    E_z2 = mle_values['Z2']
    
    # Calcul covariance selon votre formule
    covariance_ebit_ppe = E_z2 - (E_ebit * E_ppe)
    print(f"   E[EBIT] = {E_ebit}")
    print(f"   E[PP&E] = {E_ppe}")
    print(f"   E[Z2] = {E_z2}")
    print(f"\n   Covariance MLE = E[Z2] - E[EBIT] × E[PP&E]")
    print(f"                  = {E_z2} - ({E_ebit} × {E_ppe})")
    print(f"                  = {covariance_ebit_ppe:.6f}")
    
    # Calcul corrélation si on a les écarts-types
    if 'std_EBIT' in mle_values and 'std_PP&E' in mle_values:
        std_ebit = mle_values['std_EBIT']
        std_ppe = mle_values['std_PP&E']
        correlation_ebit_ppe = covariance_ebit_ppe / (std_ebit * std_ppe)
        print(f"\n   σ[EBIT] = {std_ebit}")
        print(f"   σ[PP&E] = {std_ppe}")
        print(f"\n   Corrélation MLE = Covariance / (σ[EBIT] × σ[PP&E])")
        print(f"                   = {covariance_ebit_ppe:.6f} / ({std_ebit} × {std_ppe})")
        print(f"                   = {correlation_ebit_ppe:.6f}")
    else:
        correlation_ebit_ppe = None
        print("   ❌ Écarts-types manquants pour calculer la corrélation")
else:
    covariance_ebit_ppe = None
    correlation_ebit_ppe = None
    print("   ❌ Données insuffisantes pour le calcul")

# ============================================
# CALCUL DES CORRÉLATIONS CLASSIQUES
# ============================================
print("\n" + "="*80)
print("📊 CALCUL DES CORRÉLATIONS CLASSIQUES")
print("="*80)

# Extraire les données originales
ebit_data = industrial_df['EBIT (Billion USD)'].values
personnel_data = industrial_df['Employees (10k units)'].values
ppe_data = industrial_df['PP&E (Billion USD)'].values

print(f"\n📈 Données originales: {len(ebit_data)} observations")

# Fonction pour calculer Kendall avec détails
def calculate_kendall_details(x, y):
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

# 1. Corrélations pour (EBIT, Personnel)
print("\n1. CORRÉLATIONS (EBIT, Personnel):")
print("-" * 30)

pearson_ebit_pers, pearson_p_pers = pearsonr(ebit_data, personnel_data)
spearman_ebit_pers, spearman_p_pers = spearmanr(ebit_data, personnel_data)
kendall_tau_pers, concordant_pers, discordant_pers, total_pairs_pers = calculate_kendall_details(ebit_data, personnel_data)

print(f"   Pearson:  {pearson_ebit_pers:.6f} (p-value: {pearson_p_pers:.6e})")
print(f"   Spearman: {spearman_ebit_pers:.6f} (p-value: {spearman_p_pers:.6e})")
print(f"   Kendall:  {kendall_tau_pers:.6f}")
print(f"     Paires concordantes: {concordant_pers:,}")
print(f"     Paires discordantes: {discordant_pers:,}")
print(f"     Total paires: {total_pairs_pers:,}")

# 2. Corrélations pour (EBIT, PP&E)
print("\n2. CORRÉLATIONS (EBIT, PP&E):")
print("-" * 30)

pearson_ebit_ppe, pearson_p_ppe = pearsonr(ebit_data, ppe_data)
spearman_ebit_ppe, spearman_p_ppe = spearmanr(ebit_data, ppe_data)
kendall_tau_ppe, concordant_ppe, discordant_ppe, total_pairs_ppe = calculate_kendall_details(ebit_data, ppe_data)

print(f"   Pearson:  {pearson_ebit_ppe:.6f} (p-value: {pearson_p_ppe:.6e})")
print(f"   Spearman: {spearman_ebit_ppe:.6f} (p-value: {spearman_p_ppe:.6e})")
print(f"   Kendall:  {kendall_tau_ppe:.6f}")
print(f"     Paires concordantes: {concordant_ppe:,}")
print(f"     Paires discordantes: {discordant_ppe:,}")
print(f"     Total paires: {total_pairs_ppe:,}")

# ============================================
# TABLEAU COMPARATIF FINAL
# ============================================
print("\n" + "="*80)
print("📋 TABLEAU COMPARATIF DES CORRÉLATIONS")
print("="*80)

# Créer le tableau de résultats
results_data = {
    'Paire': ['(EBIT, Personnel)', '(EBIT, PP&E)'],
    'Covariance MLE': [
        f"{covariance_ebit_personnel:.6f}" if covariance_ebit_personnel is not None else "N/A",
        f"{covariance_ebit_ppe:.6f}" if covariance_ebit_ppe is not None else "N/A"
    ],
    'Corrélation MLE': [
        f"{correlation_ebit_personnel:.6f}" if correlation_ebit_personnel is not None else "N/A",
        f"{correlation_ebit_ppe:.6f}" if correlation_ebit_ppe is not None else "N/A"
    ],
    'Pearson': [f"{pearson_ebit_pers:.6f}", f"{pearson_ebit_ppe:.6f}"],
    'p-value Pearson': [f"{pearson_p_pers:.6e}", f"{pearson_p_ppe:.6e}"],
    'Spearman': [f"{spearman_ebit_pers:.6f}", f"{spearman_ebit_ppe:.6f}"],
    'p-value Spearman': [f"{spearman_p_pers:.6e}", f"{spearman_p_ppe:.6e}"],
    'Kendall tau': [f"{kendall_tau_pers:.6f}", f"{kendall_tau_ppe:.6f}"],
    'Paires concordantes': [f"{concordant_pers:,}", f"{concordant_ppe:,}"],
    'Paires discordantes': [f"{discordant_pers:,}", f"{discordant_ppe:,}"],
    'Total paires': [f"{total_pairs_pers:,}", f"{total_pairs_ppe:,}"]
}

results_df = pd.DataFrame(results_data)
print("\n" + results_df.to_string(index=False))

# ============================================
# ANALYSE ET INTERPRÉTATION
# ============================================
print("\n" + "="*80)
print("📊 INTERPRÉTATION DES RÉSULTATS")
print("="*80)

print("\n🔍 ÉCHELLE DE CORRÉLATION (valeur absolue):")
print("  0.00 - 0.19 : Très faible")
print("  0.20 - 0.39 : Faible")
print("  0.40 - 0.59 : Modérée")
print("  0.60 - 0.79 : Forte")
print("  0.80 - 1.00 : Très forte")

print("\n📊 SIGNIFICATIVITÉ STATISTIQUE (p-value):")
print("  p < 0.001 : Extrêmement significatif (***)")
print("  p < 0.01  : Très significatif (**)")
print("  p < 0.05  : Significatif (*)")
print("  p ≥ 0.05  : Non significatif")

print("\n🎯 COMPARAISON DES MÉTHODES:")
print("  1. Pearson:  Mesure la relation LINÉAIRE")
print("  2. Spearman: Mesure la relation MONOTONE (basée sur les rangs)")
print("  3. Kendall:  Mesure la CONCORDANCE (robuste aux outliers)")
print("  4. MLE:      Estimation par MAXIMUM DE VRAISEMBLANCE")

# Analyser spécifiquement vos résultats
print("\n📈 ANALYSE DE VOS RÉSULTATS:")
for i, row in results_df.iterrows():
    paire = row['Paire']
    pearson_val = float(row['Pearson'])
    spearman_val = float(row['Spearman'])
    kendall_val = float(row['Kendall tau'])
    
    # Déterminer la force
    abs_pearson = abs(pearson_val)
    if abs_pearson >= 0.8:
        force = "très forte"
    elif abs_pearson >= 0.6:
        force = "forte"
    elif abs_pearson >= 0.4:
        force = "modérée"
    elif abs_pearson >= 0.2:
        force = "faible"
    else:
        force = "très faible"
    
    # Déterminer la direction
    direction = "positive" if pearson_val > 0 else "négative"
    
    print(f"\n  {paire}:")
    print(f"    • Pearson: Corrélation {force} et {direction} (r = {pearson_val:.3f})")
    print(f"    • Spearman: ρ = {spearman_val:.3f}")
    print(f"    • Kendall: τ = {kendall_val:.3f}")
    
    # Comparer Pearson et Spearman
    diff = abs(pearson_val - spearman_val)
    if diff > 0.1:
        print(f"    • Note: Différence notable entre Pearson et Spearman ({diff:.3f})")
        print(f"      → La relation n'est pas parfaitement linéaire")

# ============================================
# SAUVEGARDE DES RÉSULTATS
# ============================================
print("\n" + "="*80)
print("💾 SAUVEGARDE DES RÉSULTATS")
print("="*80)

try:
    # Sauvegarder le tableau principal
    results_path = '/content/drive/MyDrive/RESULTATS_CORRELATIONS_COMPARATIF.csv'
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    print(f"✅ Tableau principal sauvegardé: {results_path}")
    
    # Sauvegarder les détails MLE
    details_path = '/content/drive/MyDrive/DETAILS_CALCULS_MLE.txt'
    with open(details_path, 'w', encoding='utf-8') as f:
        f.write("DÉTAILS DES CALCULS MLE\n")
        f.write("="*60 + "\n\n")
        
        f.write("Valeurs extraites des fichiers MLE:\n")
        f.write("-"*40 + "\n")
        for key, value in mle_values.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\n\nCalculs détaillés:\n")
        f.write("="*60 + "\n\n")
        
        f.write("1. COVARIANCE MLE (EBIT, Personnel):\n")
        f.write("   covariance = E[Z1] - E[EBIT] × E[Personnel]\n")
        if all(k in mle_values for k in ['EBIT', 'Personnel', 'Z1']):
            f.write(f"              = {mle_values['Z1']} - ({mle_values['EBIT']} × {mle_values['Personnel']})\n")
            f.write(f"              = {covariance_ebit_personnel}\n")
        
        f.write("\n2. COVARIANCE MLE (EBIT, PP&E):\n")
        f.write("   covariance = E[Z2] - E[EBIT] × E[PP&E]\n")
        if all(k in mle_values for k in ['EBIT', 'PP&E', 'Z2']):
            f.write(f"              = {mle_values['Z2']} - ({mle_values['EBIT']} × {mle_values['PP&E']})\n")
            f.write(f"              = {covariance_ebit_ppe}\n")
    
    print(f"✅ Détails MLE sauvegardés: {details_path}")
    
    # Sauvegarder un rapport complet
    report_path = '/content/drive/MyDrive/RAPPORT_ANALYSE_CORRELATIONS.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("RAPPORT D'ANALYSE DE CORRÉLATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("DONNÉES ANALYSÉES:\n")
        f.write("-"*40 + "\n")
        f.write(f"• Période: 2005-2023\n")
        f.write(f"• Observations: {len(industrial_df)} entreprises\n")
        f.write(f"• Variables: EBIT, Personnel, PP&E\n\n")
        
        f.write("MÉTHODES UTILISÉES:\n")
        f.write("-"*40 + "\n")
        f.write("1. Corrélation de Pearson: Relation linéaire\n")
        f.write("2. Corrélation de Spearman: Relation monotone (rangs)\n")
        f.write("3. Tau de Kendall: Concordance entre paires\n")
        f.write("4. Maximum de Vraisemblance (MLE): Estimation paramétrique\n\n")
        
        f.write("FORMULES MLE:\n")
        f.write("-"*40 + "\n")
        f.write("• Covariance(EBIT, Personnel) = E[Z1] - E[EBIT] × E[Personnel]\n")
        f.write("• Covariance(EBIT, PP&E) = E[Z2] - E[EBIT] × E[PP&E]\n")
        f.write("• Corrélation MLE = Covariance / (σ_X × σ_Y)\n\n")
        
        f.write("RÉSULTATS:\n")
        f.write("-"*40 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("INTERPRÉTATION:\n")
        f.write("-"*40 + "\n")
        for i, row in results_df.iterrows():
            paire = row['Paire']
            pearson_val = float(row['Pearson'])
            
            if pearson_val > 0:
                direction = "positive"
            else:
                direction = "négative"
            
            abs_val = abs(pearson_val)
            if abs_val >= 0.8:
                force = "très forte"
            elif abs_val >= 0.6:
                force = "forte"
            elif abs_val >= 0.4:
                force = "modérée"
            elif abs_val >= 0.2:
                force = "faible"
            else:
                force = "très faible"
            
            f.write(f"• {paire}: Corrélation {force} et {direction} (Pearson = {pearson_val:.3f})\n")
    
    print(f"✅ Rapport complet sauvegardé: {report_path}")
    
except Exception as e:
    print(f"⚠️  Erreur lors de la sauvegarde: {e}")

# ============================================
# VISUALISATION
# ============================================
print("\n" + "="*80)
print("📊 VISUALISATION GRAPHIQUE")
print("="*80)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Créer la figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Graphique comparatif des méthodes
    pairs = ['EBIT-Personnel', 'EBIT-PP&E']
    methods = ['Pearson', 'Spearman', 'Kendall']
    
    pearson_vals = [pearson_ebit_pers, pearson_ebit_ppe]
    spearman_vals = [spearman_ebit_pers, spearman_ebit_ppe]
    kendall_vals = [kendall_tau_pers, kendall_tau_ppe]
    
    x = np.arange(len(pairs))
    width = 0.25
    
    axes[0, 0].bar(x - width, pearson_vals, width, label='Pearson', color='skyblue')
    axes[0, 0].bar(x, spearman_vals, width, label='Spearman', color='lightgreen')
    axes[0, 0].bar(x + width, kendall_vals, width, label='Kendall', color='salmon')
    axes[0, 0].set_xlabel('Paire de variables')
    axes[0, 0].set_ylabel('Valeur de corrélation')
    axes[0, 0].set_title('Comparaison des méthodes de corrélation')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(pairs)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0, color='black', linewidth=0.5)
    
    # 2. Nuage de points EBIT vs Personnel
    axes[0, 1].scatter(ebit_data, personnel_data, alpha=0.5, s=20, color='blue')
    axes[0, 1].set_xlabel('EBIT (Milliards USD)')
    axes[0, 1].set_ylabel('Personnel (10k unités)')
    axes[0, 1].set_title(f'EBIT vs Personnel\nr = {pearson_ebit_pers:.3f}')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Nuage de points EBIT vs PP&E
    axes[1, 0].scatter(ebit_data, ppe_data, alpha=0.5, s=20, color='green')
    axes[1, 0].set_xlabel('EBIT (Milliards USD)')
    axes[1, 0].set_ylabel('PP&E (Milliards USD)')
    axes[1, 0].set_title(f'EBIT vs PP&E\nr = {pearson_ebit_ppe:.3f}')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Heatmap des corrélations
    corr_matrix = pd.DataFrame({
        'EBIT': ebit_data,
        'Personnel': personnel_data,
        'PP&E': ppe_data
    }).corr()
    
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[1, 1])
    axes[1, 1].set_title('Matrice de corrélation (Pearson)')
    
    plt.tight_layout()
    
    # Sauvegarder le graphique
    plot_path = '/content/drive/MyDrive/VISUALISATION_CORRELATIONS.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé: {plot_path}")
    
    plt.show()
    
except ImportError:
    print("⚠️  Bibliothèques de visualisation non disponibles")
except Exception as e:
    print(f"⚠️  Erreur lors de la visualisation: {e}")

print("\n" + "="*80)
print("✅ ANALYSE COMPLÈTEMENT TERMINÉE !")
print("="*80)
print("\n🎯 RÉSUMÉ DES FICHIERS CRÉÉS:")
print("  1. RESULTATS_CORRELATIONS_COMPARATIF.csv - Tableau principal")
print("  2. DETAILS_CALCULS_MLE.txt - Détails des calculs MLE")
print("  3. RAPPORT_ANALYSE_CORRELATIONS.txt - Rapport complet")
print("  4. VISUALISATION_CORRELATIONS.png - Graphiques (si disponible)")
print("\n📍 Tous les fichiers sont dans votre Google Drive: /content/drive/MyDrive/")