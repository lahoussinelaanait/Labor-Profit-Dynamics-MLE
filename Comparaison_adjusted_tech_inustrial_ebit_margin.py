# Montage de Google Drive et importation des bibliothèques
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Chemins des fichiers
industrial_file = '/content/drive/MyDrive/ANNUAL_GAMMA_EXPECTATIONS_INDUSTRIAL_2005_2023.csv'
technology_file = '/content/drive/MyDrive/ANNUAL_GAMMA_EXPECTATIONS_TECHNOLOGY_2005_2023.csv'

# Lire les fichiers
df_industrial = pd.read_csv(industrial_file)
df_technology = pd.read_csv(technology_file)

print("=== FICHIERS CHARGÉS AVEC SUCCÈS ===")
print(f"Fichier industriel : {industrial_file}")
print(f"Fichier technologie : {technology_file}")
print(f"Dimensions industriel : {df_industrial.shape}")
print(f"Dimensions technologie : {df_technology.shape}")

# Extraire les données z5 pour l'industriel
z5_industrial = df_industrial[df_industrial['Variable'] == 'z5'].copy()
z5_industrial = z5_industrial[['Year', 'Gamma_Expectation']].rename(
    columns={'Gamma_Expectation': 'z5_industrial'}
)

# Extraire les données z5 pour la technologie
z5_technology = df_technology[df_technology['Variable'] == 'z5'].copy()
z5_technology = z5_technology[['Year', 'Gamma_Expectation']].rename(
    columns={'Gamma_Expectation': 'z5_technology'}
)

# Fusionner les données
df_z5 = pd.merge(z5_industrial, z5_technology, on='Year', how='inner')

# Calculer ebit margin adjusted TECH (z5_tech - 0.1054)
# Note: Si z5 est en pourcentage (ex: 8.0 pour 8%), alors 0.1054% = 0.001054 en ratio
# Nous allons d'abord vérifier si les valeurs sont en pourcentage
print("\n=== VÉRIFICATION DES VALEURS Z5 ===")
print(f"z5 Industrial - Min: {df_z5['z5_industrial'].min():.4f}, Max: {df_z5['z5_industrial'].max():.4f}, Moyenne: {df_z5['z5_industrial'].mean():.4f}")
print(f"z5 Technology - Min: {df_z5['z5_technology'].min():.4f}, Max: {df_z5['z5_technology'].max():.4f}, Moyenne: {df_z5['z5_technology'].mean():.4f}")

# Si les valeurs sont > 1, elles sont probablement en pourcentage (ex: 8.0 = 8%)
# Nous les convertirons en ratio (8.0 -> 0.08)
convert_to_ratio = df_z5['z5_industrial'].max() > 1
if convert_to_ratio:
    print("\nLes valeurs Z5 semblent être en pourcentage. Conversion en ratio (division par 100)...")
    df_z5['z5_industrial'] = df_z5['z5_industrial'] / 100
    df_z5['z5_technology'] = df_z5['z5_technology'] / 100
    adjustment = 0.1054 / 100  # 0.1054% = 0.001054 en ratio
else:
    print("\nLes valeurs Z5 semblent déjà être en ratio. Aucune conversion nécessaire.")
    adjustment = 0.1054  # Supposons que 0.1054 est déjà en ratio

# Calculer ebit margin adjusted TECH
df_z5['ebit_margin_adjusted_TECH'] = df_z5['z5_technology'] - adjustment

# Afficher les données finales
print("\n=== DONNÉES FINALES POUR ANALYSE ===")
print(df_z5.head())
print(f"\nPériode couverte : {df_z5['Year'].min()} à {df_z5['Year'].max()}")
print(f"Nombre d'années : {len(df_z5)}")

# 1. Graphique de 2005 à 2023
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(df_z5['Year'], df_z5['ebit_margin_adjusted_TECH'], 
         label='EBIT Margin Adjusted TECH (z5_tech - 0.1054)', 
         marker='o', linewidth=2, color='blue')
plt.plot(df_z5['Year'], df_z5['z5_industrial'], 
         label='z5 Industrial (EBIT/Revenue)', 
         marker='s', linewidth=2, color='orange', alpha=0.7)
plt.xlabel('Année')
plt.ylabel('Ratio EBIT/Revenue' + (' (%)' if not convert_to_ratio else ''))
plt.title('Comparaison: EBIT Margin Adjusted TECH vs z5 Industrial\n(2005-2023)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(df_z5['Year'], rotation=45)

# 2. Graphique de 2012 à 2023
df_2012_2023 = df_z5[df_z5['Year'] >= 2012].copy()

plt.subplot(1, 2, 2)
plt.plot(df_2012_2023['Year'], df_2012_2023['ebit_margin_adjusted_TECH'], 
         label='EBIT Margin Adjusted TECH (z5_tech - 0.1054)', 
         marker='o', linewidth=2, color='blue')
plt.plot(df_2012_2023['Year'], df_2012_2023['z5_industrial'], 
         label='z5 Industrial (EBIT/Revenue)', 
         marker='s', linewidth=2, color='orange', alpha=0.7)
plt.xlabel('Année')
plt.ylabel('Ratio EBIT/Revenue' + (' (%)' if not convert_to_ratio else ''))
plt.title('Comparaison: EBIT Margin Adjusted TECH vs z5 Industrial\n(2012-2023)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(df_2012_2023['Year'], rotation=45)

plt.tight_layout()
plt.show()

# Analyse statistique détaillée
print("\n" + "="*80)
print("ANALYSE STATISTIQUE COMPARATIVE")
print("="*80)

# Fonction pour afficher les statistiques
def print_stats(name, data):
    print(f"{name}:")
    print(f"  Moyenne: {data.mean():.6f}")
    print(f"  Écart-type: {data.std():.6f}")
    print(f"  Min: {data.min():.6f}")
    print(f"  Max: {data.max():.6f}")
    print(f"  Médiane: {np.median(data):.6f}")
    print(f"  25e percentile: {np.percentile(data, 25):.6f}")
    print(f"  75e percentile: {np.percentile(data, 75):.6f}")

# Statistiques pour 2005-2023
print("\n=== PÉRIODE 2005-2023 ===")
print_stats("EBIT Margin Adjusted TECH", df_z5['ebit_margin_adjusted_TECH'])
print()
print_stats("z5 Industrial", df_z5['z5_industrial'])

# Statistiques pour 2012-2023
print("\n=== PÉRIODE 2012-2023 ===")
print_stats("EBIT Margin Adjusted TECH", df_2012_2023['ebit_margin_adjusted_TECH'])
print()
print_stats("z5 Industrial", df_2012_2023['z5_industrial'])

# Tests statistiques
print("\n=== TESTS STATISTIQUES DE COMPARAISON ===")

# 1. Test de corrélation
corr_full, p_full = stats.pearsonr(df_z5['ebit_margin_adjusted_TECH'], df_z5['z5_industrial'])
corr_2012, p_2012 = stats.pearsonr(df_2012_2023['ebit_margin_adjusted_TECH'], df_2012_2023['z5_industrial'])

print(f"\n1. CORRÉLATIONS:")
print(f"   Période 2005-2023: r = {corr_full:.4f}, p-value = {p_full:.4f}")
print(f"   Période 2012-2023: r = {corr_2012:.4f}, p-value = {p_2012:.4f}")

# Test de différence entre corrélations (Fisher z-test)
def fisher_z_test(r1, r2, n1, n2):
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z2 = 0.5 * np.log((1 + r2) / (1 - r2))
    SE = np.sqrt(1/(n1-3) + 1/(n2-3))
    z = (z1 - z2) / SE
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p

z_corr, p_corr_diff = fisher_z_test(corr_full, corr_2012, len(df_z5), len(df_2012_2023))
print(f"   Test de différence des corrélations: z = {z_corr:.4f}, p = {p_corr_diff:.4f}")

# 2. Test t pour comparer les moyennes entre périodes
t_tech, p_tech = stats.ttest_ind(df_z5['ebit_margin_adjusted_TECH'], 
                                 df_2012_2023['ebit_margin_adjusted_TECH'],
                                 equal_var=False)
t_ind, p_ind = stats.ttest_ind(df_z5['z5_industrial'], 
                               df_2012_2023['z5_industrial'],
                               equal_var=False)

print(f"\n2. COMPARAISON DES MOYENNES (test t):")
print(f"   EBIT Margin Adjusted TECH: t = {t_tech:.4f}, p = {p_tech:.4f}")
print(f"   z5 Industrial: t = {t_ind:.4f}, p = {p_ind:.4f}")

# 3. Test de Levene pour comparer les variances (plus robuste que le test F)
levene_tech, p_levene_tech = stats.levene(df_z5['ebit_margin_adjusted_TECH'], 
                                          df_2012_2023['ebit_margin_adjusted_TECH'])
levene_ind, p_levene_ind = stats.levene(df_z5['z5_industrial'], 
                                        df_2012_2023['z5_industrial'])

print(f"\n3. COMPARAISON DES VARIANCES (test de Levene):")
print(f"   EBIT Margin Adjusted TECH: F = {levene_tech:.4f}, p = {p_levene_tech:.4f}")
print(f"   z5 Industrial: F = {levene_ind:.4f}, p = {p_levene_ind:.4f}")

# 4. Différences absolues entre les deux séries
df_z5['diff_absolute'] = np.abs(df_z5['ebit_margin_adjusted_TECH'] - df_z5['z5_industrial'])
df_2012_2023['diff_absolute'] = np.abs(df_2012_2023['ebit_margin_adjusted_TECH'] - df_2012_2023['z5_industrial'])

print(f"\n4. DIFFÉRENCES ABSOLUES ENTRE LES DEUX SÉRIES:")
print(f"   Période 2005-2023:")
print(f"     Moyenne: {df_z5['diff_absolute'].mean():.6f}")
print(f"     Écart-type: {df_z5['diff_absolute'].std():.6f}")
print(f"     Max: {df_z5['diff_absolute'].max():.6f}")
print(f"   Période 2012-2023:")
print(f"     Moyenne: {df_2012_2023['diff_absolute'].mean():.6f}")
print(f"     Écart-type: {df_2012_2023['diff_absolute'].std():.6f}")
print(f"     Max: {df_2012_2023['diff_absolute'].max():.6f}")

# Test t sur les différences
t_diff, p_diff = stats.ttest_ind(df_z5['diff_absolute'], 
                                 df_2012_2023['diff_absolute'],
                                 equal_var=False)
print(f"   Test de différence des écarts: t = {t_diff:.4f}, p = {p_diff:.4f}")

# 5. Test de Mann-Whitney (non paramétrique) pour comparer les distributions
mw_tech, p_mw_tech = stats.mannwhitneyu(df_z5['ebit_margin_adjusted_TECH'], 
                                        df_2012_2023['ebit_margin_adjusted_TECH'])
mw_ind, p_mw_ind = stats.mannwhitneyu(df_z5['z5_industrial'], 
                                      df_2012_2023['z5_industrial'])

print(f"\n5. TEST NON PARAMÉTRIQUE (Mann-Whitney U):")
print(f"   EBIT Margin Adjusted TECH: U = {mw_tech:.4f}, p = {p_mw_tech:.4f}")
print(f"   z5 Industrial: U = {mw_ind:.4f}, p = {p_mw_ind:.4f}")

# Conclusion statistique
print("\n" + "="*80)
print("CONCLUSION STATISTIQUE")
print("="*80)

alpha = 0.05  # Seuil de significativité

# Évaluation des différences
significant_corr_diff = p_corr_diff <= alpha
significant_mean_diff_tech = p_tech <= alpha
significant_mean_diff_ind = p_ind <= alpha
significant_var_diff_tech = p_levene_tech <= alpha
significant_var_diff_ind = p_levene_ind <= alpha
significant_diff_diff = p_diff <= alpha

print(f"\nAvec un seuil de significativité de α = {alpha}:")
print(f"1. Différence de corrélations significative? {'OUI' if significant_corr_diff else 'NON'}")
print(f"2. Différence des moyennes TECH significative? {'OUI' if significant_mean_diff_tech else 'NON'}")
print(f"3. Différence des moyennes INDUSTRIEL significative? {'OUI' if significant_mean_diff_ind else 'NON'}")
print(f"4. Différence des variances TECH significative? {'OUI' if significant_var_diff_tech else 'NON'}")
print(f"5. Différence des variances INDUSTRIEL significative? {'OUI' if significant_var_diff_ind else 'NON'}")
print(f"6. Différence des écarts entre séries significative? {'OUI' if significant_diff_diff else 'NON'}")

# Décision finale
any_significant = (significant_corr_diff or significant_mean_diff_tech or 
                   significant_mean_diff_ind or significant_var_diff_tech or 
                   significant_var_diff_ind or significant_diff_diff)

print("\n" + "="*80)
print("RÉCAPITULATIF FINAL")
print("="*80)

if not any_significant:
    print("CONCLUSION: Les deux graphiques (2005-2023 vs 2012-2023) sont")
    print("STATISTIQUEMENT IDENTIQUES. Il n'y a pas de différences")
    print("statistiquement significatives entre les deux périodes.")
else:
    print("CONCLUSION: Les deux graphiques (2005-2023 vs 2012-2023) présentent")
    print("des DIFFÉRENCES STATISTIQUEMENT SIGNIFICATIVES.")
    print("\nDifférences significatives détectées dans:")
    if significant_corr_diff:
        print("  - La corrélation entre TECH et Industrial")
    if significant_mean_diff_tech:
        print("  - La moyenne de EBIT Margin Adjusted TECH")
    if significant_mean_diff_ind:
        print("  - La moyenne de z5 Industrial")
    if significant_var_diff_tech:
        print("  - La variance de EBIT Margin Adjusted TECH")
    if significant_var_diff_ind:
        print("  - La variance de z5 Industrial")
    if significant_diff_diff:
        print("  - L'écart moyen entre les deux séries")

# Graphique récapitulatif des différences
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
periods = ['2005-2023', '2012-2023']
x_pos = np.arange(len(periods))
width = 0.35

tech_means = [df_z5['ebit_margin_adjusted_TECH'].mean(), df_2012_2023['ebit_margin_adjusted_TECH'].mean()]
tech_stds = [df_z5['ebit_margin_adjusted_TECH'].std(), df_2012_2023['ebit_margin_adjusted_TECH'].std()]

ind_means = [df_z5['z5_industrial'].mean(), df_2012_2023['z5_industrial'].mean()]
ind_stds = [df_z5['z5_industrial'].std(), df_2012_2023['z5_industrial'].std()]

plt.bar(x_pos - width/2, tech_means, width, yerr=tech_stds, 
        capsize=5, label='TECH Adjusted', color='blue', alpha=0.7)
plt.bar(x_pos + width/2, ind_means, width, yerr=ind_stds, 
        capsize=5, label='Industrial', color='orange', alpha=0.7)
plt.xlabel('Période')
plt.ylabel('Moyenne EBIT/Revenue')
plt.title('Moyennes et écarts-types par période')
plt.xticks(x_pos, periods)
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
correlations = [corr_full, corr_2012]
plt.bar(periods, correlations, color=['blue', 'orange'], alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
plt.xlabel('Période')
plt.ylabel('Coefficient de corrélation')
plt.title('Corrélations entre TECH et Industrial')
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
diff_means = [df_z5['diff_absolute'].mean(), df_2012_2023['diff_absolute'].mean()]
diff_stds = [df_z5['diff_absolute'].std(), df_2012_2023['diff_absolute'].std()]
plt.bar(periods, diff_means, yerr=diff_stds, capsize=5, color=['green', 'red'], alpha=0.7)
plt.xlabel('Période')
plt.ylabel('Différence absolue moyenne')
plt.title('Écarts moyens entre les séries')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Tableau de données final pour export
print("\n=== DONNÉES POUR EXPORT ===")
export_df = df_z5.copy()
print(export_df.head())
print(f"\nDimensions: {export_df.shape}")

# Option pour sauvegarder les résultats
try:
    output_path = '/content/drive/MyDrive/z5_comparison_results.csv'
    export_df.to_csv(output_path, index=False)
    print(f"\nDonnées sauvegardées à: {output_path}")
except:
    print("\nLes données peuvent être copiées manuellement depuis le tableau ci-dessus.")