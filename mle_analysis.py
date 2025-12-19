# GLOBAL GAMMA MLE ANALYSIS - SECTOR-SPECIFIC (TECHNOLOGY OR INDUSTRIAL)
# All data analyzed together (global analysis)
# ========================================================

from google.colab import drive
import pandas as pd
import numpy as np
from scipy.special import gammaln, digamma, polygamma, logsumexp
from scipy.stats import gamma, kstest, anderson, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ========== 1. USER SELECTION: CHOOSE DATASET ==========
print("=" * 80)
print("GLOBAL GAMMA MLE ANALYSIS - SECTOR SELECTION")
print("=" * 80)
print("\nPlease choose the dataset to analyze:")
print("1. Technology Sector (tech_intensive_panel.csv)")
print("2. Industrial Sector (industrial_sector_panel.csv)")

while True:
    try:
        choice = int(input("\nEnter your choice (1 or 2): "))
        if choice in [1, 2]:
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
    except ValueError:
        print("Invalid input. Please enter 1 or 2.")

# Set parameters based on user choice
if choice == 1:
    sector = "TECHNOLOGY"
    file_path = '/content/drive/MyDrive/tech_intensive_panel.csv'
    delimiter = ';'  # Technology file uses semicolon
    # For Technology sector, we analyze all 8 variables
    variables_to_analyze = [
        ('ebit_positif', 'EBIT (positive only)'),
        ('rd', 'R&D'),
        ('ppe', 'PP&E'),
        ('personnel', 'Personnel'),
        ('z1', 'Z1 = EBIT × Personnel'),
        ('z2', 'Z2 = EBIT × PP&E'),
        ('z3', 'Z3 = EBIT × R&D'),
        ('z4', 'Z4 = R&D × Personnel')
    ]
else:  # choice == 2
    sector = "INDUSTRIAL"
    file_path = '/content/drive/MyDrive/industrial_sector_panel.csv'
    delimiter = ','  # Industrial file uses comma
    # For Industrial sector, we analyze only 6 variables (no R&D, Z3, Z4)
    variables_to_analyze = [
        ('ebit_positif', 'EBIT (positive only)'),
        ('ppe', 'PP&E'),
        ('personnel', 'Personnel'),
        ('z1', 'Z1 = EBIT × Personnel'),
        ('z2', 'Z2 = EBIT × PP&E')
    ]

print(f"\nSelected sector: {sector}")
print(f"File path: {file_path}")
print(f"Number of variables to analyze: {len(variables_to_analyze)}")

# ---------- 2. MOUNT DRIVE ----------
drive.mount('/content/drive')

# ---------- 3. COMPLETE GAMMA MLE CLASS WITH VALIDATION TESTS ----------

class GammaMixtureMLE_Validation:
    """Gamma Mixture with complete validation tests"""
    
    def __init__(self, K_min=1, K_max=3, max_iter=200, tol=1e-6):
        self.K_min = K_min
        self.K_max = K_max
        self.max_iter = max_iter
        self.tol = tol
        self.reg_param = 1e-8
        self.validation_results = {}
    
    def _gamma_logpdf(self, x, alpha, beta):
        """Log PDF Gamma with regularization"""
        x = np.maximum(x, self.reg_param)
        alpha = max(alpha, self.reg_param)
        beta = max(beta, self.reg_param)
        return (alpha - 1) * np.log(x) - x/beta - alpha * np.log(beta) - gammaln(alpha)
    
    def _cdf_mixture(self, x):
        """CDF of Gamma mixture"""
        cdf_val = 0
        for k in range(self.n_components):
            cdf_val += self.weights[k] * gamma.cdf(x, self.alphas[k], scale=self.betas[k])
        return cdf_val
    
    def _fit_single_gamma_robuste(self, X, weights=None, scale_factor=1.0):
        """Robust single Gamma fit"""
        if weights is None:
            weights = np.ones(len(X))
        weights = weights / np.sum(weights)
        
        X_scaled = X / scale_factor
        
        # Weighted statistics
        mean_w = np.sum(weights * X_scaled)
        log_mean_w = np.sum(weights * np.log(np.maximum(X_scaled, self.reg_param)))
        var_w = np.sum(weights * (X_scaled - mean_w)**2)
        
        if var_w > 0:
            alpha_init = min(max(mean_w**2 / var_w, 0.1), 500)
            beta_init = min(max(var_w / mean_w, 0.01), 1000)
        else:
            alpha_init, beta_init = 2.0, mean_w/2
        
        # Optimization
        def neg_log_likelihood(params):
            alpha, beta = params[0], params[1]
            if alpha <= 0 or beta <= 0:
                return 1e10
            beta_original = beta * scale_factor
            log_lik = np.sum(weights * self._gamma_logpdf(X, alpha, beta_original))
            return -log_lik
        
        bounds = [(0.05, 1000), (0.001, 5000)]
        result = minimize(neg_log_likelihood, [alpha_init, beta_init], 
                         method='L-BFGS-B', bounds=bounds, 
                         options={'maxiter': 200, 'ftol': 1e-8})
        
        if result.success:
            alpha = max(0.05, result.x[0])
            beta = max(0.001, result.x[1] * scale_factor)
            return alpha, beta
        else:
            mean_orig = np.sum(weights * X)
            var_orig = np.sum(weights * (X - mean_orig)**2)
            if var_orig > 0:
                alpha = min(max(mean_orig**2 / var_orig, 0.05), 1000)
                beta = min(max(var_orig / mean_orig, 0.001), 5000)
            else:
                alpha, beta = 2.0, mean_orig/2
            return alpha, beta
    
    def _run_tests_validation(self, X):
        """Execute all validation tests"""
        if len(X) < 10:
            return {}
        
        # 1. Kolmogorov-Smirnov test
        ks_stat, ks_pvalue = kstest(X, self._cdf_mixture)
        
        # 2. Information criteria calculation
        n_params = 3 * self.n_components - 1
        aic = 2 * n_params - 2 * self.logL
        bic = np.log(len(X)) * n_params - 2 * self.logL
        
        # 3. Adapted Chi2 test
        try:
            n_bins = min(10, max(3, int(np.sqrt(len(X)))))
            hist, bin_edges = np.histogram(X, bins=n_bins, density=False)
            
            # Theoretical frequencies
            expected = np.zeros(n_bins)
            for i in range(n_bins):
                prob = self._cdf_mixture(bin_edges[i+1]) - self._cdf_mixture(bin_edges[i])
                expected[i] = prob * len(X)
            
            # Avoid too low frequencies
            mask = expected >= 1
            if np.sum(mask) >= 2:
                chi2_stat = np.sum((hist[mask] - expected[mask])**2 / expected[mask])
                df = np.sum(mask) - n_params - 1
                df = max(1, df)
                chi2_pvalue = 1 - chi2.cdf(chi2_stat, df) if df > 0 else np.nan
            else:
                chi2_stat = np.nan
                chi2_pvalue = np.nan
                df = 0
        except:
            chi2_stat = np.nan
            chi2_pvalue = np.nan
            df = 0
        
        # 4. Wasserstein distance (approximation)
        X_sorted = np.sort(X)
        n = len(X)
        cdf_emp = np.arange(1, n + 1) / n
        cdf_theo = self._cdf_mixture(X_sorted)
        wasserstein = np.mean(np.abs(cdf_emp - cdf_theo))
        
        return {
            'KS_statistic': ks_stat,
            'KS_pvalue': ks_pvalue,
            'AIC': aic,
            'BIC': bic,
            'Chi2_statistic': chi2_stat,
            'Chi2_pvalue': chi2_pvalue,
            'Chi2_df': df,
            'Wasserstein': wasserstein,
            'logLikelihood': self.logL,
            'n_params': n_params
        }
    
    def _initialize_components(self, X, K):
        """Component initialization"""
        if K == 1:
            scale_factor = np.median(X) if np.median(X) > 0 else np.mean(X)
            alpha, beta = self._fit_single_gamma_robuste(X, scale_factor=scale_factor)
            return np.array([alpha]), np.array([beta]), np.array([1.0])
        
        from sklearn.cluster import KMeans
        
        X_log = np.log(np.maximum(X, self.reg_param))
        
        # Outlier detection
        q25, q75 = np.percentile(X_log, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        mask = (X_log >= lower_bound) & (X_log <= upper_bound)
        X_filtered = X[mask] if np.sum(mask) > 10 else X
        
        if len(X_filtered) < K * 3:
            scale_factor = np.median(X) if np.median(X) > 0 else np.mean(X)
            alpha, beta = self._fit_single_gamma_robuste(X, scale_factor=scale_factor)
            return np.array([alpha]), np.array([beta]), np.array([1.0])
        
        X_log_filtered = np.log(np.maximum(X_filtered, self.reg_param))
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels_filtered = kmeans.fit_predict(X_log_filtered.reshape(-1, 1))
        
        if len(X) > len(X_filtered):
            centroids = kmeans.cluster_centers_.flatten()
            X_log_all = np.log(np.maximum(X, self.reg_param))
            distances = np.abs(X_log_all.reshape(-1, 1) - centroids.reshape(1, -1))
            labels_all = np.argmin(distances, axis=1)
        else:
            labels_all = labels_filtered
        
        alphas, betas, weights = [], [], []
        for k in range(K):
            X_k = X[labels_all == k]
            if len(X_k) >= 3:
                scale_factor_k = np.median(X_k) if np.median(X_k) > 0 else np.mean(X_k)
                alpha_k, beta_k = self._fit_single_gamma_robuste(X_k, scale_factor=scale_factor_k)
                alphas.append(alpha_k)
                betas.append(beta_k)
                weights.append(len(X_k))
            else:
                scale_factor = np.median(X) if np.median(X) > 0 else np.mean(X)
                alpha_k, beta_k = self._fit_single_gamma_robuste(X, scale_factor=scale_factor)
                alphas.append(alpha_k)
                betas.append(beta_k)
                weights.append(1)
        
        weights = np.array(weights) / np.sum(weights)
        return np.array(alphas), np.array(betas), weights
    
    def _EM_iteration(self, X, alphas, betas, weights):
        """EM iteration"""
        N = len(X)
        K = len(alphas)
        
        # E-step
        log_resp = np.zeros((N, K))
        for k in range(K):
            log_resp[:, k] = self._gamma_logpdf(X, alphas[k], betas[k]) + np.log(max(weights[k], 1e-10))
        
        log_sum = logsumexp(log_resp, axis=1, keepdims=True)
        resp = np.exp(log_resp - log_sum)
        
        # M-step
        Nk = resp.sum(axis=0)
        weights_new = Nk / N
        
        alphas_new = np.zeros(K)
        betas_new = np.zeros(K)
        
        for k in range(K):
            if Nk[k] > 2:
                scale_factor = np.median(X[resp[:, k] > 0.1]) if len(X[resp[:, k] > 0.1]) > 0 else np.median(X)
                alpha_k, beta_k = self._fit_single_gamma_robuste(X, weights=resp[:, k], scale_factor=scale_factor)
                alphas_new[k] = alpha_k
                betas_new[k] = beta_k
            else:
                alphas_new[k] = alphas[k]
                betas_new[k] = betas[k]
        
        return alphas_new, betas_new, weights_new
    
    def _log_likelihood(self, X, alphas, betas, weights):
        """Log-likelihood calculation"""
        K = len(alphas)
        log_probs = np.zeros((len(X), K))
        for k in range(K):
            log_probs[:, k] = self._gamma_logpdf(X, alphas[k], betas[k]) + np.log(max(weights[k], 1e-10))
        return logsumexp(log_probs, axis=1).sum()
    
    def fit(self, X):
        """Fit with validation tests"""
        X_clean = X[(~np.isnan(X)) & (X > 0)]
        
        if len(X_clean) < 10:
            self.n_components = 1
            scale_factor = np.median(X_clean) if np.median(X_clean) > 0 else np.mean(X_clean)
            alpha, beta = self._fit_single_gamma_robuste(X_clean, scale_factor=scale_factor)
            self.alphas = np.array([alpha])
            self.betas = np.array([beta])
            self.weights = np.array([1.0])
            self.logL = self._log_likelihood(X_clean, self.alphas, self.betas, self.weights)
            self.validation_results = self._run_tests_validation(X_clean) if len(X_clean) >= 5 else {}
            return self
        
        # Try different K values
        best_bic = np.inf
        best_model = None
        
        for K in range(self.K_min, min(self.K_max, len(X_clean)//5) + 1):
            try:
                alphas, betas, weights = self._initialize_components(X_clean, K)
                
                logL_old = -np.inf
                for iteration in range(self.max_iter):
                    alphas, betas, weights = self._EM_iteration(X_clean, alphas, betas, weights)
                    logL = self._log_likelihood(X_clean, alphas, betas, weights)
                    
                    if iteration > 5 and abs(logL - logL_old) < self.tol:
                        break
                    logL_old = logL
                
                n_params = 3 * K - 1
                bic = np.log(len(X_clean)) * n_params - 2 * logL
                
                if bic < best_bic and not np.any(alphas > 900):
                    best_bic = bic
                    best_model = (K, alphas.copy(), betas.copy(), weights.copy(), logL, bic)
            except:
                continue
        
        if best_model:
            self.n_components, self.alphas, self.betas, self.weights, self.logL, self.bic = best_model
        else:
            self.n_components = 1
            scale_factor = np.median(X_clean) if np.median(X_clean) > 0 else np.mean(X_clean)
            alpha, beta = self._fit_single_gamma_robuste(X_clean, scale_factor=scale_factor)
            self.alphas = np.array([alpha])
            self.betas = np.array([beta])
            self.weights = np.array([1.0])
            self.logL = self._log_likelihood(X_clean, self.alphas, self.betas, self.weights)
        
        # Execute validation tests
        self.validation_results = self._run_tests_validation(X_clean)
        
        return self
    
    def expectation(self):
        """Mixture expectation"""
        return np.sum(self.weights * self.alphas * self.betas)
    
    def print_validation_summary(self):
        """Display validation test summary"""
        if not self.validation_results:
            print("  No validation tests available")
            return
        
        print("\n  Validation Tests:")
        print(f"    • KS: D = {self.validation_results.get('KS_statistic', 'NA'):.4f}, "
              f"p = {self.validation_results.get('KS_pvalue', 'NA'):.4f}")
        
        ks_pval = self.validation_results.get('KS_pvalue', 1)
        if ks_pval < 0.05:
            print(f"      KS: Rejected (p={ks_pval:.4f}) - Poor fit")
        elif ks_pval < 0.10:
            print(f"      KS: Borderline (p={ks_pval:.4f}) - Acceptable fit")
        else:
            print(f"      KS: Accepted (p={ks_pval:.4f}) - Good fit")
        
        print(f"    • AIC: {self.validation_results.get('AIC', 'NA'):.2f}, "
              f"BIC: {self.validation_results.get('BIC', 'NA'):.2f}")

# ---------- 4. LOAD AND PREPARE DATA ----------

print(f"\nLoading file: {file_path}")

# Read file
try:
    df = pd.read_csv(file_path, delimiter=delimiter)
    print("File loaded successfully!")
except Exception as e:
    print(f"Error loading file: {e}")
    raise

# Display first lines for verification
print("\nFile structure:")
print(f"Columns: {list(df.columns)}")
print(f"Number of observations: {len(df)}")

# Display column names
print("\nDetected column names:")
for col in df.columns:
    print(f" - '{col}'")

# Rename columns for consistency
# Map to standardized column names used in the rest of the code
column_mapping = {}
for col in df.columns:
    col_lower = col.lower()
    if 'company' in col_lower:
        column_mapping[col] = 'compagny'
    elif 'fiscal' in col_lower or 'date' in col_lower or 'year' in col_lower:
        column_mapping[col] = 'year'
    elif 'revenue' in col_lower or ('ca' in col_lower and ' ' in col):
        column_mapping[col] = 'ca'
    elif 'ppe' in col_lower or 'pp&e' in col_lower:
        column_mapping[col] = 'ppe'
    elif 'ebit' in col_lower:
        column_mapping[col] = 'ebit'
    elif 'r&d' in col_lower or 'rd' in col_lower:
        column_mapping[col] = 'rd'
    elif 'employee' in col_lower or 'personnel' in col_lower:
        column_mapping[col] = 'personnel'

print("\nColumn mapping applied:")
for old, new in column_mapping.items():
    print(f"  '{old}' -> '{new}'")

df = df.rename(columns=column_mapping)

print("\nColumns after renaming:")
print(f"{list(df.columns)}")

# Convert numeric columns
df['ca'] = pd.to_numeric(df['ca'], errors='coerce')
df['ppe'] = pd.to_numeric(df['ppe'], errors='coerce')
df['ebit'] = pd.to_numeric(df['ebit'], errors='coerce')
df['personnel'] = pd.to_numeric(df['personnel'], errors='coerce')
df['year'] = pd.to_numeric(df['year'], errors='coerce')

# For technology sector, also convert R&D
if sector == "TECHNOLOGY" and 'rd' in df.columns:
    df['rd'] = pd.to_numeric(df['rd'], errors='coerce')
elif sector == "INDUSTRIAL" and 'rd' in df.columns:
    # Industrial sector doesn't have R&D data, remove the column if it exists
    df = df.drop(columns=['rd'])

# ===== CORRECTION FOR ACTUAL DATA PERIOD =====
# Determine actual period from data
min_year = int(df['year'].min())
max_year = int(df['year'].max())

# Filter for the actual data period
df = df[(df['year'] >= min_year) & (df['year'] <= max_year)].copy()

print(f"\nData filtered for ACTUAL period: {len(df)} observations")
print(f"  Actual period in data: {min_year}-{max_year}")
print(f"  Note: Analysis will cover {max_year - min_year + 1} years ({min_year} to {max_year})")
# ===== END OF CORRECTION =====

print(f"  Number of unique companies: {df['compagny'].nunique()}")

# ---------- 5. GLOBAL ANALYSIS FUNCTION ----------

def analyze_global_variable(df, variable_name, display_name, filter_positive=True):
    """Global analysis of a variable over the actual data period"""
    print(f"\nGLOBAL MLE ANALYSIS of {display_name} ({min_year}-{max_year})...")
    
    # Prepare data
    if variable_name == 'ebit_positif':
        # Positive EBIT only
        data = df['ebit'].dropna().values
        data = data[data > 0]
        
    elif variable_name == 'z1':
        # Z1 = EBIT × Personnel (positive EBIT only)
        df['ebit_positif'] = df['ebit'].where(df['ebit'] > 0, np.nan)
        data = (df['ebit_positif'] * df['personnel']).dropna().values
        data = data[data > 0]
        
    elif variable_name == 'z2':
        # Z2 = EBIT × PP&E (positive EBIT only)
        df['ebit_positif'] = df['ebit'].where(df['ebit'] > 0, np.nan)
        data = (df['ebit_positif'] * df['ppe']).dropna().values
        data = data[data > 0]
        
    elif variable_name == 'z3':
        # Z3 = EBIT × R&D (positive EBIT only)
        df['ebit_positif'] = df['ebit'].where(df['ebit'] > 0, np.nan)
        data = (df['ebit_positif'] * df['rd']).dropna().values
        data = data[data > 0]
        
    elif variable_name == 'z4':
        # Z4 = R&D × Personnel (R&D and Personnel positive)
        mask = (df['rd'] > 0) & (df['personnel'] > 0)
        data = (df.loc[mask, 'rd'] * df.loc[mask, 'personnel']).dropna().values
        data = data[data > 0]
        
    elif variable_name == 'rd':
        # R&D only
        data = df['rd'].dropna().values
        data = data[data > 0]
        
    else:
        # Other variables
        data = df[variable_name].dropna().values
        if filter_positive:
            data = data[data > 0]
    
    print(f"  Data: N = {len(data):,}")
    
    if len(data) < 10:
        print(f"  Insufficient data for analysis")
        return None
    
    # Fit Gamma mixture model
    model = GammaMixtureMLE_Validation(K_min=1, K_max=3)
    model.fit(data)
    
    # Calculate statistics
    expectation = model.expectation()
    mean = np.mean(data)
    diff = abs(expectation - mean)
    diff_rel = diff / mean * 100 if mean != 0 else 0
    
    # Display results
    print(f"\n  GLOBAL RESULTS:")
    print(f"  {'─' * 60}")
    print(f"  • Gamma Expectation: {expectation:.4f}")
    print(f"  • Empirical Mean: {mean:.4f}")
    print(f"  • Difference: {diff:.4f} ({diff_rel:.4f}%)")
    print(f"  • Number of components (K): {model.n_components}")
    print(f"  • Number of observations: {len(data):,}")
    
    for k in range(model.n_components):
        print(f"\n  Component {k+1}:")
        print(f"    • Weight: {model.weights[k]:.4f}")
        print(f"    • α (alpha): {model.alphas[k]:.4f}")
        print(f"    • β (beta): {model.betas[k]:.4f}")
        print(f"    • Component expectation: {model.alphas[k] * model.betas[k]:.4f}")
    
    # Display test summary
    model.print_validation_summary()
    
    # Prepare result
    result = {
        'variable': variable_name,
        'display_name': display_name,
        'n_observations': len(data),
        'gamma_expectation': expectation,
        'empirical_mean': mean,
        'absolute_difference': diff,
        'percentage_difference': diff_rel,
        'n_components': model.n_components,
        'alphas': model.alphas.tolist(),
        'betas': model.betas.tolist(),
        'weights': model.weights.tolist(),
        'logL': model.logL,
        'bic': model.bic,
        'validation': model.validation_results
    }
    
    return result

# ---------- 6. EXECUTE GLOBAL ANALYSES ----------

print("\n" + "=" * 80)
print(f"GLOBAL GAMMA MLE ANALYSIS - {sector} SECTOR ({min_year}-{max_year})")
print("All data analyzed together")
print("=" * 80)

global_results = []

# Analyze each variable globally
for var_name, display_name in variables_to_analyze:
    # Skip R&D related variables for industrial sector
    if sector == "INDUSTRIAL" and var_name in ['rd', 'z3', 'z4']:
        continue
        
    result = analyze_global_variable(df, var_name, display_name)
    if result:
        global_results.append(result)

# ---------- 7. SAVE RESULTS ----------

print("\n" + "=" * 80)
print("SAVING GLOBAL RESULTS")
print("=" * 80)

# Main global results file
global_results_path = f'/content/drive/MyDrive/GLOBAL_MLE_RESULTS_{sector}_{min_year}_{max_year}.csv'

if global_results:
    # Convert to DataFrame
    df_results = pd.DataFrame(global_results)
    
    # Flatten lists for CSV
    for i in range(3):  # For K up to 3
        df_results[f'alpha_{i+1}'] = df_results['alphas'].apply(
            lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan)
        df_results[f'beta_{i+1}'] = df_results['betas'].apply(
            lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan)
        df_results[f'weight_{i+1}'] = df_results['weights'].apply(
            lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan)
    
    # Extract validation results
    validation_cols = ['KS_pvalue', 'AIC', 'BIC', 'logLikelihood', 'KS_statistic', 'Wasserstein']
    for col in validation_cols:
        df_results[f'valid_{col}'] = df_results['validation'].apply(
            lambda x: x.get(col, np.nan) if isinstance(x, dict) else np.nan)
    
    # Keep only important columns
    cols_to_keep = ['variable', 'display_name', 'n_observations', 'gamma_expectation', 
                     'empirical_mean', 'percentage_difference', 'n_components']
    cols_to_keep.extend([f'valid_{col}' for col in validation_cols])
    cols_to_keep.extend([f'alpha_{i+1}' for i in range(3)])
    cols_to_keep.extend([f'beta_{i+1}' for i in range(3)])
    cols_to_keep.extend([f'weight_{i+1}' for i in range(3)])
    
    df_results_final = df_results[cols_to_keep]
    df_results_final.to_csv(global_results_path, index=False, encoding='utf-8')
    print(f"Global results saved: {global_results_path}")

# Expectations only file
expectations_path = f'/content/drive/MyDrive/GAMMA_EXPECTATIONS_{sector}_{min_year}_{max_year}.csv'

expectations_data = []
for result in global_results:
    # Determine unit
    name = result['display_name']
    if 'EBIT' in name and '×' not in name:
        unit = "Billion USD"
    elif 'PP&E' in name:
        unit = "Billion USD"
    elif 'R&D' in name:
        unit = "Billion USD"
    elif 'Personnel' in name and '×' not in name:
        unit = "10k employees"
    elif 'Z1' in name:
        unit = "Billion USD × 10k employees"
    elif 'Z2' in name:
        unit = "Billion USD²"
    elif 'Z3' in name:
        unit = "Billion USD²"
    elif 'Z4' in name:
        unit = "Billion USD × 10k employees"
    else:
        unit = ""
    
    expectations_data.append({
        'Variable': result['display_name'],
        'Gamma_Expectation': result['gamma_expectation'],
        'Empirical_Mean': result['empirical_mean'],
        'Difference_%': result['percentage_difference'],
        'N_Observations': result['n_observations'],
        'N_Components': result['n_components'],
        'Unit': unit,
        'Period': f'{min_year}-{max_year}',
        'Analysis_Type': 'Global (all years combined)',
        'KS_pvalue': result['validation'].get('KS_pvalue', np.nan),
        'AIC': result['validation'].get('AIC', np.nan),
        'BIC': result['validation'].get('BIC', np.nan)
    })

if expectations_data:
    df_expectations = pd.DataFrame(expectations_data)
    df_expectations.to_csv(expectations_path, index=False, encoding='utf-8')
    print(f"Gamma expectations saved: {expectations_path}")

# ---------- 8. FINAL REPORT ----------

print("\n" + "=" * 80)
print(f"FINAL REPORT - GLOBAL MLE ANALYSIS {sector} SECTOR ({min_year}-{max_year})")
print("=" * 80)

print(f"\nGLOBAL GAMMA EXPECTATIONS ({min_year}-{max_year}):")
print("-" * 80)
for result in global_results:
    name = result['display_name']
    expectation = result['gamma_expectation']
    k = result['n_components']
    
    # Determine unit
    if 'EBIT' in name and '×' not in name:
        unit = "Billion USD"
    elif 'PP&E' in name:
        unit = "Billion USD"
    elif 'R&D' in name:
        unit = "Billion USD"
    elif 'Personnel' in name and '×' not in name:
        unit = "10k employees"
    elif 'Z1' in name:
        unit = "Billion USD × 10k employees"
    elif 'Z2' in name:
        unit = "Billion USD²"
    elif 'Z3' in name:
        unit = "Billion USD²"
    elif 'Z4' in name:
        unit = "Billion USD × 10k employees"
    else:
        unit = ""
    
    print(f" • {name:<30} = {expectation:>10.4f} {unit:<25} (K={k})")

print("\nGOODNESS OF FIT (KS Test):")
print("-" * 80)
for result in global_results:
    name = result['display_name']
    ks_pval = result['validation'].get('KS_pvalue', np.nan)
    if not np.isnan(ks_pval):
        if ks_pval >= 0.05:
            status = "Accepted"
        elif ks_pval >= 0.01:
            status = "Borderline"
        else:
            status = "Rejected"
        print(f" • {name:<30} : p = {ks_pval:.4f} {status}")

print("\nGENERATED FILES:")
print(f"  1. {global_results_path}")
print(f"  2. {expectations_path}")

print("\nDATA CHARACTERISTICS:")
print(f" • Analyzed period: {min_year}-{max_year}")
print(f" • Sector: {sector}")
print(f" • Companies: {df['compagny'].nunique()}")
print(f" • Total observations: {len(df)}")
print(f" • Globally analyzed variables: {len(global_results)}")

print("\n" + "=" * 80)
print("GLOBAL ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 80)

# ---------- 9. CALCULATE STANDARD DEVIATIONS ----------

print("\n" + "=" * 80)
print("CALCULATING MLE STANDARD DEVIATIONS")
print("=" * 80)

# Function to calculate standard deviation
def calculate_expectation_std_mixture(alphas, betas, weights):
    """Calculates expectation and standard deviation of a Gamma mixture"""
    alphas = np.array(alphas)
    betas = np.array(betas)
    weights = np.array(weights)
    
    # Expectation of each component
    component_expectations = alphas * betas
    
    # Total mixture expectation
    total_expectation = np.sum(weights * component_expectations)
    
    # Variance of each Gamma component: Var = α × β²
    component_variances = alphas * (betas ** 2)
    
    # Total mixture variance
    total_variance = np.sum(weights * (component_variances + (component_expectations - total_expectation) ** 2))
    
    # Standard deviation
    std_deviation = np.sqrt(total_variance)
    
    return total_expectation, std_deviation

# Calculate standard deviations for important variables
print("\nCALCULATED MLE STANDARD DEVIATIONS:")
print("-" * 80)

std_results = []

# Define which variables to calculate std for based on sector
if sector == "TECHNOLOGY":
    std_variables = [
        ('ebit_positif', 'EBIT', 'Billion USD'),
        ('rd', 'R&D', 'Billion USD'),
        ('ppe', 'PP&E', 'Billion USD'),
        ('personnel', 'Personnel', '10k employees'),
        ('z4', 'Z4 = R&D × Personnel', 'Billion USD × 10k employees')
    ]
else:  # INDUSTRIAL
    std_variables = [
        ('ebit_positif', 'EBIT', 'Billion USD'),
        ('ppe', 'PP&E', 'Billion USD'),
        ('personnel', 'Personnel', '10k employees')
    ]

for var_code, var_name, unit in std_variables:
    # Find corresponding result
    for result in global_results:
        if result['variable'] == var_code:
            alphas = result['alphas']
            betas = result['betas']
            weights = result['weights']
            
            # Calculate expectation and standard deviation
            expectation, std_deviation = calculate_expectation_std_mixture(alphas, betas, weights)
            cv = std_deviation / expectation if expectation > 0 else np.nan
            
            print(f"\n{var_name} ({unit}):")
            print(f"  MLE Expectation: {expectation:.4f}")
            print(f"  MLE Standard Deviation: {std_deviation:.4f}")
            print(f"  Coefficient of variation: {cv:.4f}")
            print(f"  Interval [μ-σ, μ+σ]: [{expectation-std_deviation:.2f}, {expectation+std_deviation:.2f}]")
            
            std_results.append({
                'Variable': var_name,
                'Code': var_code,
                'Unit': unit,
                'Components': len(alphas),
                'MLE_Expectation': expectation,
                'MLE_Std_Deviation': std_deviation,
                'Coefficient_Variation': cv
            })
            break

# Save standard deviations
if std_results:
    df_std_results = pd.DataFrame(std_results)
    std_path = f'/content/drive/MyDrive/MLE_STANDARD_DEVIATIONS_{sector}_{min_year}_{max_year}.csv'
    df_std_results.to_csv(std_path, index=False, encoding='utf-8')
    print(f"\nStandard deviations saved in: {std_path}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE!")
print("=" * 80)