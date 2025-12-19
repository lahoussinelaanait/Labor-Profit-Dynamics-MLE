# GLOBAL GAMMA MLE ANALYSIS
# Reproducibility script for arXiv submission
# ========================================================

import pandas as pd
import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.stats import gamma, kstest, chi2
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ---------- 1. FILE SELECTION ----------
# Uncomment the line for the sector you want to analyze
# FICHIER = 'industrial_sector_panel.csv'
FICHIER = 'tech_intensive_panel.csv' 

print(f"File path: {FICHIER}")

# ---------- 2. GAMMA MIXTURE MLE CLASS WITH VALIDATION ----------

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
        """Gamma Log PDF with regularization"""
        x = np.maximum(x, self.reg_param)
        alpha = max(alpha, self.reg_param)
        beta = max(beta, self.reg_param)
        return (alpha - 1) * np.log(x) - x/beta - alpha * np.log(beta) - gammaln(alpha)
    
    def _cdf_mixture(self, x):
        """Mixture CDF"""
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
        
        # 1. Kolmogorov-Smirnov Test
        ks_stat, ks_pvalue = kstest(X, self._cdf_mixture)
        
        # 2. Information Criteria
        n_params = 3 * self.n_components - 1
        aic = 2 * n_params - 2 * self.logL
        bic = np.log(len(X)) * n_params - 2 * self.logL
        
        # 3. Adapted Chi2 Test
        try:
            n_bins = min(10, max(3, int(np.sqrt(len(X)))))
            hist, bin_edges = np.histogram(X, bins=n_bins, density=False)
            expected = np.zeros(n_bins)
            for i in range(n_bins):
                prob = self._cdf_mixture(bin_edges[i+1]) - self._cdf_mixture(bin_edges[i])
                expected[i] = prob * len(X)
            mask = expected >= 1
            if np.sum(mask) >= 2:
                chi2_stat = np.sum((hist[mask] - expected[mask])**2 / expected[mask])
                df = max(1, np.sum(mask) - n_params - 1)
                chi2_pvalue = 1 - chi2.cdf(chi2_stat, df) if df > 0 else np.nan
            else:
                chi2_stat, chi2_pvalue, df = np.nan, np.nan, 0
        except:
            chi2_stat, chi2_pvalue, df = np.nan, np.nan, 0
        
        # 4. Wasserstein Distance
        X_sorted = np.sort(X)
        wasserstein = np.mean(np.abs(np.arange(1, len(X) + 1) / len(X) - self._cdf_mixture(X_sorted)))
        
        return {
            'KS_statistic': ks_stat, 'KS_pvalue': ks_pvalue,
            'AIC': aic, 'BIC': bic, 'Chi2_pvalue': chi2_pvalue,
            'Wasserstein': wasserstein, 'logLikelihood': self.logL
        }
    
    def _initialize_components(self, X, K):
        """Components initialization"""
        from sklearn.cluster import KMeans
        X_log = np.log(np.maximum(X, self.reg_param))
        kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
        labels_all = kmeans.fit_predict(X_log.reshape(-1, 1))
        
        alphas, betas, weights = [], [], []
        for k in range(K):
            X_k = X[labels_all == k]
            if len(X_k) >= 3:
                alpha_k, beta_k = self._fit_single_gamma_robuste(X_k, scale_factor=np.mean(X_k))
                alphas.append(alpha_k); betas.append(beta_k); weights.append(len(X_k))
            else:
                alpha_k, beta_k = self._fit_single_gamma_robuste(X, scale_factor=np.mean(X))
                alphas.append(alpha_k); betas.append(beta_k); weights.append(1)
        
        return np.array(alphas), np.array(betas), np.array(weights) / np.sum(weights)
    
    def _EM_iteration(self, X, alphas, betas, weights):
        """EM Iteration"""
        N, K = len(X), len(alphas)
        log_resp = np.zeros((N, K))
        for k in range(K):
            log_resp[:, k] = self._gamma_logpdf(X, alphas[k], betas[k]) + np.log(max(weights[k], 1e-10))
        resp = np.exp(log_resp - logsumexp(log_resp, axis=1, keepdims=True))
        Nk = resp.sum(axis=0)
        weights_new = Nk / N
        alphas_new, betas_new = np.zeros(K), np.zeros(K)
        for k in range(K):
            alphas_new[k], betas_new[k] = self._fit_single_gamma_robuste(X, weights=resp[:, k])
        return alphas_new, betas_new, weights_new
    
    def _log_likelihood(self, X, alphas, betas, weights):
        K = len(alphas)
        log_probs = np.zeros((len(X), K))
        for k in range(K):
            log_probs[:, k] = self._gamma_logpdf(X, alphas[k], betas[k]) + np.log(max(weights[k], 1e-10))
        return logsumexp(log_probs, axis=1).sum()
    
    def fit(self, X):
        """Fit with validation tests"""
        X_clean = X[X > 0]
        best_bic = np.inf
        for K in range(self.K_min, self.K_max + 1):
            alphas, betas, weights = self._initialize_components(X_clean, K)
            logL_old = -np.inf
            for _ in range(self.max_iter):
                alphas, betas, weights = self._EM_iteration(X_clean, alphas, betas, weights)
                logL = self._log_likelihood(X_clean, alphas, betas, weights)
                if abs(logL - logL_old) < self.tol: break
                logL_old = logL
            bic = np.log(len(X_clean)) * (3*K-1) - 2*logL
            if bic < best_bic:
                best_bic = bic
                self.n_components, self.alphas, self.betas, self.weights, self.logL = K, alphas, betas, weights, logL
        self.validation_results = self._run_tests_validation(X_clean)
        return self

    def expectation(self):
        return np.sum(self.weights * self.alphas * self.betas)

# ---------- 3. LOAD AND PREPARE DATA ----------

try:
    df = pd.read_csv(FICHIER, delimiter=';')
    # Internal mapping to match English CSV headers
    df = df.rename(columns={
        'Company': 'compagny', 'Fiscal Year': 'year',
        'Revenue (Billion USD)': 'ca', 'PP&E (Billion USD)': 'ppe',
        'EBIT (Billion USD)': 'ebit', 'R&D (Billion USD)': 'rd',
        'Employees (10k units)': 'personnel'
    })
    print("File loaded successfully!")
except Exception as e:
    print(f"Loading error: {e}"); exit()

# ---------- 4. GLOBAL ANALYSIS EXECUTION ----------

def analyser_variable_globale(df, variable_nom, nom_affichage):
    print(f"\nGLOBAL MLE ANALYSIS: {nom_affichage}")
    if variable_nom == 'z4':
        df['z4'] = df['rd'] * df['personnel']
        donnees = df['z4'].dropna().values
    else:
        donnees = df[variable_nom].dropna().values
    
    donnees = donnees[donnees > 0]
    modele = GammaMixtureMLE_Validation().fit(donnees)
    
    print(f"  Observations: {len(donnees)}")
    print(f"  MLE Expectation: {modele.expectation():.4f}")
    print(f"  Components (K): {modele.n_components}")
    print(f"  KS p-value: {modele.validation_results.get('KS_pvalue', 0):.4f}")

# Run analysis
analyser_variable_globale(df, 'ebit', 'EBIT (Billion USD)')
analyser_variable_globale(df, 'rd', 'R&D (Billion USD)')
analyser_variable_globale(df, 'z4', 'Z4 = R&D x Personnel')

print("\nAnalysis completed.")