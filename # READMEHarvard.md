# README: Replication Framework for Structural Rent Analysis (MLE)

This repository contains the official Python implementation of the Unified Gamma Mixture Maximum Likelihood Estimation (MLE) framework. These scripts replicate the structural analysis, goodness-of-fit diagnostics, and correlation matrices used to identify technological rents (2005–2023).

## 1. Core Estimation & Diagnostic Scripts

### 1.1 Structural Estimation (MLE_Global_Annual_Analyzis.py)
This is the primary engine for high-precision econometric modeling. It fits corporate financial data (EBIT, R&D, Personnel, PP&E) to Gamma mixture distributions.
- **Inputs**: tech_intensive_panel.csv and industrial_sector_panel.csv.
- **Key Functions**: EM Algorithm, L-BFGS-B optimization, and automated validation suite (AIC, BIC, Wasserstein).
- **Outputs**: Structural parameters (alpha, beta, weights) and annual expectations.

### 1.2 Visual Diagnostics (mle_analyzis_global_histogram_fit.py)
Generates high-resolution diagnostic plots to verify the model's accuracy.
- **Role**: Overlays the theoretical Gamma mixture density over the empirical histograms of the data.
- **Functions**: `create_histogram_with_fit_bw` generates publication-ready plots in black and white.
- **Outputs**: Histograms_Gamma_Fits_...PDF/PNG. These provide visual proof of the model's robustness.

## 2. Correlation & Interaction Scripts (Article Tables)
These scripts compute the structural interactions between variables to produce the final results presented in the study.

### 2.1 Industrial Sector Analysis (correlations_industrial.py)
- **Role**: Generates the correlation matrix for the industrial sample (Table 1 of the article).
- **Output**: Comparative table of MLE-based correlations vs. classical methods (Pearson, Spearman, Kendall).

### 2.2 Technology Sector Analysis (correlations_technology.py)
- **Role**: Generates the interaction matrix for the technology-intensive sample (Table 2 of the article).
- **Output**: Detailed matrix including R&D interactions (EBIT, Personnel, PP&E, R&D).

## 3. Synthesis & Hypothesis Validation

### 3.1 Rent Adjustment Comparison (comparaison_adjusted_tech_industrial_ebit_margin.py)
- **Role**: Tests the core hypothesis by comparing the industrial profit margin with the tech margin adjusted for the identified 10.54% structural rent.
- **Logic**: Executes t-tests, Levene variance tests, and Mann-Whitney U tests to verify convergence.
- **Output**: Time-series plots showing the convergence of adjusted margins.

## 4. Methodology: Structural Correlation Formula
Unlike standard linear methods, these scripts derive correlations from the underlying Gamma mixture structure:
1. MLE Covariance: Cov(X, Y) = E[Z] - (E[X] * E[Y]), where Z is the cross-product variable.
2. MLE Correlation: rho = Cov(X, Y) / (sigma_X * sigma_Y).

## 5. Usage Instructions
1. **Setup**: Place all CSV datasets in your /content/drive/MyDrive/ directory.
2. **Step 1**: Run `MLE_Global_Annual_Analyzis.py` to generate expectations.
3. **Step 2**: Run `mle_analyzis_global_histogram_fit.py` to validate the distribution fits.
4. **Step 3**: Run the sector-specific correlation scripts to populate the article's tables.
5. **Step 4**: Run the comparison script to finalize the hypothesis testing.

## 6. Technical Requirements
- **Language**: Python 3.x
- **Libraries**: scipy, pandas, numpy, matplotlib, seaborn, sklearn.
- **Reproducibility**: All random seeds are fixed at 42 to ensure exact replication of the results reported in the manuscript.