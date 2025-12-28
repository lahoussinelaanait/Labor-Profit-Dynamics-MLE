# Labor-Profit Dynamics via Unified Gamma Mixture MLE (2005–2023)

This repository contains the Python source code and computational framework developed to analyze the distributional characteristics and structural dependencies of the Industrial and Technology sectors.

## Overview

The core of this project is a **Unified Gamma Mixture Maximum Likelihood Estimation (MLE)** framework. It is designed to:
1.  Model the empirical distributions of EBIT, R&D, Fixed Capital (PP&E), and Personnel using Gamma mixtures.
2.  Independently estimate cross-moments ($E[XY]$) to quantify structural dependencies (covariances) without assuming linear correlations or specific copula forms.
3.  Validate model fit using Kolmogorov-Smirnov (KS) tests and Information Criteria (AIC/BIC).

## Repository Structure

- `/data`: Contains the sectoral datasets (CSV format) for the 2005–2023 period.
- `/scripts`: 
    - `main_analysis.py`: The primary script to run the global MLE analysis.
    - `gamma_mixtures.py`: Core functions for the Expectation-Maximization (EM) algorithm and Gamma PDF fitting.
    - `plotting_tool.py`: Script used to generate the Black & White histograms (Figures 3 and 4 in the manuscript).
- `requirements.txt`: List of necessary Python libraries.

## Requirements

To run the scripts, you will need Python 3.8+ and the following libraries:
- `numpy`
- `scipy`
- `pandas`
- `matplotlib`

You can install the dependencies using:
```bash

pip install -r requirements.txt
