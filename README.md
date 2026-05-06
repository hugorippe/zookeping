# Zookeping

Replication code for the article "Zookeping: A Comparison of Factor Pruning Methods" by Hugo Rippe and Axel Riley.  The paper investigates among other things whether dimensionality reduction improves out-of-sample portfolio performance in a high-dimensional asset-pricing environment.

## Models

| Model | Class | Description |
|---|---|---|
| FF3 / FF5 | `FF3Model`, `FF5Model` | Fama-French 3- and 5-factor OLS benchmarks |
| 51-factor | `StandardOLSModel` | OLS on all 51 factors |
| PCA | `PCAModel` | Principal component regression |
| RP-PCA | `RPPCA` | Risk-Premium PCA (Pelger & Xiong 2022) |
| Bayesian | `BayesianModel` | Spike-and-slab SDF à la Bryzgalova et al. (2023) |
| DLFM | `DLFMModel` | Deep latent factor model (autoencoder) |

## Data

The empirical analysis uses 51 monthly factors and 60 test portfolios corresponding to the dataset in Bryzgalova et al. (2023). The data can be obtained in accordance with that paper, reconstructed using the Kenneth French factor data libraries, or an entirely different data set could be used to obtain results beyond the scope of the paper.

The expected files are:
- `data/MonthlyFactors.xlsx` — 51 factors (tradable and nontradable), sheet `51 factors - ranked`
- `data/MonthlyPortfolios.xlsx` — 60 cross-sectional test portfolios, sheet `Sheet1`

Both files must be present before running `main.py`.

## Installation

Python 3.10+ is recommended. Install dependencies:

```bash
pip install -r requrements.txt
```

## Reproducing results

```bash
python main.py
```

Figures (300 DPI PNGs) are written to `results/figures/` and LaTeX tables to `results/tables/`.

**Auxiliary scripts:**

```bash
python summary_stats.py       # Descriptive statistics table
python sensitivity.py         # Regularisation sensitivity sweep (also called from main.py)
python replot_from_csv.py     # Regenerate plots from saved CSVs without re-running models
```

All experiment parameters (window lengths, hyperparameters, regularisation grids) are in `params.yaml`.

## Repository structure

```
models/             one file per model family, all subclass BaseModel
config.py           code-level constants (colours, figure sizes, factor lists)
params.yaml         experiment parameters and hyperparameters
data_utils.py       data loading and rolling-window construction
plots.py            all plotting functions
tables.py           LaTeX table generation
sensitivity.py      regularisation sensitivity analysis
summary_stats.py    descriptive statistics
replot_from_csv.py  regenerate figures from saved CSVs
main.py             entry point — reproduces all results
results/
  figures/          300 DPI PNGs, \includegraphics{}-ready
  tables/           .tex files, \input{}-ready with booktabs
```

