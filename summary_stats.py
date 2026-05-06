import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from data_utils import load_data

NW_LAGS = 5

# Factor classification — the 17 non-tradable macro/uncertainty/sentiment series
NON_TRADABLE = [
    "LIQ_NT",
    "INTERM_CAP_RATIO",
    "FIN_UNC",
    "REAL_UNC",
    "MACRO_UNC",
    "BW_ISENT",
    "HJTZ_ISENT",
    "TERM",
    "DEFAULT",
    "DIV",
    "UNRATE",
    "PE",
    "NONDUR",
    "SERV",
    "IPGrowth",
    "Oil",
    "DeltaSLOPE",
]
FF5 = ["MKT", "SMB", "HML", "RMW", "CMA"]


def nw_tstat(series: pd.Series, lags: int = NW_LAGS) -> float:
    # Newey-West t-stat corrects for autocorrelation and heteroskedasticity
    y = series.values.astype(float)
    X = np.ones_like(y).reshape(-1, 1)
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(res.tvalues[0])


def series_stats(series: pd.Series, tradable: bool) -> dict:
    mean = series.mean()  # time-series
    sd = series.std(ddof=1)
    tstat = nw_tstat(series)
    # annualize with sqrt(12)
    sharpe = (mean / sd) * np.sqrt(12) if tradable else np.nan
    return {"mean": mean, "sd": sd, "tstat": tstat, "sharpe": sharpe}


factors, portfolios = load_data()

date_col = "Date"
factor_cols = [c for c in factors.columns if c != date_col]
port_cols = [c for c in portfolios.columns if c != date_col]

print(f"Sample period: {factors[date_col].min()} - {factors[date_col].max()}")
print(f"Observations: {len(factors)}")
print(f"Factors: {len(factor_cols)} (expected 51)")
print(f"Test portfolios: {len(port_cols)} (expected 60)")
print()


rows = {}
for col in factor_cols:
    is_tradable = col not in NON_TRADABLE
    rows[col] = {
        **series_stats(factors[col], tradable=is_tradable),
        "tradable": is_tradable,
    }
fstats = pd.DataFrame(rows).T
fstats["tradable"] = fstats["tradable"].astype(bool)
for c in ["mean", "sd", "tstat", "sharpe"]:
    fstats[c] = fstats[c].astype(float)

# Sanity check
n_tradable = int(fstats["tradable"].sum())
n_nontrad = len(fstats) - n_tradable
print(f"Tradable factors: {n_tradable} (expected 34)")
print(f"Non-tradable factors: {n_nontrad} (expected 17)")
print()

prows = {}
for col in port_cols:
    prows[col] = series_stats(portfolios[col], tradable=True)
pstats = pd.DataFrame(prows).T


def fmt(x: float, precision: int = 2) -> str:
    if pd.isna(x):
        return "--"  # N/A fallback
    return f"{x:.{precision}f}"


def factor_table_latex() -> str:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Summary statistics for the factor set.}")
    lines.append(r"\label{tab:factor-summary}")
    lines.append(r"\begin{tabular}{l r r r r}")
    lines.append(r"\toprule")
    lines.append(r" & Mean & SD & $t$-stat & Sharpe \\")
    lines.append(r"\midrule")

    # Panel A: FF5
    lines.append(r"\multicolumn{5}{l}{\textit{Panel A: Fama-French factors}} \\")
    for f in FF5:
        s = fstats.loc[f]
        lines.append(
            f"{f} & {fmt(s['mean'])} & {fmt(s['sd'])} & "
            f"{fmt(s['tstat'])} & {fmt(s['sharpe'])} \\\\"
        )

    # Panel B: Other tradable
    other_trad = fstats[(fstats["tradable"]) & (~fstats.index.isin(FF5))]
    n = len(other_trad)
    sig_share = (other_trad["tstat"].abs() > 2).mean()
    lines.append(r"\midrule")
    lines.append(
        r"\multicolumn{5}{l}{\textit{Panel B: Other tradable factors (cross-sectional summary)}} \\"
    )
    lines.append(f"Number of factors & \\multicolumn{{4}}{{r}}{{{n}}} \\\\")
    lines.append(
        f"Mean of monthly means & "
        f"\\multicolumn{{4}}{{r}}{{{fmt(other_trad['mean'].mean())}}} \\\\"
    )
    lines.append(
        f"Range of means & "
        f"\\multicolumn{{4}}{{r}}{{[{fmt(other_trad['mean'].min())}, "
        f"{fmt(other_trad['mean'].max())}]}} \\\\"
    )
    lines.append(
        f"Mean SD & " f"\\multicolumn{{4}}{{r}}{{{fmt(other_trad['sd'].mean())}}} \\\\"
    )
    lines.append(
        f"Mean annualized Sharpe & "
        f"\\multicolumn{{4}}{{r}}{{{fmt(other_trad['sharpe'].mean())}}} \\\\"
    )
    lines.append(
        f"Fraction with $|t|>2$ & "
        f"\\multicolumn{{4}}{{r}}{{{fmt(sig_share, 2)}}} \\\\"
    )

    # Panel C: Non-tradable factors; heterogeneous scales
    nontrad = fstats[~fstats["tradable"]]
    n_nt = len(nontrad)
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{l}{\textit{Panel C: Non-tradable factors}} \\")
    lines.append(r" & Mean & SD & Min & Max \\")
    lines.append(r"\cmidrule(lr){2-5}")

    # Need raw min/max of each non-tradable series
    for f in NON_TRADABLE:
        s = factors[f]  # non-tradable factor series
        lines.append(
            f"{f.replace('_', r'\_')} & {fmt(s.mean(), 3)} & {fmt(s.std(), 3)} & "
            f"{fmt(s.min(), 2)} & {fmt(s.max(), 2)} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def portfolio_table_latex() -> str:
    means = pstats["mean"]
    sds = pstats["sd"]
    sharpes = pstats["sharpe"]
    tstats = pstats["tstat"]

    def row(label: str, s: pd.Series) -> str:
        # .quantile(0.x) gives the x-th percentile
        return (
            f"{label} & {fmt(s.mean())} & {fmt(s.std())} & "
            f"{fmt(s.quantile(0.10))} & {fmt(s.quantile(0.50))} & "
            f"{fmt(s.quantile(0.90))} & "
            f"{fmt(s.min())} & {fmt(s.max())} \\\\"
        )

    sig_share = (tstats.abs() > 2).mean()

    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Cross-sectional summary statistics for the 60 test portfolios.}"
    )
    lines.append(r"\label{tab:portfolio-summary}")
    lines.append(r"\begin{tabular}{l r r r r r r r}")
    lines.append(r"\toprule")
    lines.append(r" & Mean & SD & $P_{10}$ & Median & $P_{90}$ & Min & Max \\")
    lines.append(r"\midrule")
    lines.append(row("Monthly mean return (\\%)", means))
    lines.append(row("Standard deviation (\\%)", sds))
    lines.append(row("Annualised Sharpe ratio", sharpes))
    lines.append(row("$t$-statistic of mean", tstats))
    lines.append(r"\midrule")
    lines.append(
        f"\\multicolumn{{8}}{{l}}{{Fraction of portfolios with $|t|>2$: {fmt(sig_share, 2)}}} \\\\"
    )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


factor_tex = factor_table_latex()
port_tex = portfolio_table_latex()


out_dir = "results/tables/data_summary"
os.makedirs(out_dir, exist_ok=True)
with open(os.path.join(out_dir, "factor_summary.tex"), "w") as f:
    f.write(factor_tex)
with open(os.path.join(out_dir, "portfolio_summary.tex"), "w") as f:
    f.write(port_tex)


print("Data summary tables written to results/tables/data_summary.")
