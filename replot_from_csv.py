"""
Reconstruct cross-sectional and sensitivity figures from saved CSV exports.

This avoids rerunning model fits when only plots need refreshing. It calls the
same functions in plots.py as main.py.

Limitation: sensitivity CSVs do not include Lasso active-factor counts, so
plot_sensitivity_lasso will not show the small numeric annotations above the
51 factors line that a full run would produce.
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

from config import (
    CONFIGS,
    SENSITIVITY_LASSO_ALPHAS,
    SENSITIVITY_RIDGE_ALPHAS,
    config_key,
)
from plots import (
    plot_r2_bar,
    plot_r2_vs_factors,
    plot_sensitivity_lasso,
    plot_sensitivity_ridge,
)
from sensitivity import SensitivityResult
from tables import save_ols_gls_comparison_table

_DEFAULT_CS_SUMMARY = os.path.join(
    "results", "tables", "cross_sectional", "config_summary_with_ci.csv"
)
_DEFAULT_SENS_RIDGE = os.path.join(
    "results", "tables", "sensitivity", "sensitivity_ridge_metrics.csv"
)
_DEFAULT_SENS_LASSO = os.path.join(
    "results", "tables", "sensitivity", "sensitivity_lasso_metrics.csv"
)


def _alphas_in_sweep_order(df: pd.DataFrame, config_alphas: list[float]) -> list[float]:
    """Order alpha grid like params.yaml, intersected with values present in CSV."""
    present = df["alpha"].astype(float).unique()
    ordered = [
        float(present[np.isclose(present, a, rtol=0, atol=1e-6)][0])
        for a in config_alphas
        if np.any(np.isclose(present, a, rtol=0, atol=1e-6))
    ]
    return ordered or sorted(float(x) for x in present)


def replot_cross_sectional(summary_csv: str) -> None:
    """Rebuild bar and vs-factors plots from config_summary_with_ci.csv."""
    df = pd.read_csv(summary_csv)
    required = {"model", "config", "mean_oos_r2", "oos_r2_ci_lo_95", "oos_r2_ci_hi_95"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    for cfg in CONFIGS:
        k = config_key(cfg)
        sub = df[df["config"] == k]
        if sub.empty:
            warnings.warn(f"No rows for config {k!r} in {summary_csv}", stacklevel=2)
            continue
        if sub["model"].duplicated().any():
            warnings.warn(
                f"Duplicate model rows for config {k!r}; using first occurrence.",
                stacklevel=2,
            )
            sub = sub.drop_duplicates(subset=["model"], keep="first")

        sub = sub.set_index("model")
        oos_means = sub["mean_oos_r2"].to_dict()
        cis = {
            m: (float(row.oos_r2_ci_lo_95), float(row.oos_r2_ci_hi_95))
            for m, row in sub.iterrows()
        }

        plot_r2_bar(oos_means, tag=k, ci=cis)
        plot_r2_vs_factors(oos_means, tag=k, ci=cis)

    best = df.loc[df.groupby("model")["mean_oos_r2"].idxmax()].set_index("model")
    oos_best = best["mean_oos_r2"].to_dict()
    cis_best = {
        m: (float(row.oos_r2_ci_lo_95), float(row.oos_r2_ci_hi_95))
        for m, row in best.iterrows()
    }
    plot_r2_bar(oos_best, tag="best_config", ci=cis_best)


def _sensitivity_result_from_frame(
    df: pd.DataFrame,
    reg_label: str,
    config_alphas: list[float],
) -> SensitivityResult:
    sub = df.loc[df["regularization"] == reg_label].copy()
    if sub.empty:
        raise ValueError(f"No rows with regularization={reg_label!r}")

    sub["alpha"] = sub["alpha"].astype(float)
    alphas = _alphas_in_sweep_order(sub, config_alphas)

    pivot = (
        sub.groupby(["model", "alpha"])["oos_r2"]
        .mean()
        .unstack()
        .reindex(columns=alphas, fill_value=np.nan)
    )
    r2 = {str(name): row.values.copy() for name, row in pivot.iterrows()}
    return SensitivityResult(regularization=reg_label, alphas=alphas, r2=r2)


def replot_ols_gls_table(summary_csv: str) -> None:
    """Write ols_gls_comparison.tex from config_summary_with_ci.csv."""
    df = pd.read_csv(summary_csv)
    required = {"model", "config", "mean_oos_r2", "oos_r2_ci_lo_95", "oos_r2_ci_hi_95"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")
    path = os.path.join("results", "tables", "cross_sectional", "ols_gls_comparison.tex")
    save_ols_gls_comparison_table(df, path)


def replot_sensitivity(ridge_csv: str, lasso_csv: str) -> None:
    """Rebuild sensitivity figures from long-format ridge/lasso metric CSVs."""
    ridge_df = pd.read_csv(ridge_csv)
    lasso_df = pd.read_csv(lasso_csv)
    for col in ("regularization", "model", "alpha", "oos_r2"):
        if col not in ridge_df.columns or col not in lasso_df.columns:
            raise ValueError(f"Expected column {col!r} in both sensitivity CSVs")

    ridge_result = _sensitivity_result_from_frame(
        ridge_df, "ridge", list(SENSITIVITY_RIDGE_ALPHAS)
    )
    lasso_result = _sensitivity_result_from_frame(
        lasso_df, "lasso", list(SENSITIVITY_LASSO_ALPHAS)
    )

    plot_sensitivity_ridge(ridge_result)
    plot_sensitivity_lasso(lasso_result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate cross-sectional and sensitivity plots from saved CSVs."
    )
    parser.add_argument(
        "--cs-summary",
        default=_DEFAULT_CS_SUMMARY,
        help=f"config_summary_with_ci.csv (default: {_DEFAULT_CS_SUMMARY})",
    )
    parser.add_argument(
        "--sensitivity-ridge",
        default=_DEFAULT_SENS_RIDGE,
        help=f"Ridge metrics CSV (default: {_DEFAULT_SENS_RIDGE})",
    )
    parser.add_argument(
        "--sensitivity-lasso",
        default=_DEFAULT_SENS_LASSO,
        help=f"Lasso metrics CSV (default: {_DEFAULT_SENS_LASSO})",
    )
    scope = parser.add_mutually_exclusive_group()
    scope.add_argument(
        "--cross-sectional-only",
        action="store_true",
        help="Only replot bar and vs-factors figures.",
    )
    scope.add_argument(
        "--sensitivity-only",
        action="store_true",
        help="Only replot sensitivity figures.",
    )
    scope.add_argument(
        "--ols-gls-only",
        action="store_true",
        help="Only write the OLS vs GLS comparison table.",
    )
    args = parser.parse_args()

    run_cs = not args.sensitivity_only and not args.ols_gls_only
    run_sens = not args.cross_sectional_only and not args.ols_gls_only
    run_ols_gls = not args.cross_sectional_only and not args.sensitivity_only

    if run_cs:
        if not os.path.isfile(args.cs_summary):
            print(f"Missing file: {args.cs_summary}", file=sys.stderr)
            sys.exit(1)
        replot_cross_sectional(args.cs_summary)

    if run_sens:
        for path in (args.sensitivity_ridge, args.sensitivity_lasso):
            if not os.path.isfile(path):
                print(f"Missing file: {path}", file=sys.stderr)
                sys.exit(1)
        replot_sensitivity(args.sensitivity_ridge, args.sensitivity_lasso)

    if run_ols_gls:
        if not os.path.isfile(args.cs_summary):
            print(f"Missing file: {args.cs_summary}", file=sys.stderr)
            sys.exit(1)
        replot_ols_gls_table(args.cs_summary)

    print("Done.")


if __name__ == "__main__":
    main()
