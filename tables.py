import os
import numpy as np
import pandas as pd

_TABLES_CS = "results/tables/cross_sectional"
_CSV_CS_SUMMARY = os.path.join(_TABLES_CS, "config_summary_with_ci.csv")
_CSV_SENS_RIDGE = "results/tables/sensitivity/sensitivity_ridge_metrics.csv"
_CSV_SENS_LASSO = "results/tables/sensitivity/sensitivity_lasso_metrics.csv"
_TABLES_MAIN = "results/tables"
_TABLES_SENS = "results/tables/sensitivity"

_N_BOOTSTRAP = 1000
_CI_LEVEL = 0.95


def bootstrap_ci(
    window_r2s: np.ndarray,
    n_boot: int = _N_BOOTSTRAP,
    level: float = _CI_LEVEL,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Percentile bootstrap CI for the mean OOS R² across rolling windows.

    Each bootstrap resample draws T windows with replacement and computes their
    mean, approximating the sampling distribution of the estimator.
    """
    rng = np.random.default_rng(seed)
    T = len(window_r2s)
    means = np.array(
        [rng.choice(window_r2s, size=T, replace=True).mean() for _ in range(n_boot)]
    )
    alpha = (1 - level) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def compute_cis(
    oos_windows: dict[str, np.ndarray],
    **kwargs,
) -> dict[str, tuple[float, float]]:
    """Bootstrap CIs for all models. Returns {model: (lo, hi)}."""
    return {name: bootstrap_ci(arr, **kwargs) for name, arr in oos_windows.items()}


def save_cross_sectional_csv(
    oos_windows: dict[str, dict[str, np.ndarray]],
    ins_windows: dict[str, dict[str, np.ndarray]],
    mape_windows: dict[str, dict[str, np.ndarray]],
    config_keys: list[str],
) -> None:
    """
    Write two CSVs: (1) long-format per rolling window; (2) per model×config means
    with bootstrap 95% CIs on mean OOS R².
    """
    os.makedirs(_TABLES_CS, exist_ok=True)

    win_rows: list[dict] = []
    sum_rows: list[dict] = []
    for name in oos_windows:
        for k in config_keys:
            ins = ins_windows[name][k]
            oos = oos_windows[name][k]
            mape = mape_windows[name][k]
            for w_idx in range(len(oos)):
                win_rows.append(
                    {
                        "model": name,
                        "config": k,
                        "window_index": w_idx,
                        "ins_r2": float(ins[w_idx]),
                        "oos_r2": float(oos[w_idx]),
                        "mape": float(mape[w_idx]),
                    }
                )
            ci_lo, ci_hi = bootstrap_ci(oos)
            sum_rows.append(
                {
                    "model": name,
                    "config": k,
                    "mean_ins_r2": float(ins.mean()),
                    "mean_oos_r2": float(oos.mean()),
                    "oos_r2_ci_lo_95": ci_lo,
                    "oos_r2_ci_hi_95": ci_hi,
                    "mean_mape": float(mape.mean()),
                    "n_windows": int(len(oos)),
                }
            )

    pd.DataFrame(sum_rows).to_csv(_CSV_CS_SUMMARY, index=False)
    print(f"Saved {_CSV_CS_SUMMARY}")


def save_sensitivity_csvs(ridge_result, lasso_result) -> None:
    """
    Long-format CSV per regularization: one row per (model, alpha, window) with
    OOS R², IS R², MAPE. No confidence intervals.
    """
    os.makedirs(os.path.dirname(_CSV_SENS_RIDGE), exist_ok=True)

    def _build_frame(result, reg_label: str) -> pd.DataFrame:
        rows: list[dict] = []
        for name in result.r2_windows:
            rw = result.r2_windows[name]
            ins_w = result.ins_r2_windows[name]
            mape_w = result.mape_windows[name]
            n_alpha, n_win = rw.shape
            for a_idx in range(n_alpha):
                alpha_val = float(result.alphas[a_idx])
                for w_idx in range(n_win):
                    rows.append(
                        {
                            "regularization": reg_label,
                            "model": name,
                            "alpha": alpha_val,
                            "window_index": w_idx,
                            "oos_r2": float(rw[a_idx, w_idx]),
                            "ins_r2": float(ins_w[a_idx, w_idx]),
                            "mape": float(mape_w[a_idx, w_idx]),
                        }
                    )
        return pd.DataFrame(rows)

    _build_frame(ridge_result, "ridge").to_csv(_CSV_SENS_RIDGE, index=False)
    _build_frame(lasso_result, "lasso").to_csv(_CSV_SENS_LASSO, index=False)
    print(f"Saved {_CSV_SENS_RIDGE}")
    print(f"Saved {_CSV_SENS_LASSO}")


def save_config_table(
    oos_results: dict[str, np.ndarray],
    ins_results: dict[str, np.ndarray],
    mape_results: dict[str, np.ndarray],
    path: str,
) -> None:
    """
    Per-config LaTeX table: IS R², OOS R² (mean, median, % pos), MAPE.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    for name in oos_results:
        oos, ins, mape = oos_results[name], ins_results[name], mape_results[name]
        ci_lo, ci_hi = bootstrap_ci(oos)
        rows.append(
            (
                name.replace("_", " "),
                f"{ins.mean():.3f}",
                f"{oos.mean():.3f}",
                f"[{ci_lo:.3f}, {ci_hi:.3f}]",
                f"{np.median(oos):.3f}",
                f"{(oos > 0).mean() * 100:.1f}\\%",
                f"{mape.mean():.4f}",
            )
        )

    header = (
        "\\begin{tabular}{lrrrrrrr}\n"
        "\\toprule\n"
        "Model & IS $R^2$ & OOS $R^2$ & 95\\% CI & Median $R^2$ & \\% Pos & MAPE \\\\\n"
        "\\midrule\n"
    )
    body = "\n".join(
        f"{m} & {ins} & {oos} & {ci} & {med} & {pos} & {mape} \\\\"
        for m, ins, oos, ci, med, pos, mape in rows
    )
    footer = "\n\\bottomrule\n\\end{tabular}\n"
    with open(path, "w") as f:
        f.write(header + body + footer)
    print(f"Saved {path}")


def save_sensitivity_table(
    ridge_result, lasso_result, model_display_names: dict[str, str]
) -> None:
    """
    Two-panel sensitivity table (Panel A: Ridge, Panel B: Lasso).

    Columns = alpha values. Rows = models. Lasso cells additionally show the mean
    active factor count in parentheses: R² (K_active).

    model_display_names: maps model key (e.g. "PCA_5") to a display label.
    """
    os.makedirs(_TABLES_SENS, exist_ok=True)

    def _panel(result, include_active: bool) -> str:
        alphas = result.alphas
        col_header = " & ".join(
            f"$\\alpha_{{\\mathrm{{reg}}}}={a:.3g}$" for a in alphas
        )
        lines = [
            f"\\begin{{tabular}}{{l{'c' * len(alphas)}}}\n\\toprule",
            f"Model & {col_header} \\\\\n\\midrule",
        ]
        for key, label in model_display_names.items():
            if key not in result.r2:
                continue
            r2s = result.r2[key]
            afs = result.active_factors.get(key) if include_active else None
            cells = []
            for i, r2v in enumerate(r2s):
                cell = f"{r2v:.3f}"
                if afs is not None:
                    cell += f" ({int(round(afs[i]))})"
                cells.append(cell)
            lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
        lines.append("\\bottomrule\n\\end{tabular}")
        return "\n".join(lines)

    ridge_tex = _panel(ridge_result, include_active=False)
    lasso_tex = _panel(lasso_result, include_active=True)

    content = (
        "\\textbf{Panel A: Ridge}\\\\\n"
        + ridge_tex
        + "\n\n\\bigskip\n\\textbf{Panel B: Lasso (active factors in parentheses)}\\\\\n"
        + lasso_tex
        + "\n"
    )
    path = os.path.join(_TABLES_SENS, "sensitivity.tex")
    with open(path, "w") as f:
        f.write(content)
    print(f"Saved {path}")


_MODEL_DISPLAY = {
    "FF3": "FF3",
    "FF5": "FF5",
    "StandardOLS": "51 factors",
    "BAYESIAN": "BMA",
}

# Families rendered as "Label, $k = n$"
_FAMILY_DISPLAY = {
    "PCA": "PCA",
    "RPPCA": "RP-PCA",
    "DLFM": "DLFM",
}

# Models that are config-agnostic (best config shown as N/A)
_CONFIG_AGNOSTIC = {"BAYESIAN"}


def _model_display_name(key: str) -> str:
    """Human-readable LaTeX model name for the main table."""
    if key in _MODEL_DISPLAY:
        return _MODEL_DISPLAY[key]
    for prefix, label in _FAMILY_DISPLAY.items():
        if key.startswith(f"{prefix}_"):
            k = key.split("_", 1)[1]
            return f"{label}, $k = {k}$"
    return key.replace("_", " ")


def _config_display(key: str) -> str:
    """
    Convert a config_key string to a readable LaTeX string.
    e.g. "ols_lasso_0.05" → "OLS, Lasso, $\\alpha_{\mathrm{reg}} = 0.05$"
         "gls_none"       → "GLS, none"
    """
    parts = key.split("_")
    estimator = parts[0].upper()
    regularization = parts[1].capitalize() if len(parts) > 1 else "None"
    if len(parts) > 2:
        alpha_val = parts[2]
        return f"{estimator}, {regularization}, $\\alpha_{{\\mathrm{{reg}}}} = {alpha_val}$"
    return f"{estimator}, {regularization}"


def _reg_display(config_key: str) -> str:
    """Config string without the estimator prefix, e.g. 'Ridge, $\\alpha = 10.0$'."""
    parts = config_key.split("_")
    regularization = parts[1].capitalize() if len(parts) > 1 else "None"
    if len(parts) > 2:
        return f"{regularization}, $\\alpha_{{\\mathrm{{reg}}}} = {parts[2]}$"
    return regularization


def save_ols_gls_comparison_table(df: pd.DataFrame, path: str) -> None:
    """
    For each model, compare best GLS config against best OLS config.

    Columns: Model | GLS best config | OOS R² | 95% CI || OLS best config | OOS R² | 95% CI

    df must have columns: model, config, mean_oos_r2, oos_r2_ci_lo_95, oos_r2_ci_hi_95.
    """
    from config import PCA_N_VALUES, RPPCA_N_VALUES, DLFM_N_VALUES

    os.makedirs(os.path.dirname(path), exist_ok=True)

    _BENCH_ORDER = ("FF3", "FF5", "StandardOLS", "BAYESIAN")
    _FAMILIES = [
        ("PCA", PCA_N_VALUES),
        ("RPPCA", RPPCA_N_VALUES),
        ("DLFM", DLFM_N_VALUES),
    ]

    def _best(model_df: pd.DataFrame, prefix: str):
        sub = model_df[model_df["config"].str.startswith(prefix)]
        if sub.empty:
            return None
        return sub.loc[sub["mean_oos_r2"].idxmax()]

    def _fmt(row, bold: bool) -> tuple[str, str, str]:
        if row is None:
            return "---", "---", "---"
        cfg = _reg_display(row["config"])
        oos_val = f"{row['mean_oos_r2']:.3f}"
        if bold:
            oos_val = f"\\textbf{{{oos_val}}}"
        ci = f"[{row['oos_r2_ci_lo_95']:.3f},\\ {row['oos_r2_ci_hi_95']:.3f}]"
        return cfg, oos_val, ci

    def _tex_row(name: str) -> str | None:
        model_df = df[df["model"] == name]
        if model_df.empty:
            return None
        gls_row = _best(model_df, "gls_")
        ols_row = _best(model_df, "ols_")
        gls_val = gls_row["mean_oos_r2"] if gls_row is not None else float("-inf")
        ols_val = ols_row["mean_oos_r2"] if ols_row is not None else float("-inf")
        gls_better = gls_val > ols_val
        gc, go, gci = _fmt(gls_row, bold=gls_better)
        oc, oo, oci = _fmt(ols_row, bold=not gls_better)
        label = _model_display_name(name)
        return f"{label} & {gc} & {go} & {gci} & {oc} & {oo} & {oci} \\\\"

    bench_lines = [r for name in _BENCH_ORDER if (r := _tex_row(name)) is not None]
    family_lines = [
        r
        for prefix, n_values in _FAMILIES
        for n in n_values
        if (r := _tex_row(f"{prefix}_{n}")) is not None
    ]

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\begin{tabular}{l l r l || l r l}\n"
        "\\toprule\n"
        " & \\multicolumn{3}{c||}{GLS} & \\multicolumn{3}{c}{OLS} \\\\\n"
        "\\cmidrule(lr){2-4} \\cmidrule(l){5-7}\n"
        "Model & Best config & OOS $R^2$ & 95\\% CI"
        " & Best config & OOS $R^2$ & 95\\% CI \\\\\n"
        "\\midrule\n"
    )
    body = "\n".join(bench_lines) + "\n\\midrule\n" + "\n".join(family_lines)
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"

    with open(path, "w") as f:
        f.write(header + body + footer)
    print(f"Saved {path}")


def save_main_table(
    oos_windows: dict[str, dict[str, np.ndarray]],
    ins_windows: dict[str, dict[str, np.ndarray]],
    mape_windows: dict[str, dict[str, np.ndarray]],
    path: str,
) -> None:
    """
    Main summary table: for each model, select the config with the highest mean
    OOS R², then report IS R², OOS R² ± 95% CI, MAPE, and the winning config.

    oos_windows[model][config_key] = array of per-window R² values
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = []
    for name in oos_windows:
        is_agnostic = name in _CONFIG_AGNOSTIC
        best_key = max(oos_windows[name], key=lambda k: oos_windows[name][k].mean())
        oos = oos_windows[name][best_key]
        ins = ins_windows[name][best_key]
        mape = mape_windows[name][best_key]
        ci_lo, ci_hi = bootstrap_ci(oos)
        rows.append(
            (
                _model_display_name(name),
                f"{ins.mean():.3f}",
                f"{oos.mean():.3f}",
                f"[{ci_lo:.3f},\\ {ci_hi:.3f}]",
                f"{mape.mean():.4f}",
                "N/A" if is_agnostic else _config_display(best_key),
            )
        )

    header = (
        "\\begin{table}[h]\n"
        "\\centering\n"
        "\\begin{tabular}{lrrrrr}\n"
        "\\toprule\n"
        "Model & IS $R^2$ & OOS $R^2$ & 95\\% CI & MAPE & Best config \\\\\n"
        "\\midrule\n"
    )
    body = "\n".join(
        f"{m} & {ins} & {oos} & {ci} & {mape} & {cfg} \\\\"
        for m, ins, oos, ci, mape, cfg in rows
    )
    footer = "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    with open(path, "w") as f:
        f.write(header + body + footer)
    print(f"Saved {path}")
