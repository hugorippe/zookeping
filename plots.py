from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from config import (
    COL_FF3,
    COL_FF5,
    COL_PCA,
    COL_DLFM,
    COL_RPPCA,
    COL_BAYESIAN,
    FIGSIZE,
    PCA_N_VALUES,
    RPPCA_N_VALUES,
    DLFM_N_VALUES,
    model_color,
)

_FIGS_BAR = "results/figures/cross_sectional/bar_plots"
_FIGS_VS = "results/figures/cross_sectional/vs_factors"

_FIGS_SENS = "results/figures/cross_sectional/sensitivity"

_BENCH_ORDER = ("FF3", "FF5", "StandardOLS", "BAYESIAN")
_BENCH_TICK = {
    "FF3": "FF3",
    "FF5": "FF5",
    "StandardOLS": "51 factors",
    "BAYESIAN": "BMA",
}
_COMPONENT_TICK = (
    ("RPPCA", RPPCA_N_VALUES, "RP-PCA"),
    ("PCA", PCA_N_VALUES, "PCA"),
    ("DLFM", DLFM_N_VALUES, "DLFM"),
)


def _bar_tick_label(key: str) -> str:
    if key in _BENCH_TICK:
        return _BENCH_TICK[key]
    for prefix, _, title in _COMPONENT_TICK:
        if key.startswith(f"{prefix}_"):
            n = int(key.rsplit("_", 1)[-1])
            return f"{title}, $K={n}$"
    return key.replace("_", " ")


def plot_r2_bar(
    results: dict[str, float],
    tag: str | None = None,
    ci: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    Horizontal bar plot of mean OOS CS R² across models.

    ci: optional dict mapping model name → (lower, upper) CI bounds.
        When provided, error bars are drawn centred on the mean value.
    """
    os.makedirs(_FIGS_BAR, exist_ok=True)

    groups = [[k for k in _BENCH_ORDER if k in results]]
    for prefix, n_list, _ in _COMPONENT_TICK:
        groups.append([f"{prefix}_{n}" for n in n_list if f"{prefix}_{n}" in results])

    y_pos, values, colors, labels, xerr_lo, xerr_hi = [], [], [], [], [], []
    current_y = 0.0

    for group in groups:
        if not group:
            continue
        for k in group:
            y_pos.append(current_y)
            values.append(results[k])
            colors.append(model_color(k))
            labels.append(_bar_tick_label(k))
            if ci and k in ci:
                lo, hi = ci[k]
                xerr_lo.append(results[k] - lo)
                xerr_hi.append(hi - results[k])
            else:
                xerr_lo.append(0.0)
                xerr_hi.append(0.0)
            current_y -= 1.0
        current_y -= 0.8

    if not values:
        return

    fig_h = float(max(FIGSIZE[1], min(22.0, 0.26 * len(values) + 2.2)))
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], fig_h), layout="constrained")
    ax.set_axisbelow(True)

    ax.barh(
        y_pos,
        values,
        height=0.72,
        color=colors,
        edgecolor="0.98",
        linewidth=0.85,
        zorder=2,
    )

    if ci:
        ax.errorbar(
            values,
            y_pos,
            xerr=[xerr_lo, xerr_hi],
            fmt="none",
            ecolor="0.25",
            elinewidth=1.1,
            capsize=2,
            capthick=1.1,
            zorder=3,
        )

    ax.axvline(0, color="0.2", linewidth=1.0, zorder=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_xlabel("Cross-sectional OOS $R^2$", labelpad=8)

    lo_val, hi_val = min(values), max(values)
    span = hi_val - lo_val
    pad = max(span * 0.07, 0.015) if span > 0 else 0.04
    # extend xlim to fit error bars if present
    if ci:
        all_lo = [values[i] - xerr_lo[i] for i in range(len(values))]
        all_hi = [values[i] + xerr_hi[i] for i in range(len(values))]
        lo_val, hi_val = min(all_lo), max(all_hi)
    ax.set_xlim(min(lo_val, 0.0) - pad, max(hi_val, 0.0) + pad)

    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, min_n_ticks=4))
    ax.xaxis.grid(True, which="major", zorder=0)
    sns.despine(ax=ax)

    fname = f"cross_sectional_r2_bar_{tag}.png" if tag else "cross_sectional_r2_bar.png"
    path = os.path.join(_FIGS_BAR, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


def plot_r2_vs_factors(
    results: dict[str, float],
    tag: str | None = None,
    ci: dict[str, tuple[float, float]] | None = None,
) -> None:
    """
    R² vs. number of latent factors for PCA, DLFM, RP-PCA families.
    Benchmark models shown as horizontal dashed lines.

    ci: optional dict mapping model name → (lower, upper) CI bounds (shaded band).
    """
    os.makedirs(_FIGS_VS, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE)

    all_n_vals = set()
    model_configs = [
        ("PCA", COL_PCA, "o", "PCA"),
        ("DLFM", COL_DLFM, "s", "DLFM"),
        ("RPPCA", COL_RPPCA, "^", "RP-PCA"),
    ]

    for prefix, color, marker, label in model_configs:
        keys = [k for k in results if k.startswith(f"{prefix}_")]
        if not keys:
            continue
        n_vals = [int(k.split("_")[1]) for k in keys]
        r2_vals = [results[k] for k in keys]
        n_vals, r2_vals = zip(*sorted(zip(n_vals, r2_vals)))

        ax.plot(n_vals, r2_vals, marker=marker, color=color, label=label)
        all_n_vals.update(n_vals)

        if ci:
            lo_band = [
                ci[f"{prefix}_{n}"][0] if f"{prefix}_{n}" in ci else r
                for n, r in zip(n_vals, r2_vals)
            ]
            hi_band = [
                ci[f"{prefix}_{n}"][1] if f"{prefix}_{n}" in ci else r
                for n, r in zip(n_vals, r2_vals)
            ]
            ax.fill_between(n_vals, lo_band, hi_band, color=color, alpha=0.15)

    for key, color, label in [
        ("FF3", COL_FF3, "FF3"),
        ("FF5", COL_FF5, "FF5"),
        ("BAYESIAN", COL_BAYESIAN, "BMA"),
    ]:
        if key in results:
            val = results[key]
            ax.axhline(val, color=color, linestyle="--", label=f"{label} ({val:.3f})")

    ax.set_xlabel("Number of Latent Factors")
    ax.set_ylabel("Cross-sectional OOS $R^2$")
    ax.legend(loc="lower right")

    if all_n_vals:
        ax.set_xticks(sorted(all_n_vals))

    sns.despine(ax=ax)

    fname = (
        f"cross_sectional_r2_vs_factors_{tag}.png"
        if tag
        else "cross_sectional_r2_vs_factors.png"
    )
    path = os.path.join(_FIGS_VS, fname)
    plt.savefig(path)
    plt.close()
    print(f"Saved {path}")


_FIGS_TIME = "results/figures/cross_sectional"


def plot_cs_r2_over_time(
    oos_arrays: dict[str, dict[str, np.ndarray]],
    line_plot_models: list[str],
    line_plot_cs: dict[str, tuple[str | None, list]],
    window_labels: list | None = None,
) -> None:
    """OOS CS R² per rolling window for a selected set of (model, config) pairs."""
    os.makedirs(_FIGS_TIME, exist_ok=True)
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")

    for model in line_plot_models:
        if model not in oos_arrays:
            continue
        cfg_key, _ = line_plot_cs[model]
        if cfg_key is None:
            cfg_key = next(iter(oos_arrays[model]))
        if cfg_key not in oos_arrays[model]:
            continue
        y = oos_arrays[model][cfg_key]
        x = window_labels if window_labels is not None else list(range(len(y)))
        ax.plot(x, y, color=model_color(model), label=_bar_tick_label(model), linewidth=1.5)

    ax.axhline(0, color="0.5", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Test period start (year)")
    ax.set_ylabel("OOS Cross-Sectional $R^2$")
    ax.legend(loc="best")
    if window_labels is not None:
        ax.tick_params(axis="x", rotation=45)
    sns.despine(ax=ax)

    path = os.path.join(_FIGS_TIME, "cs_r2_over_time.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


# ---------------------------------------------------------------------------
# Sensitivity plots
# ---------------------------------------------------------------------------

# Line colours for the benchmark/scalar models in the sensitivity plots
_SENS_LINE_COLORS = {
    "FF3": COL_FF3,
    "FF5": COL_FF5,
    "StandardOLS": model_color("StandardOLS"),
}

# Family configs: (prefix, K values list, display label, heatmap colormap)
_SENS_FAMILIES = [
    ("PCA", PCA_N_VALUES, "PCA", "coolwarm"),
    ("RPPCA", RPPCA_N_VALUES, "RP-PCA", "coolwarm"),
    ("DLFM", DLFM_N_VALUES, "DLFM", "coolwarm"),
]


def _build_heatmap_grid(
    result_dict: dict[str, np.ndarray],
    prefix: str,
    n_values: list[int],
) -> np.ndarray:
    """
    Assemble a (len(n_values) × len(alphas)) matrix from the flat result dict.
    Rows = K values (ascending), columns = alpha values.
    """
    rows = []
    for n in n_values:
        key = f"{prefix}_{n}"
        rows.append(
            result_dict[key]
            if key in result_dict
            else np.full(len(next(iter(result_dict.values()))), np.nan)
        )
    return np.array(rows)


_FS_TICK = 20  # axis tick labels
_FS_LABEL = 22  # axis labels and colorbar labels
_FS_TITLE = 26  # subplot titles
_FS_CELL = 20  # heatmap cell annotations
_FS_ANNOT = 20  # active-factor annotations on line plots
_FS_LEGEND = 20  # legend text
_MARKER_SZ = 7


def _draw_heatmap(
    ax: plt.Axes,
    data: np.ndarray,
    row_labels: list,
    col_labels: list,
    title: str,
    cbar_label: str,
    cmap: str = "coolwarm",
    vcenter: float | None = 0.0,
    fmt: str = ".2f",
) -> None:
    """Draw an annotated heatmap on ax."""
    from matplotlib.colors import TwoSlopeNorm, Normalize

    if vcenter is not None:
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        vmin = min(vmin, vcenter - 1e-6)
        vmax = max(vmax, vcenter + 1e-6)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm = Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))

    cmap_obj = plt.get_cmap(cmap)
    im = ax.imshow(data, aspect="auto", cmap=cmap_obj, norm=norm)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(
        [f"{a:.3g}" for a in col_labels], rotation=45, ha="right", fontsize=_FS_TICK
    )
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"$K={k}$" for k in row_labels], fontsize=_FS_TICK)
    ax.set_xlabel("$\\alpha_{\\mathrm{reg}}$", fontsize=_FS_LABEL)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            if not np.isnan(val):
                rgba = cmap_obj(norm(val))
                lum = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
                ax.text(
                    c, r, f"{val:{fmt}}",
                    ha="center", va="center",
                    fontsize=_FS_CELL, color="w" if lum < 0.45 else "0.15",
                )

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label, fontsize=_FS_LABEL)
    cbar.ax.tick_params(labelsize=_FS_TICK)


def _draw_sensitivity_lines(
    ax: plt.Axes,
    result,
    line_keys: list[str],
    ylabel: str,
    annotate_active_for: set[str] | None = None,
) -> None:
    """
    Plot scalar (benchmark) models as lines on ax.

    annotate_active_for: set of model keys whose active-factor counts are
        annotated above each point. Pass None or empty set to suppress all.
    """
    for name in line_keys:
        if name not in result.r2:
            continue
        ax.plot(
            range(len(result.alphas)),
            result.r2[name],
            marker="o",
            markersize=_MARKER_SZ,
            linewidth=2.0,
            color=_SENS_LINE_COLORS.get(name, "0.4"),
            label=_BENCH_TICK.get(name, name),
        )
        if (
            annotate_active_for
            and name in annotate_active_for
            and result.active_factors.get(name) is not None
        ):
            for x, (y, n_act) in enumerate(
                zip(result.r2[name], result.active_factors[name])
            ):
                ax.annotate(
                    f"{int(round(n_act))}",
                    xy=(x, y),
                    xytext=(0, 8),
                    textcoords="offset points",
                    ha="center",
                    fontsize=_FS_ANNOT,
                    color="0.3",
                )

    ax.set_xticks(range(len(result.alphas)))
    ax.set_xticklabels(
        [f"{a:.3g}" for a in result.alphas], rotation=45, ha="right", fontsize=_FS_TICK
    )
    ax.tick_params(axis="y", labelsize=_FS_TICK)
    ax.set_xlabel("$\\alpha_{\\mathrm{reg}}$", fontsize=_FS_LABEL)
    ax.set_ylabel(ylabel, fontsize=_FS_LABEL)
    ax.axhline(0, color="0.5", linewidth=0.9, linestyle="--")
    ax.legend(fontsize=_FS_LEGEND)
    sns.despine(ax=ax)


def _save_sensitivity(fig: plt.Figure, fname: str) -> None:
    path = os.path.join(_FIGS_SENS, fname)
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def plot_sensitivity_ridge(result) -> None:
    """
    Three separate PNGs:
      sensitivity_ridge_benchmarks.png   — line plot: FF3, FF5, StandardOLS vs α
      sensitivity_ridge_pca.png          — PCA R² heatmap (K × α)
      sensitivity_ridge_dlfm.png           — DLFM  R² heatmap (K × α)
    """
    os.makedirs(_FIGS_SENS, exist_ok=True)
    line_keys = [k for k in _SENS_LINE_COLORS if k in result.r2]

    # Benchmarks line plot
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    _draw_sensitivity_lines(ax, result, line_keys, ylabel="OOS CS $R^2$")
    _save_sensitivity(fig, "sensitivity_ridge_benchmarks.png")

    # Per-family heatmaps
    for prefix, n_values, label, cmap in _SENS_FAMILIES:
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        grid = _build_heatmap_grid(result.r2, prefix, n_values)
        _draw_heatmap(
            ax,
            grid,
            row_labels=n_values,
            col_labels=result.alphas,
            title=f"Ridge — {label} OOS CS $R^2$",
            cbar_label="OOS $R^2$",
            cmap=cmap,
            vcenter=0.0,
        )
        _save_sensitivity(fig, f"sensitivity_ridge_{prefix.lower()}.png")


def plot_sensitivity_lasso(result) -> None:
    """
    Three PNGs:
      sensitivity_lasso_benchmarks.png        — line plot with annotation for standard OLS
      sensitivity_lasso_pca_r2.png            — PCA R² heatmap
      sensitivity_lasso_dlfm_r2.png             — DLFM  R² heatmap
    """
    os.makedirs(_FIGS_SENS, exist_ok=True)
    line_keys = [k for k in _SENS_LINE_COLORS if k in result.r2]

    # Benchmarks line plot with annotation for standard OLS
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    _draw_sensitivity_lines(
        ax,
        result,
        line_keys,
        ylabel="OOS CS $R^2$",
        annotate_active_for={"StandardOLS"},
    )
    _save_sensitivity(fig, "sensitivity_lasso_benchmarks.png")

    # Per-family heatmaps for R²
    for prefix, n_values, label, cmap in _SENS_FAMILIES:
        slug = prefix.lower()

        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        r2_grid = _build_heatmap_grid(result.r2, prefix, n_values)
        _draw_heatmap(
            ax,
            r2_grid,
            row_labels=n_values,
            col_labels=result.alphas,
            title=f"Lasso — {label} OOS CS $R^2$",
            cbar_label="OOS $R^2$",
            cmap=cmap,
            vcenter=0.0,
        )
        _save_sensitivity(fig, f"sensitivity_lasso_{slug}_r2.png")
