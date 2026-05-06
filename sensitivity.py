"""
Regularization sensitivity analysis.

Sweeps a dense grid of alpha values for Ridge and Lasso (OLS estimator, first-pass
beta regularization) across the designated sensitivity models. A single fixed seed is
used per window — this is a diagnostic, not a final result, so seed averaging is
unnecessary.

For PCA and DLFM families, all K variants are swept, producing a (K × alpha) result
grid per family — visualised as a heatmap. For benchmark models (FF3, FF5,
51 factors), results are a 1-D array over alpha values.

Active factor count (Lasso only): number of columns of beta with at least one
non-zero loading across assets, i.e. the group-sparsity metric from MultiTaskLasso.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from config import (
    SEED,
    SENSITIVITY_MODELS,
    SENSITIVITY_RIDGE_ALPHAS,
    SENSITIVITY_LASSO_ALPHAS,
)


@dataclass
class SensitivityResult:
    """Aggregated sensitivity results for one regularization type."""

    regularization: str  # "ridge" or "lasso"
    alphas: list[float]
    # r2[model_key]            = array of shape (n_alphas,)  — mean OOS R² across windows
    r2: dict[str, np.ndarray] = field(default_factory=dict)
    # Per-window arrays for CSV export: shape (n_alphas, n_windows)
    r2_windows: dict[str, np.ndarray] = field(default_factory=dict)
    ins_r2_windows: dict[str, np.ndarray] = field(default_factory=dict)
    mape_windows: dict[str, np.ndarray] = field(default_factory=dict)
    # active_factors[model_key] = array of shape (n_alphas,) — mean active factor count
    #                             None for Ridge (sparsity not applicable)
    active_factors: dict[str, np.ndarray | None] = field(default_factory=dict)


def _active_factor_count(beta: np.ndarray) -> int:
    """Columns of beta (factors) with at least one non-zero asset loading."""
    return int((np.abs(beta).sum(axis=0) > 0).sum())


def run_sensitivity_sweep(
    models: dict,
    windows: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]],
    sensitivity_model_names: list[str] | None = None,
) -> tuple[SensitivityResult, SensitivityResult]:
    """
    Sweep Ridge and Lasso alpha grids for the sensitivity models.

    models: full model dict from main (keyed by display name).
    windows: rolling windows from get_rolling_windows().
    sensitivity_model_names: base names to include. Family prefixes ("PCA", "DLFM")
        are expanded to all matching keys in models (e.g. PCA_1 … PCA_8).

    Returns (ridge_result, lasso_result).
    """
    if sensitivity_model_names is None:
        sensitivity_model_names = SENSITIVITY_MODELS

    # Expand family prefixes → full model keys
    sweep_keys: list[str] = []
    for name in sensitivity_model_names:
        if name in models:
            sweep_keys.append(name)
        else:
            sweep_keys.extend(k for k in models if k.startswith(f"{name}_"))

    results = []
    for reg, alphas in [
        ("ridge", SENSITIVITY_RIDGE_ALPHAS),
        ("lasso", SENSITIVITY_LASSO_ALPHAS),
    ]:
        result = SensitivityResult(regularization=reg, alphas=alphas)

        r2_buffer: dict[str, list[list[float]]] = {
            k: [[] for _ in alphas] for k in sweep_keys
        }
        ins_buffer: dict[str, list[list[float]]] = {
            k: [[] for _ in alphas] for k in sweep_keys
        }
        mape_buffer: dict[str, list[list[float]]] = {
            k: [[] for _ in alphas] for k in sweep_keys
        }
        active_factors_buffer: dict[str, list[list[int]]] = {
            k: [[] for _ in alphas] for k in sweep_keys
        }

        for w_idx, (train_f, _, train_r, test_r) in enumerate(windows):
            for name in sweep_keys:
                model = models[name]
                for a_idx, alpha_val in enumerate(alphas):
                    cfg = {
                        "estimator": "ols",
                        "regularization": reg,
                        "alpha": alpha_val,
                    }
                    model.fit(train_f, train_r, seed=SEED, config=cfg)
                    r2_buffer[name][a_idx].append(
                        model.cross_sectional_r2(test_r, config=cfg)
                    )
                    ins_buffer[name][a_idx].append(
                        model.cross_sectional_r2_insample(config=cfg)
                    )
                    mape_buffer[name][a_idx].append(
                        model.cross_sectional_mape(test_r, config=cfg)
                    )
                    active_factors_buffer[name][a_idx].append(
                        _active_factor_count(model.beta)
                    )
                    print(
                        f"[sensitivity/{reg}] W{w_idx+1}/{len(windows)} {name} α={alpha_val}"
                    )

        for name in sweep_keys:
            result.r2[name] = np.array(
                [np.mean(r2_buffer[name][i]) for i in range(len(alphas))]
            )
            result.r2_windows[name] = np.array(
                [
                    np.asarray(r2_buffer[name][i], dtype=float)
                    for i in range(len(alphas))
                ]
            )
            result.ins_r2_windows[name] = np.array(
                [
                    np.asarray(ins_buffer[name][i], dtype=float)
                    for i in range(len(alphas))
                ]
            )
            result.mape_windows[name] = np.array(
                [
                    np.asarray(mape_buffer[name][i], dtype=float)
                    for i in range(len(alphas))
                ]
            )
            result.active_factors[name] = (
                np.array(
                    [
                        np.mean(active_factors_buffer[name][i])
                        for i in range(len(alphas))
                    ]
                )
                if reg == "lasso"
                else None
            )

        results.append(result)

    return results[0], results[1]
