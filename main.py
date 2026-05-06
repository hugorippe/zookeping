import numpy as np
from config import (
    DLFM_N_VALUES,
    LINE_PLOT_CS,
    LINE_PLOT_MODELS,
    PCA_N_VALUES,
    RPPCA_GAMMA,
    RPPCA_N_VALUES,
    MIN_TRAIN_YEARS,
    MIN_TEST_YEARS,
    N_RUNS,
    CONFIGS,
    config_key,
)
from data_utils import load_data, get_rolling_windows
from models.dlfm import DLFMModel
from models.bayesian import BayesianModel
from models.fama_french import FF3Model, FF5Model, StandardOLSModel
from models.pca import PCAModel
from models.rp_pca import RPPCA
from plots import (
    plot_cs_r2_over_time,
    plot_r2_bar,
    plot_r2_vs_factors,
    plot_sensitivity_ridge,
    plot_sensitivity_lasso,
)
from sensitivity import run_sensitivity_sweep
from tables import (
    _TABLES_CS,
    _TABLES_MAIN,
    compute_cis,
    save_config_table,
    save_cross_sectional_csv,
    save_main_table,
    save_sensitivity_csvs,
    save_sensitivity_table,
)


def main():
    factors_df, portfolios_df = load_data()

    models = {
        "FF3": FF3Model(),
        "FF5": FF5Model(),
        "StandardOLS": StandardOLSModel(),
        "BAYESIAN": BayesianModel(),
        **{f"PCA_{n}": PCAModel(n_components=n) for n in PCA_N_VALUES},
        **{
            f"RPPCA_{n}": RPPCA(n_components=n, gamma=RPPCA_GAMMA)
            for n in RPPCA_N_VALUES
        },
        **{f"DLFM_{k}": DLFMModel(n_factors=k) for k in DLFM_N_VALUES},
    }

    windows = get_rolling_windows(
        factors_df, portfolios_df, MIN_TRAIN_YEARS, MIN_TEST_YEARS
    )
    print(
        f"Rolling windows: {len(windows)} (min_train={MIN_TRAIN_YEARS}y, min_test={MIN_TEST_YEARS}y)"
    )

    keys = [config_key(cfg) for cfg in CONFIGS]

    # [model][config_key] = list of per-window mean R² / MAPE values
    cs_oos: dict[str, dict[str, list[float]]] = {
        n: {k: [] for k in keys} for n in models
    }
    cs_ins: dict[str, dict[str, list[float]]] = {
        n: {k: [] for k in keys} for n in models
    }
    cs_mape: dict[str, dict[str, list[float]]] = {
        n: {k: [] for k in keys} for n in models
    }

    for w_idx, (train_f, _, train_r, test_r) in enumerate(windows):
        for name, model in models.items():
            oos_runs: dict[str, list[float]] = {k: [] for k in keys}
            ins_runs: dict[str, list[float]] = {k: [] for k in keys}
            mape_runs: dict[str, list[float]] = {k: [] for k in keys}

            if isinstance(model, BayesianModel):
                # bayesian is agnostic w.r.t config
                for i in range(N_RUNS):
                    print(f"[W{w_idx+1}/{len(windows)}] {name} seed {i+1}/{N_RUNS}")
                    model.fit(train_f, train_r, seed=i)
                    oos_val = model.cross_sectional_r2(test_r)
                    ins_val = model.cross_sectional_r2_insample()
                    mape_val = model.cross_sectional_mape(test_r)
                    for k in keys:
                        oos_runs[k].append(oos_val)
                        ins_runs[k].append(ins_val)
                        mape_runs[k].append(mape_val)
            else:
                for cfg in CONFIGS:
                    k = config_key(cfg)
                    for i in range(N_RUNS):
                        print(
                            f"[W{w_idx+1}/{len(windows)}] {name} cfg={k} seed {i+1}/{N_RUNS}"
                        )
                        model.fit(train_f, train_r, seed=i, config=cfg)
                        oos_runs[k].append(model.cross_sectional_r2(test_r, config=cfg))
                        ins_runs[k].append(
                            model.cross_sectional_r2_insample(config=cfg)
                        )
                        mape_runs[k].append(
                            model.cross_sectional_mape(test_r, config=cfg)
                        )

            for k in keys:
                cs_oos[name][k].append(np.mean(oos_runs[k]))
                cs_ins[name][k].append(np.mean(ins_runs[k]))
                cs_mape[name][k].append(np.mean(mape_runs[k]))

    # Convert to numpy arrays
    oos_arrays = {n: {k: np.array(cs_oos[n][k]) for k in keys} for n in models}
    ins_arrays = {n: {k: np.array(cs_ins[n][k]) for k in keys} for n in models}
    mape_arrays = {n: {k: np.array(cs_mape[n][k]) for k in keys} for n in models}

    # Print summary
    print(f"\n  {'Model':<16s}  " + "  ".join(f"{k:>22s}" for k in keys))
    for name in models:
        row = "  ".join(f"{oos_arrays[name][k].mean():+22.4f}" for k in keys)
        print(f"  {name:<16s}  {row}")

    # Per-config tables and plots
    for cfg in CONFIGS:
        k = config_key(cfg)
        oos_k = {n: oos_arrays[n][k] for n in models}
        ins_k = {n: ins_arrays[n][k] for n in models}
        mape_k = {n: mape_arrays[n][k] for n in models}

        cis = compute_cis(oos_k)
        oos_means = {n: arr.mean() for n, arr in oos_k.items()}

        plot_r2_bar(oos_means, tag=k, ci=cis)
        plot_r2_vs_factors(oos_means, tag=k, ci=cis)
        save_config_table(oos_k, ins_k, mape_k, path=f"{_TABLES_CS}/cs_r2_{k}.tex")

    save_main_table(
        oos_arrays, ins_arrays, mape_arrays, path=f"{_TABLES_MAIN}/main_results.tex"
    )
    save_cross_sectional_csv(oos_arrays, ins_arrays, mape_arrays, keys)

    all_years = sorted((factors_df["Date"] // 100).unique())
    window_test_start_years = [all_years[MIN_TRAIN_YEARS + i] for i in range(len(windows))]
    plot_cs_r2_over_time(oos_arrays, LINE_PLOT_MODELS, LINE_PLOT_CS, window_test_start_years)

    ridge_result, lasso_result = run_sensitivity_sweep(models, windows)

    sens_display = {
        "FF3": "FF3",
        "FF5": "FF5",
        "StandardOLS": "51 factors",
        **{f"PCA_{n}": f"PCA $K={n}$" for n in PCA_N_VALUES},
        **{f"DLFM_{k}": f"DLFM $K={k}$" for k in DLFM_N_VALUES},
        **{f"RPPCA_{n}": f"RPPCA $K={n}$" for n in RPPCA_N_VALUES},
    }
    plot_sensitivity_ridge(ridge_result)
    plot_sensitivity_lasso(lasso_result)
    save_sensitivity_table(ridge_result, lasso_result, sens_display)
    save_sensitivity_csvs(ridge_result, lasso_result)

    print("\nDone.")


if __name__ == "__main__":
    main()
