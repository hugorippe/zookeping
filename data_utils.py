import pandas as pd


OTHER_TEST_ASSETS_PATH = "data/other_test_assets.xlsx"


def get_expanding_windows(factors_df, portfolios_df, min_train_years=5, min_test_years=5):
    """
    Expanding initial sample: train on calendar years [y_0, …, y_{n-1}], test on all
    remaining years up to t_max. n runs from min_train_years to total_years − min_test_years.

    Date column is YYYYMM int64 (e.g. 197310); year = Date // 100.
    """
    years = sorted((factors_df["Date"] // 100).unique())
    total_years = len(years)
    windows = []
    for n_train in range(min_train_years, total_years - min_test_years + 1):
        train_years = set(years[:n_train])
        mask_f = (factors_df["Date"] // 100).isin(train_years)
        mask_r = (portfolios_df["Date"] // 100).isin(train_years)
        train_f = factors_df[mask_f].drop(columns=["Date"]).reset_index(drop=True)
        test_f = factors_df[~mask_f].drop(columns=["Date"]).reset_index(drop=True)
        train_r = portfolios_df[mask_r].drop(columns=["Date"]).reset_index(drop=True)
        test_r = portfolios_df[~mask_r].drop(columns=["Date"]).reset_index(drop=True)
        windows.append((train_f, test_f, train_r, test_r))
    return windows


def get_fixed_rolling_windows(
    factors_df, portfolios_df, train_years=8, min_test_years=5
):
    """
    Walk-forward rolling: train on `train_years` consecutive calendar years; OOS on **all**
    later years through t_max (disjoint from train, same as expanding-OOS definition).
    Slides forward one year at a time. Only windows with at least `min_test_years` of
    OOS data are included.
    """
    years = sorted((factors_df["Date"] // 100).unique())
    total_years = len(years)
    if total_years < train_years + min_test_years:
        return []
    windows = []
    for start in range(0, total_years - train_years - min_test_years + 1):
        train_year_set = set(years[start : start + train_years])
        test_year_set = set(years[start + train_years :])
        mask_f_tr = (factors_df["Date"] // 100).isin(train_year_set)
        mask_f_te = (factors_df["Date"] // 100).isin(test_year_set)
        mask_r_tr = (portfolios_df["Date"] // 100).isin(train_year_set)
        mask_r_te = (portfolios_df["Date"] // 100).isin(test_year_set)
        train_f = factors_df[mask_f_tr].drop(columns=["Date"]).reset_index(drop=True)
        test_f = factors_df[mask_f_te].drop(columns=["Date"]).reset_index(drop=True)
        train_r = portfolios_df[mask_r_tr].drop(columns=["Date"]).reset_index(drop=True)
        test_r = portfolios_df[mask_r_te].drop(columns=["Date"]).reset_index(drop=True)
        windows.append((train_f, test_f, train_r, test_r))
    return windows


def get_rolling_windows(factors_df, portfolios_df, min_train_years=5, min_test_years=5):
    """
    Backwards-compatible alias for :func:`get_expanding_windows` (expanding initial
    train, test = remainder of sample).
    """
    return get_expanding_windows(
        factors_df, portfolios_df, min_train_years, min_test_years
    )


def load_data():
    """
    Load the data; corresponds to Bryzgalova et al. (2023).
    Note: The data is ranked such that the first 17 factors are nontradable.
    """
    factors_df = pd.read_excel(
        "data/MonthlyFactors.xlsx", sheet_name="51 factors - ranked"
    )
    portfolios_df = pd.read_excel("data/MonthlyPortfolios.xlsx", sheet_name="Sheet1")
    return factors_df, portfolios_df


def load_other_test_assets(
    path: str = OTHER_TEST_ASSETS_PATH, sheet_name: str = "test assets"
) -> pd.DataFrame:
    """
    Load an alternate portfolio panel for rolling evaluation.
    This is what Bryzgalova et al. (2023) use for their out-of-sample portfolios.
    Processes the data to adjust for the risk-free rate.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df = df.rename(columns={"ID": "Date"})
    return_cols = [c for c in df.columns if c not in ("Date", "RF")]
    df[return_cols] = df[return_cols].subtract(df["RF"], axis=0)
    df = df.drop(columns=["RF"])
    return df
