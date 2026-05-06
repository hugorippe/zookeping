from pathlib import Path
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

_ROOT = Path(__file__).parent

with open(_ROOT / "params.yaml") as f:
    PARAMS = yaml.safe_load(f)

MIN_TRAIN_YEARS = PARAMS["rolling"]["min_train_years"]
MIN_TEST_YEARS = PARAMS["rolling"]["min_test_years"]
N_RUNS = PARAMS["rolling"]["n_runs"]
PCA_N_VALUES = PARAMS["pca"]["n_values"]
DLFM_N_VALUES = PARAMS["dlfm"]["n_values"]
RPPCA_N_VALUES = PARAMS["rp_pca"]["n_values"]
RPPCA_GAMMA = PARAMS["rp_pca"]["gamma"]


FF3_COLS = ["MKT", "SMB", "HML"]
FF5_COLS = ["MKT", "SMB", "HML", "RMW", "CMA"]


COL_FF3 = "#4C72B0"
COL_FF5 = "#2A3F5F"
COL_STANDARD_OLS = "#C44E52"
COL_PCA = "#8172B3"
COL_RPPCA = "#E1A654"
COL_DLFM = "#4B9368"
COL_BAYESIAN = "#1B7C6E"
COL_COMBINED = "#D65F5F"


FIGSIZE = (9.5, 8.0)


_EXACT_COLORS = {
    "FF3": COL_FF3,
    "FF5": COL_FF5,
    "StandardOLS": COL_STANDARD_OLS,
    "BAYESIAN": COL_BAYESIAN,
    "COMBINED": COL_COMBINED,
}

# RPPCA before PCA to avoid partial matches
_PREFIX_COLORS = [
    ("RPPCA", COL_RPPCA),
    ("PCA", COL_PCA),
    ("DLFM", COL_DLFM),
]


SEED = 5

CONFIGS = PARAMS["configs"]
SENSITIVITY_RIDGE_ALPHAS = PARAMS["sensitivity"]["ridge_alphas"]
SENSITIVITY_LASSO_ALPHAS = PARAMS["sensitivity"]["lasso_alphas"]
SENSITIVITY_MODELS = PARAMS["sensitivity"]["models"]


LINE_PLOT_MODELS = [
    "BAYESIAN",
    "FF3",
    "FF5",
    "StandardOLS",
    "PCA_8",
    "RPPCA_8",
    "DLFM_8",
]

LINE_PLOT_CS: dict[str, tuple[str | None, list]] = {
    "BAYESIAN":    (None,             []),
    "FF3":         ("gls_ridge_10.0", []),
    "FF5":         ("ols_ridge_2.5",  []),
    "StandardOLS": ("ols_ridge_1.0",  []),
    "PCA_8":       ("ols_ridge_2.5",  []),
    "RPPCA_8":     ("ols_ridge_1.0",  []),
    "DLFM_8":      ("gls_ridge_5.0",  []),
}


def config_key(cfg: dict) -> str:
    key = f"{cfg['estimator']}_{cfg.get('regularization', 'none')}"
    if cfg.get("regularization", "none") not in ("none", None):
        key += f"_{cfg.get('alpha', 1.0)}"
    return key


def model_color(key: str) -> str:
    """Map each model name to its colour."""
    if key in _EXACT_COLORS:
        return _EXACT_COLORS[key]
    for prefix, color in _PREFIX_COLORS:
        if key.startswith(prefix):
            return color
    raise ValueError(f"Invalid model key: {key}")


sns.set_theme(style="ticks", font_scale=1.2)


plt.rcParams.update(
    {
        "figure.dpi": 300,
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.labelsize": 12,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "grid.linewidth": 0.4,
        "grid.color": "0.85",
    }
)
