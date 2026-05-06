import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import BaseModel
from config import SEED

"""
Matlab code from Markus Pelger available at
https://www.dropbox.com/scl/fi/3jesv2bmr505bxe2xu4pm/Code.zip?dl=0&e=1&file_subpath=%2FCode%2FRPPCA.m&rlkey=ohljzyjhjewuh9dn3xi2kekyp
"""


class RPPCA(BaseModel):
    def __init__(self, n_components: int, gamma: float = 2.0):
        self.n_components = n_components
        self.gamma = gamma

    def fit(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        seed: int = SEED,
        config: dict | None = None,
    ) -> None:
        np.random.seed(seed)
        R = returns.values.astype(np.float64)
        T, N = R.shape

        mu = R.mean(axis=0)
        M = (R.T @ R) / T
        M_rp = M + self.gamma * np.outer(mu, mu)

        eigvals, eigvecs = np.linalg.eigh(M_rp)
        idx = np.argsort(eigvals)[::-1]
        self.transform_vecs = eigvecs[:, idx[: self.n_components]]

        signs = np.sign(np.mean(R @ self.transform_vecs, axis=0))
        self.transform_vecs = self.transform_vecs * signs

        self.estimate_alpha_beta(factors, returns, config=config)

    def get_transformed_factors(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        return returns.values @ self.transform_vecs

    def risk_prices(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        cs_res = sm.OLS(self.avg_returns.T, self.beta).fit()
        self.lam = cs_res.params
        return self.lam
