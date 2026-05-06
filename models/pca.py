import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from .base import BaseModel
from config import SEED


class PCAModel(BaseModel):
    def __init__(self, n_components: int):
        self.n_components = n_components
        self._scaler = StandardScaler()
        self._pca = PCA(n_components=n_components)

    def fit(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        seed: int = SEED,
        config: dict | None = None,
    ) -> None:
        np.random.seed(seed)
        self._pca.fit_transform(self._scaler.fit_transform(returns.values))
        self.estimate_alpha_beta(factors, returns, config=config)

    def get_transformed_factors(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        return self._pca.transform(self._scaler.transform(returns.values))

    def risk_prices(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        cs_res = sm.OLS(self.avg_returns.T, self.beta).fit()
        self.lam = cs_res.params
        return self.lam
