import numpy as np
import pandas as pd
import statsmodels.api as sm
from .base import BaseModel
from config import FF3_COLS, FF5_COLS, SEED


class _OLSModel(BaseModel):
    def __init__(self, factor_cols: list[str] | None = None):
        self.factor_cols = factor_cols

    def get_transformed_factors(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        if self.factor_cols is None:
            return factors.values
        return factors[self.factor_cols].values

    def fit(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        seed: int = SEED,
        config: dict | None = None,
    ) -> None:
        self.estimate_alpha_beta(factors, returns, config=config)
        cs_res = sm.OLS(self.avg_returns.T, self.beta).fit()
        self.lam = cs_res.params

    def risk_prices(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        return self.lam


class FF3Model(_OLSModel):
    def __init__(self):
        super().__init__(FF3_COLS)


class FF5Model(_OLSModel):
    def __init__(self):
        super().__init__(FF5_COLS)


class StandardOLSModel(_OLSModel):
    def __init__(self):
        super().__init__(None)
