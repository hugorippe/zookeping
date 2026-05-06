from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from config import SEED

from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge, Lasso, MultiTaskLasso


class BaseModel(ABC):

    @abstractmethod
    def __init__(self) -> None: ...

    @abstractmethod
    def fit(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        seed: int = SEED,
        config: dict | None = None,
    ) -> None:
        """
        Fit the model on training data.

        config: regularization config dict with keys estimator, regularization, alpha.
            Controls first-pass beta estimation (ridge/lasso) and second-pass λ estimation.
        """
        np.random.seed(seed)

    @abstractmethod
    def risk_prices(
        self, factors: pd.DataFrame, returns: pd.DataFrame
    ) -> np.ndarray: ...

    def get_transformed_factors(
        self, factors: pd.DataFrame, returns: pd.DataFrame
    ) -> np.ndarray:
        """
        Returns transformed factors of shape (T, K). Default: identity (use factors directly).
        PCA and RP-PCA override this to use the returns matrix instead.
        """
        return factors.values

    def estimate_alpha_beta(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        config: dict | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimates alpha and beta via vectorized OLS (or ridge/lasso if config specifies).

        config: dict with keys regularization ('none'|'ridge'|'lasso') and alpha (strength).
            GLS estimator is not applied to the first-pass TS regression.
        """
        F = self.get_transformed_factors(factors, returns)
        Y = np.asarray(returns)

        regularization = config.get("regularization", "none") if config else "none"
        alpha_reg = float(config.get("alpha", 1.0)) if config else 1.0

        if regularization == "ridge":
            reg = Ridge(alpha=alpha_reg, fit_intercept=True).fit(F, Y)
            self.alpha = reg.intercept_  # (N,)
            self.beta = reg.coef_  # (N, K)
        elif regularization == "lasso":
            reg = MultiTaskLasso(
                alpha=alpha_reg, fit_intercept=True, max_iter=10000
            ).fit(F, Y)
            self.alpha = reg.intercept_  # (N,)
            self.beta = reg.coef_  # (N, K)
        else:
            X = np.hstack([np.ones((F.shape[0], 1)), F])
            params, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
            self.alpha = params[0, :]  # (N,)
            self.beta = params[1:, :].T  # (N, K)

        self.avg_returns = Y.mean(axis=0)  # (N,)
        self.resid_ts = Y - (F @ self.beta.T + self.alpha)  # (T, N) — for GLS

        return self.alpha, self.beta

    def _fit_cs_regression(
        self,
        beta: np.ndarray,
        avg_returns: np.ndarray,
        estimator: str = "ols",
        regularization: str = "none",
        alpha: float = 1.0,
    ) -> np.ndarray:
        """
        Second-pass CS regression: OLS or GLS and regularization (none/ridge/lasso).
        GLS uses Ledoit-Wolf Σ estimated from self.resid_ts.
        """
        avg_returns = np.asarray(avg_returns)

        if estimator == "gls":
            Sigma = LedoitWolf().fit(self.resid_ts).covariance_
            L = np.linalg.cholesky(Sigma)
            X_eff = np.linalg.solve(L, beta)
            y_eff = np.linalg.solve(L, avg_returns)
        else:
            X_eff, y_eff = beta, avg_returns

        if regularization == "ridge":
            return Ridge(alpha=alpha, fit_intercept=False).fit(X_eff, y_eff).coef_
        elif regularization == "lasso":
            return (
                Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
                .fit(X_eff, y_eff)
                .coef_
            )
        else:
            lam, _, _, _ = np.linalg.lstsq(X_eff, y_eff, rcond=None)
            return lam

    def _lam_from_config(self, config: dict | None) -> np.ndarray:
        """Fit and return λ from the second-pass CS regression using config."""
        return self._fit_cs_regression(
            self.beta,
            self.avg_returns,
            estimator=config.get("estimator", "ols") if config else "ols",
            regularization=config.get("regularization", "none") if config else "none",
            alpha=float(config.get("alpha", 1.0)) if config else 1.0,
        )

    def cross_sectional_r2(
        self,
        test_returns: pd.DataFrame,
        config: dict | None = None,
    ) -> float:
        """
        Cross-sectional OOS R²: how well β λ̂ predicts actual OOS mean returns.
        Requires fit() to have been called first.
        """
        lam = self._lam_from_config(config)
        mu_hat = self.beta @ lam
        mu_test = test_returns.values.mean(axis=0)
        ss_res = np.sum((mu_test - mu_hat) ** 2)
        ss_tot = np.sum((mu_test - mu_test.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    def cross_sectional_r2_insample(self, config: dict | None = None) -> float:
        """Cross-sectional in-sample R²: β λ̂ vs training mean returns."""
        lam = self._lam_from_config(config)
        mu_hat = self.beta @ lam
        ss_res = np.sum((self.avg_returns - mu_hat) ** 2)
        ss_tot = np.sum((self.avg_returns - self.avg_returns.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    def cross_sectional_mape(
        self, test_returns: pd.DataFrame, config: dict | None = None
    ) -> float:
        """Mean absolute pricing error (OOS): mean |μ_i -μ̂_i| across assets."""
        lam = self._lam_from_config(config)
        mu_hat = self.beta @ lam
        mu_test = test_returns.values.mean(axis=0)
        return float(np.mean(np.abs(mu_test - mu_hat)))
