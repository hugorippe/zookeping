from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from models.base import BaseModel
from config import SEED

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "params.yaml") as f:
    _BAYES_PARAMS = yaml.safe_load(f)["bayesian"]


class BayesianModel(BaseModel):

    def __init__(self, num_factors: int = 10) -> None:
        self.num_factors = num_factors

    def fit(
        self,
        training_factors: pd.DataFrame,
        training_returns: pd.DataFrame,
        num_iters: int = _BAYES_PARAMS["num_iters"],
        seed: int = SEED,
        config: dict | None = None,
    ) -> None:
        """
        Spike-and-slab Gibbs sampler for cross-sectional SDF risk prices.

        Pricing equation: Ȳ = X λ + ε, where Ȳ is (N,) mean returns and
        X = β is the (N, K) matrix of factor betas.

        Gibbs steps:
          1. Draw λ | γ, Y from its multivariate-normal posterior.
          2. Draw γ_k | λ_k from the Bernoulli (spike vs. slab).
        Post burn-in draws are averaged → self.lam, self.gamma.
        """
        np.random.seed(seed)
        T, N = training_returns.shape
        _, K = training_factors.shape

        sigma_sq = float(_BAYES_PARAMS["sigma_sq"])
        v_slab = float(_BAYES_PARAMS["v_slab"])
        v_spike = float(_BAYES_PARAMS["v_spike"])
        prior_prob = float(_BAYES_PARAMS["prior_prob"])

        Y = training_returns.values.mean(axis=0, keepdims=True).T  # (N, 1)
        _, X = self.estimate_alpha_beta(training_factors, training_returns, config=config)

        XX = (X.T @ X) / sigma_sq   # (K, K)
        XY = (X.T @ Y) / sigma_sq   # (K, 1)

        gamma = (np.random.rand(K) < prior_prob).astype(float)

        burn_in = num_iters // 2
        lam_samples, gamma_samples = [], []

        for i in range(num_iters):
            v_prior = gamma * v_slab + (1 - gamma) * v_spike
            post_prec = XX + np.diag(1.0 / v_prior)
            post_cov = np.linalg.inv(post_prec)
            post_mean = post_cov @ XY

            lam = (post_mean + np.linalg.cholesky(post_cov) @ np.random.randn(K, 1)).flatten()

            log_p_slab  = -0.5 * (lam**2 / v_slab)  - 0.5 * np.log(v_slab)  + np.log(prior_prob)
            log_p_spike = -0.5 * (lam**2 / v_spike) - 0.5 * np.log(v_spike) + np.log(1.0 - prior_prob)
            gamma = (np.random.rand(K) < 1.0 / (1.0 + np.exp(log_p_spike - log_p_slab))).astype(float)

            if i >= burn_in:
                lam_samples.append(lam)
                gamma_samples.append(gamma)

        self.lam = np.array(lam_samples).mean(axis=0)
        self.gamma = np.array(gamma_samples).mean(axis=0)

    def risk_prices(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        return self.lam

    def cross_sectional_r2(self, test_returns: pd.DataFrame, config: dict | None = None) -> float:
        """Uses posterior mean λ (config ignored — Bayesian prior already regularises λ)."""
        mu_hat = self.beta @ self.lam
        mu_test = test_returns.values.mean(axis=0)
        ss_res = np.sum((mu_test - mu_hat) ** 2)
        ss_tot = np.sum((mu_test - mu_test.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    def cross_sectional_r2_insample(self, config: dict | None = None) -> float:
        mu_hat = self.beta @ self.lam
        ss_res = np.sum((self.avg_returns - mu_hat) ** 2)
        ss_tot = np.sum((self.avg_returns - self.avg_returns.mean()) ** 2)
        return float(1 - ss_res / ss_tot)

    def cross_sectional_mape(self, test_returns: pd.DataFrame, config: dict | None = None) -> float:
        mu_hat = self.beta @ self.lam
        mu_test = test_returns.values.mean(axis=0)
        return float(np.mean(np.abs(mu_test - mu_hat)))
