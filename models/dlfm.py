from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from config import SEED
from models.base import BaseModel

_ROOT = Path(__file__).parent.parent
with open(_ROOT / "params.yaml") as f:
    _DLFM_PARAMS = yaml.safe_load(f)["dlfm"]


class _Encoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dims: list[int], latent_dim: int, dropout: float
    ):
        super().__init__()
        layers, in_dim = [], input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _DLFMNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoder = _Encoder(input_dim, hidden_dims, latent_dim, dropout)
        self.decoder = nn.Linear(latent_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


class DLFMModel(BaseModel):
    """Deep Latent Factor Model (DLFM); developed by us. Autoencoder-inspired.

    Encodes the L macro factors to K latent pricing factors, then decodes to
    N portfolio returns.
    """

    def __init__(
        self,
        n_factors: int,
        hidden_dims: list[int] = _DLFM_PARAMS["hidden_dims"],
        lr: float = _DLFM_PARAMS["lr"],
        epochs: int = _DLFM_PARAMS["epochs"],
        dropout: float = _DLFM_PARAMS["dropout"],
        weight_decay: float = _DLFM_PARAMS["weight_decay"],
        seed: int = SEED,
    ):
        self.n_factors = n_factors
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.seed = seed
        self._factors_scaler = StandardScaler()
        self.model: _DLFMNet | None = None

    def fit(
        self,
        factors: pd.DataFrame,
        returns: pd.DataFrame,
        seed: int = SEED,
        config: dict | None = None,
    ) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        factors_scaled = self._factors_scaler.fit_transform(factors.values).astype(
            np.float32
        )
        returns_t = torch.from_numpy(returns.values.astype(np.float32))
        _, K = factors_scaled.shape
        N = returns_t.shape[1]
        self.model = _DLFMNet(K, self.hidden_dims, self.n_factors, N, self.dropout)
        opt = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.MSELoss()

        factors_scaled_t = torch.from_numpy(factors_scaled)
        self.model.train()
        for _ in range(self.epochs):
            opt.zero_grad()
            loss_fn(self.model(factors_scaled_t), returns_t).backward()
            opt.step()

        self.model.eval()
        with torch.no_grad():
            self.estimate_alpha_beta(factors, returns, config=config)

    def get_transformed_factors(
        self, factors: pd.DataFrame, returns: pd.DataFrame
    ) -> np.ndarray:
        factors_scaled = torch.from_numpy(
            self._factors_scaler.transform(factors.values).astype(np.float32)
        )
        self.model.eval()
        with torch.no_grad():
            return self.model.encoder(factors_scaled).detach().numpy()

    def risk_prices(self, factors: pd.DataFrame, returns: pd.DataFrame) -> np.ndarray:
        cs_res = sm.OLS(self.avg_returns.T, self.beta).fit()
        return cs_res.params
