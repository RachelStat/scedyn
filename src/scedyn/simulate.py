"""Data-generating processes of the simulation study.

The cohort is generated once (covariates, random effects, visit times,
longitudinal measurements, censoring times and the train/test split), and
each replication regenerates the event times only. With the default
parameters and seeds, the generated data reproduce the simulation data of
the paper exactly.

Survival models, with g(X, b) = (x_1^2 + x_2^2) / (2 r_x^2) + (b_1^2 + b_2^2) / (2 r_b^2):

* AFT: log T = c g(X, b) + eps, eps ~ N(0, sigma^2)
* PH:  T ~ Exponential(rate = lambda0 exp{-c g(X, b)})
* PO:  S(t | X, b) = [1 + {t exp(-c g(X, b) + k)}^{1 / s_po}]^{-1}

The longitudinal trajectories are generated on the spline time scale
t in [0, visit_max], and the visit times on the time scale of T and of the
landmark are t / time_scale.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .data import LandmarkData

# --------------------------------------------------------------------------
# Defaults (the values used in the paper)
# --------------------------------------------------------------------------

MODEL_DEFAULTS: dict[str, dict] = {
    "AFT": {
        "n_patients": 10000,
        "c": 0.25,
        "r_x": 0.35,
        "r_b": 0.7,
        "sigma": 1.0,
        "censor_max": {"low": 37.0, "high": 9.0},
    },
    "PH": {
        "n_patients": 13000,
        "c": 0.65,
        "r_x": 0.5,
        "r_b": 1.0,
        "lambda0": 1.55,
        "censor_max": {"low": 22.0, "high": 6.0},
    },
    "PO": {
        "n_patients": 11000,
        "c": 0.75,
        "r_x": 0.5,
        "r_b": 1.0,
        "k": 0.6,
        "s_po": 0.8,
        "censor_max": {"low": 54.0, "high": 11.0},
    },
}

COMMON_DEFAULTS: dict = {
    "n_baseline": 50,
    "spline_intercept": 1.0,
    "spline_coef": (3.0, 2.0, -9.0, 3.0, 5.0, -7.0, 6.0, -6.0, 7.0, -8.0),
    "random_effect_var": (1.0, 1.0, 0.8, 0.8, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5, 0.5),
    "z_noise_sd": 0.5,
    "nuisance_strength": 9.0,
    "jump_prob": 0.05,
    "jump_scale": 0.9,
    "n_visits": 19,
    "visit_max": 9.0,
    "visit_jitter_sd": 0.05,
    "time_scale": 5.0,
    "train_ratio": 0.8,
    "train_val_ratio": 0.8,
}

LANDMARKS = (0.6, 0.9, 1.2)
CENSORING_INDEX = {"low": 0, "high": 1}
CENSORING_SEED_OFFSET = {"low": 3, "high": 6}


def default_seeds(seed: int = 2025, censoring: str = "low") -> dict:
    """Seeds used by the paper, derived from a base seed.

    * covariates:   seed            (X and b)
    * visit_times:  seed + 1
    * longitudinal: seed + 2
    * censoring:    seed + 3 (low) or seed + 6 (high)
    * split:        seed + 4        (train pool / test)
    * event_times:  seed * 1000 + r for replication r
    * train_val:    seed * 1000 + r + (0 for low, 1 for high censoring)
    * oracle:       seed + 7        (Monte Carlo draws of the true curves)
    """
    return {
        "covariates": seed,
        "visit_times": seed + 1,
        "longitudinal": seed + 2,
        "censoring": seed + CENSORING_SEED_OFFSET[censoring],
        "split": seed + 4,
        "event_times_base": seed * 1000,
        "train_val_base": seed * 1000 + CENSORING_INDEX[censoring],
        "oracle": seed + 7,
    }


# --------------------------------------------------------------------------
# Model components
# --------------------------------------------------------------------------


def g_function(x1, x2, b1, b2, r_x, r_b):
    return (x1**2 + x2**2) / (2 * r_x**2) + (b1**2 + b2**2) / (2 * r_b**2)


def f_function(model: str, g, p: dict):
    """Model-specific f(X, b) with S(t | X, b) as in the supplement."""
    if model == "AFT":
        return p["c"] * g
    if model == "PH":
        return -p["c"] * g
    if model == "PO":
        return -p["c"] * g
    msg = f"Unknown model {model!r}."
    raise ValueError(msg)


def survival_function(model: str, t, f, p: dict):
    """S(t | X, b) given f(X, b); broadcasts over t and f."""
    if model == "AFT":
        t = torch.clamp(t, min=1e-30)
        return 1.0 - torch.distributions.Normal(0.0, p["sigma"]).cdf(torch.log(t) - f)
    if model == "PH":
        return torch.exp(-p["lambda0"] * t * torch.exp(f))
    if model == "PO":
        inner = torch.clamp(t * torch.exp(f + p["k"]), min=1e-30)
        return 1.0 / (1.0 + inner ** (1.0 / p["s_po"]))
    msg = f"Unknown model {model!r}."
    raise ValueError(msg)


def sample_event_times(model: str, f, p: dict, generator: torch.Generator):
    n = f.shape[0]
    if model == "AFT":
        eps = torch.randn(1, n, generator=generator) * p["sigma"]
        return torch.exp(f.unsqueeze(0) + eps).squeeze(0)
    if model == "PH":
        u = torch.clamp(torch.rand(1, n, generator=generator), min=1e-10, max=1.0)
        return (-torch.log(u) / (p["lambda0"] * torch.exp(f)).unsqueeze(0)).squeeze(0)
    if model == "PO":
        u = torch.clamp(torch.rand(1, n, generator=generator), 1e-10, 1.0 - 1e-10)
        return (
            torch.exp(-(f + p["k"])).unsqueeze(0) * (u / (1.0 - u)) ** p["s_po"]
        ).squeeze(0)
    msg = f"Unknown model {model!r}."
    raise ValueError(msg)


def _spline_basis(t, n_knots: int = 9):
    cols = [torch.ones_like(t), t] + [
        torch.clamp(t - k, min=0.0) for k in range(1, n_knots + 1)
    ]
    return torch.stack(cols, dim=-1)


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------


@dataclass
class LandmarkSplit:
    """Train, validation and test risk sets at one landmark.

    ``eval_grid`` is the evaluation grid of the test IMSE, and
    ``true_survival`` holds the oracle residual survival curves of the test
    subjects on ``eval_grid`` (computed on request).
    """

    tau: float
    train: LandmarkData
    val: LandmarkData
    test: LandmarkData
    eval_grid: np.ndarray
    true_survival: np.ndarray | None = None


class Simulation:
    """Simulation design of the paper for one survival model and censoring level.

    Parameters
    ----------
    model : {"AFT", "PH", "PO"}
    censoring : {"low", "high"}
    seed : int
        Base seed. The individual seeds default to :func:`default_seeds`.
    seeds : dict, optional
        Overrides of individual seeds, with keys as in :func:`default_seeds`.
    **params
        Overrides of the model parameters (see ``MODEL_DEFAULTS`` and
        ``COMMON_DEFAULTS``), e.g. ``n_patients``, ``c``, ``censor_max``.
    """

    def __init__(
        self,
        model: str = "AFT",
        censoring: str = "low",
        seed: int = 2025,
        seeds: dict | None = None,
        **params,
    ):
        model = model.upper()
        if model not in MODEL_DEFAULTS:
            msg = "model must be one of 'AFT', 'PH', 'PO'."
            raise ValueError(msg)
        if censoring not in CENSORING_INDEX:
            msg = "censoring must be 'low' or 'high'."
            raise ValueError(msg)
        p = {**COMMON_DEFAULTS, **MODEL_DEFAULTS[model]}
        unknown = set(params) - set(p)
        if unknown:
            msg = f"Unknown parameters: {sorted(unknown)}"
            raise ValueError(msg)
        p.update(params)
        if isinstance(p["censor_max"], dict):
            p["censor_max"] = float(p["censor_max"][censoring])
        self.model, self.censoring, self.params = model, censoring, p
        self.seeds = {**default_seeds(seed, censoring), **(seeds or {})}
        self._generate_cohort()
        self._test_cache: dict[tuple, tuple[LandmarkData, np.ndarray]] = {}

    # ------------------------------------------------------------------
    def _generate_cohort(self):
        p, s = self.params, self.seeds
        n = int(p["n_patients"])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(s["covariates"])
            X = 2 * torch.rand(n, p["n_baseline"]) - 1
            D = torch.diag(torch.tensor(p["random_effect_var"], dtype=torch.float32))
            b = torch.distributions.MultivariateNormal(
                torch.zeros(D.shape[0]), D
            ).sample((n,))
        g = g_function(X[:, 0], X[:, 1], b[:, 0], b[:, 1], p["r_x"], p["r_b"])
        f = f_function(self.model, g, p)

        grid = torch.linspace(0, p["visit_max"], p["n_visits"]).unsqueeze(0)
        gen = torch.Generator().manual_seed(s["visit_times"])
        t_raw = torch.clamp(
            grid + torch.randn(n, p["n_visits"], generator=gen) * p["visit_jitter_sd"],
            min=0.0,
        )
        meas_time = t_raw / p["time_scale"]

        gen = torch.Generator().manual_seed(s["longitudinal"])
        coef = torch.tensor(
            [p["spline_intercept"], *p["spline_coef"]], dtype=torch.float32
        )
        C = coef.unsqueeze(0) + b
        B1 = C[:, 0:1] + C[:, 1:2] * t_raw
        for j in range(2, C.shape[1]):
            B1 = B1 + C[:, j : j + 1] * (t_raw - (j - 1.0)).clamp(min=0)
        Z1 = B1 + torch.randn(n, p["n_visits"], generator=gen) * p["z_noise_sd"]
        t_center = (t_raw - 3.0) / 6.0
        u1 = torch.randn(n, generator=gen)
        B2 = (
            0.5
            + p["nuisance_strength"] * u1.unsqueeze(1) * torch.abs(t_center) ** 1.5
            + 0.25 * torch.sin(t_raw / 1.7)
        )
        Z2 = B2 + torch.randn(n, p["n_visits"], generator=gen) * p["z_noise_sd"]
        u2 = torch.randn(n, generator=gen)
        B3 = (
            0.5
            - p["nuisance_strength"] * u2.unsqueeze(1) * torch.abs(t_center) ** 1.5
            + 0.25 * torch.cos(t_raw / 1.7)
        )
        Z3 = B3 + torch.randn(n, p["n_visits"], generator=gen) * p["z_noise_sd"]
        s4 = torch.randn(n, generator=gen)
        B4 = 0.5 + s4.unsqueeze(1) * torch.exp(2.0 * (t_raw - 2.0) / 6.0)
        jump = (torch.rand(n, p["n_visits"], generator=gen) < p["jump_prob"]).float()
        jump_noise = (
            jump
            * torch.randn(n, p["n_visits"], generator=gen)
            * p["nuisance_strength"]
            * p["jump_scale"]
        )
        Z4 = (
            B4
            + jump_noise
            + torch.randn(n, p["n_visits"], generator=gen) * p["z_noise_sd"]
        )

        gen = torch.Generator().manual_seed(s["censoring"])
        censor = p["censor_max"] * torch.rand(n, generator=gen)

        ids = torch.arange(1, n + 1)
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(s["split"]))
        n_train = int(p["train_ratio"] * n)
        self.train_pool_ids = ids[perm[:n_train]].sort().values.numpy()
        self.test_ids = ids[perm[n_train:]].sort().values.numpy()

        self.X = X.numpy()
        self.b = b.numpy()
        self.f = f
        self.t_raw = t_raw.numpy()
        self.meas_time = meas_time.numpy()
        self.Z = torch.stack([Z1, Z2, Z3, Z4], dim=-1).numpy()
        self.censor_time = censor.numpy()

    # ------------------------------------------------------------------
    def event_times(self, replication: int) -> np.ndarray:
        gen = torch.Generator().manual_seed(
            self.seeds["event_times_base"] + replication
        )
        return sample_event_times(self.model, self.f, self.params, gen).numpy()

    def _risk_set(self, ids, T, tau, covariates):
        idx = ids - 1
        obs = np.minimum(T[idx], self.censor_time[idx])
        delta = (T[idx] <= self.censor_time[idx]).astype(np.float32)
        keep = obs > tau
        visible = self.meas_time[idx] <= tau
        keep &= visible.any(axis=1)
        idx, obs, delta, visible = idx[keep], obs[keep], delta[keep], visible[keep]
        if covariates == "all":
            bcols, lcols = list(range(self.X.shape[1])), [0, 1, 2, 3]
        elif covariates == "informative":
            bcols, lcols = [0, 1], [0]
        else:
            msg = "covariates must be 'all' or 'informative'."
            raise ValueError(msg)
        Z = [self.Z[i][v][:, lcols] for i, v in zip(idx, visible, strict=True)]
        return LandmarkData(
            X=self.X[idx][:, bcols],
            Z=Z,
            time=obs - tau,
            event=delta,
            ids=idx + 1,
            baseline_names=[f"X_{j + 1}" for j in bcols],
            longitudinal_names=[f"Z_{j + 1}" for j in lcols],
            extra={
                "t_raw": [self.t_raw[i][v] for i, v in zip(idx, visible, strict=True)],
                "z_signal": [
                    self.Z[i][v][:, 0] for i, v in zip(idx, visible, strict=True)
                ],
                "x_signal": [self.X[i, :2] for i in idx],
            },
        )

    @staticmethod
    def _quantile_grid(time, lo, hi, k):
        t = torch.tensor(time, dtype=torch.float32)
        t = t[t > 0]
        return torch.quantile(t, torch.linspace(lo, hi, k)).numpy()

    def landmark_splits(
        self,
        replication: int,
        landmarks: Sequence[float] = LANDMARKS,
        covariates: str = "all",
        true_survival: bool = True,
        n_mc: int = 2000,
        grid_size: int = 100,
        grid_quantiles=(0.01, 0.95),
    ) -> dict[float, LandmarkSplit]:
        """Train, validation and test sets of one replication at each landmark.

        The train/validation split of the risk set uses one random stream per
        replication across the landmarks in the given order, as in the paper.
        The test set and its evaluation grid are fixed across replications
        (built from replication 1).
        """
        T = self.event_times(replication)
        rng = np.random.RandomState(self.seeds["train_val_base"] + replication)
        out = {}
        for tau in landmarks:
            pool = self._risk_set(self.train_pool_ids, T, tau, covariates)
            order = list(pool.ids)
            rng.shuffle(order)
            n_train = int(self.params["train_val_ratio"] * len(order))
            pos = {pid: i for i, pid in enumerate(pool.ids)}
            train = pool.subset(sorted(pos[i] for i in order[:n_train]))
            val = pool.subset(sorted(pos[i] for i in order[n_train:]))
            test, grid = self._test_set(tau, covariates, grid_size, grid_quantiles)
            truth = None
            if true_survival:
                truth = self.true_residual_survival(test, tau, grid, n_mc=n_mc)
            out[tau] = LandmarkSplit(
                tau=tau,
                train=train,
                val=val,
                test=test,
                eval_grid=grid,
                true_survival=truth,
            )
        return out

    def _test_set(self, tau, covariates, grid_size, grid_quantiles):
        key = (tau, covariates, grid_size, tuple(grid_quantiles))
        if key not in self._test_cache:
            T1 = self.event_times(1)
            test = self._risk_set(self.test_ids, T1, tau, covariates)
            pool1 = self._risk_set(self.train_pool_ids, T1, tau, covariates)
            lo, hi = grid_quantiles
            grid = self._quantile_grid(pool1.time, lo, hi, grid_size)
            self._test_cache[key] = (test, grid)
        return self._test_cache[key]

    # ------------------------------------------------------------------
    def true_residual_survival(
        self,
        data: LandmarkData,
        tau: float,
        grid,
        n_mc: int = 2000,
        seed: int | None = None,
    ) -> np.ndarray:
        """Oracle residual survival S*_tau(s | H) on ``grid`` for each subject.

        The posterior of the random effects given the signal trajectory is
        normal (normal--normal conjugacy), and the ratio
        E[S(tau + s | X, b)] / E[S(tau | X, b)] is approximated with
        ``n_mc`` posterior draws.
        """
        p = self.params
        gen = torch.Generator().manual_seed(
            self.seeds["oracle"] if seed is None else seed
        )
        coef = torch.tensor(
            [p["spline_intercept"], *p["spline_coef"]], dtype=torch.float64
        )
        D_inv = torch.diag(
            1.0 / torch.tensor(p["random_effect_var"], dtype=torch.float64)
        )
        s2 = p["z_noise_sd"] ** 2
        grid_t = torch.as_tensor(np.asarray(grid), dtype=torch.float64)
        t_all = torch.cat([torch.tensor([tau], dtype=torch.float64), tau + grid_t])
        out = np.empty((data.n, len(grid_t)))
        for i in range(data.n):
            t = torch.as_tensor(data.extra["t_raw"][i], dtype=torch.float64)
            z = torch.as_tensor(data.extra["z_signal"][i], dtype=torch.float64)
            H = _spline_basis(t)
            prec = D_inv + H.T @ H / s2
            cov = torch.linalg.solve(
                prec, torch.eye(prec.shape[0], dtype=torch.float64)
            )
            mean = cov @ H.T @ (z - H @ coef) / s2
            L = torch.linalg.cholesky(
                cov + 1e-6 * torch.eye(cov.shape[0], dtype=torch.float64)
            )
            eps = torch.randn(n_mc, cov.shape[0], generator=gen, dtype=torch.float64)
            b = mean + eps @ L.T
            x = torch.as_tensor(data.extra["x_signal"][i], dtype=torch.float64)
            g = g_function(x[0], x[1], b[:, 0], b[:, 1], p["r_x"], p["r_b"])
            f = f_function(self.model, g, p)
            S = survival_function(
                self.model, t_all.unsqueeze(1), f.unsqueeze(0), p
            ).mean(dim=1)
            out[i] = (S[1:] / torch.clamp(S[0], min=1e-16)).numpy()
        return out
