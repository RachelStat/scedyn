"""The SCEDyn estimator: self-consistent min--max training at one landmark."""

from __future__ import annotations

import math
import warnings
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .data import LandmarkData
from .metrics import brier_scores, km_survival
from .networks import Generator, WeightNetwork


def _check_penalty_arguments(variable_selection, lambda_grid, lambda_value) -> None:
    """Check the penalty arguments against ``variable_selection``."""
    if not variable_selection:
        unused = [
            name
            for name, value in (
                ("lambda_grid", lambda_grid),
                ("lambda_value", lambda_value),
            )
            if value is not None
        ]
        if unused:
            msg = (
                f"{' and '.join(unused)} {'is' if len(unused) == 1 else 'are'} not used "
                "because variable_selection=False."
            )
            warnings.warn(msg, UserWarning, stacklevel=3)
        return
    if lambda_grid is None and lambda_value is None:
        msg = (
            "variable_selection=True requires lambda_grid (candidate penalty levels "
            "for the grid search) or lambda_value (a fixed penalty level)."
        )
        raise ValueError(msg)
    if lambda_grid is not None and len(lambda_grid) == 0 and lambda_value is None:
        msg = "lambda_grid must contain at least one value."
        raise ValueError(msg)
    if lambda_grid is not None and lambda_value is not None:
        msg = (
            "lambda_grid is ignored because lambda_value is given; variable "
            "selection uses lambda_value."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)


def _pick_device(device):
    if device is not None and device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if (
        getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _quantile_grid(time: np.ndarray, lo: float, hi: float, k: int) -> torch.Tensor:
    t = torch.as_tensor(np.asarray(time), dtype=torch.float32)
    t = t[t > 0]
    if t.numel() == 0:
        msg = "No positive residual times to build a time grid from."
        raise ValueError(msg)
    return torch.quantile(t, torch.linspace(lo, hi, k))


@torch.no_grad()
def _step_lookup(
    grid: torch.Tensor, values: torch.Tensor, query: torch.Tensor
) -> torch.Tensor:
    """Value of the step function (grid, values) at ``query``; below the grid the first value is used."""
    idx = torch.bucketize(query.reshape(-1), grid, right=True) - 1
    idx = idx.clamp(min=0, max=grid.numel() - 1)
    return values[idx].view(-1, 1)


@dataclass
class _Tensors:
    """Covariates of a data set on the training device."""

    X: torch.Tensor
    Z: torch.Tensor
    lengths: torch.Tensor

    @property
    def n(self) -> int:
        return int(self.X.shape[0])


@dataclass
class _Labeled(_Tensors):
    """Covariates and outcomes of a data set on the training device."""

    time: torch.Tensor
    event: torch.Tensor


@dataclass
class SelectionResult:
    """Outcome of variable selection."""

    baseline: list[str]
    longitudinal: list[str]
    baseline_index: list[int]
    longitudinal_index: list[int]
    lambda_value: float
    actual_lambda: float
    w_L: float
    scale: float
    converge_epoch: int | None
    grid: list[dict] = field(default_factory=list)


class SCEDyn:
    """Self-consistent dynamic survival estimator at a single landmark.

    The model is fitted on the risk set at one landmark time, stored in a
    :class:`~scedyn.LandmarkData`: baseline covariates, the longitudinal history
    up to the landmark, and the observed residual time and event indicator. It
    estimates the conditional residual survival curve of new subjects.

    All tuning constants are keyword arguments. Their defaults are the values of
    the simulation study of the paper (low censoring, landmark 0.9).

    Parameters
    ----------
    noise_dim : int, default 5
        Dimension of the latent noise ``U`` of the generator.
    noise_range : float, default 1.0
        ``U`` is uniform on ``[-noise_range, noise_range] ** noise_dim``.
    hidden_dim : int, default 128
        Width of the feed-forward head of the generator.
    lstm_hidden_dim : int, default 216
        Hidden size of the LSTM encoders.
    lstm_layers : int, default 1
        Number of LSTM layers.
    embedding_dim : int, default 6
        Dimension of the history embedding of the generator.
    phi_hidden_dim : int, default 1280
        Width of the feed-forward part of the weight network.
    phi_lstm_hidden_dim : int, optional
        Hidden size of the LSTM of the weight network; ``lstm_hidden_dim`` if None.
    phi_embedding_dim : int, optional
        Embedding dimension of the weight network; ``embedding_dim`` if None.
    n_generated : int, default 500
        Number of generated residual times per subject in training.
    n_eval_times : int, default 100
        Number of evaluation times drawn from the pool in each epoch.
    pool_size : int, default 400
        Size of the pool of evaluation times.
    pool_quantiles : tuple of float, default (0.001, 0.999)
        The pool consists of quantiles, at levels equally spaced on this range,
        of the positive training residual times.
    indicator_temperature : float, default 1.0
        Temperature of the smoothed indicator in the criterion.
    censor_temperature : float, default 3.0
        Temperature of the smoothed indicator in the model-based estimate of
        the survival probability at the observed time, used in the denominator.
    censor_mix : {"mix", "km", "model"}, default "mix"
        Estimate of the survival probability in the denominator: Kaplan--Meier
        (``"km"``), model-based (``"model"``), or a mixture whose Kaplan--Meier
        weight decreases linearly to ``censor_mix_end`` (``"mix"``).
    censor_mix_end : float, default 0.0
        Kaplan--Meier weight after the warm-up of the mixture.
    censor_warmup_epochs : int, default 30
        Length of the warm-up of the mixture and of the clipping bounds.
    censor_clip_start : tuple of float, default (0.10, 0.90)
        Lower and upper clipping bounds of the denominator in the first epoch.
    censor_clip_warmup_end : tuple of float, default (0.05, 0.99)
        Clipping bounds reached at the end of the warm-up.
    censor_clip_after : tuple of float, default (0.02, 0.995)
        Clipping bounds after the warm-up.
    eps_model : float, default 1e-6
        Threshold subtracted from the model-based estimate.
    floor_model : float, default 1e-3
        Floor added to the model-based estimate and lower bound of the denominator.
    lr_generator : float, default 1e-4
        Adam learning rate of the generator.
    lr_phi : float, default 2e-4
        Adam learning rate of the weight network.
    betas_generator : tuple of float, default (0.9, 0.999)
        Adam coefficients of the generator.
    betas_phi : tuple of float, default (0.5, 0.999)
        Adam coefficients of the weight network.
    batch_size : int, default 16
        Minibatch size.
    max_epochs : int, default 200
        Number of training epochs.
    grad_clip : float, default 5.0
        Maximum gradient norm of the generator.
    lr_decay_generator : float, default 0.05
        After early stopping starts, the learning rate of the generator is
        ``lr_generator / (1 + k * lr_decay_generator)`` in the ``k``-th epoch.
    lr_decay_phi : float, default 5e-4
        Same decay for the weight network.
    es_start_fraction : float, default 0.6
        Early stopping starts after ``ceil(es_start_fraction * max_epochs)`` epochs.
    patience : int, default 30
        Epochs without improvement before the best state is kept. Training then
        continues to ``max_epochs``, and the kept state is returned.
    es_delta : float, default 2e-4
        An epoch improves if its validation score is below the best score plus
        ``es_delta``.
    val_samples : int, default 500
        Generated residual times per subject for the validation score.
    val_grid_size : int, default 100
        Number of evaluation times of the validation score.
    val_grid_quantiles : tuple of float, default (0.01, 0.95)
        Quantile levels of the validation residual times defining the
        evaluation times of the validation score.
    time_clip : float, optional
        If given, generated times are clipped to ``[0, time_clip]`` in the
        validation score.
    variable_selection : bool, default False
        Whether to select variables with stochastic gates.
    lambda_grid : sequence of float, optional
        Candidate normalized penalty levels for variable selection; the level
        with the best validation score is selected. The simulation study of the
        paper uses ``(0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)``. With
        ``variable_selection=True``, ``lambda_grid`` or ``lambda_value`` is
        required.
    lambda_value : float, optional
        A fixed normalized penalty level for variable selection, on the same
        scale as ``lambda_grid``, which skips the grid search; for example the
        level selected by a previous grid search,
        ``model.selection_.lambda_value``. If both are given, ``lambda_value``
        is used and ``lambda_grid`` is ignored.
    gate_sigma_baseline : float, default 0.3
        Standard deviation of the gate noise of the baseline covariates.
    gate_sigma_longitudinal : float, default 0.3
        Standard deviation of the gate noise of the longitudinal variables.
    gate_init_mu_baseline : float, default 0.65
        Initial gate mean of the baseline covariates.
    gate_init_mu_longitudinal : float, default 0.6
        Initial gate mean of the longitudinal variables.
    gate_threshold : float, default 0.5
        A variable is selected if its gate mean is at least this value.
    vs_warmup_epochs : int, default 50
        Epochs of training without penalty before selection.
    lambda_anneal_start : float, default 0.3
        Initial fraction of the penalty, increased linearly to one.
    lambda_anneal_epochs : int, default 60
        Length of the annealing of the penalty.
    drop_patience : int, default 2
        Consecutive epochs below the threshold after which a variable is
        eliminated.
    stable_epochs : int, default 5
        A group of gates converges when its selected set is unchanged for this
        many epochs.
    min_vs_epochs : int, default 15
        Minimum number of selection epochs before convergence.
    max_vs_epochs : int, default 150
        Maximum number of selection epochs.
    require_drop : bool, default True
        Whether a group of gates can converge only after at least one of its
        variables has been eliminated.
    post_selection_epochs : int, default 30
        Epochs of training after selection used to score a candidate level.
    top_k_ibs : int, default 3
        A candidate level is scored by the mean of its ``top_k_ibs`` smallest
        validation scores.
    w_L_clip : tuple of float, default (0.1, 10.0)
        Clipping range of the weight of the longitudinal penalty.
    lambda_scale_window : int, default 5
        Number of final warm-up epochs whose criterion sets the penalty scale.
    seed : int, optional
        Random seed set at the start of :meth:`fit`.
    device : str, default "auto"
        ``"auto"``, ``"cpu"``, ``"cuda"`` or ``"mps"``.
    verbose : int, default 1
        0 is silent, 1 reports the phases and 2 every epoch.

    Attributes
    ----------
    selection_ : SelectionResult or None
        Result of variable selection, None without selection.
    history_ : dict
        Training criterion, weight-network loss and validation scores by epoch.
    best_epoch_ : int
        Epoch of the returned state.

    Examples
    --------
    >>> from scedyn import SCEDyn
    >>> model = SCEDyn(seed=1).fit(train, val)  # doctest: +SKIP
    >>> S = model.predict_survival(test, times)  # doctest: +SKIP
    """

    def __init__(
        self,
        *,
        # networks
        noise_dim: int = 5,
        noise_range: float = 1.0,
        hidden_dim: int = 128,
        lstm_hidden_dim: int = 216,
        lstm_layers: int = 1,
        embedding_dim: int = 6,
        phi_hidden_dim: int = 1280,
        phi_lstm_hidden_dim: int | None = None,
        phi_embedding_dim: int | None = None,
        # criterion
        n_generated: int = 500,
        n_eval_times: int = 100,
        pool_size: int = 400,
        pool_quantiles: tuple[float, float] = (0.001, 0.999),
        indicator_temperature: float = 1.0,
        censor_temperature: float = 3.0,
        censor_mix: str = "mix",
        censor_mix_end: float = 0.0,
        censor_warmup_epochs: int = 30,
        censor_clip_start: tuple[float, float] = (0.10, 0.90),
        censor_clip_warmup_end: tuple[float, float] = (0.05, 0.99),
        censor_clip_after: tuple[float, float] = (0.02, 0.995),
        eps_model: float = 1e-6,
        floor_model: float = 1e-3,
        # optimization
        lr_generator: float = 1e-4,
        lr_phi: float = 2e-4,
        betas_generator: tuple[float, float] = (0.9, 0.999),
        betas_phi: tuple[float, float] = (0.5, 0.999),
        batch_size: int = 16,
        max_epochs: int = 200,
        grad_clip: float = 5.0,
        lr_decay_generator: float = 5e-2,
        lr_decay_phi: float = 5e-4,
        es_start_fraction: float = 0.6,
        patience: int = 30,
        es_delta: float = 2e-4,
        val_samples: int = 500,
        val_grid_size: int = 100,
        val_grid_quantiles: tuple[float, float] = (0.01, 0.95),
        time_clip: float | None = None,
        # variable selection
        variable_selection: bool = False,
        lambda_grid: Sequence[float] | None = None,
        lambda_value: float | None = None,
        gate_sigma_baseline: float = 0.3,
        gate_sigma_longitudinal: float = 0.3,
        gate_init_mu_baseline: float = 0.65,
        gate_init_mu_longitudinal: float = 0.6,
        gate_threshold: float = 0.5,
        vs_warmup_epochs: int = 50,
        lambda_anneal_start: float = 0.3,
        lambda_anneal_epochs: int = 60,
        drop_patience: int = 2,
        stable_epochs: int = 5,
        min_vs_epochs: int = 15,
        max_vs_epochs: int = 150,
        require_drop: bool = True,
        post_selection_epochs: int = 30,
        top_k_ibs: int = 3,
        w_L_clip: tuple[float, float] = (0.1, 10.0),
        lambda_scale_window: int = 5,
        # other
        seed: int | None = None,
        device: str | None = "auto",
        verbose: int = 1,
    ):
        if censor_mix not in ("mix", "km", "model"):
            msg = "censor_mix must be 'mix', 'km' or 'model'."
            raise ValueError(msg)
        _check_penalty_arguments(variable_selection, lambda_grid, lambda_value)
        params = dict(locals())
        params.pop("self")
        self.params = params
        self.noise_dim = noise_dim
        self.noise_range = noise_range
        self.hidden_dim = hidden_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.lstm_layers = lstm_layers
        self.embedding_dim = embedding_dim
        self.phi_hidden_dim = phi_hidden_dim
        self.n_generated = n_generated
        self.n_eval_times = n_eval_times
        self.pool_size = pool_size
        self.pool_quantiles = pool_quantiles
        self.indicator_temperature = indicator_temperature
        self.censor_temperature = censor_temperature
        self.censor_mix = censor_mix
        self.censor_mix_end = censor_mix_end
        self.censor_warmup_epochs = censor_warmup_epochs
        self.censor_clip_start = censor_clip_start
        self.censor_clip_warmup_end = censor_clip_warmup_end
        self.censor_clip_after = censor_clip_after
        self.eps_model = eps_model
        self.floor_model = floor_model
        self.lr_generator = lr_generator
        self.lr_phi = lr_phi
        self.betas_generator = betas_generator
        self.betas_phi = betas_phi
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.grad_clip = grad_clip
        self.lr_decay_generator = lr_decay_generator
        self.lr_decay_phi = lr_decay_phi
        self.es_start_fraction = es_start_fraction
        self.patience = patience
        self.es_delta = es_delta
        self.val_samples = val_samples
        self.val_grid_size = val_grid_size
        self.val_grid_quantiles = val_grid_quantiles
        self.time_clip = time_clip
        self.variable_selection = variable_selection
        self.lambda_grid = lambda_grid
        self.lambda_value = lambda_value
        self.gate_sigma_baseline = gate_sigma_baseline
        self.gate_sigma_longitudinal = gate_sigma_longitudinal
        self.gate_init_mu_baseline = gate_init_mu_baseline
        self.gate_init_mu_longitudinal = gate_init_mu_longitudinal
        self.gate_threshold = gate_threshold
        self.vs_warmup_epochs = vs_warmup_epochs
        self.lambda_anneal_start = lambda_anneal_start
        self.lambda_anneal_epochs = lambda_anneal_epochs
        self.drop_patience = drop_patience
        self.stable_epochs = stable_epochs
        self.min_vs_epochs = min_vs_epochs
        self.max_vs_epochs = max_vs_epochs
        self.require_drop = require_drop
        self.post_selection_epochs = post_selection_epochs
        self.top_k_ibs = top_k_ibs
        self.w_L_clip = w_L_clip
        self.lambda_scale_window = lambda_scale_window
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.phi_lstm_hidden_dim = phi_lstm_hidden_dim or lstm_hidden_dim
        self.phi_embedding_dim = phi_embedding_dim or embedding_dim
        self.device_ = _pick_device(device)
        self.generator_: Generator | None = None
        self.phi_: WeightNetwork | None = None
        self.selection_: SelectionResult | None = None
        self.history_: dict[str, list] = {}

    @property
    def _G(self) -> Generator:
        if self.generator_ is None:
            msg = "Call fit() first."
            raise RuntimeError(msg)
        return self.generator_

    @property
    def _phi(self) -> WeightNetwork:
        if self.phi_ is None:
            msg = "Call fit() first."
            raise RuntimeError(msg)
        return self.phi_

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _log(self, level, msg):
        if self.verbose >= level:
            print(msg, flush=True)  # noqa: T201

    def _to_tensors(self, data: LandmarkData) -> _Tensors:
        dev = self.device_
        return _Tensors(
            X=torch.as_tensor(data.X, dtype=torch.float32, device=dev),
            Z=torch.as_tensor(data.padded_Z(), dtype=torch.float32, device=dev),
            lengths=torch.as_tensor(data.lengths, dtype=torch.long),
        )

    def _to_labeled(self, data: LandmarkData) -> _Labeled:
        if data.time is None or data.event is None:
            msg = "time and event are required."
            raise ValueError(msg)
        t = self._to_tensors(data)
        dev = self.device_
        return _Labeled(
            X=t.X,
            Z=t.Z,
            lengths=t.lengths,
            time=torch.as_tensor(data.time, dtype=torch.float32, device=dev),
            event=torch.as_tensor(data.event, dtype=torch.float32, device=dev),
        )

    def _indicator(self, x):
        return 0.5 + 0.5 * torch.tanh(self.indicator_temperature * x)

    def _smooth_censor(self, x):
        return 0.5 + 0.5 * torch.tanh(self.censor_temperature * x)

    def _noise(self, n):
        r = self.noise_range
        return 2 * r * torch.rand(n, self.noise_dim, device=self.device_) - r

    def _alpha_warmup(self, epoch):
        """Kaplan--Meier weight of the denominator during the warm-up schedule."""
        if self.censor_mix == "km":
            return 1.0
        if self.censor_mix == "model":
            return 0.0
        W = self.censor_warmup_epochs
        if epoch < W:
            a = 1.0 - (1.0 - float(self.censor_mix_end)) * (epoch / max(1, W - 1))
        else:
            a = float(self.censor_mix_end)
        return float(np.clip(a, 0.0, 1.0))

    def _alpha_after(self):
        if self.censor_mix == "km":
            return 1.0
        if self.censor_mix == "model":
            return 0.0
        return float(np.clip(float(self.censor_mix_end), 0.0, 1.0))

    def _clip_after_epoch(self, epoch):
        """Denominator clipping bounds used after ``epoch`` (warm-up schedule)."""
        W = self.censor_warmup_epochs
        if epoch < W:
            (lo0, hi0), (lo1, hi1) = self.censor_clip_start, self.censor_clip_warmup_end
            frac = epoch / max(1, W - 1)
            return lo0 + (lo1 - lo0) * frac, hi0 + (hi1 - hi0) * frac
        return self.censor_clip_after

    def _set_lr(self, epoch_count, es_start):
        if epoch_count > es_start:
            k = epoch_count - es_start
            for g in self.opt_gen_.param_groups:
                g["lr"] = self.lr_generator / (1 + k * self.lr_decay_generator)
            for g in self.opt_phi_.param_groups:
                g["lr"] = self.lr_phi / (1 + k * self.lr_decay_phi)

    # ------------------------------------------------------------------
    # criterion
    # ------------------------------------------------------------------
    def _prepare(self, train: LandmarkData, val: LandmarkData):
        self.n_baseline_ = train.n_baseline
        self.n_longitudinal_ = train.n_longitudinal
        self.baseline_names_ = list(train.baseline_names or [])
        self.longitudinal_names_ = list(train.longitudinal_names or [])
        tr = self._to_labeled(train)
        va = self._to_labeled(val)
        dev = self.device_

        pool = _quantile_grid(
            tr.time.cpu().numpy(), *self.pool_quantiles, self.pool_size
        ).to(dev)
        self.pool_ = pool
        self.t_mean_ = pool.mean()
        self.t_std_ = pool.std() + 1e-6

        T_cpu, E_cpu, pool_cpu = tr.time.cpu(), tr.event.cpu(), pool.cpu()
        S_km_pool = km_survival(T_cpu, E_cpu, pool_cpu)
        self.cw_km_ = _step_lookup(pool_cpu, S_km_pool, T_cpu).to(dev)
        self.right1_ = self._indicator(tr.time.view(-1, 1) - pool.view(1, -1)).detach()
        self.right2_ = self._indicator(pool.view(1, -1) - tr.time.view(-1, 1)).detach()

        self.val_grid_ = _quantile_grid(
            va.time.cpu().numpy(), *self.val_grid_quantiles, self.val_grid_size
        )
        self.G_val_ = km_survival(va.time.cpu(), 1.0 - va.event.cpu(), self.val_grid_)
        self.train_t_, self.val_t_ = tr, va

    def _batch_terms(self, rows, tgrid, cols, fake, alpha_km, lo, hi):
        """Left and Right sides of the self-consistent equation on a batch."""
        tr = self.train_t_
        y = tr.time[rows].view(-1, 1)
        delta = tr.event[rows].view(-1, 1)
        M = fake.shape[1]
        cw_model = (
            (
                F.relu(self._smooth_censor(fake - y).sum(1) / M - self.eps_model)
                + self.floor_model
            )
            .view(-1, 1)
            .detach()
        )
        cw = alpha_km * self.cw_km_[rows] + (1.0 - alpha_km) * cw_model
        cw = torch.clamp(cw, min=max(self.floor_model, lo), max=hi)
        S = self._indicator(fake.unsqueeze(2) - tgrid.view(1, 1, -1)).mean(1)
        right = (
            self.right1_[rows][:, cols]
            + self.right2_[rows][:, cols] * (1 - delta) / cw * S.detach()
        )
        return S, right

    def _sample_eval_times(self):
        K = self.pool_.numel()
        if self.n_eval_times <= K:
            cols = torch.randperm(K, device=self.device_)[: self.n_eval_times]
        else:
            cols = torch.randint(0, K, (self.n_eval_times,), device=self.device_)
        tgrid = self.pool_[cols] + 1e-6
        tstd = ((tgrid - self.t_mean_) / (self.t_std_ + 1e-6)).view(-1, 1)
        return tgrid, cols, tstd

    def _run_epoch(self, alpha_km, lo, hi, penalty_coef=0.0, enforce_eliminated=False):
        G, phi, tr = self._G, self._phi, self.train_t_
        M = self.n_generated
        tgrid, cols, tstd = self._sample_eval_times()
        K = tgrid.numel()
        order = torch.randperm(tr.n)
        totals = {"phi": 0.0, "loss": 0.0, "criterion": 0.0}
        n_batches = 0
        for start in range(0, tr.n, self.batch_size):
            rows_cpu = order[start : start + self.batch_size]
            rows = rows_cpu.to(self.device_)
            B = rows.numel()
            x, z, lens = tr.X[rows], tr.Z[rows], tr.lengths[rows_cpu]
            noise = self._noise(B * M)
            s_rep = tstd.repeat(B, 1)

            # weight network step (ascent on the criterion)
            with torch.no_grad():
                fake = G(noise, x, z, lens, repeats=M).view(B, M)
                left, right = self._batch_terms(
                    rows, tgrid, cols, fake, alpha_km, lo, hi
                )
            g = phi(s_rep, x, z, lens, repeats=K).view(B, K)
            phi_loss = -F.mse_loss((left * g).mean(0), (right * g).mean(0))
            self.opt_phi_.zero_grad()
            phi_loss.backward()
            self.opt_phi_.step()

            # generator step (descent on the criterion)
            fake = G(noise, x, z, lens, repeats=M).view(B, M)
            left, right = self._batch_terms(rows, tgrid, cols, fake, alpha_km, lo, hi)
            with torch.no_grad():
                g = phi(s_rep, x, z, lens, repeats=K).view(B, K)
            crit = F.mse_loss((left * g).mean(0), (right * g).mean(0))
            loss = crit
            if penalty_coef > 0.0:
                loss = loss + penalty_coef * G.penalty()
            self.opt_gen_.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=self.grad_clip)
            self.opt_gen_.step()
            if enforce_eliminated:
                G.enforce_eliminated()

            totals["phi"] += phi_loss.item()
            totals["loss"] += loss.item()
            totals["criterion"] += crit.item()
            n_batches += 1
        return {k: v / max(n_batches, 1) for k, v in totals.items()}

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _generate(self, t: _Tensors, n_samples, chunk=None):
        """Generated residual times, shape (n, n_samples), on the CPU."""
        G = self._G
        was_training = G.training
        G.eval()
        chunk = chunk or max(1, 8000 // max(n_samples, 1))
        out = []
        for start in range(0, t.n, chunk):
            sl = slice(start, start + chunk)
            x, z, lens = t.X[sl], t.Z[sl], t.lengths[sl]
            n = x.shape[0]
            fake = G(self._noise(n * n_samples), x, z, lens, repeats=n_samples).view(
                n, n_samples
            )
            out.append(fake.cpu())
        G.train(was_training)
        return torch.cat(out, 0)

    def _clip_times(self, fake, time_clip):
        if time_clip is None:
            return fake
        s_max = max(float(time_clip), 0.0)
        return (
            torch.clamp(fake, min=0.0, max=s_max)
            if s_max > 0
            else torch.zeros_like(fake)
        )

    def _val_ibs(self):
        va = self.val_t_
        fake = self._clip_times(self._generate(va, self.val_samples), self.time_clip)
        S_hat = (fake.unsqueeze(2) > self.val_grid_.view(1, 1, -1)).float().mean(1)
        bs = brier_scores(
            S_hat,
            va.time.cpu().numpy(),
            va.event.cpu().numpy(),
            self.val_grid_.numpy(),
            self.G_val_,
        )
        return float(bs.mean())

    # ------------------------------------------------------------------
    # training phases
    # ------------------------------------------------------------------
    def _init_networks(self):
        p, q = self.n_baseline_, self.n_longitudinal_
        self.generator_ = Generator(
            self.noise_dim,
            p,
            q,
            hidden_dim=self.hidden_dim,
            lstm_hidden_dim=self.lstm_hidden_dim,
            lstm_layers=self.lstm_layers,
            embedding_dim=self.embedding_dim,
            gate_sigma_baseline=self.gate_sigma_baseline,
            gate_sigma_longitudinal=self.gate_sigma_longitudinal,
            gate_init_mu_baseline=self.gate_init_mu_baseline,
            gate_init_mu_longitudinal=self.gate_init_mu_longitudinal,
            gate_threshold=self.gate_threshold,
        ).to(self.device_)
        self.phi_ = WeightNetwork(
            p,
            q,
            hidden_dim=self.phi_hidden_dim,
            lstm_hidden_dim=self.phi_lstm_hidden_dim,
            lstm_layers=self.lstm_layers,
            embedding_dim=self.phi_embedding_dim,
        ).to(self.device_)
        self.opt_gen_ = torch.optim.Adam(
            self._G.parameters(),
            lr=self.lr_generator,
            betas=(float(self.betas_generator[0]), float(self.betas_generator[1])),
        )
        self.opt_phi_ = torch.optim.Adam(
            self._phi.parameters(),
            lr=self.lr_phi,
            betas=(float(self.betas_phi[0]), float(self.betas_phi[1])),
        )

    def _freeze_all_and_mask(self):
        G = self._G
        mask_B = (G.mu_B.data.clone()).float()
        mask_L = (G.mu_L.data.clone()).float()
        self._phi.set_masks(mask_B, mask_L)

    def _train_with_early_stopping(self, start_count, history):
        """Final training phase with learning-rate decay and early stopping."""
        G = self._G
        E = self.max_epochs
        es_start = math.ceil(E * self.es_start_fraction)
        lo, hi = self.censor_clip_after
        alpha = self._alpha_after()
        count = start_count
        best_ibs, best_epoch, patience = float("inf"), 0, 0
        best_state: dict[str, Any] = deepcopy(G.state_dict())
        cached: tuple[dict[str, Any], int] | None = None
        while count < E:
            count += 1
            self._set_lr(count, es_start)
            res = self._run_epoch(alpha, lo, hi)
            history["criterion"].append(res["criterion"])
            history["phi"].append(res["phi"])
            msg = f"[epoch {count}/{E}] criterion={res['criterion']:.6f}"
            if count > es_start:
                ibs = self._val_ibs()
                history["val_ibs"].append((count, ibs))
                msg += f" val_ibs={ibs:.6f}"
                if count == es_start + 1 or ibs < best_ibs + self.es_delta:
                    best_ibs, best_epoch, best_state, patience = (
                        ibs,
                        count,
                        deepcopy(G.state_dict()),
                        0,
                    )
                else:
                    patience += 1
                    if patience >= self.patience and cached is None:
                        cached = (deepcopy(best_state), best_epoch)
                        msg += f" (early stop: best epoch {best_epoch})"
            self._log(2, msg)
        if cached is not None:
            G.load_state_dict(cached[0])
            self.best_epoch_ = cached[1]
            self._log(1, f"Early stopping kept the state of epoch {cached[1]}.")
        else:
            self.best_epoch_ = count
            self._log(1, "Early stopping not triggered; the last epoch is kept.")

    def _train_without_selection(self, history):
        """Training with all inputs (Algorithm 1)."""
        G = self._G
        G.freeze("both", threshold=0.0)
        self._freeze_all_and_mask()
        E = self.max_epochs
        es_start = math.ceil(E * self.es_start_fraction)
        lo, hi = self.censor_clip_start
        best_ibs, best_epoch, patience = float("inf"), 0, 0
        best_state: dict[str, Any] = deepcopy(G.state_dict())
        cached: tuple[dict[str, Any], int] | None = None
        for epoch in range(1, E + 1):
            self._set_lr(epoch, es_start)
            res = self._run_epoch(self._alpha_warmup(epoch), lo, hi)
            history["criterion"].append(res["criterion"])
            history["phi"].append(res["phi"])
            lo, hi = self._clip_after_epoch(epoch)
            msg = f"[epoch {epoch}/{E}] criterion={res['criterion']:.6f}"
            if epoch > es_start:
                ibs = self._val_ibs()
                history["val_ibs"].append((epoch, ibs))
                msg += f" val_ibs={ibs:.6f}"
                if epoch == es_start + 1 or ibs < best_ibs + self.es_delta:
                    best_ibs, best_epoch, best_state, patience = (
                        ibs,
                        epoch,
                        deepcopy(G.state_dict()),
                        0,
                    )
                else:
                    patience += 1
                    if patience >= self.patience and cached is None:
                        cached = (deepcopy(best_state), best_epoch)
                        msg += f" (early stop: best epoch {best_epoch})"
            self._log(2, msg)
        if cached is not None:
            G.load_state_dict(cached[0])
            self.best_epoch_ = cached[1]
            self._log(1, f"Early stopping kept the state of epoch {cached[1]}.")
        else:
            self.best_epoch_ = E
            self._log(1, "Early stopping not triggered; the last epoch is kept.")

    def _calibrate_w_L(self, lo, hi):
        """w_L = mean |d criterion / d mu_L| / mean |d criterion / d mu_B| over one pass."""
        G, phi, tr = self._G, self._phi, self.train_t_
        G.train()
        M = self.n_generated
        alpha = self._alpha_after()
        gB, gL = [], []
        order = torch.randperm(tr.n)
        for start in range(0, tr.n, self.batch_size):
            rows_cpu = order[start : start + self.batch_size]
            rows = rows_cpu.to(self.device_)
            B = rows.numel()
            x, z, lens = tr.X[rows], tr.Z[rows], tr.lengths[rows_cpu]
            self.opt_gen_.zero_grad()
            fake = G(self._noise(B * M), x, z, lens, repeats=M).view(B, M)
            tgrid, cols, tstd = self._sample_eval_times()
            K = tgrid.numel()
            left, right = self._batch_terms(rows, tgrid, cols, fake, alpha, lo, hi)
            with torch.no_grad():
                g = phi(tstd.repeat(B, 1), x, z, lens, repeats=K).view(B, K)
            F.mse_loss((left * g).mean(0), (right * g).mean(0)).backward()
            if G.mu_B.grad is not None and G.alive_B.any():
                gB.append(G.mu_B.grad[G.alive_B].abs().mean().item())
            if G.mu_L.grad is not None and G.alive_L.any():
                gL.append(G.mu_L.grad[G.alive_L].abs().mean().item())
        self.opt_gen_.zero_grad()
        grad_B = float(np.mean(gB)) if gB else 1e-12
        grad_L = float(np.mean(gL)) if gL else 1e-12
        return float(np.clip(grad_L / (grad_B + 1e-12), *self.w_L_clip))

    def _selection_loop(self, actual_lambda):
        """Penalized training until both gate groups converge (Algorithm 2, inner loop)."""
        G = self._G
        p, q = self.n_baseline_, self.n_longitudinal_
        thr = self.gate_threshold
        below_B, below_L = np.zeros(p, dtype=int), np.zeros(q, dtype=int)
        prev_B = prev_L = None
        stable_B = stable_L = 0
        dropped_B = dropped_L = 0
        conv_B = conv_L = False
        lo, hi = self.censor_clip_after
        alpha = self._alpha_after()
        for ep in range(self.max_vs_epochs):
            if ep < self.lambda_anneal_epochs:
                anneal = self.lambda_anneal_start + (1.0 - self.lambda_anneal_start) * (
                    ep / self.lambda_anneal_epochs
                )
            else:
                anneal = 1.0
            res = self._run_epoch(
                alpha,
                lo,
                hi,
                penalty_coef=anneal * actual_lambda,
                enforce_eliminated=True,
            )

            muB = G.mu_B.data.cpu().numpy().copy()
            muL = G.mu_L.data.cpu().numpy().copy()
            aliveB = G.alive_B.cpu().numpy()
            aliveL = G.alive_L.cpu().numpy()
            for counter, mu, alive in ((below_B, muB, aliveB), (below_L, muL, aliveL)):
                for j in range(len(mu)):
                    if alive[j]:
                        counter[j] = counter[j] + 1 if mu[j] < thr else 0
            drop_B = [
                j for j in range(p) if aliveB[j] and below_B[j] >= self.drop_patience
            ]
            drop_L = [
                k for k in range(q) if aliveL[k] and below_L[k] >= self.drop_patience
            ]
            # never eliminate every alive input of a group
            if len(drop_B) >= int(aliveB.sum()) > 0:
                keep = max(drop_B, key=lambda j: muB[j])
                drop_B = [j for j in drop_B if j != keep]
            if len(drop_L) >= int(aliveL.sum()) > 0:
                keep = max(drop_L, key=lambda k: muL[k])
                drop_L = [k for k in drop_L if k != keep]
            if drop_B or drop_L:
                nB = sum(bool(G.alive_B[j]) for j in drop_B)
                nL = sum(bool(G.alive_L[k]) for k in drop_L)
                G.drop(drop_B, drop_L)
                dropped_B += nB
                dropped_L += nL
                for j in drop_B:
                    below_B[j] = 0
                for k in drop_L:
                    below_L[k] = 0

            aliveB = G.alive_B.cpu().numpy()
            aliveL = G.alive_L.cpu().numpy()
            sel_B = frozenset(j for j in range(p) if aliveB[j] and muB[j] >= thr)
            sel_L = frozenset(k for k in range(q) if aliveL[k] and muL[k] >= thr)
            # never leave a group with no selected input
            if not sel_B and not conv_B:
                alive_idx = [j for j in range(p) if aliveB[j]]
                if alive_idx:
                    best = max(alive_idx, key=lambda j: muB[j])
                    G.mu_B.data[best] = thr + 0.01
                    sel_B = frozenset([best])
            if not sel_L and not conv_L:
                alive_idx = [k for k in range(q) if aliveL[k]]
                if alive_idx:
                    best = max(alive_idx, key=lambda k: muL[k])
                    G.mu_L.data[best] = thr + 0.01
                    sel_L = frozenset([best])
            if not conv_B:
                stable_B = stable_B + 1 if sel_B == prev_B else 0
                prev_B = sel_B
            if not conv_L:
                stable_L = stable_L + 1 if sel_L == prev_L else 0
                prev_L = sel_L

            self._log(
                2,
                f"[selection epoch {ep + 1}] criterion={res['criterion']:.6f} "
                f"selected B={len(sel_B)} L={len(sel_L)} dropped B={dropped_B} L={dropped_L}",
            )

            if (
                not conv_B
                and stable_B >= self.stable_epochs
                and ep >= self.min_vs_epochs
                and (dropped_B > 0 or not self.require_drop)
            ):
                conv_B = True
                G.freeze("baseline")
            if (
                not conv_L
                and stable_L >= self.stable_epochs
                and ep >= self.min_vs_epochs
                and (dropped_L > 0 or not self.require_drop)
            ):
                conv_L = True
                G.freeze("longitudinal")
            if conv_B and conv_L:
                self._freeze_all_and_mask()
                return ep + 1
        return None

    def _restore(self, ckpt):
        G = self._G
        G.load_state_dict(ckpt["generator"])
        self._phi.load_state_dict(ckpt["phi"])
        self.opt_gen_.load_state_dict(ckpt["opt_gen"])
        self.opt_phi_.load_state_dict(ckpt["opt_phi"])
        G.unfreeze()
        G.w_L = ckpt["w_L"]
        self._phi.reset_masks()

    def _snapshot(self):
        return {
            "generator": deepcopy(self._G.state_dict()),
            "phi": deepcopy(self._phi.state_dict()),
            "opt_gen": deepcopy(self.opt_gen_.state_dict()),
            "opt_phi": deepcopy(self.opt_phi_.state_dict()),
            "w_L": self._G.w_L,
        }

    def _set_frozen(self):
        G = self._G
        G.sigma_B = G.sigma_L = 0.0
        G.frozen_B = G.frozen_L = True
        G.mu_B.requires_grad_(False)
        G.mu_L.requires_grad_(False)
        self._phi.mask_active = True

    def _train_with_selection(self, history):
        """Training with variable selection (Algorithm 2)."""
        G = self._G
        lo, hi = self.censor_clip_start
        warm_losses = []
        for epoch in range(self.vs_warmup_epochs):
            res = self._run_epoch(self._alpha_warmup(epoch), lo, hi)
            history["criterion"].append(res["criterion"])
            history["phi"].append(res["phi"])
            warm_losses.append(res["loss"])
            lo, hi = self._clip_after_epoch(epoch)
            self._log(
                2,
                f"[warm-up {epoch + 1}/{self.vs_warmup_epochs}] criterion={res['criterion']:.6f}",
            )

        w_L = self._calibrate_w_L(lo, hi)
        G.w_L = w_L
        L_warm = float(np.mean(warm_losses[-self.lambda_scale_window :]))
        R_warm = float(G.penalty().item())
        scale = L_warm / (R_warm + 1e-12)
        k_scale = (self.n_baseline_ + self.n_longitudinal_) / 5
        self._log(1, f"Warm-up done: w_L={w_L:.4f}, scale={scale:.6g}")

        grid_records = []
        if self.lambda_value is not None:
            lam = float(self.lambda_value)
            actual = lam * k_scale * scale
            conv = self._selection_loop(actual)
            if conv is None:
                msg = (
                    f"Variable selection did not converge within {self.max_vs_epochs} epochs "
                    f"(lambda_value={lam})."
                )
                raise RuntimeError(msg)
            start_count = 0
        else:
            ckpt = self._snapshot()
            best: tuple[dict[str, Any], dict[str, Any]] | None = None
            grid_values = [] if self.lambda_grid is None else list(self.lambda_grid)
            for lam in grid_values:
                actual = lam * k_scale * scale
                self._restore(ckpt)
                G.train()
                self._phi.train()
                conv = self._selection_loop(actual)
                if conv is None:
                    grid_records.append(
                        {
                            "lambda": lam,
                            "actual_lambda": actual,
                            "converged": False,
                            "score": float("inf"),
                        }
                    )
                    self._log(1, f"lambda={lam}: selection did not converge, skipped.")
                    continue
                ibs_list: list[float] = []
                best_ibs = float("inf")
                best_snap = self._snapshot()
                lo_a, hi_a = self.censor_clip_after
                for _ in range(self.post_selection_epochs):
                    self._run_epoch(self._alpha_after(), lo_a, hi_a)
                    ibs = self._val_ibs()
                    ibs_list.append(ibs)
                    if ibs < best_ibs:
                        best_ibs, best_snap = ibs, self._snapshot()
                score = float(np.mean(sorted(ibs_list)[: self.top_k_ibs]))
                rec = {
                    "lambda": lam,
                    "actual_lambda": actual,
                    "converged": True,
                    "converge_epoch": conv,
                    "score": score,
                    "best_ibs": best_ibs,
                    "baseline_index": self._selected_index("B"),
                    "longitudinal_index": self._selected_index("L"),
                }
                grid_records.append(rec)
                self._log(
                    1,
                    f"lambda={lam}: score={score:.6f}, selected "
                    f"{len(rec['baseline_index'])} baseline and "
                    f"{len(rec['longitudinal_index'])} longitudinal inputs",
                )
                if best is None or score < best[0]["score"]:
                    best = (rec, best_snap)
            if best is None:
                msg = (
                    "Variable selection did not converge for any value in lambda_grid."
                )
                raise RuntimeError(msg)
            rec, snap = best
            lam, actual, conv = (
                rec["lambda"],
                rec["actual_lambda"],
                rec["converge_epoch"],
            )
            G.load_state_dict(snap["generator"])
            self._phi.load_state_dict(snap["phi"])
            self.opt_gen_.load_state_dict(snap["opt_gen"])
            self.opt_phi_.load_state_dict(snap["opt_phi"])
            self._set_frozen()
            start_count = self.post_selection_epochs

        self._set_frozen()
        G.train()
        self._phi.train()
        self.selection_ = SelectionResult(
            baseline=[self.baseline_names_[j] for j in self._selected_index("B")],
            longitudinal=[
                self.longitudinal_names_[k] for k in self._selected_index("L")
            ],
            baseline_index=self._selected_index("B"),
            longitudinal_index=self._selected_index("L"),
            lambda_value=lam,
            actual_lambda=actual,
            w_L=w_L,
            scale=scale,
            converge_epoch=conv,
            grid=grid_records,
        )
        self._log(1, f"Selected baseline: {self.selection_.baseline}")
        self._log(1, f"Selected longitudinal: {self.selection_.longitudinal}")
        self._train_with_early_stopping(start_count, history)

    def _selected_index(self, group):
        mu = self._G.mu_B if group == "B" else self._G.mu_L
        return [int(j) for j in torch.where(mu.data > 0.5)[0].cpu()]

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def fit(self, train: LandmarkData, val: LandmarkData) -> SCEDyn:
        """Fit the model on the training risk set, using ``val`` for early stopping
        and for the choice of the penalty level."""
        if (
            train.n_baseline != val.n_baseline
            or train.n_longitudinal != val.n_longitudinal
        ):
            msg = "train and val must have the same covariates."
            raise ValueError(msg)
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
        self._prepare(train, val)
        self._init_networks()
        self._G.train()
        self._phi.train()
        history: dict[str, list[Any]] = {"criterion": [], "phi": [], "val_ibs": []}
        if self.variable_selection:
            self._train_with_selection(history)
        else:
            self._train_without_selection(history)
            self.selection_ = None
        self.history_ = history
        self._G.eval()
        # free training tensors
        del self.train_t_, self.val_t_, self.right1_, self.right2_, self.cw_km_
        return self

    def sample(self, data: LandmarkData, n_samples: int = 1000) -> np.ndarray:
        """Generated residual times, shape (n, n_samples)."""
        self._check_fitted(data)
        return self._generate(self._to_tensors(data), n_samples).numpy()

    def predict_survival(
        self,
        data: LandmarkData,
        times,
        n_samples: int = 1000,
        time_clip: float | None = None,
    ) -> np.ndarray:
        """Estimated residual survival P(T - tau > s | history) at ``times``.

        Returns an array of shape (n, len(times)). The estimate is the fraction
        of ``n_samples`` generated residual times exceeding each time. If
        ``time_clip`` is given, generated times are first clipped to
        ``[0, time_clip]``.
        """
        self._check_fitted(data)
        fake = self._generate(self._to_tensors(data), n_samples)
        fake = self._clip_times(fake, time_clip)
        grid = torch.as_tensor(np.asarray(times), dtype=torch.float32).view(1, 1, -1)
        return (fake.unsqueeze(2) > grid).float().mean(1).numpy()

    @property
    def selected_variables_(self) -> dict[str, list[str]] | None:
        """Selected baseline and longitudinal variables, or None without selection."""
        if self.selection_ is None:
            return None
        return {
            "baseline": list(self.selection_.baseline),
            "longitudinal": list(self.selection_.longitudinal),
        }

    def _check_fitted(self, data: LandmarkData):
        if self.generator_ is None:
            msg = "Call fit() first."
            raise RuntimeError(msg)
        if (
            data.n_baseline != self.n_baseline_
            or data.n_longitudinal != self.n_longitudinal_
        ):
            msg = "data must have the same covariates as the training data."
            raise ValueError(msg)

    # ------------------------------------------------------------------
    def save(self, path: str):
        """Save the fitted model to ``path``."""
        torch.save(
            {
                "params": self.params,
                "generator": self._G.state_dict(),
                "phi": self._phi.state_dict(),
                "w_L": self._G.w_L,
                "meta": {
                    "n_baseline": self.n_baseline_,
                    "n_longitudinal": self.n_longitudinal_,
                    "baseline_names": self.baseline_names_,
                    "longitudinal_names": self.longitudinal_names_,
                    "selection": None
                    if self.selection_ is None
                    else self.selection_.__dict__,
                    "best_epoch": getattr(self, "best_epoch_", None),
                },
            },
            path,
        )

    @classmethod
    def load(cls, path: str, device: str | None = "auto") -> SCEDyn:
        """Load a model saved with :meth:`save`."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        params = dict(ckpt["params"])
        params["device"] = device
        model = cls(**params)
        meta = ckpt["meta"]
        model.n_baseline_ = meta["n_baseline"]
        model.n_longitudinal_ = meta["n_longitudinal"]
        model.baseline_names_ = meta["baseline_names"]
        model.longitudinal_names_ = meta["longitudinal_names"]
        model.best_epoch_ = meta["best_epoch"]
        model._init_networks()
        model._G.load_state_dict(ckpt["generator"])
        model._phi.load_state_dict(ckpt["phi"])
        model._G.w_L = ckpt["w_L"]
        sel = meta["selection"]
        model.selection_ = None if sel is None else SelectionResult(**sel)
        model._G.sigma_B = model._G.sigma_L = 0.0
        model._G.eval()
        return model
