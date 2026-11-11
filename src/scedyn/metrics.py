"""Evaluation metrics: integrated Brier score and integrated mean squared error."""

from __future__ import annotations

import numpy as np
import torch


def km_survival(times, events, eval_times):
    """Kaplan--Meier estimate of P(T > t) at ``eval_times``."""
    times = torch.as_tensor(times, dtype=torch.float32)
    events = torch.as_tensor(events, dtype=torch.float32)
    eval_times = torch.as_tensor(eval_times, dtype=torch.float32)
    order = torch.argsort(times)
    t, d = times[order], events[order]
    uniq, counts = torch.unique_consecutive(t, return_counts=True)
    pos = torch.bucketize(t, uniq, right=True) - 1
    d_by_t = torch.zeros_like(uniq).scatter_add_(0, pos, d)
    at_risk = torch.flip(torch.cumsum(torch.flip(counts.float(), [0]), 0), [0])
    hazard = torch.where(at_risk > 0, d_by_t / at_risk, torch.zeros_like(d_by_t))
    S = torch.cumprod(1.0 - hazard, 0)
    idx = torch.bucketize(eval_times, uniq, right=True) - 1
    return torch.where(idx >= 0, S[idx.clamp(min=0)], torch.ones_like(eval_times))


def brier_scores(S_hat, time, event, grid, G_grid=None):
    """IPCW Brier score at each grid point.

    ``S_hat`` has shape (n, K) and holds the predicted survival on ``grid``.
    The censoring survival function is estimated by Kaplan--Meier on the
    same subjects unless ``G_grid`` is given.
    """
    S_hat = torch.as_tensor(np.asarray(S_hat), dtype=torch.float32)
    T = torch.as_tensor(np.asarray(time), dtype=torch.float32).view(-1, 1)
    E = torch.as_tensor(np.asarray(event), dtype=torch.float32).view(-1, 1)
    u_grid = torch.as_tensor(np.asarray(grid), dtype=torch.float32)
    if G_grid is None:
        G_grid = km_survival(T.view(-1), 1.0 - E.view(-1), u_grid)
    K = S_hat.shape[1]
    u = u_grid.view(1, -1)
    Y = (u < T).float()
    case1 = ((u >= T) & (E == 1)).float()
    idx_T = (torch.bucketize(T.squeeze(1), u_grid, right=True) - 1).clamp(min=0)
    G_T = G_grid[idx_T].view(-1, 1)
    w1 = (
        torch.where(G_T > 1e-8, 1.0 / G_T, torch.full_like(G_T, 1e8)).expand(-1, K)
        * case1
    )
    G_u = G_grid.view(1, -1)
    w2 = torch.where(G_u > 1e-8, 1.0 / G_u, torch.full_like(G_u, 1e8)) * (u < T).float()
    w = w1 + w2
    return (w * (Y - S_hat) ** 2).sum(0) / torch.clamp(w.sum(0), min=1e-8)


def integrated_brier_score(S_hat, time, event, grid, G_grid=None) -> float:
    """IPCW integrated Brier score, the average Brier score over ``grid``."""
    return float(brier_scores(S_hat, time, event, grid, G_grid).mean())


def imse(S_hat, S_true) -> float:
    """Integrated mean squared error between predicted and true curves on a common grid."""
    S_hat = np.asarray(S_hat, dtype=float)
    S_true = np.asarray(S_true, dtype=float)
    return float(np.nanmean(np.nanmean((S_hat - S_true) ** 2, axis=1)))
