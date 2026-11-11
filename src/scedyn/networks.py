"""Residual-time generator and weight network."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _encode(lstm: nn.LSTM, proj: nn.Linear, z, lengths):
    packed = pack_padded_sequence(
        z, lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    _, (h_n, _) = lstm(packed)
    return proj(h_n[-1])


class Generator(nn.Module):
    """Residual-time generator G(U, X, Z) with stochastic gates on the inputs.

    The sequential encoder is an LSTM followed by a linear projection to the
    embedding of dimension ``embedding_dim``. The feed-forward head maps
    (U, gated X, embedding) to log residual time.
    """

    alive_B: torch.Tensor
    alive_L: torch.Tensor

    def __init__(
        self,
        noise_dim,
        n_baseline,
        n_longitudinal,
        hidden_dim=128,
        lstm_hidden_dim=216,
        lstm_layers=1,
        embedding_dim=6,
        gate_sigma_baseline=0.3,
        gate_sigma_longitudinal=0.3,
        gate_init_mu_baseline=0.65,
        gate_init_mu_longitudinal=0.6,
        gate_threshold=0.5,
    ):
        super().__init__()
        self.noise_dim = noise_dim
        self.lstm = nn.LSTM(
            n_longitudinal,
            lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.proj = nn.Linear(lstm_hidden_dim, embedding_dim)
        self.fc_ux = nn.Linear(noise_dim + n_baseline + embedding_dim, hidden_dim)
        self.fc_1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

        self.sigma_B = float(gate_sigma_baseline)
        self.sigma_L = float(gate_sigma_longitudinal)
        self.init_sigma = (self.sigma_B, self.sigma_L)
        self.mu_B = nn.Parameter(
            torch.full((n_baseline,), float(gate_init_mu_baseline))
        )
        self.mu_L = nn.Parameter(
            torch.full((n_longitudinal,), float(gate_init_mu_longitudinal))
        )
        self.gate_threshold = float(gate_threshold)
        self.w_L = 1.0
        self.frozen_B = False
        self.frozen_L = False
        self.register_buffer("alive_B", torch.ones(n_baseline, dtype=torch.bool))
        self.register_buffer("alive_L", torch.ones(n_longitudinal, dtype=torch.bool))

    # gates ---------------------------------------------------------------
    def _gate(self, mu, sigma, alive):
        if (not self.training) or sigma <= 0.0:
            g = torch.clamp(mu, 0.0, 1.0)
        else:
            g = torch.clamp(mu + torch.randn_like(mu) * sigma, 0.0, 1.0)
        return g * alive.float()

    def sample_gates(self):
        return (
            self._gate(self.mu_B, self.sigma_B, self.alive_B),
            self._gate(self.mu_L, self.sigma_L, self.alive_L),
        )

    def penalty(self):
        """sum_j Phi(mu_j / sigma_B) + w_L sum_k Phi(mu_k / sigma_L) over alive gates."""
        normal = torch.distributions.Normal(0.0, 1.0)
        reg = torch.zeros((), device=self.mu_B.device)
        if self.sigma_B > 0 and self.alive_B.any():
            reg = reg + normal.cdf(self.mu_B[self.alive_B] / self.sigma_B).sum()
        if self.sigma_L > 0 and self.alive_L.any():
            reg = (
                reg
                + self.w_L * normal.cdf(self.mu_L[self.alive_L] / self.sigma_L).sum()
            )
        return reg

    @torch.no_grad()
    def drop(self, idx_B=(), idx_L=()):
        for j in idx_B:
            self.alive_B[j] = False
            self.mu_B.data[j] = 0.0
        for k in idx_L:
            self.alive_L[k] = False
            self.mu_L.data[k] = 0.0

    @torch.no_grad()
    def enforce_eliminated(self):
        self.mu_B.data[~self.alive_B] = 0.0
        self.mu_L.data[~self.alive_L] = 0.0

    @torch.no_grad()
    def freeze(self, group, threshold=None):
        thr = self.gate_threshold if threshold is None else threshold
        if group in ("baseline", "both"):
            self.mu_B.data = (self.mu_B.data >= thr).float()
            self.mu_B.requires_grad_(False)
            self.sigma_B = 0.0
            self.frozen_B = True
        if group in ("longitudinal", "both"):
            self.mu_L.data = (self.mu_L.data >= thr).float()
            self.mu_L.requires_grad_(False)
            self.sigma_L = 0.0
            self.frozen_L = True

    def unfreeze(self):
        self.sigma_B, self.sigma_L = self.init_sigma
        self.frozen_B = self.frozen_L = False
        self.mu_B.requires_grad_(True)
        self.mu_L.requires_grad_(True)

    # forward -------------------------------------------------------------
    def forward(self, u, x, z, lengths, repeats: int = 1):
        """Generate residual times.

        ``x``, ``z`` and ``lengths`` describe ``n`` subjects, and ``u`` holds
        ``n * repeats`` noise vectors ordered subject by subject, so that rows
        ``i * repeats, ..., (i + 1) * repeats - 1`` of the output are draws for
        subject ``i``. The history is encoded once per subject, which is
        equivalent to encoding each repeated row because one gate sample is
        shared by the whole call.
        """
        zB, zL = self.sample_gates()
        x = x * zB.unsqueeze(0)
        z = z * zL.view(1, 1, -1)
        emb = _encode(self.lstm, self.proj, z, lengths)
        if repeats > 1:
            x = x.repeat_interleave(repeats, dim=0)
            emb = emb.repeat_interleave(repeats, dim=0)
        h = self.fc_ux(torch.cat((u, x, emb), dim=1))
        h = self.relu(self.fc_1(h))
        h = self.relu(self.fc_2(h))
        return torch.exp(self.fc_3(h))


class WeightNetwork(nn.Module):
    """Weight network phi(s, X, Z) with output in (1e-4, 1 + 1e-4).

    After variable selection the unselected inputs are masked out.
    """

    mask_B: torch.Tensor
    mask_L: torch.Tensor

    def __init__(
        self,
        n_baseline,
        n_longitudinal,
        hidden_dim=1280,
        lstm_hidden_dim=216,
        lstm_layers=1,
        embedding_dim=6,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            n_longitudinal,
            lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
        )
        self.proj = nn.Linear(lstm_hidden_dim, embedding_dim)
        self.fc_x1 = nn.Linear(1 + n_baseline + embedding_dim, hidden_dim)
        self.fc_x2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_x3 = nn.Linear(hidden_dim, 1)
        self.act = nn.Tanh()
        self.register_buffer("mask_B", torch.ones(n_baseline))
        self.register_buffer("mask_L", torch.ones(n_longitudinal))
        self.mask_active = False

    def set_masks(self, mask_B, mask_L):
        self.mask_B.copy_(mask_B)
        self.mask_L.copy_(mask_L)
        self.mask_active = True

    def reset_masks(self):
        self.mask_B.fill_(1.0)
        self.mask_L.fill_(1.0)
        self.mask_active = False

    def forward(self, s, x, z, lengths, repeats: int = 1, gates=None):
        """Weights phi(s, X, Z).

        ``s`` holds ``n * repeats`` standardized times ordered subject by
        subject, matching ``repeats`` copies of each of the ``n`` subjects.
        """
        if self.mask_active:
            x = x * self.mask_B.unsqueeze(0)
            z = z * self.mask_L.view(1, 1, -1)
        elif gates is not None:
            x = x * gates[0].unsqueeze(0)
            z = z * gates[1].view(1, 1, -1)
        emb = _encode(self.lstm, self.proj, z, lengths)
        if repeats > 1:
            x = x.repeat_interleave(repeats, dim=0)
            emb = emb.repeat_interleave(repeats, dim=0)
        h = self.act(self.fc_x1(torch.cat((s, x, emb), dim=1)))
        h = self.act(self.fc_x2(h))
        return torch.sigmoid(self.fc_x3(h) - 2) + 1e-4
