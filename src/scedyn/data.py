"""Data container for one landmark.

The package works on data that have already been processed at a single
landmark time tau: every subject is in the risk set at tau, the outcome is
the residual time after tau, and the longitudinal history is truncated at tau.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LandmarkData:
    """Risk-set data at one landmark.

    Parameters
    ----------
    X : array of shape (n, p)
        Baseline covariates.
    Z : list of ndarray
        One array of shape (m_i, q) per subject: the longitudinal measurements
        observed up to the landmark, ordered by visit. The number of visits
        m_i may differ across subjects.
    time : array of shape (n,)
        Observed residual time after the landmark.
    event : array of shape (n,)
        Event indicator, 1 for an observed event and 0 for censoring.
    ids : array of shape (n,), optional
        Subject identifiers. Defaults to 0, ..., n - 1.
    baseline_names, longitudinal_names : list of str, optional
        Variable names, used to report the selected variables.
    """

    X: np.ndarray
    Z: list[np.ndarray]
    time: np.ndarray | None = None
    event: np.ndarray | None = None
    ids: np.ndarray | None = None
    baseline_names: list[str] | None = None
    longitudinal_names: list[str] | None = None
    extra: dict = field(default_factory=dict, repr=False)

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float32)
        if self.X.ndim != 2:
            msg = "X must be a 2-d array of shape (n, p)."
            raise ValueError(msg)
        n = self.X.shape[0]
        if len(self.Z) != n:
            msg = "Z must contain one array per subject."
            raise ValueError(msg)
        self.Z = [np.asarray(z, dtype=np.float32).reshape(len(z), -1) for z in self.Z]
        q = {z.shape[1] for z in self.Z if len(z) > 0}
        if len(q) > 1:
            msg = "All longitudinal arrays must have the same number of columns."
            raise ValueError(msg)
        if any(len(z) == 0 for z in self.Z):
            msg = "Every subject needs at least one longitudinal measurement."
            raise ValueError(msg)
        if self.time is not None:
            self.time = np.asarray(self.time, dtype=np.float32).reshape(-1)
            if len(self.time) != n:
                msg = "time must have length n."
                raise ValueError(msg)
        if self.event is not None:
            self.event = np.asarray(self.event, dtype=np.float32).reshape(-1)
            if len(self.event) != n:
                msg = "event must have length n."
                raise ValueError(msg)
        self.ids = np.arange(n) if self.ids is None else np.asarray(self.ids)
        if self.baseline_names is None:
            self.baseline_names = [f"X_{j + 1}" for j in range(self.X.shape[1])]
        if self.longitudinal_names is None:
            self.longitudinal_names = [f"Z_{j + 1}" for j in range(self.n_longitudinal)]

    @property
    def n(self) -> int:
        return self.X.shape[0]

    @property
    def n_baseline(self) -> int:
        return self.X.shape[1]

    @property
    def n_longitudinal(self) -> int:
        return self.Z[0].shape[1]

    @property
    def lengths(self) -> np.ndarray:
        return np.array([len(z) for z in self.Z], dtype=np.int64)

    def padded_Z(self) -> np.ndarray:
        """Zero-padded array of shape (n, max_i m_i, q)."""
        m = int(self.lengths.max())
        out = np.zeros((self.n, m, self.n_longitudinal), dtype=np.float32)
        for i, z in enumerate(self.Z):
            out[i, : len(z)] = z
        return out

    def subset(self, index: Sequence[int]) -> LandmarkData:
        """Data of the subjects at positions ``index``."""
        idx = np.asarray(index)
        return LandmarkData(
            X=self.X[idx],
            Z=[self.Z[i] for i in idx],
            time=None if self.time is None else self.time[idx],
            event=None if self.event is None else self.event[idx],
            ids=np.asarray(self.ids)[idx],
            baseline_names=list(self.baseline_names or []),
            longitudinal_names=list(self.longitudinal_names or []),
            extra={
                k: [v[i] for i in idx] if isinstance(v, list) else v
                for k, v in self.extra.items()
            },
        )

    def select_columns(
        self, baseline: Sequence[int], longitudinal: Sequence[int]
    ) -> LandmarkData:
        """Keep only the given baseline and longitudinal columns."""
        baseline = list(baseline)
        longitudinal = list(longitudinal)
        b_names = list(self.baseline_names or [])
        l_names = list(self.longitudinal_names or [])
        return LandmarkData(
            X=self.X[:, baseline],
            Z=[z[:, longitudinal] for z in self.Z],
            time=self.time,
            event=self.event,
            ids=self.ids,
            baseline_names=[b_names[j] for j in baseline],
            longitudinal_names=[l_names[j] for j in longitudinal],
            extra=dict(self.extra),
        )
