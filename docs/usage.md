# Usage

## Input data

The estimator works at one landmark time $\tau$. The input is the risk set at
$\tau$, already processed: every subject has survived to $\tau$, the outcome is
the residual time after $\tau$, and the longitudinal history is truncated at
$\tau$. It is stored in a {class}`scedyn.LandmarkData`:

```python
from scedyn import LandmarkData

train = LandmarkData(
    X=X_train,  # (n, p) baseline covariates
    Z=Z_train,  # list of n arrays (m_i, q): measurements up to tau, in visit order
    time=time_train,  # (n,) observed residual time after tau
    event=event_train,  # (n,) 1 = event, 0 = censoring
)
```

The number of visits $m_i$ may differ across subjects. Inputs are used as given,
so scale them beforehand if their ranges differ widely.

## Fitting and prediction

```python
import numpy as np
from scedyn import SCEDyn

model = SCEDyn(seed=1).fit(train, val)
S = model.predict_survival(test, np.linspace(0.1, 10, 50))
```

`val` is a held-out set used for early stopping on the integrated Brier score.
`predict_survival` returns the estimated $P(T - \tau > s \mid H)$ for each
subject and each time $s$.

## Variable selection

```python
grid = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
model = SCEDyn(variable_selection=True, lambda_grid=grid, seed=1).fit(train, val)
model.selected_variables_
```

With `variable_selection=True`, either `lambda_grid` (candidate penalty levels,
chosen by the validation integrated Brier score) or `lambda_value` (a fixed
penalty level on the same scale) is required. If both are given, `lambda_value`
is used. Without variable selection both are ignored, with a warning.

## Tuning constants

Every constant of the method is an argument of {class}`scedyn.SCEDyn`. The
defaults are the values of the simulation study of the paper (low censoring,
landmark 0.9).
