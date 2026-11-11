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
model = SCEDyn(variable_selection=True, seed=1).fit(train, val)
model.selected_variables_
```

The penalty level is chosen on `lambda_grid` by the validation integrated Brier
score, or fixed with `lambda_value`.

## Tuning constants

Every constant of the method is an argument of {class}`scedyn.SCEDyn`. The
defaults are the values of the simulation study of the paper (low censoring,
landmark 0.9).
