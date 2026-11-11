# scedyn

`scedyn` implements the self-consistent estimator for dynamic survival
prediction of _Dynamic prediction via self-consistent estimation_ (Du, Kim, Wang
and Lu). At a landmark time τ it estimates, for every subject still at risk, the
whole conditional residual survival curve

S_τ(s | H) = P(T − τ > s | T > τ, X, Z(t), t ≤ τ)

from baseline covariates X and the longitudinal history Z up to τ, without a
model for the longitudinal process and without a parametric survival family. A
residual-time generator and a weight network, each with an LSTM encoder of the
history, are trained by a min–max problem built on the self-consistent equation,
and optional stochastic gates select baseline and longitudinal variables.

The package contains

| Module                | Content                                                                        |
| --------------------- | ------------------------------------------------------------------------------ |
| `scedyn.SCEDyn`       | the estimator: training, prediction and variable selection                     |
| `scedyn.LandmarkData` | the input format, the risk set at one landmark                                 |
| `scedyn.simulate`     | the data-generating processes of the paper (AFT, PH, PO) and the oracle curves |
| `scedyn.metrics`      | integrated Brier score (IPCW) and integrated mean squared error                |

## 1. Installation

```bash
pip install git+https://github.com/RachelStat/scedyn.git
# or, from a local copy
pip install .
```

Python ≥ 3.10 and PyTorch ≥ 2.0 are required. A GPU is used automatically when
available.

## 2. Analysing your own data

The input is the risk set at one landmark τ, already processed: every subject
has survived to τ, the outcome is the residual time after τ, and the
longitudinal history is truncated at τ. Fit one model per landmark.

```python
import numpy as np
from scedyn import LandmarkData, SCEDyn

train = LandmarkData(
    X=X_train,  # (n, p) baseline covariates
    Z=Z_train,  # list of n arrays of shape (m_i, q): measurements up to tau, in visit order
    time=time_train,  # (n,) observed residual time after tau
    event=event_train,  # (n,) 1 = event, 0 = censoring
    baseline_names=["age", "sex", ...],  # optional
    longitudinal_names=["sofa", "gcs", ...],  # optional
)
val = LandmarkData(...)  # held-out subjects, for early stopping
test = LandmarkData(X=X_test, Z=Z_test)  # outcomes not needed for prediction

model = SCEDyn(seed=1).fit(train, val)
times = np.linspace(0.1, 10, 50)
S = model.predict_survival(test, times)  # array (n_test, 50)
```

The number of visits m_i may differ across subjects. Inputs are used as given,
so scale them beforehand if their ranges differ widely.

**Variable selection.** With `variable_selection=True`, give either
`lambda_grid`, candidate penalty levels among which the one with the best
validation integrated Brier score is chosen, or `lambda_value`, a fixed penalty
level on the same scale. The simulation study of the paper uses the grid 0.40,
0.45, …, 0.70.

```python
grid = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
model = SCEDyn(variable_selection=True, lambda_grid=grid, seed=1).fit(train, val)
print(model.selected_variables_)  # {'baseline': [...], 'longitudinal': [...]}
print(model.selection_.lambda_value)  # the chosen level, reusable as lambda_value
```

**Tuning.** Every constant of the method is an argument of `SCEDyn`, and the
defaults are the values of the simulation study (low censoring, landmark 0.9).
See `help(SCEDyn)` for the full list. The ones most often changed are

| Argument          | Default | Meaning                                          |
| ----------------- | ------- | ------------------------------------------------ |
| `embedding_dim`   | 6       | dimension of the history embedding               |
| `lstm_hidden_dim` | 216     | LSTM hidden size                                 |
| `hidden_dim`      | 128     | width of the generator head                      |
| `max_epochs`      | 200     | number of training epochs                        |
| `batch_size`      | 16      | minibatch size                                   |
| `n_generated`     | 500     | generated residual times per subject in training |
| `time_clip`       | None    | upper bound on generated times in validation     |

`model.save(path)` and `SCEDyn.load(path)` store and restore a fitted model.

## 3. Reproducing the simulation study

`scedyn.simulate.Simulation` regenerates the simulated data of the paper. With
the default arguments the covariates, longitudinal measurements, event and
censoring times, train/validation/test splits and evaluation grids are identical
to those of the paper. All parameters and seeds are arguments
(`help(Simulation)`, `scedyn.simulate.default_seeds`).

A cell of the simulation tables is a survival model (`"AFT"`, `"PH"`, `"PO"`), a
censoring level (`"low"`, `"high"`) and a landmark τ ∈ {0.6, 0.9, 1.2},
summarized over replications r = 1, …, 20. The settings that change with the
landmark are

| τ   | `embedding_dim` | `max_epochs` |
| --- | --------------- | ------------ |
| 0.6 | 4               | 180          |
| 0.9 | 6               | 200          |
| 1.2 | 8               | 220          |

and in every cell the training seed is `2025 + r`, generated times are clipped
at `censor_max − τ`, and test curves use 1000 generated times.

```python
import numpy as np
from scedyn import SCEDyn, imse
from scedyn.simulate import Simulation

model_name, censoring, tau = "PO", "low", 0.6
settings = {0.6: (4, 180), 0.9: (6, 200), 1.2: (8, 220)}
emb, epochs = settings[tau]

sim = Simulation(model_name, censoring)
clip = sim.params["censor_max"] - tau


def run(r, **kwargs):
    split = sim.landmark_splits(r)[tau]  # all landmarks are split, as in the paper
    fit = SCEDyn(
        embedding_dim=emb, max_epochs=epochs, time_clip=clip, seed=2025 + r, **kwargs
    ).fit(split.train, split.val)
    S_hat = fit.predict_survival(
        split.test, split.eval_grid, n_samples=1000, time_clip=clip
    )
    return fit, imse(S_hat, split.true_survival)


# main setting: all 54 inputs with variable selection; the penalty level is
# chosen by grid search in replication 1 and reused in replications 2-20
grid = (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
fit1, imse1 = run(1, variable_selection=True, lambda_grid=grid)
lam = fit1.selection_.lambda_value
scores = [imse1] + [
    run(r, variable_selection=True, lambda_value=lam)[1] for r in range(2, 21)
]
print(np.mean(scores), np.std(scores))
```

For the comparison without variable selection, generate the informative inputs
only with `sim.landmark_splits(r, covariates="informative")` and fit with the
default `variable_selection=False`.

**Exact and statistical reproduction.** The simulated data are reproduced bit
for bit, and the oracle curves up to Monte Carlo error. Training a neural
network on a GPU is not bitwise deterministic, so a rerun with the same seed
gives a slightly different fitted model; the IMSE of a cell agrees with the
paper in distribution over replications rather than digit for digit.

## Citation

> Du, Q., Kim, S., Wang, R. and Lu, W. Dynamic prediction via self-consistent
> estimation.
