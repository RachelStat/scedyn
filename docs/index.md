# scedyn

`scedyn` estimates the conditional residual survival curve at a landmark time
from baseline covariates and longitudinal histories, by self-consistent min–max
training of a residual-time generator and a weight network with LSTM encoders.

```{toctree}
:maxdepth: 2
:hidden:

usage
simulation
api
changelog
```

- {doc}`usage`: fitting the estimator to your own data.
- {doc}`simulation`: the data-generating processes and reproduction of the
  paper.
- {doc}`api`: reference of all public functions and classes.
