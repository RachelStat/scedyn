import numpy as np
import pytest
import torch

from scedyn import LandmarkData, SCEDyn, imse, integrated_brier_score, km_survival
from scedyn.networks import Generator, WeightNetwork
from scedyn.simulate import Simulation


def _toy_data(n, p=3, q=2, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    Z = [rng.normal(size=(rng.integers(1, 6), q)) for _ in range(n)]
    time = rng.exponential(1.0, size=n) + 0.01
    event = (rng.random(n) < 0.7).astype(float)
    return LandmarkData(X=X, Z=Z, time=time, event=event)


SMALL = {
    "n_generated": 10,
    "n_eval_times": 8,
    "hidden_dim": 8,
    "lstm_hidden_dim": 4,
    "phi_hidden_dim": 8,
    "max_epochs": 4,
    "patience": 1,
    "device": "cpu",
    "seed": 0,
    "verbose": 0,
}


def test_landmark_data_checks():
    with pytest.raises(ValueError, match="one array per subject"):
        LandmarkData(X=np.zeros((2, 1)), Z=[np.zeros((1, 1))])
    d = _toy_data(5)
    assert d.padded_Z().shape[0] == 5
    sub = d.subset([0, 2])
    assert sub.n == 2
    assert np.allclose(sub.X, d.X[[0, 2]])


def test_kaplan_meier_without_censoring():
    t = torch.tensor([1.0, 2.0, 3.0, 4.0])
    S = km_survival(t, torch.ones(4), torch.tensor([0.5, 1.0, 2.5, 4.0]))
    assert torch.allclose(S, torch.tensor([1.0, 0.75, 0.5, 0.0]))


def test_generator_repeats_matches_tiled_inputs():
    torch.manual_seed(0)
    G = Generator(3, 4, 2, hidden_dim=8, lstm_hidden_dim=5, embedding_dim=3).eval()
    x, z = torch.randn(3, 4), torch.randn(3, 6, 2)
    lens = torch.tensor([6, 2, 4])
    u = torch.rand(3 * 5, 3)
    fast = G(u, x, z, lens, repeats=5)
    slow = G(
        u,
        x.repeat_interleave(5, 0),
        z.repeat_interleave(5, 0),
        lens.repeat_interleave(5, 0),
    )
    assert torch.allclose(fast, slow, atol=1e-6)


def test_weight_network_repeats_matches_tiled_inputs():
    torch.manual_seed(0)
    phi = WeightNetwork(4, 2, hidden_dim=8, lstm_hidden_dim=5, embedding_dim=3).eval()
    x, z = torch.randn(3, 4), torch.randn(3, 6, 2)
    lens = torch.tensor([6, 2, 4])
    s = torch.randn(3 * 7, 1)
    fast = phi(s, x, z, lens, repeats=7)
    slow = phi(
        s,
        x.repeat_interleave(7, 0),
        z.repeat_interleave(7, 0),
        lens.repeat_interleave(7, 0),
    )
    assert torch.allclose(fast, slow, atol=1e-6)


def test_fit_and_predict():
    train, val, test = (
        _toy_data(80, seed=1),
        _toy_data(30, seed=2),
        _toy_data(10, seed=3),
    )
    model = SCEDyn(**SMALL).fit(train, val)
    times = np.linspace(0.1, 3, 12)
    S = model.predict_survival(test, times, n_samples=50)
    assert S.shape == (10, 12)
    assert np.all(np.diff(S, axis=1) <= 1e-12)
    assert 0.0 <= integrated_brier_score(S, test.time, test.event, times) <= 1.0
    assert model.selected_variables_ is None


def test_variable_selection_runs():
    train, val = _toy_data(80, seed=1), _toy_data(30, seed=2)
    model = SCEDyn(
        **SMALL,
        variable_selection=True,
        vs_warmup_epochs=2,
        lambda_grid=(0.5,),
        max_vs_epochs=10,
        min_vs_epochs=1,
        stable_epochs=1,
        lambda_anneal_epochs=2,
        post_selection_epochs=1,
        require_drop=False,
    ).fit(train, val)
    sel = model.selected_variables_
    assert len(sel["baseline"]) >= 1
    assert len(sel["longitudinal"]) >= 1


def test_simulation_is_deterministic():
    a = Simulation("AFT", "low", n_patients=300)
    b = Simulation("AFT", "low", n_patients=300)
    assert np.array_equal(a.X, b.X)
    assert np.array_equal(a.Z, b.Z)
    sa = a.landmark_splits(2, true_survival=False)
    sb = b.landmark_splits(2, true_survival=False)
    for tau in sa:
        assert np.array_equal(sa[tau].train.ids, sb[tau].train.ids)
        assert np.array_equal(sa[tau].eval_grid, sb[tau].eval_grid)


def test_oracle_is_a_survival_curve():
    sim = Simulation("PH", "low", n_patients=300)
    split = sim.landmark_splits(1, landmarks=[0.6], true_survival=True, n_mc=200)[0.6]
    S = split.true_survival
    assert S.shape == (split.test.n, len(split.eval_grid))
    assert np.all((S >= 0) & (S <= 1 + 1e-9))
    assert imse(S, S) == 0.0
