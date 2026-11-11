# Changelog

All notable changes to this project are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the
project uses [Semantic Versioning](https://semver.org/).

## [0.1.0] - unreleased

### Added

- `SCEDyn`: self-consistent min–max estimator of the conditional residual
  survival curve at one landmark, with optional variable selection by stochastic
  gates.
- `LandmarkData`: input container for the risk set at one landmark.
- `scedyn.simulate`: the AFT, PH and PO data-generating processes of the paper,
  with exact reproduction of the simulated data and Monte Carlo oracle curves.
- `scedyn.metrics`: Kaplan–Meier estimator, IPCW Brier score, integrated Brier
  score and integrated mean squared error.
