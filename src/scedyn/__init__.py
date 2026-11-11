"""scedyn: self-consistent estimation for dynamic survival prediction."""

from . import simulate
from .data import LandmarkData
from .metrics import brier_scores, imse, integrated_brier_score, km_survival
from .model import SCEDyn, SelectionResult

__version__ = "0.1.0"

__all__ = [
    "LandmarkData",
    "SCEDyn",
    "SelectionResult",
    "__version__",
    "brier_scores",
    "imse",
    "integrated_brier_score",
    "km_survival",
    "simulate",
]
