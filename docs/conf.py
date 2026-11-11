"""Sphinx configuration."""

from __future__ import annotations

import importlib.metadata

project = "scedyn"
copyright = "2026, Qiurui Du, Sehwan Kim, Rui Wang and Wenbin Lu"
author = "Qiurui Du, Sehwan Kim, Rui Wang and Wenbin Lu"
version = release = importlib.metadata.version("scedyn")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "numpydoc",
]

source_suffix = [".rst", ".md"]
exclude_patterns = [
    "_build",
    "**.ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
    ".env",
    ".venv",
]

html_theme = "furo"

myst_enable_extensions = ["colon_fence", "dollarmath"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://pytorch.org/docs/stable", None),
}

autodoc_member_order = "bysource"
autodoc_typehints = "none"

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    "LandmarkData": "scedyn.LandmarkData",
    "SelectionResult": "scedyn.SelectionResult",
    "Simulation": "scedyn.simulate.Simulation",
    "ndarray": "numpy.ndarray",
    "array": "numpy.ndarray",
    "Tensor": "torch.Tensor",
}
numpydoc_xref_ignore = {
    "of",
    "shape",
    "optional",
    "default",
    "or",
    "each",
    "n",
    "p",
    "q",
    "m_i",
    "K",
}

nitpick_ignore = [
    ("py:class", "_io.StringIO"),
    ("py:class", "_io.BytesIO"),
]
