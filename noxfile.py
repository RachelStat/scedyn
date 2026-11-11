#!/usr/bin/env -S uv run --script
# /// script
# dependencies = ["nox>=2025.2.9"]
# ///
"""Task runner: ``nox -s lint``, ``nox -s tests``, ``nox -s docs``, ``nox -s build``."""

from __future__ import annotations

import shutil
from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session(default=True)
def lint(session: nox.Session) -> None:
    """Run the linters through pre-commit."""
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session(default=False)
def pylint(session: nox.Session) -> None:
    """Run PyLint."""
    session.install(".", "pylint>=3.2")
    session.run("pylint", "scedyn", *session.posargs)


@nox.session(default=True)
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    session.install(".", "pytest>=6")
    session.run("pytest", *session.posargs)


@nox.session(reuse_venv=True, default=False)
def docs(session: nox.Session) -> None:
    """Build the documentation into ``docs/_build/html``."""
    session.install(
        ".", "sphinx>=7", "myst-parser>=2", "numpydoc>=1.6", "furo>=2023.08.17"
    )
    shutil.rmtree(DIR / "docs" / "_build", ignore_errors=True)
    session.chdir("docs")
    session.run("sphinx-build", "-n", "-T", "-b", "html", ".", "_build/html")


@nox.session(default=False)
def build(session: nox.Session) -> None:
    """Build the sdist and the wheel and check their metadata."""
    shutil.rmtree(DIR / "dist", ignore_errors=True)
    session.install("build", "twine")
    session.run("python", "-m", "build")
    session.run("twine", "check", "--strict", "dist/*")


if __name__ == "__main__":
    nox.main()
