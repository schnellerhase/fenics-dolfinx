# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Adaptive FEM for the Poisson equation
#
# This demo illustrates how to:
#
# - Use residual-type error estimators
# - Handle adaptively refined mesh hierachies
# - Compute convergence rates
#
#
# ```{admonition} Download sources
# :class: download
# * {download}`Python script <./demo_poisson_adaptive.py>`
# * {download}`Jupyter notebook <./demo_poisson_adaptive.ipynb>`
# ```
# ## TODO
#
# ...
#
# +
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType  # type: ignore

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem

