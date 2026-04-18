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
# Copyright (C) 2026 Maximilian Brodbeck and Paul T. Kühner
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
#
# ## TODO
# ...
#
# +
from pathlib import Path

from mpi4py import MPI
from petsc4py.PETSc import ScalarType as stype  # type: ignore

import numpy as np

import basix.ufl
import ufl
from dolfinx import fem, io, mesh
from dolfinx.fem.petsc import LinearProblem

rtype = np.real(stype(0)).dtype

comm = MPI.COMM_WORLD

meshes = []
meshes.append(mesh.create_unit_square(comm, n := 10, n, dtype=stype))

tdim = meshes[0].topology.dim

ufl_domain = ufl.Mesh(basix.ufl.element("Lagrange", "triangle", 1, shape=(2,), dtype=rtype))
el = basix.ufl.element("Lagrange", "triangle", 1, dtype=stype)
V = ufl.FunctionSpace(ufl_domain, el)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = ufl.inner(u, v) * ufl.dx(domain=ufl_domain)
L = ufl.inner(-6, v) * ufl.dx(domain=ufl_domain)

a_compiled = fem.compile_form(comm, a, form_compiler_options={"scalar_type": stype})
L_compiled = fem.compile_form(comm, L, form_compiler_options={"scalar_type": stype})

for it in range(max_it := 3):
    print(f"AFEM it. {it}:")

    msh = meshes[it]

    V = fem.functionspace(msh, el)

    _a = fem.create_form(a_compiled, [V, V], msh, {}, {}, {})
    _L = fem.create_form(L_compiled, [V], msh, {}, {}, {})

    print(" SOLVE:", end="")
    fem.assemble_matrix(_a)
    fem.assemble_vector(_L)
    # problem = LinearProblem(_a, _L, petsc_options_prefix="solver")
    # problem.solve()
    print(" TODO")

    print(" ESTIMATE:")
    im_cell = msh.topology.index_map(tdim)
    marker = np.random.default_rng(0).random(im_cell.size_local + im_cell.num_ghosts)

    print(" MARK: ", end="")
    marked_cells = np.argwhere(
        marker >= (theta := 0.5) * comm.allreduce(np.max(marker), MPI.MAX)
    ).flatten()

    msh.topology.create_entities(1)
    marked_edges = mesh.compute_incident_entities(msh.topology, marked_cells, tdim, 1)
    print(f"{marked_cells.size} cells ({marked_edges.size} edges) marked for refinement")

    print(" REFINE:")
    msh_refined, _, _ = mesh.refine(msh, marked_edges)
    meshes.append(msh_refined)

    with io.XDMFFile(comm, f"output/it_{it}.xdmf", "w") as file:
        file.write_mesh(msh)
