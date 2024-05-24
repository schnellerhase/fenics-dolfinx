from mpi4py import MPI

from petsc4py.PETSc import ScalarType 

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

comm = MPI.COMM_WORLD

msh_fine = mesh.create_rectangle(
    comm=comm,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(16, 16),
    cell_type=mesh.CellType.triangle,
)

V_fine = fem.functionspace(msh_fine, ("Lagrange", 1))

facets_fine = mesh.locate_entities_boundary(
    msh_fine,
    dim=(msh_fine.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
)

dofs_fine = fem.locate_dofs_topological(V=V_fine, entity_dim=1, entities=facets_fine)

bc_fine = fem.dirichletbc(value=ScalarType(0), dofs=dofs_fine, V=V_fine)

def variational_problem(V):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(V.mesh)
    f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    g = ufl.sin(5 * x[0])
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx + inner(g, v) * ds
    return a, L

a_fine, L_fine = variational_problem(V_fine)
problem = LinearProblem(a_fine, L_fine, bcs=[bc_fine], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()

with io.XDMFFile(comm, "out_twogrid/poisson.xdmf", "w") as file:
    file.write_mesh(msh_fine)
    file.write_function(uh)
