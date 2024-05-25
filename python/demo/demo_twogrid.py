from mpi4py import MPI

from petsc4py.PETSc import ScalarType 

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

comm = MPI.COMM_WORLD

create_mesh = lambda n: mesh.create_rectangle(
    comm=comm,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(n, n),
    cell_type=mesh.CellType.triangle,
)

msh_coarse = create_mesh(8)
msh_fine = create_mesh(16)

V_coarse = fem.functionspace(msh_coarse, ("Lagrange", 1))
V_fine = fem.functionspace(msh_fine, ("Lagrange", 1))

def create_bc(V):
    facets_fine = mesh.locate_entities_boundary(
        V.mesh,
        dim=(V.mesh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 1.0),
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets_fine)
    return fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)

bc_fine = create_bc(V_fine)

def variational_problem(V):
    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    x = ufl.SpatialCoordinate(V.mesh)
    f = 10 * ufl.exp(-((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
    g = ufl.sin(5 * x[0])
    a = inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx + inner(g, v) * ds
    return a, L

a_coarse, L_coarse = variational_problem(V_coarse)
a_fine, L_fine = variational_problem(V_fine)

problem_fine = LinearProblem(a_fine, L_fine, bcs=[bc_fine], petsc_options= {"ksp_type": "chebyshev", "pc_type": "none", "ksp_max_it": 10, "ksp_monitor": ""})
u_fine = problem_fine.solve()

with io.XDMFFile(comm, "out_twogrid/fine.xdmf", "w") as file:
    file.write_mesh(msh_fine)
    file.write_function(u_fine)

problem_coarse = LinearProblem(a_fine, L_fine, bcs=[bc_fine], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_coarse = problem_coarse.solve()

with io.XDMFFile(comm, "out_twogrid/coarse.xdmf", "w") as file:
    file.write_mesh(msh_coarse)
    file.write_function(u_coarse)
