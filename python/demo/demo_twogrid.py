from mpi4py import MPI

from petsc4py.PETSc import ScalarType 

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, la
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner

comm = MPI.COMM_WORLD

create_mesh = lambda n: mesh.create_rectangle(
    comm=comm,
    points=((0.0, 0.0), (1.0, 1.0)),
    n=(n, n),
    cell_type=mesh.CellType.triangle,
)

msh_coarse = create_mesh(16)
msh_fine = create_mesh(32)

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

# implement the restriction of fine problem to coarse via non matching mesh interpolation, i.e. interpolate residual of fine problem on coarse mesh
A_fine = problem_fine.A
b_fine = problem_fine.b

# residual_fine = A_fine u_fine - b_fine
residual_fine = fem.Function(V_fine)
b_fine *= -1
A_fine.multAdd(u_fine.vector, b_fine, residual_fine.x.petsc_vec)

print(f"Fine residual: {residual_fine.vector.norm():e}")

# Petform restriction by non matching cell interpolation
# checkout test_interpolation.py for this in further detail
coarse_mesh_cell_map = msh_coarse.topology.index_map(msh_coarse.topology.dim)
num_cells_on_proc = coarse_mesh_cell_map.size_local + coarse_mesh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)
interpolation_data = fem.create_interpolation_data(V_coarse, V_fine, cells, padding=1e-14)

residual_coarse = fem.Function(V_coarse)
residual_coarse.interpolate_nonmatching(residual_fine, cells, interpolation_data=interpolation_data)

print(f"Coarse residual (pre solve): {residual_coarse.vector.norm():e}")

problem_coarse = LinearProblem(a_fine, L_fine, bcs=[bc_fine], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
u_coarse = problem_coarse.solve()

print(f"Coarse residual (post solve): {u_coarse.vector.norm():e}")

with io.XDMFFile(comm, "out_twogrid/coarse.xdmf", "w") as file:
    file.write_mesh(msh_coarse)
    file.write_function(u_coarse)

# Petform prolongation by non matching cell interpolation
fine_mesh_cell_map = msh_fine.topology.index_map(msh_fine.topology.dim)
num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)
interpolation_data = fem.create_interpolation_data(V_fine, V_coarse, cells, padding=1e-14)

residual_fine.interpolate_nonmatching(residual_coarse, cells, interpolation_data=interpolation_data)

# Update fine solution: u_fine -= residual_fine
u_fine.vector.axpy(-1, residual_fine.vector)

# recompute residual
A_fine.multAdd(u_fine.vector, b_fine, residual_fine.x.petsc_vec)
print(f"Fine residual (pre smoothing): {residual_fine.vector.norm():e}")

# smoothing
# problem_fine.solve()
problem_fine = LinearProblem(a_fine, L_fine, bcs=[bc_fine], u=u_fine, petsc_options= {"ksp_type": "chebyshev", "pc_type": "none", "ksp_max_it": 10, "ksp_monitor": ""})
u_fine = problem_fine.solve()
A_fine = problem_fine.A
b_fine = problem_fine.b
# recompute residual
A_fine.multAdd(u_fine.vector, b_fine, residual_fine.x.petsc_vec)
print(f"Fine residual (post smoothing): {residual_fine.vector.norm():e}")

with io.XDMFFile(comm, "out_twogrid/fine.xdmf", "w") as file:
    file.write_mesh(msh_fine)
    file.write_function(u_fine)