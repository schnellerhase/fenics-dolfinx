from mpi4py import MPI

from petsc4py.PETSc import ScalarType 
from petsc4py import PETSc

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

def create_bc(V, is_coarse=False):
    facets_fine = mesh.locate_entities_boundary(
        V.mesh,
        dim=(V.mesh.topology.dim - 1),
        marker=lambda x: np.isclose(x[0], 0.0)
                       | np.isclose(x[0], 1.0)
                       | np.isclose(x[1], 0.0)
                       | np.isclose(x[1], 1.0),
    )

    dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets_fine)
    if is_coarse:
        return fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
    else:
        bc_func = fem.Function(V)
        bc_func.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
        return fem.dirichletbc(bc_func, dofs)

bc_fine = create_bc(V_fine)
bc_coarse = create_bc(V_coarse, is_coarse=True) # they 

def variational_problem(V):
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a = inner(grad(u), grad(v)) * dx
    f = -6
    L = inner(f, v) * dx
    return a, L

a_coarse, L_coarse = variational_problem(V_coarse)
a_fine, L_fine = variational_problem(V_fine)

smoother_options = {"ksp_type": "richardson", "ksp_richardson_scale": 0.1, "pc_type": "none", "ksp_monitor": "", "ksp_max_it":10, "ksp_initial_guess_nonzero": True}
problem_fine = LinearProblem(a_fine, L_fine, bcs=[bc_fine], petsc_options=smoother_options)
u_fine = problem_fine.solve()

with io.XDMFFile(comm, "out_twogrid/fine-init.xdmf", "w") as file:
    file.write_mesh(msh_fine)
    file.write_function(u_fine)

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

with io.XDMFFile(comm, "out_twogrid/restricted-residual.xdmf", "w") as file:
    file.write_mesh(msh_coarse)
    file.write_function(residual_coarse)

# problem_coarse = LinearProblem(a_coarse, L_coarse, bcs=[bc_coarse], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
# u_coarse = problem_coarse.solve()

A_coarse = fem.petsc.create_matrix(fem.form(a_coarse))
A_coarse.zeroEntries()
fem.petsc.assemble_matrix_mat(A_coarse, fem.form(a_coarse), bcs=[bc_coarse])
A_coarse.assemble()

coarse_solver = PETSc.KSP().create(comm)
coarse_solver.setType("preonly")
coarse_solver.getPC().setType("lu")
coarse_solver.setOperators(A_coarse)

# apply boundary conditions to coarse vector 
fem.petsc.apply_lifting(residual_coarse.x.petsc_vec, [fem.form(a_coarse)], bcs=[[bc_coarse]])
residual_coarse.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.petsc.set_bc(residual_coarse.x.petsc_vec, [bc_coarse])

coarse_grid_correction = fem.Function(V_coarse)
coarse_solver.solve(residual_coarse.x.petsc_vec, coarse_grid_correction.x.petsc_vec)
coarse_grid_correction.x.scatter_forward()

print(f"Coarse grid correction (post solve): {coarse_grid_correction.vector.norm():e}")

with io.XDMFFile(comm, "out_twogrid/coarse-correction.xdmf", "w") as file:
    file.write_mesh(msh_coarse)
    file.write_function(coarse_grid_correction)

# Petform prolongation by non matching cell interpolation
fine_mesh_cell_map = msh_fine.topology.index_map(msh_fine.topology.dim)
num_cells_on_proc = fine_mesh_cell_map.size_local + fine_mesh_cell_map.num_ghosts
cells = np.arange(num_cells_on_proc, dtype=np.int32)
interpolation_data = fem.create_interpolation_data(V_fine, V_coarse, cells, padding=1e-14)

residual_fine.interpolate_nonmatching(coarse_grid_correction, cells, interpolation_data=interpolation_data)

# Update fine solution: u_fine -= residual_fine
u_fine.vector.axpy(-1, residual_fine.vector)

# recompute residual
A_fine.multAdd(u_fine.vector, b_fine, residual_fine.x.petsc_vec)
print(f"Fine residual (pre smoothing): {residual_fine.vector.norm():e}")

# smoothing
# problem_fine.solve()
# TODO: can't use this one here either -> must use smoothing by hand, this overwrites 
problem_fine = LinearProblem(a_fine, L_fine, bcs=[bc_fine], u=u_fine, petsc_options=smoother_options)
u_fine = problem_fine.solve()
A_fine = problem_fine.A
b_fine = problem_fine.b
# recompute residual
A_fine.multAdd(u_fine.vector, b_fine, residual_fine.x.petsc_vec)
print(f"Fine residual (post smoothing): {residual_fine.vector.norm():e}")

with io.XDMFFile(comm, "out_twogrid/fine.xdmf", "w") as file:
    file.write_mesh(msh_fine)
    file.write_function(u_fine)