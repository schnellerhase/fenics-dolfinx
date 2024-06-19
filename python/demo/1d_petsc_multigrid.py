from mpi4py import MPI
from petsc4py import PETSc
import ufl
import numpy as np
from dolfinx import fem, mesh

from dolfinx.fem import petsc

comm = MPI.COMM_WORLD

n = 32
msh = mesh.create_interval(comm, (n), (0, 1))
V = fem.functionspace(msh, ("Lagrange", 1))

facets = mesh.locate_entities_boundary(
    V.mesh,
    dim=(V.mesh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0)
                    | np.isclose(x[0], 1.0)
                    | np.isclose(x[1], 0.0)
                    | np.isclose(x[1], 1.0),
)

dofs = fem.locate_dofs_topological(V=V, entity_dim=0, entities=facets)
bc_func = fem.Function(V)
bc_func.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)
bc = fem.dirichletbc(bc_func, dofs)

u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
a = fem.form(ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx)
L = fem.form(ufl.inner(-6, v) * ufl.dx)

A = petsc.assemble_matrix(a, bcs=[bc])
A.assemble()

print(dir(A))
A.view()
print(A.getSize())

b = petsc.assemble_vector(L)
petsc.apply_lifting(b, [a], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
petsc.set_bc(b, [bc])

log = PETSc.Log()
log.begin()

opts = PETSc.Options()
opts["ksp_type"] = "preonly"
opts["ksp_monitor"] = ""
opts["ksp_view"] = ""
opts["pc_type"] = "mg"

## TODO: mg should be the correct PC


opts["pc_mg_cycle_type"] = "w"
opts["pc_mg_levels"] = "2"

# opts["pc_gamg_log"] = ""
# opts["log_view"] = ""


ksp = PETSc.KSP().create(comm)
ksp.setFromOptions()

dmda = PETSc.DMDA().create(dim=1, dof=1, sizes=[n+1], proc_sizes=None, boundary_type=None, stencil_type=None, stencil_width=None, setup=True, ownership_ranges=None, comm=comm)

ksp.setDM(dmda)
def compute_operators(ksp, A_i, P_i) -> None:
    # produce A_coarse here
    # print(dir(ksp.getDM()))
    # print(ksp.getDM().dof)
    # print(dir(dmda))
    # print(dir(dmda))
    R = dmda.createRestriction()
    # print(A_i.getSize())
    # exit()
    A.copy(A_i)
    A.copy(P_i)
    # print(A_i.getSize())
    pass

ksp.setComputeOperators(compute_operators)
ksp.setUp()
u = fem.Function(V)
ksp.solve(b, u.x.petsc_vec)

# log.view()