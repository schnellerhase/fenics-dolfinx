import math

from mpi4py import MPI

import numpy as np

import ufl
from dolfinx import fem, io, mesh

comm = MPI.COMM_WORLD

n_coarse = 16
n_fine = 32

assert 2 * n_coarse == n_fine, "Require exact containment"

msh_coarse = mesh.create_interval(comm, (n_coarse), (0, 1))
msh_fine = mesh.create_interval(comm, (n_fine), (0, 1))

V_coarse = fem.functionspace(msh_coarse, ("Lagrange", 1))
V_fine = fem.functionspace(msh_fine, ("Lagrange", 1))

def f(x):
    return 1 + x[0] ** 2

f_fine = fem.Function(V_fine, name="f_fine")
f_fine.interpolate(f)

f_coarse = fem.Function(V_coarse, name="f_coarse")
f_coarse.interpolate(f)

dofs_coarse = V_coarse.dofmap.index_map.size_local
dofs_fine = V_fine.dofmap.index_map.size_local

assert dofs_coarse == V_coarse.dofmap.index_map.size_global, "Not parallel ready"
assert dofs_fine == V_fine.dofmap.index_map.size_global, "Not parallel ready"

# TODO: make sparse
P = np.zeros((dofs_fine, dofs_coarse))
for i in range(dofs_fine):
    if i % 2 == 0:
        P[i, int(i / 2)] = 1
    else:
        P[i, math.floor(i / 2)] = 0.5
        P[i, math.ceil(i / 2)] = 0.5


R = 1/2 * P.transpose()

f_restricted = fem.Function(V_coarse, name="f_restricted")
f_prolongated = fem.Function(V_fine)

f_restricted.x.array[:] = R.dot(f_fine.x.array)[:]
f_prolongated.x.array[:] = P.dot(f_coarse.x.array)[:]

# with io.XDMFFile(comm, "prolongated.xdmf", "w") as file:
#     file.write_mesh(msh_fine)
#     file.write_function(f_prolongated)

# with io.XDMFFile(comm, "fine.xdmf", "w") as file:
#     file.write_mesh(msh_fine)
#     file.write_function(f_fine)

print(f"<I_coarse(f), R I_fine(f)> = {np.inner(f_coarse.x.array, f_restricted.x.array)}")
print(f"<P I_coarse(f), I_fine(f)> = {np.inner(f_prolongated.x.array, f_fine.x.array)}")

def L2_norm(f):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(f, f) * ufl.dx)), op=MPI.SUM))

print(f"||I_fine(f) - f||_L2 = {L2_norm(f_fine - f(ufl.SpatialCoordinate(msh_fine)))}")
print(f"||R I_fine(f) - f||_L2 = {L2_norm(f_restricted - f(ufl.SpatialCoordinate(msh_coarse)))}")
print(f"||P I_coarse(f) - f||_L2 = {L2_norm(f_prolongated - f(ufl.SpatialCoordinate(msh_fine)))}")
