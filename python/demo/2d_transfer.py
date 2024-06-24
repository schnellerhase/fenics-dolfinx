import math

from mpi4py import MPI

import numpy as np

import ufl
from dolfinx import fem, io, mesh

comm = MPI.COMM_WORLD

np.set_printoptions(threshold=np.inf, linewidth=np.inf)
n_coarse = 2
n_fine = 4

# n_coarse = 16
# n_fine = 32

assert 2 * n_coarse == n_fine, "Require exact containment"

msh_coarse = mesh.create_unit_square(comm, n_coarse, n_coarse, cell_type=mesh.CellType.quadrilateral)
msh_fine = mesh.create_unit_square(comm, n_fine, n_fine, cell_type=mesh.CellType.quadrilateral)

V_coarse = fem.functionspace(msh_coarse, ("Lagrange", 1))
V_fine = fem.functionspace(msh_fine, ("Lagrange", 1))

def f(x):
    return 1 + x[0] ** 2 + 2 * x[1]**2

f_fine = fem.Function(V_fine, name="f_fine")
f_fine.interpolate(f)

f_coarse = fem.Function(V_coarse, name="f_coarse")
f_coarse.interpolate(f)

dofs_coarse = V_coarse.dofmap.index_map.size_local
dofs_fine = V_fine.dofmap.index_map.size_local

print(dofs_fine)
print(dofs_coarse)

assert dofs_coarse == V_coarse.dofmap.index_map.size_global, "Not parallel ready"
assert dofs_fine == V_fine.dofmap.index_map.size_global, "Not parallel ready"

# TODO: make sparse
# P = np.zeros((n_fine + 1, n_coarse+1))
# for i in range(n_fine + 1):
#     if i % 2 == 0:
#         P[i, int(i / 2)] = 1
#     else:
#         P[i, math.floor(i / 2)] = 0.5
#         P[i, math.ceil(i / 2)] = 0.5

# # P = 1/4 * P
# P = np.kron(P, P)
# print(P)


P = np.zeros((dofs_fine, dofs_coarse))
for ix in range(n_coarse+1):
    for iy in range(n_coarse+1):
        fine_ix = 2*ix
        fine_iy = 2*iy

        flat_coarse = ix + (n_coarse+1) * iy

        P[fine_ix + (n_fine + 1) * fine_iy, flat_coarse] = 1

        if (fine_ix > 0):
            P[(fine_ix - 1) + (n_fine+1) * fine_iy, flat_coarse] = .5
        if (fine_ix < n_fine):
            P[(fine_ix + 1) + (n_fine+1) * fine_iy, flat_coarse] = .5

        if (fine_iy > 0):
            P[fine_ix + (n_fine+1) * (fine_iy-1), flat_coarse] = .5
        if (fine_iy < n_fine):
            P[fine_ix + (n_fine+1) * (fine_iy+1), flat_coarse] = .5

        if (fine_ix > 0 and fine_iy > 0):
            P[(fine_ix-1) + (n_fine+1) * (fine_iy-1), flat_coarse] = .25

        if (fine_ix > 0 and fine_iy < n_fine):
            P[(fine_ix-1) + (n_fine+1) * (fine_iy+1), flat_coarse] = .25

        if (fine_ix < n_fine and fine_iy > 0):
            P[(fine_ix+1) + (n_fine+1) * (fine_iy-1), flat_coarse] = .25

        if (fine_ix < n_fine and fine_iy < n_fine):
            P[(fine_ix+1) + (n_fine+1) * (fine_iy+1), flat_coarse] = .25

print(P)

R = 1/4 * P.transpose()

print(R)

f_restricted = fem.Function(V_coarse, name="f_restricted")
f_prolongated = fem.Function(V_fine)

f_restricted.x.array[:] = R.dot(f_fine.x.array)[:]
f_prolongated.x.array[:] = P.dot(f_coarse.x.array)[:]

with io.XDMFFile(comm, "prolongated.xdmf", "w") as file:
    file.write_mesh(msh_fine)
    file.write_function(f_prolongated)

with io.XDMFFile(comm, "fine.xdmf", "w") as file:
    file.write_mesh(msh_fine)
    file.write_function(f_fine)

with io.XDMFFile(comm, "restricted.xdmf", "w") as file:
    file.write_mesh(msh_coarse)
    file.write_function(f_restricted)

print(f"<I_coarse(f), R I_fine(f)> = {np.inner(f_coarse.x.array, f_restricted.x.array)}")
print(f"<P I_coarse(f), I_fine(f)> = {np.inner(f_prolongated.x.array, f_fine.x.array)}")

def L2_norm(f):
    return np.sqrt(comm.allreduce(fem.assemble_scalar(fem.form(ufl.inner(f, f) * ufl.dx)), op=MPI.SUM))

print(f"||I_fine(f) - f||_L2 = {L2_norm(f_fine - f(ufl.SpatialCoordinate(msh_fine)))}")
print(f"||R I_fine(f) - f||_L2 = {L2_norm(f_restricted - f(ufl.SpatialCoordinate(msh_coarse)))}")
print(f"||P I_coarse(f) - f||_L2 = {L2_norm(f_prolongated - f(ufl.SpatialCoordinate(msh_fine)))}")
