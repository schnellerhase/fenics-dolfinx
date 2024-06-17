import math

from mpi4py import MPI

import numpy as np

from dolfinx import fem, mesh

comm = MPI.COMM_WORLD

n_coarse = 16
n_fine = 32

assert 2 * n_coarse == n_fine, "Require exact containment"

msh_coarse = mesh.create_interval(comm, (n_coarse), (0, 1))
msh_fine = mesh.create_interval(comm, (n_fine), (0, 1))

V_coarse = fem.functionspace(msh_coarse, ("Lagrange", 1))
V_fine = fem.functionspace(msh_fine, ("Lagrange", 1))

f_fine = fem.Function(V_fine)
f_fine.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

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

R = P.transpose()

f_restricted = R.dot(f_fine.x.array)
f_prolongated = P.dot(f_restricted)

print(f"<f_coarse, R f_fine>={np.inner(f_restricted, R.dot(f_fine.x.array))}")
print(f"<P f_coarse, f_fine>={np.inner(f_prolongated, f_fine.x.array)}")
