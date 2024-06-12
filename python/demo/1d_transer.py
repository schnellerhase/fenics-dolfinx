import math

from mpi4py import MPI

import numpy as np

from dolfinx import fem, mesh

comm = MPI.COMM_WORLD

msh_coarse = mesh.create_interval(comm, (16), (0, 1))
msh_fine = mesh.create_interval(comm, (32), (0, 1))

V_coarse = fem.functionspace(msh_coarse, ("Lagrange", 1))
V_fine = fem.functionspace(msh_fine, ("Lagrange", 1))

f_fine = fem.Function(V_fine)
f_fine.interpolate(lambda x: 1 + x[0] ** 2 + 2 * x[1] ** 2)

P = np.zeros((33, 17))
for i in range(33):
    if i % 2 == 0:
        P[i, int(i / 2)] = 1
    else:
        P[i, math.floor(i / 2)] = 0.5
        P[i, math.ceil(i / 2)] = 0.5

R = P.transpose()
# print(R)

# f_coarse = fem.Function(V_coarse, )
f_coarse = R.dot(f_fine.x.array)

print(f"<f_coarse, R f_fine>={np.inner(f_coarse, R.dot(f_fine.x.array))}")
print(f"<P f_coarse, f_fine>={np.inner(P.dot(f_coarse), f_fine.x.array)}")
