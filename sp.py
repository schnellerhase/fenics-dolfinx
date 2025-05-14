import ufl
import dolfinx
from mpi4py import MPI

comm = MPI.COMM_WORLD

if comm.size > 1:
    raise RuntimeError("Parallel not supported for checking")

mesh = dolfinx.mesh.create_unit_square(comm, 1, 1)

V = dolfinx.fem.functionspace(mesh, ("P", 1))

sp = dolfinx.cpp.la.SparsityPattern(
    comm,
    [V.dofmap.index_map, V.dofmap.index_map],
    [V.dofmap.index_map_bs, V.dofmap.index_map_bs],
)

for i in range(V.dofmap.index_map.size_local * V.dofmap.index_map_bs):
    sp.insert(i, i)
    print(f"Inserting {(i,i)=}")

sp.finalize()

D = dolfinx.la.matrix_csr(sp)

a = ufl.TrialFunction(V) * ufl.TestFunction(V) * ufl.dx
a = dolfinx.fem.form(a)

dolfinx.fem.assemble_matrix(D, a)
D = D.to_dense()
A = dolfinx.fem.assemble_matrix(a).to_dense()

print(f"{D=}")
print(f"{A=}")

for i in range(A.shape[0]):
    for j in range(A.shape[1]):
        if i == j:
            print(f"{i=}, {j=}, {D[i,j]=}, {A[i,j]=}")
            assert A[i,j] == D[i,j]
        else:
            assert D[i,j] == 0




