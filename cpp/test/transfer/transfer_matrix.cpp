#include <cstdint>
#include <memory>
#include <mpi.h>

#include <catch2/catch_test_macros.hpp>

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/refinement/refine.h>
// #include <dolfinx/transfer/transfer_matrix.h>
#include <dolfinx/refinement/plaza.h>
#include <optional>

using namespace dolfinx;

/// from = row
/// to = column
template <typename U>
dolfinx::la::SparsityPattern
create_sparsity(const dolfinx::fem::FunctionSpace<U>& V_from,
                const dolfinx::fem::FunctionSpace<U>& V_to,
                [[maybe_unused]] const std::vector<std::int32_t>& parent_cells)
{
  auto mesh_from = V_from.mesh();
  auto mesh_to = V_to.mesh();

  assert(mesh_from);
  assert(mesh_to);

  MPI_Comm comm = mesh_from->comm();
  {
    // Check comms equal
    int result;
    MPI_Comm_compare(comm, mesh_to->comm(), &result);
    assert(result == MPI_CONGRUENT);
  }

  auto dofmap_from = V_from.dofmap();
  auto dofmap_to = V_to.dofmap();

  assert(dofmap_from);
  assert(dofmap_to);

  // Create and build  sparsity pattern
  assert(dofmap_from->index_map);
  assert(dofmap_to->index_map);

  dolfinx::la::SparsityPattern sp(
      comm, {dofmap_from->index_map, dofmap_to->index_map},
      {dofmap_from->index_map_bs(), dofmap_to->index_map_bs()});

  int tdim = mesh_from->topology()->dim();
  assert(tdim == mesh_to->topology()->dim());

  auto cell_map_from = mesh_from->topology()->index_map(tdim);
  auto cell_map_to = mesh_to->topology()->index_map(tdim);

  assert(cell_map_from);
  assert(cell_map_to);

  std::vector<int32_t> c_to_f { 0, 2, 4 };
  for (int dof_from = 0; dof_from < mesh_from->topology()->index_map(0)->size_local(); dof_from++)
  {
    int32_t dof_to = c_to_f[dof_from];

    auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
    for (auto e : to_v_to_f->links(dof_to))
    {
      auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
      for (auto n : to_f_to_v->links(e))
      {
        // if (n == dof_to)
        //   continue;
        std::cout << "coarse " << dof_from << " has fine neighbors " <<  n << std::endl;
         std::vector<int32_t> a{dof_from};
        std::vector<int32_t> b{n};
        sp.insert(a, b);
      }
    }
  }

  // auto c_to_v_from = mesh_from->topology()->connectivity(tdim, 0);
  // auto c_to_v_to = mesh_to->topology()->connectivity(tdim, 0);
  // for (int cell_idx = 0; cell_idx < cell_map_to->size_local(); cell_idx++)
  // {
  //   auto parent_cell = parent_cells[cell_idx];

  //   for (auto v_to : c_to_v_to->links(cell_idx))
  //   {
  //     for (auto v_from : c_to_v_from->links(parent_cell))
  //     {
  //       std::vector<int32_t> parent(1, v_from);
  //       std::vector<int32_t> e_span{v_to};
  //       sp.insert(parent, e_span);
  //     }
  //   }
  // }
  sp.finalize();

  return sp;
}

TEST_CASE("Transfer Matrix", "transfer_matrix")
{
  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex);
  auto mesh_coarse = std::make_shared<mesh::Mesh<double>>(
      dolfinx::mesh::create_interval<double>(MPI_COMM_SELF, 2, {0.0, 1.0}, mesh::GhostMode::none,
                                             part));

  bool redistribute = false;
  auto [mesh_fine, parent_cell, parent_facet] = refinement::refine(
      *mesh_coarse, std::nullopt, redistribute, mesh::GhostMode::none,
      refinement::Option::parent_cell);

  std::cout << "parent_cell = ";
  for (auto pc : parent_cell.value())
    std::cout << pc << ", ";
  std::cout << std::endl;

  // std::cout << "parent_facet = ";
  // for (auto pc : parent_facet.value())
  //   std::cout << pc << ", ";
  // std::cout << std::endl;

  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::interval, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(mesh_coarse, element, {}));
  auto V_fine = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          std::make_shared<mesh::Mesh<double>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0,1);

  auto sparsity_pattern
      = create_sparsity<double>(*V_coarse, *V_fine, parent_cell.value());

  la::MatrixCSR<double> transfer_matrix(sparsity_pattern,
                                        la::BlockMode::compact);
  transfer_matrix.set(1);

  std::vector<double> dense = transfer_matrix.to_dense();
  for (int i = 0; i < transfer_matrix.index_map(0)->size_local(); i++)
  {
    for (int j = 0; j < transfer_matrix.index_map(1)->size_local(); j++)
    {
      std::cout << dense[i * transfer_matrix.index_map(1)->size_local() + j]
                << " ";
    }
    std::cout << "\n";
  }
  // create_1d_transfer_matrix(V0, V1);
}