// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <memory>
#include <optional>

#include <mpi.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/refinement/plaza.h>
#include <dolfinx/refinement/refine.h>

using namespace dolfinx;
using namespace Catch::Matchers;

template <typename T>
constexpr auto EPS = std::numeric_limits<T>::epsilon();

/// from = row
/// to = column
template <typename U>
dolfinx::la::SparsityPattern
create_sparsity(const dolfinx::fem::FunctionSpace<U>& V_from,
                const dolfinx::fem::FunctionSpace<U>& V_to,
                const std::vector<std::int32_t>& from_to_map)
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

  assert(mesh_from->topology()->dim() == mesh_to->topology()->dim());

  assert(mesh_from->topology()->index_map(0));

  for (int dof_from = 0;
       dof_from < mesh_from->topology()->index_map(0)->size_local(); dof_from++)
  {
    int32_t dof_to = from_to_map[dof_from];

    auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
    assert(to_v_to_f);
    for (auto e : to_v_to_f->links(dof_to))
    {
      auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
      assert(to_f_to_v);
      for (auto n : to_f_to_v->links(e))
      {
        std::cout << "coarse " << dof_from << " has fine neighbors " << n
                  << std::endl;
        sp.insert(std::vector<int32_t>{dof_from}, std::vector<int32_t>{n});
      }
    }
  }
  sp.finalize();
  return sp;
}

TEST_CASE("Transfer Matrix 1D", "transfer_matrix")
{
  using T = double;

  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex);
  auto mesh_coarse = std::make_shared<mesh::Mesh<double>>(
      dolfinx::mesh::create_interval<double>(MPI_COMM_SELF, 2, {0.0, 1.0},
                                             mesh::GhostMode::none, part));

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
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<int32_t> from_to_map{0, 2, 4}; // TODO: general computation!

  auto sparsity_pattern
      = create_sparsity<double>(*V_coarse, *V_fine, from_to_map);

  la::MatrixCSR<double> transfer_matrix(sparsity_pattern,
                                        la::BlockMode::compact);
  // transfer_matrix.set(1);

  auto mesh_from = mesh_coarse;
  auto mesh_to = mesh_fine;
  for (int dof_from = 0;
       dof_from < mesh_from->topology()->index_map(0)->size_local(); dof_from++)
  {
    int32_t dof_to = from_to_map[dof_from];

    auto to_v_to_f = mesh_to.topology()->connectivity(0, 1);
    for (auto e : to_v_to_f->links(dof_to))
    {
      auto to_f_to_v = mesh_to.topology()->connectivity(1, 0);
      for (auto n : to_f_to_v->links(e))
      {
        double value = n == dof_to ? 1 : .5;
        transfer_matrix.set<1, 1>(std::vector<double>{value},
                                  std::vector<int32_t>{dof_from},
                                  std::vector<int32_t>{n});
      }
    }
  }

  std::vector<double> expected{1.0, .5, 0, 0, 0, 0,  .5, 1,
                               .5,  0,  0, 0, 0, .5, 1};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
  // for (int i = 0; i < transfer_matrix.index_map(0)->size_local(); i++)
  // {
  //   for (int j = 0; j < transfer_matrix.index_map(1)->size_local(); j++)
  //   {
  //     std::cout << dense[i * transfer_matrix.index_map(1)->size_local() + j]
  //               << " ";
  //   }
  //   std::cout << "\n";
  // }
}

TEST_CASE("Transfer Matrix 2D", "transfer_matrix")
{
  using T = double;

  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex);
  auto mesh_coarse = std::make_shared<mesh::Mesh<double>>(
      dolfinx::mesh::create_rectangle<double>(
          MPI_COMM_SELF, {{{0, 0}, {1, 1}}}, {1, 1}, mesh::CellType::triangle));
  mesh_coarse->topology()->create_entities(1);

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
      basix::element::family::P, basix::cell::type::triangle, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(mesh_coarse, element, {}));
  auto V_fine = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          std::make_shared<mesh::Mesh<double>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<int32_t> from_to_map{4, 1, 5, 8};

  auto sparsity_pattern
      = create_sparsity<double>(*V_coarse, *V_fine, from_to_map);

  la::MatrixCSR<double> transfer_matrix(sparsity_pattern,
                                        la::BlockMode::compact);
  // transfer_matrix.set(1);

  auto mesh_from = mesh_coarse;
  auto mesh_to = mesh_fine;
  for (int dof_from = 0;
       dof_from < mesh_from->topology()->index_map(0)->size_local(); dof_from++)
  {
    int32_t dof_to = from_to_map[dof_from];

    auto to_v_to_f = mesh_to.topology()->connectivity(0, 1);
    for (auto e : to_v_to_f->links(dof_to))
    {
      auto to_f_to_v = mesh_to.topology()->connectivity(1, 0);
      for (auto n : to_f_to_v->links(e))
      {
        double value = n == dof_to ? 1 : .5;
        transfer_matrix.set<1, 1>(std::vector<double>{value},
                                  std::vector<int32_t>{dof_from},
                                  std::vector<int32_t>{n});
      }
    }
  }

  std::vector<double> expected{0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0,
                               0.5, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.5, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0,
                               0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}