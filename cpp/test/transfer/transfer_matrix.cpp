// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <cstdint>
#include <memory>
#include <optional>

#include <mpi.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/fem/FunctionSpace.h>
#include <dolfinx/fem/utils.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <dolfinx/mesh/cell_types.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/plaza.h>
#include <dolfinx/refinement/refine.h>
#include <dolfinx/transfer/transfer_matrix.h>

#include "../mesh/util.h"

using namespace dolfinx;
using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("Transfer Matrix 1D", "[transfer_matrix]", double) // TODO: float
{
  using T = TestType;

  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex);
  auto mesh_coarse
      = std::make_shared<mesh::Mesh<T>>(dolfinx::mesh::create_interval<T>(
          MPI_COMM_SELF, 2, {0.0, 1.0}, mesh::GhostMode::none, part));

  bool redistribute = false;
  auto [mesh_fine, parent_cell, parent_facet] = refinement::refine(
      *mesh_coarse, std::nullopt, redistribute, mesh::GhostMode::none,
      refinement::Option::parent_cell);

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::interval, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<T>>(
      fem::create_functionspace<T>(mesh_coarse, element, {}));
  auto V_fine
      = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
          std::make_shared<mesh::Mesh<T>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<std::int64_t> from_to_map{0, 2, 4}; // TODO: general computation!

  la::MatrixCSR<T> transfer_matrix
      = transfer::create_transfer_matrix<T>(*V_coarse, *V_fine, from_to_map);

  std::vector<T> expected{1.0, .5, 0, 0, 0, 0, .5, 1, .5, 0, 0, 0, 0, .5, 1};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}

TEMPLATE_TEST_CASE("Transfer Matrix 1D (parallel)", "[transfer_matrix]",
                   double) // TODO: float
{
  using T = TestType;

  int comm_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

  if (comm_size != 2)
    return;

  auto mesh_coarse = std::make_shared<mesh::Mesh<T>>(
      mesh::create_interval<T>(MPI_COMM_WORLD, 5 * comm_size - 1, {0., 1.},
                               mesh::GhostMode::shared_facet));

  auto [mesh_fine, parent_cell, parent_facet] = refinement::refine(
      *mesh_coarse, std::nullopt, true, mesh::GhostMode::shared_facet,
      refinement::Option::none);

  auto element = basix::create_element<T>(
      basix::element::family::P, basix::cell::type::interval, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<T>>(
      fem::create_functionspace<T>(mesh_coarse, element, {}));
  auto V_fine
      = std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
          std::make_shared<mesh::Mesh<T>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<int64_t> from_to_map{0, 2, 4, 6, 8, 10, 12, 14, 16, 18};

  la::MatrixCSR<T> transfer_matrix
      = transfer::create_transfer_matrix(*V_coarse, *V_fine, from_to_map);

  //   auto dense = transfer_matrix.to_dense();
  //   auto cols = transfer_matrix.index_map(1)->size_global();
  //   for (int row = 0; row < transfer_matrix.num_all_rows(); row++)
  //   {
  //     std::cout << rank << " ";
  //     for (std::int32_t col = 0; col < cols; col++)
  //     {
  //       std::cout << dense[row * cols + col] << ", ";
  //     }
  //     std::cout << "\n";
  //   }
  //   std::cout << std::endl;

  // clang-format off
  std::array<std::vector<T>, 2> expected{{{/* row_0 */ 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_1 */ 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_2 */ 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_5 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_6 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                                          {/* row_0 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_1 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0, 0.0, 0.0,
                                           /* row_2 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 0.5, 0.0,
                                           /* row_3 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 1.0,
                                           /* row_4 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                           /* row_5 */ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}}};
  // clang-format on

  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected[rank], [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}

TEST_CASE("Transfer Matrix 2D", "[transfer_matrix]")
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

  std::vector<std::int64_t> from_to_map{4, 1, 5,
                                        8}; // TODO: general computation!

  la::MatrixCSR<double> transfer_matrix
      = transfer::create_transfer_matrix(*V_coarse, *V_fine, from_to_map);

  std::vector<double> expected{0.5, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0,
                               0.5, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
                               0.5, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 0.0,
                               0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}

TEST_CASE("Transfer Matrix 3D", "[transfer_matrix]")
{
  using T = double;

  auto part = mesh::create_cell_partitioner(mesh::GhostMode::shared_vertex);
  auto mesh_coarse
      = std::make_shared<mesh::Mesh<double>>(dolfinx::mesh::create_box<double>(
          MPI_COMM_SELF, {{{0, 0, 0}, {1, 1, 1}}}, {1, 1, 1},
          mesh::CellType::tetrahedron));
  mesh_coarse->topology()->create_entities(1);

  bool redistribute = false;
  auto [mesh_fine, parent_cell, parent_facet] = refinement::refine(
      *mesh_coarse, std::nullopt, redistribute, mesh::GhostMode::none,
      refinement::Option::parent_cell);

  auto element = basix::create_element<double>(
      basix::element::family::P, basix::cell::type::tetrahedron, 1,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  auto V_coarse = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(mesh_coarse, element, {}));
  auto V_fine = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          std::make_shared<mesh::Mesh<double>>(mesh_fine), element, {}));

  mesh_fine.topology()->create_connectivity(1, 0);
  mesh_fine.topology()->create_connectivity(0, 1);

  std::vector<std::int64_t> from_to_map{
      0, 6, 15, 25, 17, 9, 11, 22}; // TODO: general computation!

  la::MatrixCSR<double> transfer_matrix
      = transfer::create_transfer_matrix(*V_coarse, *V_fine, from_to_map);

  std::vector<double> expected{
      1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5,
      0.5, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5,
      0.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.5, 0.0, 0.5, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0,
      0.0, 0.5, 0.5, 0.0, 0.0, 0.5, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0,
      0.0, 0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5,
      0.5, 1.0, 0.0, 0.0, 0.0, 0.5};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));
}
