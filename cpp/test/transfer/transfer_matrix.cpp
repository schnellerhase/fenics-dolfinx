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
#include <dolfinx/transfer/transfer_matrix.h>

using namespace dolfinx;
using namespace Catch::Matchers;

template <typename T>
constexpr auto EPS = std::numeric_limits<T>::epsilon();

TEST_CASE("Transfer Matrix 1D", "[transfer_matrix]")
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

  la::MatrixCSR<double> transfer_matrix
      = transfer::create_transfer_matrix(*V_coarse, *V_fine, from_to_map);

  std::vector<double> expected{1.0, .5, 0, 0, 0, 0,  .5, 1,
                               .5,  0,  0, 0, 0, .5, 1};
  CHECK_THAT(transfer_matrix.to_dense(),
             RangeEquals(expected, [](auto a, auto b)
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

  std::vector<int32_t> from_to_map{4, 1, 5, 8}; // TODO: general computation!

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

  std::vector<int32_t> from_to_map{0,  6, 15, 25,
                                   17, 9, 11, 22}; // TODO: general computation!

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
