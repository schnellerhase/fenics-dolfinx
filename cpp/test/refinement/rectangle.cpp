// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <limits>
#include <optional>
#include <span>

#include <mpi.h>

#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dolfinx/common/MPI.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/generation.h>
#include <dolfinx/mesh/utils.h>
#include <dolfinx/refinement/interval.h>
#include <dolfinx/refinement/refine.h>

using namespace dolfinx;
using namespace Catch::Matchers;

constexpr auto EPS = std::numeric_limits<double>::epsilon();

TEST_CASE("Rectangle uniform refinement", "refinement,rectangle,uniform")
{
  if (dolfinx::MPI::rank(MPI_COMM_WORLD) > 1)
    return;

  mesh::Mesh<double> mesh =
      dolfinx::mesh::create_rectangle<double>(MPI_COMM_SELF, {{{0,0}, {1,1}}}, {1,1}, mesh::CellType::triangle);


  {
    // check mesh form -> maybe transfer to mesh test case
    
    // vertex layout:
    // 3---2
    // |  /|
    // | / |
    // |/  |
    // 0---1
    std::span<double> x = mesh.geometry().x();
    
    CHECK(x.size() == 12);

    // vertex 0
    REQUIRE_THAT(x[0], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[1], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[2], WithinAbs(0.0, EPS));

    // vertex 1
    REQUIRE_THAT(x[3], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[4], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[5], WithinAbs(0.0, EPS));

    // vertex 2
    REQUIRE_THAT(x[6], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[7], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[8], WithinAbs(0.0, EPS));

    // vertex 3
    REQUIRE_THAT(x[9], WithinAbs(0.0, EPS));
    REQUIRE_THAT(x[10], WithinAbs(1.0, EPS));
    REQUIRE_THAT(x[11], WithinAbs(0.0, EPS));


    // edge layout:
    // x-4-x
    // |  /|
    // 2 1 3
    // |/  |
    // x-0-x
    mesh.topology()->create_connectivity(1, 0);
    auto e_to_e = mesh.topology()->connectivity(1, 0);
    CHECK(e_to_e);

    CHECK(e_to_e->num_nodes() == 5);

    auto edge_0 = e_to_e->links(0);
    CHECK(edge_0.size() == 2);
    CHECK(edge_0[0] == 0);
    CHECK(edge_0[1] == 1);

    auto edge_1 = e_to_e->links(1);
    CHECK(edge_1.size() == 2);
    CHECK(edge_1[0] == 0);
    CHECK(edge_1[1] == 2);
    
    auto edge_2 = e_to_e->links(2);
    CHECK(edge_2.size() == 2);
    CHECK(edge_2[0] == 0);
    CHECK(edge_2[1] == 3);

    auto edge_3 = e_to_e->links(3);
    CHECK(edge_3.size() == 2);
    CHECK(edge_3[0] == 1);
    CHECK(edge_3[1] == 2);

    auto edge_4 = e_to_e->links(4);
    CHECK(edge_4.size() == 2);
    CHECK(edge_4[0] == 2);
    CHECK(edge_4[1] == 3);
  }

  //TODO:continue
  // plaza requires the edges to be pre initialized!
  mesh.topology()->create_entities(1);

  bool redistribute = false;
  auto [mesh_fine, parent_cell, parent_facet] = refinement::refine(
      mesh, std::nullopt, redistribute, mesh::GhostMode::none,
      refinement::Option::parent_cell_and_facet);

  
}