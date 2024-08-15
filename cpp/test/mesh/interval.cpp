// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <mpi.h>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_range_equals.hpp>

#include <dolfinx/mesh/generation.h>

#include "util.h"

using namespace dolfinx;
using namespace Catch::Matchers;

TEMPLATE_TEST_CASE("Interval mesh", "[mesh][interval]", float, double)
{
  using T = TestType;

  mesh::Mesh<T> mesh = mesh::create_interval<T>(MPI_COMM_SELF, 4, {0., 1.});

  {
    int comp_result;
    MPI_Comm_compare(mesh.comm(), MPI_COMM_SELF, &comp_result);
    CHECK(comp_result == MPI_CONGRUENT);
  }

  CHECK(mesh.geometry().dim() == 1);

  // vertex layout
  // 0 --- 1 --- 2 --- 3 --- 4
  std::vector<T> expected_x = {
      /* v_0 */ 0.0,  0.0, 0.0,
      /* v_1 */ 0.25, 0.0, 0.0,
      /* v_2 */ 0.5,  0.0, 0.0,
      /* v_3 */ 0.75, 0.0, 0.0,
      /* v_4 */ 1.0,  0.0, 0.0,
  };

  CHECK_THAT(mesh.geometry().x(),
             RangeEquals(expected_x, [](auto a, auto b)
                         { return std::abs(a - b) <= EPS<T>; }));

  // cell layout
  // x -0- x -1- x -2- x -3- x
  mesh.topology()->create_connectivity(0, 1);
  CHECK_adjacency_list_equal(*mesh.topology()->connectivity(0, 1), {{0}, {0, 1}, {1, 2}, {2, 3}, {3}});
}
