// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/mesh/utils.h>
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
  CHECK_adjacency_list_equal(*mesh.topology()->connectivity(0, 1),
                             {{0}, {0, 1}, {1, 2}, {2, 3}, {3}});
}

TEMPLATE_TEST_CASE("Interval mesh (parallel)", "[mesh][interval]", float,
                   double)
{
  using T = TestType;

  int comm_size = dolfinx::MPI::size(MPI_COMM_WORLD);
  int rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

  for (mesh::GhostMode ghost_mode :
       {mesh::GhostMode::shared_facet, mesh::GhostMode::shared_vertex})
  {
    mesh::Mesh<T> mesh = mesh::create_interval<T>(
        MPI_COMM_WORLD, 5 * comm_size - 1, {0., 1.}, ghost_mode);

    {
      int comp_result;
      MPI_Comm_compare(mesh.comm(), MPI_COMM_WORLD, &comp_result);
      CHECK(comp_result == MPI_CONGRUENT);
    }

    CHECK(mesh.geometry().dim() == 1);

    std::array<int32_t, 3> expected_local_vertex_count;
    std::array<int32_t, 3> expected_num_ghosts;
    std::array<std::vector<T>, 3> expected_x;
    std::array<std::vector<std::vector<std::int32_t>>, 3> expected_v_to_e;

    if (comm_size == 1)
    {
      // vertex layout
      //   0 --- 1 --- 2 --- 3 --- 4
      expected_local_vertex_count = {5};
      expected_num_ghosts = {0};

      expected_x[0] = {
          /* v_0 */ 0.0,  0.0, 0.0,
          /* v_1 */ 0.25, 0.0, 0.0,
          /* v_2 */ 0.5,  0.0, 0.0,
          /* v_3 */ 0.75, 0.0, 0.0,
          /* v_4 */ 1.0,  0.0, 0.0,
      };

      // cell layout
      // x -0- x -1- x -2- x -3- x
      expected_v_to_e[0] = {{0}, {0, 1}, {1, 2}, {2, 3}, {3}};
    }
    else if (comm_size == 2)
    {
      /* clang-format off
      vertex layout
      0 --- 1 --- 2 --- 3 --- 4 --- 5 --- 6                   (process 0)
      l     l     l     l     l     l     g                          
                              5 --- 4 --- 0 --- 1 --- 2 --- 3 (process 1)
                              g     g     l     l     l     l
      clang-format on */

      expected_local_vertex_count = {6, 4};
      expected_num_ghosts = {1, 2};

      expected_x[0] = {
          /* v_0 */ 0.0,    0.0, 0.0,
          /* v_1 */ 1. / 9, 0.0, 0.0,
          /* v_2 */ 2. / 9, 0.0, 0.0,
          /* v_3 */ 3. / 9, 0.0, 0.0,
          /* v_4 */ 4. / 9, 0.0, 0.0,
          /* v_5 */ 5. / 9, 0.0, 0.0,
          /* v_6 */ 6. / 9, 0.0, 0.0,
      };

      expected_x[1] = {
          /* v_0 */ 6. / 9, 0.0, 0.0,
          /* v_1 */ 7. / 9, 0.0, 0.0,
          /* v_2 */ 8. / 9, 0.0, 0.0,
          /* v_3 */ 9. / 9, 0.0, 0.0,
          /* v_3 */ 5. / 9, 0.0, 0.0,
          /* v_3 */ 4. / 9, 0.0, 0.0,
      };

      /* clang-format off
      cell layout
          x -0- x -1- x -2- x -3- x -4- x -5- x                   (process 0)
      
                                  x -4- x -0- x -1- x -2- x -3- x (process 1)
      clang-format on */
      expected_v_to_e[0] = {{0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5}};

      expected_v_to_e[1] = {{0, 1}, {1, 2}, {2, 3}, {3}, {0, 4}, {4}};
    }
    else if (comm_size == 3)
    {
      /* clang-format off
      vertex layout
      0 --- 1 --- 2 --- 3 --- 4 --- 5 --- 6                                                 (process 1)
      l     l     l     l     l     l     g                          
   
                              5 --- 4 --- 0 --- 1 --- 2 --- 3 --- 6                         (process 2)
                              g     g     l     l     l     l     g
   
                                                      6 --- 5 --- 0 --- 1 --- 2 --- 3 --- 4 (process 0)
                                                      g     g     l     l     l     l     l
      clang-format on */

      expected_local_vertex_count = {5, 6, 4};
      expected_num_ghosts = {2, 1, 3};

      expected_x[1] = {
          /* v_0 */ 0.0,     0.0, 0.0,
          /* v_1 */ 1. / 14, 0.0, 0.0,
          /* v_2 */ 2. / 14, 0.0, 0.0,
          /* v_3 */ 3. / 14, 0.0, 0.0,
          /* v_4 */ 4. / 14, 0.0, 0.0,
          /* v_5 */ 5. / 14, 0.0, 0.0,
          /* v_6 */ 6. / 14, 0.0, 0.0,
      };

      expected_x[2] = {
          /* v_0 */ 6. / 14,  0.0, 0.0,
          /* v_1 */ 7. / 14,  0.0, 0.0,
          /* v_2 */ 8. / 14,  0.0, 0.0,
          /* v_3 */ 9. / 14,  0.0, 0.0,
          /* v_4 */ 5. / 14,  0.0, 0.0,
          /* v_5 */ 4. / 14,  0.0, 0.0,
          /* v_6 */ 10. / 14, 0.0, 0.0,
      };

      expected_x[0] = {
          /* v_0 */ 10. / 14, 0.0, 0.0,
          /* v_1 */ 11. / 14, 0.0, 0.0,
          /* v_2 */ 12. / 14, 0.0, 0.0,
          /* v_3 */ 13. / 14, 0.0, 0.0,
          /* v_4 */ 14. / 14, 0.0, 0.0,
          /* v_5 */ 9. / 14,  0.0, 0.0,
          /* v_6 */ 8. / 14,  0.0, 0.0,
      };

      /* clang-format off
      vertex layout
      x -0- x -1- x -2- x -3- x -4- x -5- x                                                 (process 1)
      
                              x -4- x -0- x -1- x -2- x -3- x -5- x                         (process 2)
      
                                                      x -5- x -0- x -1- x -2- x -3- x -4- x (process 0)
      clang-format on */

      expected_v_to_e[1] = {{0}, {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5}};

      expected_v_to_e[2] = {{0, 1}, {1, 2}, {2, 3}, {3, 5}, {0, 4}, {4}, {5}};

      expected_v_to_e[0] = {{0, 1}, {1, 2}, {2, 3}, {3, 4}, {4}, {0, 5}, {5}};
    }
    else
    {
      // Test only supports np <= 3
      CHECK(false);
    }

    auto vertices = mesh.topology()->index_map(0);

    CHECK(vertices->size_local() == expected_local_vertex_count[rank]);
    CHECK(vertices->num_ghosts() == expected_num_ghosts[rank]);

    CHECK_THAT(mesh.geometry().x(),
               RangeEquals(expected_x[rank], [](auto a, auto b)
                           { return std::abs(a - b) <= EPS<T>; }));

    mesh.topology()->create_connectivity(0, 1);
    CHECK_adjacency_list_equal(*mesh.topology()->connectivity(0, 1),
                               expected_v_to_e[rank]);
  }
}