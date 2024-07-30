// Copyright (C) 2010-2023 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>

#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/log.h>
#include <dolfinx/mesh/Mesh.h>
#include <dolfinx/mesh/Topology.h>
#include <dolfinx/mesh/cell_types.h>

#include "interval.h"
#include "plaza.h"

namespace dolfinx::refinement
{
/// @brief Create a locally refined mesh.
///
/// @param[in] mesh Mesh to create a new, refined mesh from.
/// @param[in] edges Indices of the edges that should be split during
/// refinement. mesh::compute_incident_entities can be used to compute
/// the edges that are incident to other entities, e.g. incident to
/// cells.
/// @param[in] redistribute If `true` refined mesh is re-partitioned
/// across MPI ranks.
/// @return Refined mesh.
template <std::floating_point T>
mesh::Mesh<T> refine(const mesh::Mesh<T>& mesh,
                     std::optional<std::span<const std::int32_t>> edges,
                     bool redistribute = true,
                     dolfinx::mesh::GhostMode ghost_mode
                     = mesh::GhostMode::shared_facet)
{
  auto topology = mesh.topology();
  assert(topology);

  mesh::CellType cell_t = topology->cell_type();
  if (cell_t != mesh::CellType::interval and cell_t != mesh::CellType::triangle
      and cell_t != mesh::CellType::tetrahedron)
    throw std::runtime_error("Refinement only defined for simplices");

  bool oned = topology->cell_type() == mesh::CellType::interval;

  auto get_mesh = [&]()
  {
    if (oned)
      return std::get<0>(refine_interval(mesh, edges, redistribute));
    else
      return std::get<0>(
          plaza::refine(mesh, edges, redistribute, ghost_mode, Option::none));
  };
  mesh::Mesh<T> refined_mesh = get_mesh();

  // Report the number of refined cellse
  const int D = topology->dim();
  const std::int64_t n0 = topology->index_map(D)->size_global();
  const std::int64_t n1 = refined_mesh.topology()->index_map(D)->size_global();
  spdlog::info(
      "Number of cells increased from {} to {} ({}% increase).", n0, n1,
      100.0 * (static_cast<double>(n1) / static_cast<double>(n0) - 1.0));

  return refined_mesh;
}

} // namespace dolfinx::refinement
