// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <concepts>
#include <cstdint>
#include <vector>

#include "dolfinx/common/IndexMap.h"
#include "dolfinx/fem/FunctionSpace.h"
#include "dolfinx/la/MatrixCSR.h"
#include "dolfinx/la/SparsityPattern.h"

namespace dolfinx::transfer
{

template <std::floating_point T, typename F>
  requires std::invocable<F&, std::int32_t>
la::MatrixCSR<T>
create_transfer_matrix(const dolfinx::fem::FunctionSpace<T>& V_from,
                       const dolfinx::fem::FunctionSpace<T>& V_to,
                       const std::vector<std::int64_t>& from_to_map, F weight)
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
  assert(mesh_from->topology()->dim() == mesh_to->topology()->dim());

  auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
  auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
  assert(to_v_to_f);
  assert(to_f_to_v);

  auto dofmap_from = V_from.dofmap();
  auto dofmap_to = V_to.dofmap();
  assert(dofmap_from);
  assert(dofmap_to);

  assert(mesh_to->topology()->index_map(0));
  assert(mesh_from->topology()->index_map(0));
  const common::IndexMap& im_to = *mesh_to->topology()->index_map(0);
  const common::IndexMap& im_from = *mesh_from->topology()->index_map(0);

  dolfinx::la::SparsityPattern sp(
      comm, {dofmap_from->index_map, dofmap_to->index_map},
      {dofmap_from->index_map_bs(), dofmap_to->index_map_bs()});

  assert(from_to_map.size() == im_from.size_global());
  for (int dof_from_global = 0; dof_from_global < im_from.size_global();
       dof_from_global++)
  {
    std::int64_t dof_to_global = from_to_map[dof_from_global];

    std::vector<std::int32_t> local_dof_to_v{0};
    im_to.global_to_local(std::vector<std::int64_t>{dof_to_global},
                          local_dof_to_v);

    auto local_dof_to = local_dof_to_v[0];

    bool is_remote = (local_dof_to == -1);
    bool is_ghost = local_dof_to >= im_to.size_local();
    if (is_remote || is_ghost)
      continue;

    std::vector<std::int32_t> dof_from_v{0};
    im_from.global_to_local(std::vector<std::int64_t>{dof_from_global},
                            dof_from_v);

    for (auto e : to_v_to_f->links(local_dof_to))
      sp.insert(std::vector<int32_t>(to_f_to_v->links(e).size(), dof_from_v[0]),
                to_f_to_v->links(e));
  }
  sp.finalize();

  la::MatrixCSR<double> transfer_matrix(std::move(sp), la::BlockMode::compact);

  for (int dof_from_global = 0; dof_from_global < im_from.size_global();
       dof_from_global++)
  {
    std::int64_t dof_to_global = from_to_map[dof_from_global];

    std::vector<std::int32_t> local_dof_to_v{0};
    im_to.global_to_local(std::vector<std::int64_t>{dof_to_global},
                          local_dof_to_v);

    auto local_dof_to = local_dof_to_v[0];

    bool is_remote = (local_dof_to == -1);
    bool is_ghost = local_dof_to >= im_to.size_local();
    if (is_remote || is_ghost)
      continue;

    std::vector<std::int32_t> dof_from_v{0};
    im_from.global_to_local(std::vector<std::int64_t>{dof_from_global},
                            dof_from_v);

    for (auto e : to_v_to_f->links(local_dof_to))
    {
      for (auto n : to_f_to_v->links(e))
      {
        // For now we only support distance <= 1 -> this should be somehow
        // asserted.
        std::int32_t distance = (n == local_dof_to) ? 0 : 1;
        transfer_matrix.set<1, 1>(std::vector<double>{weight(distance)},
                                  std::vector<int32_t>{dof_from_v[0]},
                                  std::vector<int32_t>{n});
      }
    }
  }

  transfer_matrix.scatter_rev();
  return transfer_matrix;
}

} // namespace dolfinx::transfer