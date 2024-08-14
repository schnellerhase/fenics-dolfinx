// Copyright (C) 2024 Paul Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "dolfinx/fem/FunctionSpace.h"
#include "dolfinx/la/MatrixCSR.h"
#include "dolfinx/la/SparsityPattern.h"

namespace dolfinx::transfer
{
namespace
{

/// from = row
/// to = column
template <typename U>
la::SparsityPattern
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

  auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
  auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
  assert(to_v_to_f);
  assert(to_f_to_v);

  for (int dof_from = 0;
       dof_from < mesh_from->topology()->index_map(0)->size_local(); dof_from++)
  {
    int32_t dof_to = from_to_map[dof_from];

    for (auto e : to_v_to_f->links(dof_to))
      for (auto n : to_f_to_v->links(e))
        sp.insert(std::vector<int32_t>{dof_from}, std::vector<int32_t>{n});
  }
  sp.finalize();
  return sp;
}

} // anonymous namespace

template <typename T>
la::MatrixCSR<T>
create_transfer_matrix(const dolfinx::fem::FunctionSpace<T>& V_from,
                       const dolfinx::fem::FunctionSpace<T>& V_to,
                       const std::vector<std::int32_t>& from_to_map)
{
  la::SparsityPattern sp = create_sparsity(V_from, V_to, from_to_map);

  la::MatrixCSR<double> transfer_matrix(sp, la::BlockMode::compact);

  auto mesh_from = V_from.mesh();
  auto mesh_to = V_to.mesh();
  for (int dof_from = 0;
       dof_from < mesh_from->topology()->index_map(0)->size_local(); dof_from++)
  {
    int32_t dof_to = from_to_map[dof_from];

    auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
    for (auto e : to_v_to_f->links(dof_to))
    {
      auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
      for (auto n : to_f_to_v->links(e))
      {
        double value = n == dof_to ? 1 : .5;
        transfer_matrix.set<1, 1>(std::vector<double>{value},
                                  std::vector<int32_t>{dof_from},
                                  std::vector<int32_t>{n});
      }
    }
  }

  auto to_v_to_f = mesh_to->topology()->connectivity(0, 1);
  auto to_f_to_v = mesh_to->topology()->connectivity(1, 0);
  assert(to_v_to_f);
  assert(to_f_to_v);

  for (int dof_from = 0;
       dof_from < mesh_from->topology()->index_map(0)->size_local(); dof_from++)
  {
    int32_t dof_to = from_to_map[dof_from];

    for (auto e : to_v_to_f->links(dof_to))
    {
      for (auto n : to_f_to_v->links(e))
      {
        double value = n == dof_to ? 1 : .5;
        transfer_matrix.set<1, 1>(std::vector<double>{value},
                                  std::vector<int32_t>{dof_from},
                                  std::vector<int32_t>{n});
      }
    }
  }

  return transfer_matrix;
}

} // namespace dolfinx::transfer