// Copyright (C) 2017-2023 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include <algorithm>
#include <dolfinx/fem/Form.h>
#include <map>
#include <nanobind/ndarray.h>
#include <span>

namespace dolfinx_wrappers
{
template <typename T>
std::map<std::pair<dolfinx::fem::IntegralType, int>,
         std::pair<std::span<const T>, int>>
py_to_cpp_coeffs(
    const std::map<std::pair<dolfinx::fem::IntegralType, int>,
                   nb::ndarray<T, nb::ndim<2>, nb::c_contig>>& coeffs)
{
  using Key_t = typename std::remove_reference_t<decltype(coeffs)>::key_type;
  std::map<Key_t, std::pair<std::span<const T>, int>> c;
  std::ranges::transform(
      coeffs, std::inserter(c, c.end()),
      [](auto& e) -> typename decltype(c)::value_type
      {
        return {
            e.first,
            {std::span(static_cast<const T*>(e.second.data()), e.second.size()),
             e.second.shape(1)}};
      });
  return c;
}
} // namespace dolfinx_wrappers
