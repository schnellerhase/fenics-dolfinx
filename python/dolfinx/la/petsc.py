# Copyright (C) 2025 Garth N. Wells, Jack S. Hale
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Functions for working with PETSc linear algebra objects"""

import functools
import typing

from petsc4py import PETSc

import numpy as np
import numpy.typing as npt

import dolfinx
from dolfinx.la import IndexMap, Vector

assert dolfinx.has_petsc4py

__all__ = ["assign", "create_vector", "create_vector_wrap"]


def create_vector(index_map: IndexMap, bs: int) -> PETSc.Vec:  # type: ignore[name-defined]
    """Create a distributed PETSc vector.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        index_map: Index map that describes the size and parallel layout of
            the vector to create.
        bs: Block size of the vector.

    Returns:
        PETSc Vec object.
    """
    ghosts = index_map.ghosts.astype(PETSc.IntType)  # type: ignore[attr-defined]
    size = (index_map.size_local * bs, index_map.size_global * bs)
    return PETSc.Vec().createGhost(ghosts, size=size, bsize=bs, comm=index_map.comm)  # type: ignore


def create_vector_wrap(x: Vector) -> PETSc.Vec:  # type: ignore[name-defined]
    """Wrap a distributed DOLFINx vector as a PETSc vector.

    Note:
        Due to subtle issues in the interaction between petsc4py memory management
        and the Python garbage collector, it is recommended that the method ``PETSc.Vec.destroy()``
        is called on the returned object once the object is no longer required. Note that
        ``PETSc.Vec.destroy()`` is collective over the object's MPI communicator.

    Args:
        x: The vector to wrap as a PETSc vector.

    Returns:
        A PETSc vector that shares data with ``x``.
    """
    index_map = x.index_map
    ghosts = index_map.ghosts.astype(PETSc.IntType)  # type: ignore[attr-defined]
    bs = x.block_size
    size = (index_map.size_local * bs, index_map.size_global * bs)
    return PETSc.Vec().createGhostWithArray(  # type: ignore[attr-defined]
        ghosts, x.array, size=size, bsize=bs, comm=index_map.comm
    )


@functools.singledispatch
def assign(x0: typing.Union[npt.NDArray[np.inexact], list[npt.NDArray[np.inexact]]], x1: PETSc.Vec):  # type: ignore
    """Assign ``x0`` values to a PETSc vector ``x1``.

    Values in ``x0``, which is possibly a stacked collection of arrays,
    are assigned ``x1``. When ``x0`` holds a sequence of ``n``` arrays
    and ``x1`` has type ``NEST``, the assignment is::

              [x0[0]]
        x1 =  [x0[1]]
              [.....]
              [x0[n-1]]

    When ``x0`` holds a sequence of ``n`` arrays and ``x1`` **does
    not** have type ``NEST``, the assignment is::

              [x0_owned[0]]
        x1 =  [.....]
              [x0_owned[n-1]]
              [x0_ghost[0]]
              [.....]
              [x0_ghost[n-1]]

    Args:
        x0: An array or list of arrays that will be assigned to ``x1``.
        x1: Vector to assign values to.
    """
    try:
        x1_nest = x1.getNestSubVecs()
        for _x0, _x1 in zip(x0, x1_nest):
            with _x1.localForm() as x:
                x.array_w[:] = _x0
    except PETSc.Error:  # type: ignore[attr-defined]
        with x1.localForm() as _x:
            try:
                start = 0
                for _x0 in x0:
                    end = start + _x0.shape[0]
                    _x.array_w[start:end] = _x0
                    start = end
            except IndexError:
                _x.array_w[:] = x0


@assign.register(PETSc.Vec)  # type: ignore[attr-defined]
def _(x0: PETSc.Vec, x1: typing.Union[npt.NDArray[np.inexact], list[npt.NDArray[np.inexact]]]):  # type: ignore
    """Assign PETSc vector ``x0`` values to (blocked) array(s) ``x1``.

    This function performs the reverse of the assigment performed by the
    version of :func:`.assign(x0: typing.Union[npt.NDArray[np.inexact],
    list[npt.NDArray[np.inexact]]], x1: PETSc.Vec)`.

    Args:
        x0: Vector that will have its values assigned to ``x1``.
        x1: An array or list of arrays to assign to.
    """
    try:
        x0_nest = x0.getNestSubVecs()
        for _x0, _x1 in zip(x0_nest, x1):
            with _x0.localForm() as x:
                _x1[:] = x.array_r[:]
    except PETSc.Error:  # type: ignore[attr-defined]
        with x0.localForm() as _x0:
            try:
                start = 0
                for _x1 in x1:
                    end = start + _x1.shape[0]
                    _x1[:] = _x0.array_r[start:end]
                    start = end
            except IndexError:
                x1[:] = _x0.array_r[:]
