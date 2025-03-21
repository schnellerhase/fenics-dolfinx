# Copyright (C) 2013-2021 Anders Logg, Jørgen S. Dokken, Chris Richardson
#
# This file is part of DOLFINx (https://www.fenicsproject.org)
#
# SPDX-License-Identifier:    LGPL-3.0-or-later
"""Unit tests for BoundingBoxTree"""

from mpi4py import MPI

import numpy as np
import pytest

from dolfinx import cpp as _cpp
from dolfinx.geometry import (
    bb_tree,
    compute_closest_entity,
    compute_colliding_cells,
    compute_collisions_points,
    compute_collisions_trees,
    compute_distance_gjk,
    create_midpoint_tree,
)
from dolfinx.mesh import (
    CellType,
    create_box,
    create_unit_cube,
    create_unit_interval,
    create_unit_square,
    exterior_facet_indices,
    locate_entities,
    locate_entities_boundary,
)


def extract_geometricial_data(mesh, dim, entities):
    """For a set of entities in a mesh, return the coordinates of the
    vertices"""
    mesh_nodes = []
    geom = mesh.geometry
    g_indices = _cpp.mesh.entities_to_geometry(
        mesh._cpp_object, dim, np.array(entities, dtype=np.int32), False
    )
    for cell in g_indices:
        nodes = np.zeros((len(cell), 3), dtype=np.float64)
        for j, entity in enumerate(cell):
            nodes[j] = geom.x[entity]
        mesh_nodes.append(nodes)
    return mesh_nodes


def expand_bbox(bbox, dtype):
    """Expand min max bbox to convex hull"""
    return np.array(
        [
            [bbox[0][0], bbox[0][1], bbox[0][2]],
            [bbox[0][0], bbox[0][1], bbox[1][2]],
            [bbox[0][0], bbox[1][1], bbox[0][2]],
            [bbox[1][0], bbox[0][1], bbox[0][2]],
            [bbox[1][0], bbox[0][1], bbox[1][2]],
            [bbox[1][0], bbox[1][1], bbox[0][2]],
            [bbox[0][0], bbox[1][1], bbox[1][2]],
            [bbox[1][0], bbox[1][1], bbox[1][2]],
        ],
        dtype=dtype,
    )


def find_colliding_cells(mesh, bbox, dtype):
    """Given a mesh and a bounding box((xmin, ymin, zmin), (xmax, ymax,
    zmax)) find all colliding cells"""

    # Find actual cells using known bounding box tree
    colliding_cells = []
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    x_indices = _cpp.mesh.entities_to_geometry(
        mesh._cpp_object, mesh.topology.dim, np.arange(num_cells, dtype=np.int32), False
    )
    points = mesh.geometry.x
    bounding_box = expand_bbox(bbox, dtype)
    for cell in range(num_cells):
        vertex_coords = points[x_indices[cell]]
        bbox_cell = np.array([vertex_coords[0], vertex_coords[0]])
        # Create bounding box for cell
        for i in range(1, vertex_coords.shape[0]):
            for j in range(3):
                bbox_cell[0, j] = min(bbox_cell[0, j], vertex_coords[i, j])
                bbox_cell[1, j] = max(bbox_cell[1, j], vertex_coords[i, j])
        distance = compute_distance_gjk(expand_bbox(bbox_cell, dtype), bounding_box)
        if np.dot(distance, distance) < 1e-16:
            colliding_cells.append(cell)

    return colliding_cells


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("padding", [True, False])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_padded_bbox(padding, dtype):
    """Test collision between two meshes separated by a distance of
    epsilon, and check if padding the mesh creates a possible
    collision"""
    eps = 1e-4
    x0 = np.array([0, 0, 0], dtype=dtype)
    x1 = np.array([1, 1, 1 - eps], dtype=dtype)
    mesh_0 = create_box(MPI.COMM_WORLD, [x0, x1], [1, 1, 2], CellType.hexahedron, dtype=dtype)
    x2 = np.array([0, 0, 1 + eps], dtype=dtype)
    x3 = np.array([1, 1, 2], dtype=dtype)
    mesh_1 = create_box(MPI.COMM_WORLD, [x2, x3], [1, 1, 2], CellType.hexahedron, dtype=dtype)
    if padding:
        pad = eps
    else:
        pad = 0

    bbox_0 = bb_tree(mesh_0, mesh_0.topology.dim, padding=pad)
    bbox_1 = bb_tree(mesh_1, mesh_1.topology.dim, padding=pad)
    collisions = compute_collisions_trees(bbox_0, bbox_1)
    if padding:
        assert len(collisions) == 1
        # Check that the colliding elements are separated by a distance
        # 2*epsilon
        element_0 = extract_geometricial_data(mesh_0, mesh_0.topology.dim, [collisions[0][0]])[0]
        element_1 = extract_geometricial_data(mesh_1, mesh_1.topology.dim, [collisions[0][1]])[0]
        distance = np.linalg.norm(compute_distance_gjk(element_0, element_1))
        assert np.isclose(distance, 2 * eps, rtol=1.0e-5, atol=1.0e-7)
    else:
        assert len(collisions) == 0


def rotation_matrix(axis, angle):
    # See https://en.wikipedia.org/wiki/Rotation_matrix,
    # Subsection: Rotation_matrix_from_axis_and_angle.
    if np.isclose(np.inner(axis, axis), 1):
        n_axis = axis
    else:
        # Normalize axis
        n_axis = axis / np.sqrt(np.inner(axis, axis))

    # Define cross product matrix of axis
    axis_x = np.array(
        [[0, -n_axis[2], n_axis[1]], [n_axis[2], 0, -n_axis[0]], [-n_axis[1], n_axis[0], 0]]
    )
    id = np.cos(angle) * np.eye(3)
    outer = (1 - np.cos(angle)) * np.outer(n_axis, n_axis)
    return np.sin(angle) * axis_x + id + outer


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_tree(dtype):
    mesh = create_unit_interval(MPI.COMM_WORLD, 16, dtype=dtype)
    bbtree = bb_tree(mesh, mesh.topology.dim, np.array([], dtype=dtype))
    assert bbtree.num_bboxes == 0


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_point_1d(dtype):
    N = 16
    p = np.array([0.3, 0, 0], dtype=dtype)
    mesh = create_unit_interval(MPI.COMM_WORLD, N, dtype=dtype)
    dx = 1 / N
    cell_index = int(p[0] // dx)
    # Vertices of cell we should collide with
    vertices = np.array([[dx * cell_index, 0, 0], [dx * (cell_index + 1), 0, 0]], dtype=dtype)

    # Compute collision
    tdim = mesh.topology.dim
    tree = bb_tree(mesh, tdim)
    entities = compute_collisions_points(tree, p)
    assert len(entities.array) == 1

    # Get the vertices of the geometry
    geom_entities = _cpp.mesh.entities_to_geometry(mesh._cpp_object, tdim, entities.array, False)[0]
    x = mesh.geometry.x
    cell_vertices = x[geom_entities]
    # Check that we get the cell with correct vertices
    assert np.allclose(cell_vertices, vertices)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("point", [np.array([0.52, 0, 0]), np.array([0.9, 0, 0])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_tree_1d(point, dtype):
    mesh_A = create_unit_interval(MPI.COMM_WORLD, 16, dtype=dtype)

    def locator_A(x):
        return x[0] >= point[0]

    # Locate all vertices of mesh A that should collide
    vertices_A = _cpp.mesh.locate_entities(mesh_A._cpp_object, 0, locator_A)
    mesh_A.topology.create_connectivity(0, mesh_A.topology.dim)
    v_to_c = mesh_A.topology.connectivity(0, mesh_A.topology.dim)

    # Find all cells connected to vertex in the collision bounding box
    cells_A = np.sort(np.unique(np.hstack([v_to_c.links(vertex) for vertex in vertices_A])))

    mesh_B = create_unit_interval(MPI.COMM_WORLD, 16, dtype=dtype)
    bgeom = mesh_B.geometry.x
    bgeom += point

    def locator_B(x):
        return x[0] <= 1

    # Locate all vertices of mesh B that should collide
    vertices_B = _cpp.mesh.locate_entities(mesh_B._cpp_object, 0, locator_B)
    mesh_B.topology.create_connectivity(0, mesh_B.topology.dim)
    v_to_c = mesh_B.topology.connectivity(0, mesh_B.topology.dim)

    # Find all cells connected to vertex in the collision bounding box
    cells_B = np.sort(np.unique(np.hstack([v_to_c.links(vertex) for vertex in vertices_B])))

    # Find colliding entities using bounding box trees
    tree_A = bb_tree(mesh_A, mesh_A.topology.dim)
    tree_B = bb_tree(mesh_B, mesh_B.topology.dim)
    entities = compute_collisions_trees(tree_A, tree_B)
    entities_A = np.sort(np.unique([q[0] for q in entities]))
    entities_B = np.sort(np.unique([q[1] for q in entities]))
    assert np.allclose(entities_A, cells_A)
    assert np.allclose(entities_B, cells_B)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("point", [np.array([0.52, 0.51, 0.0]), np.array([0.9, -0.9, 0.0])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_tree_2d(point, dtype):
    mesh_A = create_unit_square(MPI.COMM_WORLD, 3, 3, dtype=dtype)
    mesh_B = create_unit_square(MPI.COMM_WORLD, 5, 5, dtype=dtype)
    bgeom = mesh_B.geometry.x
    bgeom += point
    tree_A = bb_tree(mesh_A, mesh_A.topology.dim)
    tree_B = bb_tree(mesh_B, mesh_B.topology.dim)
    entities = compute_collisions_trees(tree_A, tree_B)

    entities_A = np.sort(np.unique([q[0] for q in entities]))
    entities_B = np.sort(np.unique([q[1] for q in entities]))
    cells_A = find_colliding_cells(mesh_A, tree_B.get_bbox(tree_B.num_bboxes - 1), dtype)
    cells_B = find_colliding_cells(mesh_B, tree_A.get_bbox(tree_A.num_bboxes - 1), dtype)
    assert np.allclose(entities_A, cells_A)
    assert np.allclose(entities_B, cells_B)


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("point", [np.array([0.52, 0.51, 0.3]), np.array([0.9, -0.9, 0.3])])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_collisions_tree_3d(point, dtype):
    mesh_A = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, dtype=dtype)
    mesh_B = create_unit_cube(MPI.COMM_WORLD, 2, 2, 2, dtype=dtype)

    bgeom = mesh_B.geometry.x
    bgeom += point

    tree_A = bb_tree(mesh_A, mesh_A.topology.dim)
    tree_B = bb_tree(mesh_B, mesh_B.topology.dim)
    entities = compute_collisions_trees(tree_A, tree_B)
    entities_A = np.sort(np.unique([q[0] for q in entities]))
    entities_B = np.sort(np.unique([q[1] for q in entities]))
    cells_A = find_colliding_cells(mesh_A, tree_B.get_bbox(tree_B.num_bboxes - 1), dtype)
    cells_B = find_colliding_cells(mesh_B, tree_A.get_bbox(tree_A.num_bboxes - 1), dtype)
    assert np.allclose(entities_A, cells_A)
    assert np.allclose(entities_B, cells_B)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_entity_1d(dim, dtype):
    ref_distance = 0.75
    N = 16
    points = np.array([[-ref_distance, 0, 0], [2 / N, 2 * ref_distance, 0]], dtype=dtype)
    mesh = create_unit_interval(MPI.COMM_WORLD, N, dtype=dtype)
    tree = bb_tree(mesh, dim)
    num_entities_local = (
        mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    )
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = create_midpoint_tree(mesh, dim, entities)
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([[0, 0, 0], [2 / N, 0, 0]], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c)
        for i in range(points.shape[0]):
            # If colliding entity is on process
            if colliding_cells.links(i).size > 0:
                assert np.isin(closest_entities[i], colliding_cells.links(i))
    else:
        for i in range(points.shape[0]):
            # Only check closest entity if any bounding box on the
            # process intersects with the point
            if colliding_entity_bboxes.links(i).size > 0:
                assert np.isin(closest_entities[i], colliding_entity_bboxes.links(i))


@pytest.mark.parametrize("dim", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_entity_2d(dim, dtype):
    points = np.array([-1.0, -0.01, 0.0], dtype=dtype)
    mesh = create_unit_square(MPI.COMM_WORLD, 15, 15, dtype=dtype)
    mesh.topology.create_entities(dim)
    tree = bb_tree(mesh, dim)
    num_entities_local = (
        mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    )
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = create_midpoint_tree(mesh, dim, entities)

    # Find which entity is colliding with known closest point on mesh
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([0, 0, 0], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c).array
        if len(colliding_cells) > 0:
            assert np.isin(closest_entities[0], colliding_cells)
    else:
        if len(colliding_entity_bboxes.links(0)) > 0:
            assert np.isin(closest_entities[0], colliding_entity_bboxes.links(0))


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_entity_3d(dim, dtype):
    points = np.array([[0.9, 0, 1.135]], dtype=dtype)
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, dtype=dtype)
    mesh.topology.create_entities(dim)

    tree = bb_tree(mesh, dim)
    num_entities_local = (
        mesh.topology.index_map(dim).size_local + mesh.topology.index_map(dim).num_ghosts
    )
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = create_midpoint_tree(mesh, dim, entities)
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([0.9, 0, 1], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c).array
        if len(colliding_cells) > 0:
            assert np.isin(closest_entities[0], colliding_cells)
    else:
        if len(colliding_entity_bboxes.links(0)) > 0:
            assert np.isin(closest_entities[0], colliding_entity_bboxes.links(0))


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compute_closest_sub_entity(dim, dtype):
    """Compute distance from subset of cells in a mesh to a point inside the mesh"""
    ref_distance = 0.31
    xc, yc, zc = 0.5, 0.5, 0.5
    points = np.array([xc + ref_distance, yc, zc], dtype=dtype)
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, dtype=dtype)
    mesh.topology.create_entities(dim)
    left_entities = locate_entities(mesh, dim, lambda x: x[0] <= xc)
    tree = bb_tree(mesh, dim, left_entities)
    midpoint_tree = create_midpoint_tree(mesh, dim, left_entities)
    closest_entities = compute_closest_entity(tree, midpoint_tree, mesh, points)

    # Find which entity is colliding with known closest point on mesh
    p_c = np.array([xc, yc, zc], dtype=dtype)
    colliding_entity_bboxes = compute_collisions_points(tree, p_c)

    # Refine search by checking for actual collision if the entities are
    # cells
    if dim == mesh.topology.dim:
        colliding_cells = compute_colliding_cells(mesh, colliding_entity_bboxes, p_c).array
        if len(colliding_cells) > 0:
            assert np.isin(closest_entities[0], colliding_cells)
    else:
        if len(colliding_entity_bboxes.links(0)) > 0:
            assert np.isin(closest_entities[0], colliding_entity_bboxes.links(0))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_surface_bbtree(dtype):
    """Test creation of BBTree on subset of entities(surface cells)"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 8, 8, 8, dtype=dtype)
    mesh.topology.create_connectivity(mesh.topology.dim - 1, mesh.topology.dim)
    sf = exterior_facet_indices(mesh.topology)
    tdim = mesh.topology.dim
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = np.array([f_to_c.links(f)[0] for f in sf], dtype=np.int32)
    bbtree = bb_tree(mesh, tdim, cells)

    # test collision (should not collide with any)
    p = np.array([0.5, 0.5, 0.5])
    assert len(compute_collisions_points(bbtree, p).array) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sub_bbtree_codim1(dtype):
    """Testing point collision with a BoundingBoxTree of sub entities"""
    mesh = create_unit_cube(MPI.COMM_WORLD, 4, 4, 4, cell_type=CellType.hexahedron, dtype=dtype)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    top_facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[2], 1))
    f_to_c = mesh.topology.connectivity(tdim - 1, tdim)
    cells = np.array([f_to_c.links(f)[0] for f in top_facets], dtype=np.int32)
    bbtree = bb_tree(mesh, tdim, cells)

    # Compute a BBtree for all processes
    process_bbtree = bbtree.create_global_tree(mesh.comm)

    # Find possible ranks for this point
    point = np.array([0.2, 0.2, 1.0], dtype=dtype)
    ranks = compute_collisions_points(process_bbtree, point)

    # Compute local collisions
    cells = compute_collisions_points(bbtree, point)
    if MPI.COMM_WORLD.rank in ranks.array:
        assert len(cells.links(0)) > 0
    else:
        assert len(cells.links(0)) == 0


@pytest.mark.parametrize("comm", [MPI.COMM_WORLD, MPI.COMM_SELF])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_serial_global_bb_tree(dtype, comm):
    # Test if global bb tree with only one node returns the correct collision
    mesh = create_unit_cube(comm, 4, 5, 3)

    # First point should not be in any tree
    # Second point should always be in the global tree, but only in
    # entity tree with a serial mesh
    x = np.array([[2.0, 2.0, 3.0], [0.3, 0.2, 0.1]], dtype=dtype)

    tree = bb_tree(mesh, mesh.topology.dim)
    global_tree = tree.create_global_tree(mesh.comm)

    tree_col = compute_collisions_points(tree, x)
    global_tree_col = compute_collisions_points(global_tree, x)
    assert len(tree_col.links(0)) == 0 and len(global_tree_col.links(0)) == 0
    assert len(global_tree_col.links(1)) > 0
    # Only guaranteed local tree collision if mesh is on one process
    if comm.size == 1:
        assert len(tree_col.links(1)) > 0


@pytest.mark.parametrize("ct", [CellType.hexahedron, CellType.tetrahedron])
@pytest.mark.parametrize("N", [7, 13])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sub_bbtree_box(ct, N, dtype):
    """Test that the bounding box of the stem of the bounding box tree is what we expect"""
    mesh = create_unit_cube(MPI.COMM_WORLD, N, N, N, cell_type=ct, dtype=dtype)
    tdim = mesh.topology.dim
    fdim = tdim - 1

    facets = locate_entities_boundary(mesh, fdim, lambda x: np.isclose(x[1], 1.0))
    f_to_c = mesh.topology.connectivity(fdim, tdim)
    cells = np.int32(np.unique([f_to_c.links(f)[0] for f in facets]))
    bbtree = bb_tree(mesh, tdim, cells)
    num_boxes = bbtree.num_bboxes
    if num_boxes > 0:
        bbox = bbtree.get_bbox(num_boxes - 1)
        assert np.isclose(bbox[0][1], (N - 1) / N)
    tree = bb_tree(mesh, tdim)
    assert num_boxes < tree.num_bboxes


@pytest.mark.skip_in_parallel
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_surface_bbtree_collision(dtype):
    """Compute collision between two meshes, where only one cell of each mesh are colliding"""
    tdim = 3
    mesh1 = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron, dtype=dtype)
    mesh2 = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, CellType.hexahedron, dtype=dtype)
    mesh2.geometry.x[:, :] += np.array([0.9, 0.9, 0.9])

    mesh1.topology.create_connectivity(mesh1.topology.dim - 1, mesh1.topology.dim)
    sf = exterior_facet_indices(mesh1.topology)
    f_to_c = mesh1.topology.connectivity(tdim - 1, tdim)

    # Compute unique set of cells (some will be counted multiple times)
    cells = np.array(list(set([f_to_c.links(f)[0] for f in sf])), dtype=np.int32)
    bbtree1 = bb_tree(mesh1, tdim, cells)

    mesh2.topology.create_connectivity(mesh2.topology.dim - 1, mesh2.topology.dim)
    sf = exterior_facet_indices(mesh2.topology)
    f_to_c = mesh2.topology.connectivity(tdim - 1, tdim)
    cells = np.array(list(set([f_to_c.links(f)[0] for f in sf])), dtype=np.int32)
    bbtree2 = bb_tree(mesh2, tdim, cells)

    collisions = compute_collisions_trees(bbtree1, bbtree2)
    assert len(collisions) == 1


@pytest.mark.parametrize("ct", [CellType.tetrahedron])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_shift_bbtree(ct, dtype):
    tdim = 3
    mesh = create_unit_cube(MPI.COMM_WORLD, 3, 3, 3, ct, dtype=dtype)
    bbtree = bb_tree(mesh, tdim, padding=0.0)
    rng = np.random.default_rng(0)
    points = rng.random((10, 3))

    # Point-tree collisions pre-motion
    collisions_pre = compute_collisions_points(bbtree, points)
    # Shift everything
    shift = np.array([1.0, 2.0, 3.0], dtype=dtype)
    points[:] += shift
    bbtree.bbox_coordinates[:] += shift

    collisions_post = compute_collisions_points(bbtree, points)
    assert (collisions_pre.array == collisions_post.array).all()
