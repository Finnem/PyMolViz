"""Regular voxel lattice stored on a Field (origin, steps, values)."""

from __future__ import annotations

import numpy as np


def has_lattice(obj) -> bool:
    return obj is not None and getattr(obj, "_values", None) is not None


def init_lattice(obj, values, positions=None, step_sizes=None, step_counts=None, origin=None):
    obj.step_sizes = step_sizes
    obj.step_counts = step_counts
    obj.origin = origin
    if getattr(obj, "A_to", None) is None:
        obj.A_to = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=float)

    values = np.array(values).flatten()

    if positions is None:
        if (obj.step_counts is None) or (obj.step_sizes is None):
            raise ValueError("Either positions or step_sizes and step_counts must be given.")
        sorted_indices = np.arange(len(values))
        if origin is None:
            obj.origin = np.array([0, 0, 0], dtype=np.float64)
    else:
        positions = np.array(positions)
        sorted_indices = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
        positions = positions[sorted_indices]
        if obj.step_sizes is None:
            obj.step_sizes = np.zeros(3)
            for i in range(3):
                unique = np.unique(positions[:, i])
                sorted_u = np.sort(unique)
                obj.step_sizes[i] = np.max(np.diff(sorted_u))
        if obj.step_counts is None:
            obj.step_counts = np.zeros(3)
            for i in range(3):
                length = np.max(positions[:, i]) - np.min(positions[:, i])
                obj.step_counts[i] = np.round(length / obj.step_sizes[i])
        if obj.origin is None:
            obj.origin = np.min(positions, axis=0).astype(np.float64)

    obj.step_counts = np.array(obj.step_counts, dtype=int)
    if len(values) != np.prod(obj.step_counts + 1):
        raise ValueError(
            f"Number of values ({len(values)}) does not match number of grid points ({np.prod(obj.step_counts + 1)})."
        )
    obj.sorted_indices = sorted_indices
    obj._values = values[sorted_indices]
    obj.step_sizes = np.array(obj.step_sizes)
    obj.is_loaded = False
    return obj


def copy_lattice_onto(dest, src):
    if dest is None or src is None or dest is src:
        return dest
    dest._values = np.asarray(getattr(src, "_values", src.values), dtype=float).reshape(-1).copy()
    dest.origin = np.asarray(src.origin, dtype=float).reshape(3).copy()
    dest.step_sizes = np.asarray(src.step_sizes, dtype=float).reshape(3).copy()
    dest.step_counts = np.asarray(src.step_counts, dtype=int).reshape(3).copy()
    dest.sorted_indices = np.asarray(
        getattr(src, "sorted_indices", np.arange(dest._values.size)), dtype=int
    ).copy()
    ttt = getattr(src, "A_to", None)
    dest.A_to = np.array(ttt, dtype=float, copy=True) if ttt is not None else np.eye(4)
    dest.is_loaded = False
    for attr in (
        "rgb_values",
        "color_stops",
        "_crystal_cell",
        "_crystal_axis_grid",
        "_crystal_lookup",
        "_crystal_copied",
    ):
        if hasattr(src, attr):
            try:
                setattr(dest, attr, getattr(src, attr))
            except Exception:
                pass
    src_name = getattr(src, "_name", None)
    if src_name and not getattr(dest, "_name", None):
        try:
            dest._name = src_name
        except Exception:
            pass
    return dest


def get_positions(obj):
    origin = np.asarray(obj.origin, dtype=float).reshape(3)
    step = np.asarray(obj.step_sizes, dtype=float).reshape(3)
    counts = np.asarray(obj.step_counts, dtype=int).reshape(3)
    x = np.linspace(origin[0], origin[0] + step[0] * counts[0], counts[0] + 1)
    y = np.linspace(origin[1], origin[1] + step[1] * counts[1], counts[1] + 1)
    z = np.linspace(origin[2], origin[2] + step[2] * counts[2], counts[2] + 1)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    xx, yy, zz = xx.flatten(), yy.flatten(), zz.flatten()
    order = getattr(obj, "sorted_indices", None)
    if order is not None:
        xx, yy, zz = xx[order], yy[order], zz[order]
    positions = np.array([xx, yy, zz]).T
    sorted_indices = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0]))
    return positions[sorted_indices]


def to_points(obj, filter=None, *args, **kwargs):
    from ..meshes.Points import Points

    positions = get_positions(obj)
    values = obj.values
    if filter is None:
        filtered_positions = positions
        filtered_values = values
    else:
        filtered_positions = positions[filter(values)]
        filtered_values = values[filter(values)]
    return Points(filtered_positions, filtered_values, *args, **kwargs)


def script_string(obj):
    values = np.asarray(obj.values).reshape(np.asarray(obj.step_counts, dtype=int) + 1)
    name = obj.name
    a_to = getattr(obj, "A_to", np.eye(4))
    return f"""
{name}_data = np.array({np.array2string(values, threshold=1e15, separator=",")})
{name} = Brick.from_numpy({name}_data, {np.array2string(np.asarray(obj.step_sizes), separator=",")}, origin={np.array2string(np.asarray(obj.origin), separator=",")})
cmd.load_brick({name}, "{name}")
cmd.set("volume_mode", 0)
cmd.set_object_ttt("{name}", {list(np.asarray(a_to).flatten())})
{name}_data = None

"""


def load_lattice(obj, cmd=None):
    if cmd is None:
        from pymol import cmd
    if getattr(obj, "is_loaded", False):
        return
    cmd.delete(obj.name)
    values = np.asarray(obj.values).reshape(np.asarray(obj.step_counts, dtype=int) + 1)
    try:
        from chempy.brick import Brick

        brick = Brick.from_numpy(values, obj.step_sizes, origin=obj.origin)
    except Exception:
        brick = obj
    cmd.load_brick(brick, obj.name)
    cmd.set("volume_mode", 0)
    cmd.set_object_ttt(obj.name, list(np.asarray(getattr(obj, "A_to", np.eye(4))).flatten()))
    obj.is_loaded = True


def _plane_intersections(obj, point, normal):
    x_dir = np.array([1, 0, 0])
    y_dir = np.array([0, 1, 0])
    z_dir = np.array([0, 0, 1])
    min_pos = obj.origin
    max_pos = obj.origin + obj.step_sizes * obj.step_counts
    x_corners = np.array(
        [
            [min_pos[0], min_pos[1], min_pos[2]],
            [min_pos[0], max_pos[1], min_pos[2]],
            [min_pos[0], min_pos[1], max_pos[2]],
            [min_pos[0], max_pos[1], max_pos[2]],
        ]
    )
    y_corners = np.array(
        [
            [min_pos[0], min_pos[1], min_pos[2]],
            [max_pos[0], min_pos[1], min_pos[2]],
            [min_pos[0], min_pos[1], max_pos[2]],
            [max_pos[0], min_pos[1], max_pos[2]],
        ]
    )
    z_corners = np.array(
        [
            [min_pos[0], min_pos[1], min_pos[2]],
            [max_pos[0], min_pos[1], min_pos[2]],
            [min_pos[0], max_pos[1], min_pos[2]],
            [max_pos[0], max_pos[1], min_pos[2]],
        ]
    )
    intersections = []
    if np.dot(x_dir, normal) != 0:
        d_x = np.dot(point - x_corners, normal) / np.dot(x_dir, normal)
        x_intersections = x_corners + d_x[:, np.newaxis] * x_dir
        intersections.extend(x_intersections[(d_x >= 0) & (d_x <= max_pos[0] - min_pos[0])])
    if np.dot(y_dir, normal) != 0:
        d_y = np.dot(point - y_corners, normal) / np.dot(y_dir, normal)
        y_intersections = y_corners + d_y[:, np.newaxis] * y_dir
        intersections.extend(y_intersections[(d_y >= 0) & (d_y <= max_pos[1] - min_pos[1])])
    if np.dot(z_dir, normal) != 0:
        d_z = np.dot(point - z_corners, normal) / np.dot(z_dir, normal)
        z_intersections = z_corners + d_z[:, np.newaxis] * z_dir
        intersections.extend(z_intersections[(d_z >= 0) & (d_z <= max_pos[2] - min_pos[2])])
    return intersections


def _projections_on_plane(obj, point, normal):
    min_pos = obj.origin
    max_pos = obj.origin + obj.step_sizes * obj.step_counts
    x_corners = np.array(
        [
            [min_pos[0], min_pos[1], min_pos[2]],
            [min_pos[0], max_pos[1], min_pos[2]],
            [min_pos[0], min_pos[1], max_pos[2]],
            [min_pos[0], max_pos[1], max_pos[2]],
        ]
    )
    y_corners = np.array(
        [
            [min_pos[0], min_pos[1], min_pos[2]],
            [max_pos[0], min_pos[1], min_pos[2]],
            [min_pos[0], min_pos[1], max_pos[2]],
            [max_pos[0], min_pos[1], max_pos[2]],
        ]
    )
    z_corners = np.array(
        [
            [min_pos[0], min_pos[1], min_pos[2]],
            [max_pos[0], min_pos[1], min_pos[2]],
            [min_pos[0], max_pos[1], min_pos[2]],
            [max_pos[0], max_pos[1], min_pos[2]],
        ]
    )
    projected_corners = []
    for corner in np.unique(
        np.concatenate((x_corners, y_corners, z_corners, np.array([max_pos, min_pos]))), axis=0
    ):
        d = np.dot((corner - point), normal)
        if d > 0:
            projected_corners.append(corner - d * normal)
    return np.array(projected_corners).reshape(-1, 3)


def cut(obj, point, normal, interpolation="NN"):
    from .field import Field
    from ..util.vector_functions import get_new_basis_vectors

    normal = np.array(normal / np.linalg.norm(normal))
    point = np.array(point)
    intersections = _plane_intersections(obj, point, normal)
    projected_corners = _projections_on_plane(obj, point, normal)
    plane_coords = np.concatenate((intersections, projected_corners))
    c1, t, c2 = get_new_basis_vectors(plane_coords, point, normal)
    M_to = np.stack((c1, c2, normal), axis=1)
    M_from = np.linalg.inv(M_to)
    x = np.linspace(obj.origin[0], obj.origin[0] + obj.step_sizes[0] * obj.step_counts[0], obj.step_counts[0] + 1)
    y = np.linspace(obj.origin[1], obj.origin[1] + obj.step_sizes[1] * obj.step_counts[1], obj.step_counts[1] + 1)
    z = np.linspace(obj.origin[2], obj.origin[2] + obj.step_sizes[2] * obj.step_counts[2], obj.step_counts[2] + 1)
    xx, yy, zz = np.meshgrid(x, y, z)
    old_positions_original_space = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    old_positions = np.matmul(M_from, (np.array(old_positions_original_space.T) - np.array(t).reshape(-1, 1))).T
    max_point = [np.max(old_positions[:, 0]), np.max(old_positions[:, 1]), np.max(old_positions[:, 2])]
    old_values = obj.values
    new_origin = [np.min(old_positions[:, 0]), np.min(old_positions[:, 1]), 0]
    new_step_count = np.ceil(np.array(np.array(max_point) - np.array(new_origin)) / np.array(obj.step_sizes))
    new_step_sizes = np.abs(np.array(np.array(max_point) - np.array(new_origin)) / np.array(new_step_count))
    x = np.arange(new_origin[0], max_point[0] + new_step_sizes[0], step=new_step_sizes[0])
    y = np.arange(new_origin[1], max_point[1] + new_step_sizes[1], step=new_step_sizes[1])
    z = np.arange(new_origin[2], max_point[2] + new_step_sizes[2], step=new_step_sizes[2])
    xx, yy, zz = np.meshgrid(x, y, z)
    new_positions = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    new_positions_original_space = np.array(np.matmul(M_to, new_positions.T) + np.array(t).reshape(-1, 1)).T
    inner_positions_index = np.where(
        (new_positions_original_space[:, 0] < np.max(old_positions_original_space[:, 0]))
        & (new_positions_original_space[:, 0] > np.min(old_positions_original_space[:, 0]))
        & (new_positions_original_space[:, 1] < np.max(old_positions_original_space[:, 1]))
        & (new_positions_original_space[:, 1] > np.min(old_positions_original_space[:, 1]))
        & (new_positions_original_space[:, 2] < np.max(old_positions_original_space[:, 2]))
        & (new_positions_original_space[:, 2] > np.min(old_positions_original_space[:, 2]))
    )
    inner_positions = new_positions_original_space[inner_positions_index]
    outer_position_index = np.where(
        (new_positions_original_space[:, 0] > np.max(old_positions_original_space[:, 0]))
        | (new_positions_original_space[:, 0] < np.min(old_positions_original_space[:, 0]))
        | (new_positions_original_space[:, 1] > np.max(old_positions_original_space[:, 1]))
        | (new_positions_original_space[:, 1] < np.min(old_positions_original_space[:, 1]))
        | (new_positions_original_space[:, 2] > np.max(old_positions_original_space[:, 2]))
        | (new_positions_original_space[:, 2] < np.min(old_positions_original_space[:, 2]))
    )
    outer_positions = new_positions_original_space[outer_position_index]
    new_values = np.zeros(len(new_positions))
    if interpolation == "Lin/NN":
        from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

        interp = LinearNDInterpolator(old_positions_original_space, old_values)
        new_values[inner_positions_index] = interp(inner_positions)
        interp = NearestNDInterpolator(old_positions_original_space, old_values)
        new_values[outer_position_index] = interp(outer_positions)
    else:
        from scipy.interpolate import NearestNDInterpolator

        interp = NearestNDInterpolator(old_positions_original_space, old_values)
        new_values = interp(new_positions_original_space)
    new_field = Field(new_values, new_positions, name="%s_cut" % obj.name)
    new_field.A_to = np.stack(
        (np.concatenate((c1, [0])), np.concatenate((c2, [0])), np.concatenate((normal, [0])), np.concatenate((t, [1]))),
        axis=1,
    )
    if getattr(obj, "is_loaded", False):
        new_field.load()
    return new_field

