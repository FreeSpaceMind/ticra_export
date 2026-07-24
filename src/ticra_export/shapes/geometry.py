"""
Mesh geometry builders for shadow-shape 3D visualization.

Pure-numpy grid builders shared by the shadow shapes tab preview and the
main 3D reflector viewer. Each returns coordinate arrays that callers
wrap in their own plotting primitives (e.g. pyvista StructuredGrid).

Rotational-copy semantics match custom_shadow: phi_deg is the shape's
phi plus the copy step, applied as a rotation about the global z-axis.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


#: Palette for structure shapes (RGB in 0..1). One color per shape
#: definition: rotational copies share it, boolean groups take the
#: first member's color, and subtracted members use the complement.
STRUCTURE_PALETTE = (
    (0.85, 0.20, 0.20),   # red
    (0.20, 0.35, 0.85),   # blue
    (0.20, 0.70, 0.30),   # green
    (0.90, 0.60, 0.10),   # orange
    (0.15, 0.70, 0.75),   # teal
    (0.75, 0.25, 0.75),   # purple
)


def structure_color(index: int) -> Tuple[float, float, float]:
    """Deterministic display color for a shape definition index."""
    return STRUCTURE_PALETTE[int(index) % len(STRUCTURE_PALETTE)]


def complementary_color(rgb) -> Tuple[float, float, float]:
    """RGB complement, used for subtracted boolean members."""
    return tuple(1.0 - float(c) for c in rgb)


def structure_color_map(shapes) -> dict:
    """Map id(shape) -> RGB for a shapes list.

    Every shape definition gets its own palette color by list position;
    members of a boolean group all share the group's color (the first
    member's), with subtracted members mapped to the complement.
    """
    from .definitions import ShapeDefinition

    group_color = {}
    colors = {}
    for index, shape in enumerate(shapes):
        base = structure_color(index)
        group = getattr(shape, 'boolean_group', '') or ''
        in_group = bool(group) and \
            shape.shape_type in ShapeDefinition.BOOLEAN_SOLID_TYPES
        if in_group:
            base = group_color.setdefault(group, base)
            if getattr(shape, 'boolean_mode', 'add') == 'subtract':
                colors[id(shape)] = complementary_color(base)
                continue
        colors[id(shape)] = base
    return colors


def strut_segment_at_phi(shape, phi_deg: float) -> Tuple[np.ndarray, np.ndarray, float]:
    """Strut axis endpoints and wire radius for one rotational copy.

    Returns (start_xyz, end_xyz, radius) with the endpoints rotated
    about the z-axis by phi_deg.
    """
    phi_rad = np.radians(phi_deg)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)

    def rotate(point):
        return np.array([
            point[0] * cos_phi - point[1] * sin_phi,
            point[0] * sin_phi + point[1] * cos_phi,
            point[2],
        ])

    start = rotate(np.asarray(shape.start_point, dtype=float))
    end = rotate(np.asarray(shape.end_point, dtype=float))
    return start, end, shape.diameter / 2.0


def strut_surface_grid(shape, phi_deg: float,
                       n_theta: int = 24) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cylindrical tube surface along the strut segment.

    Returns X, Y, Z arrays of shape (2, n_theta + 1) suitable for a
    structured grid: one ring at each strut end.
    """
    start, end, radius = strut_segment_at_phi(shape, phi_deg)

    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        axis = np.array([0.0, 0.0, 1.0])
    else:
        axis = axis / length

    # Build an orthonormal frame (u, v) perpendicular to the axis
    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, helper)) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, helper)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    ring = radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))

    points = np.stack([start + ring, end + ring])  # (2, n_theta+1, 3)
    return points[:, :, 0], points[:, :, 1], points[:, :, 2]


def bor_surface_grid(shape, phi_deg: float,
                     n_theta: int = 48) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Surface of revolution for a body-of-revolution shape.

    Revolves the piecewise-linear rho(z) profile about the vertical axis
    through the shape's (rho, phi) location. Returns X, Y, Z arrays of
    shape (n_profile, n_theta + 1).
    """
    profile = np.asarray(shape.profile_points, dtype=float)
    if profile.ndim != 2 or len(profile) < 2:
        raise ValueError("Body of revolution requires at least 2 profile points")

    phi_rad = np.radians(phi_deg)
    x_center = shape.rho * np.cos(phi_rad)
    y_center = shape.rho * np.sin(phi_rad)

    z_prof = profile[:, 0]
    rho_prof = profile[:, 1]

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    R, T = np.meshgrid(rho_prof, theta, indexing="ij")
    Z = np.broadcast_to((shape.z + z_prof)[:, np.newaxis], R.shape).copy()

    X = x_center + R * np.cos(T)
    Y = y_center + R * np.sin(T)
    return X, Y, Z


def mesh_shape_polydata_arrays(shape, phi_deg: float) -> Tuple[np.ndarray, np.ndarray]:
    """Points and pyvista-style face array for a mesh-based shape copy.

    Curved quadratic patches are tessellated to flat quads for display;
    triangle patches stay triangles. Returns (points [N x 3], faces)
    with faces in pyvista PolyData layout: [n, i0, ..., n, ...].
    """
    from .mesh_shapes import build_shape_mesh

    return build_shape_mesh(shape, phi_deg).display_faces()


def boolean_group_polydata_arrays(members,
                                  all_quad: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Points and pyvista-style face array for a boolean group result.

    Raises RuntimeError when manifold3d is unavailable and ValueError
    for invalid groups; callers fall back to drawing members
    individually.
    """
    from .mesh_shapes import combine_shape_group

    return combine_shape_group(members, all_quad=all_quad).display_faces()


def revolve_profile(r_profile, z_profile,
                    n_theta: int = 60) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Revolve a reflector radial profile z(r) about the global z-axis.

    Returns X, Y, Z arrays of shape (n_profile, n_theta + 1) for context
    surfaces (primary/subreflector) in shape previews.
    """
    r = np.asarray(r_profile, dtype=float).reshape(-1)
    z = np.asarray(z_profile, dtype=float).reshape(-1)

    theta = np.linspace(0.0, 2.0 * np.pi, n_theta + 1)
    R, T = np.meshgrid(r, theta, indexing="ij")
    Z = np.broadcast_to(z[:, np.newaxis], R.shape).copy()
    return R * np.cos(T), R * np.sin(T), Z
