"""
Quad-mesh generators for tabulated-mesh shadow shapes.

Every mesh-based shape tessellates to one common intermediate
representation -- nodes plus quad patches -- which feeds three
consumers: the generic ray-vs-mesh shadow mask, the 3D previews, and
the TICRA tabulated_mesh_table export.

Patches come in two kinds, matching GRASP's tabulated mesh:
- linear patches: 4 corner-node indices, a flat quad
- curved quadratic patches: 9 node indices forming a biquadratic
  surface -- corners 1-4 (counter-clockwise), mid-edge nodes 5-8
  (between corners 1-2, 2-3, 3-4, 4-1), and the center node 9

All node indices are 0-based here; the .tor writer converts to the
1-based indices GRASP uses. Only solid PEC surfaces are represented
(patch region -1 in the export).

Boolean combination of closed-solid shapes is provided through the
optional manifold3d package (results are triangle meshes; triangles
export as degenerate quads with the last node repeated).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

try:
    import manifold3d
    MANIFOLD_AVAILABLE = True
except ImportError:
    MANIFOLD_AVAILABLE = False


class QuadMesh:
    """Nodes plus triangle, flat quad, and curved quadratic patches."""

    def __init__(self, nodes: np.ndarray,
                 linear_patches: Optional[np.ndarray] = None,
                 curved_patches: Optional[np.ndarray] = None,
                 tri_patches: Optional[np.ndarray] = None):
        self.nodes = np.asarray(nodes, dtype=float).reshape(-1, 3)
        self.linear_patches = (
            np.asarray(linear_patches, dtype=int).reshape(-1, 4)
            if linear_patches is not None and len(linear_patches) > 0
            else np.zeros((0, 4), dtype=int))
        self.curved_patches = (
            np.asarray(curved_patches, dtype=int).reshape(-1, 9)
            if curved_patches is not None and len(curved_patches) > 0
            else np.zeros((0, 9), dtype=int))
        self.tri_patches = (
            np.asarray(tri_patches, dtype=int).reshape(-1, 3)
            if tri_patches is not None and len(tri_patches) > 0
            else np.zeros((0, 3), dtype=int))

    @property
    def patch_count(self) -> int:
        return (len(self.linear_patches) + len(self.curved_patches)
                + len(self.tri_patches))

    def rotated_about_z(self, phi_deg: float) -> "QuadMesh":
        """Copy of this mesh rotated about the global z-axis."""
        phi_rad = np.radians(phi_deg)
        cos_phi = np.cos(phi_rad)
        sin_phi = np.sin(phi_rad)
        nodes = self.nodes.copy()
        x = nodes[:, 0].copy()
        y = nodes[:, 1].copy()
        nodes[:, 0] = x * cos_phi - y * sin_phi
        nodes[:, 1] = x * sin_phi + y * cos_phi
        return QuadMesh(nodes, self.linear_patches, self.curved_patches,
                        self.tri_patches)

    def display_quads(self) -> Tuple[np.ndarray, np.ndarray]:
        """Flat-quad tessellation for shadow testing.

        Curved patches are subdivided into the four flat quads of their
        3x3 node grid; triangles become degenerate quads (last corner
        repeated). Returns (points, quads) with quads as 0-based
        corner-index rows into points.
        """
        quads: List[List[int]] = [list(p) for p in self.linear_patches]
        for patch in self.curved_patches:
            c1, c2, c3, c4, m12, m23, m34, m41, ctr = patch
            quads.extend([
                [c1, m12, ctr, m41],
                [m12, c2, m23, ctr],
                [ctr, m23, c3, m34],
                [m41, ctr, m34, c4],
            ])
        for a, b, c in self.tri_patches:
            quads.append([a, b, c, c])
        return self.nodes, np.asarray(quads, dtype=int).reshape(-1, 4)

    def display_faces(self) -> Tuple[np.ndarray, np.ndarray]:
        """Points and pyvista-style mixed face array for rendering.

        Faces layout: [n, i0, ..., in-1, n, ...] with true triangles
        kept as triangles (no degenerate quads on screen).
        """
        faces: List[int] = []
        _, quads = self.display_quads()
        for patch in quads:
            if patch[2] == patch[3]:
                faces.extend([3, patch[0], patch[1], patch[2]])
            else:
                faces.extend([4, *patch])
        return self.nodes, np.asarray(faces, dtype=np.int64)

    def triangles(self) -> np.ndarray:
        """All patches as a (T, 3, 3) triangle-vertex array."""
        points, quads = self.display_quads()
        if len(quads) == 0:
            return np.zeros((0, 3, 3))
        tris = []
        for a, b, c, d in quads:
            tris.append([points[a], points[b], points[c]])
            if c != d:
                tris.append([points[a], points[c], points[d]])
        return np.asarray(tris)

    def triangle_indices(self) -> np.ndarray:
        """All patches as (T, 3) 0-based node-index triangles."""
        _, quads = self.display_quads()
        tris = []
        for a, b, c, d in quads:
            tris.append([a, b, c])
            if c != d:
                tris.append([a, c, d])
        return (np.asarray(tris, dtype=int) if tris
                else np.zeros((0, 3), dtype=int))


def box_mesh(center, dims, rotation_z_deg: float = 0.0) -> QuadMesh:
    """Axis-aligned box (before z-rotation) of size dims about center.

    Six flat quads with outward-facing counter-clockwise winding.
    """
    cx, cy, cz = (float(v) for v in center)
    hx, hy, hz = (float(d) / 2.0 for d in dims)

    corners = np.array([
        [-hx, -hy, -hz], [hx, -hy, -hz], [hx, hy, -hz], [-hx, hy, -hz],
        [-hx, -hy, hz], [hx, -hy, hz], [hx, hy, hz], [-hx, hy, hz],
    ])
    faces = np.array([
        [0, 3, 2, 1],  # bottom (-z)
        [4, 5, 6, 7],  # top (+z)
        [0, 1, 5, 4],  # -y
        [2, 3, 7, 6],  # +y
        [1, 2, 6, 5],  # +x
        [3, 0, 4, 7],  # -x
    ])

    phi_rad = np.radians(rotation_z_deg)
    cos_phi = np.cos(phi_rad)
    sin_phi = np.sin(phi_rad)
    x = corners[:, 0].copy()
    y = corners[:, 1].copy()
    corners[:, 0] = cx + x * cos_phi - y * sin_phi
    corners[:, 1] = cy + x * sin_phi + y * cos_phi
    corners[:, 2] += cz
    return QuadMesh(corners, faces)


def rect_strut_mesh(start, end, width: float, thickness: float) -> QuadMesh:
    """Rectangular-cross-section strut between two 3D points.

    The cross section is width x thickness: width along the horizontal
    direction perpendicular to the strut axis, thickness along the
    remaining perpendicular. Closed at both ends (8 nodes, 6 patches),
    like the hand-built tabulated_mesh_table struts in Job_16.tor.
    """
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        raise ValueError("Strut start and end points coincide")
    axis = axis / length

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, helper)) > 0.99:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, axis)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    hu = float(width) / 2.0
    hv = float(thickness) / 2.0
    section = [(-hu, -hv), (hu, -hv), (hu, hv), (-hu, hv)]
    nodes = np.array(
        [a + su * u + sv * v for su, sv in section] +
        [b + su * u + sv * v for su, sv in section])
    faces = np.array([
        [0, 3, 2, 1],              # start cap
        [4, 5, 6, 7],              # end cap
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [2, 3, 7, 6],
        [3, 0, 4, 7],
    ])
    return QuadMesh(nodes, faces)


def cylinder_shell_mesh(center_xy, z_bottom: float, height: float,
                        radius: float, span_deg: float = 360.0,
                        start_deg: float = 0.0,
                        n_segments: int = 8,
                        n_axial: int = 1) -> QuadMesh:
    """Open cylindrical shell built from curved quadratic patches.

    A single-surface (sheet) PEC shell about a vertical axis: radius
    ``radius``, from ``z_bottom`` to ``z_bottom + height``, covering
    ``span_deg`` of azimuth starting at ``start_deg``. Each patch is a
    9-node biquadratic patch sampled on the true cylinder, so the
    exported surface is exactly curved rather than faceted.
    """
    if radius <= 0.0 or height <= 0.0:
        raise ValueError("Cylinder shell needs positive radius and height")
    n_segments = max(1, int(n_segments))
    n_axial = max(1, int(n_axial))
    span_deg = min(max(float(span_deg), 1e-6), 360.0)
    full_ring = abs(span_deg - 360.0) < 1e-9

    n_cols = 2 * n_segments + 1
    n_rows = 2 * n_axial + 1
    theta = np.radians(start_deg) + np.radians(span_deg) * \
        np.arange(n_cols) / (n_cols - 1)
    z_levels = z_bottom + height * np.arange(n_rows) / (n_rows - 1)

    n_unique_cols = n_cols - 1 if full_ring else n_cols

    def node_index(row: int, col: int) -> int:
        return row * n_unique_cols + (col % n_unique_cols)

    cx, cy = (float(v) for v in center_xy)
    nodes = np.array([
        [cx + radius * np.cos(theta[col]),
         cy + radius * np.sin(theta[col]),
         z_levels[row]]
        for row in range(n_rows) for col in range(n_unique_cols)
    ])

    patches = []
    for i in range(n_axial):
        for j in range(n_segments):
            r0, c0 = 2 * i, 2 * j
            patches.append([
                node_index(r0, c0),          # corner 1
                node_index(r0, c0 + 2),      # corner 2
                node_index(r0 + 2, c0 + 2),  # corner 3
                node_index(r0 + 2, c0),      # corner 4
                node_index(r0, c0 + 1),      # mid edge 1-2
                node_index(r0 + 1, c0 + 2),  # mid edge 2-3
                node_index(r0 + 2, c0 + 1),  # mid edge 3-4
                node_index(r0 + 1, c0),      # mid edge 4-1
                node_index(r0 + 1, c0 + 1),  # center
            ])
    return QuadMesh(nodes, None, patches)


def _polygon_signed_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _ear_clip(points: np.ndarray) -> List[List[int]]:
    """Triangulate a simple (possibly non-convex) CCW polygon.

    Standard ear clipping; O(n^2), fine for cross-section-sized
    polygons. Returns triangles as index triplets into points.
    """
    n = len(points)
    if n < 3:
        raise ValueError("Polygon needs at least 3 vertices")
    indices = list(range(n))
    triangles: List[List[int]] = []

    def cross_z(o, a, b):
        return ((points[a, 0] - points[o, 0]) * (points[b, 1] - points[o, 1])
                - (points[a, 1] - points[o, 1]) * (points[b, 0] - points[o, 0]))

    def point_in_triangle(p, a, b, c):
        d1 = cross_z(a, b, p)
        d2 = cross_z(b, c, p)
        d3 = cross_z(c, a, p)
        return (d1 >= -1e-15) and (d2 >= -1e-15) and (d3 >= -1e-15)

    guard = 0
    while len(indices) > 3:
        guard += 1
        if guard > 2 * n * n:
            raise ValueError("Polygon triangulation failed (self-"
                             "intersecting or degenerate cross-section?)")
        ear_found = False
        for i in range(len(indices)):
            prev_i = indices[i - 1]
            curr_i = indices[i]
            next_i = indices[(i + 1) % len(indices)]
            if cross_z(prev_i, curr_i, next_i) <= 1e-15:
                continue  # reflex or collinear corner
            if any(point_in_triangle(other, prev_i, curr_i, next_i)
                   for other in indices
                   if other not in (prev_i, curr_i, next_i)):
                continue
            triangles.append([prev_i, curr_i, next_i])
            indices.pop(i)
            ear_found = True
            break
        if not ear_found:
            raise ValueError("Polygon triangulation failed (self-"
                             "intersecting or degenerate cross-section?)")
    triangles.append(list(indices))
    return triangles


def extruded_polygon_mesh(start, end,
                          cross_section_uv) -> QuadMesh:
    """Closed prism: a 2D polygon swept between two 3D points.

    The cross-section is given as (u, v) pairs in the plane
    perpendicular to the sweep axis, using the same frame as
    rect_strut_mesh: u horizontal (perpendicular to the axis), v
    completing the right-handed frame. The polygon may be non-convex
    but must be simple (non-self-intersecting); winding is normalized
    internally. Side faces are quads; the end caps are ear-clipped
    triangles, so the solid is watertight and boolean-capable.
    """
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        raise ValueError("Extrusion start and end points coincide")
    axis = axis / length

    section = np.asarray(cross_section_uv, dtype=float).reshape(-1, 2)
    if len(section) < 3:
        raise ValueError("Cross-section needs at least 3 points")
    if abs(_polygon_signed_area(section)) < 1e-15:
        raise ValueError("Cross-section polygon is degenerate")
    if _polygon_signed_area(section) < 0.0:
        section = section[::-1].copy()

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, helper)) > 0.99:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, axis)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    n = len(section)
    nodes = np.array(
        [a + su * u + sv * v for su, sv in section] +
        [b + su * u + sv * v for su, sv in section])

    quads = []
    for i in range(n):
        j = (i + 1) % n
        # CCW section + axis toward b -> outward side winding
        quads.append([i, j, n + j, n + i])

    cap_triangles = _ear_clip(section)
    tris = []
    for t1, t2, t3 in cap_triangles:
        tris.append([t1, t3, t2])                  # start cap faces -axis
        tris.append([n + t1, n + t2, n + t3])      # end cap faces +axis
    return QuadMesh(nodes, quads, None, tris)


def solid_cylinder_mesh(center_xy, z_bottom: float, height: float,
                        radius: float, n_segments: int = 16) -> QuadMesh:
    """Closed (watertight) faceted cylinder solid.

    Flat side quads plus triangle-fan caps, wound outward so the solid
    can participate in boolean operations. Unlike cylinder_shell_mesh
    this is a faceted approximation (no curved patches) -- booleans
    operate on flat triangles only.
    """
    if radius <= 0.0 or height <= 0.0:
        raise ValueError("Solid cylinder needs positive radius and height")
    n_segments = max(3, int(n_segments))
    cx, cy = (float(v) for v in center_xy)
    z_top = z_bottom + height

    theta = 2.0 * np.pi * np.arange(n_segments) / n_segments
    ring_x = cx + radius * np.cos(theta)
    ring_y = cy + radius * np.sin(theta)

    nodes = []
    for z in (z_bottom, z_top):
        for i in range(n_segments):
            nodes.append([ring_x[i], ring_y[i], z])
    bottom_center = len(nodes)
    nodes.append([cx, cy, z_bottom])
    top_center = len(nodes)
    nodes.append([cx, cy, z_top])
    nodes = np.asarray(nodes)

    quads = []
    tris = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # Side quad, outward winding
        quads.append([i, j, n_segments + j, n_segments + i])
        # Caps: bottom faces -z, top faces +z
        tris.append([bottom_center, j, i])
        tris.append([top_center, n_segments + i, n_segments + j])
    return QuadMesh(nodes, quads, None, tris)


def _signed_volume(mesh: QuadMesh) -> float:
    """Signed volume via the divergence theorem over the triangles."""
    tris = mesh.triangles()
    if len(tris) == 0:
        return 0.0
    v0 = tris[:, 0]
    v1 = tris[:, 1]
    v2 = tris[:, 2]
    return float(np.sum(np.einsum('ij,ij->i', v0, np.cross(v1, v2)))) / 6.0


def _ensure_outward(mesh: QuadMesh) -> QuadMesh:
    """Flip face winding if the mesh volume comes out negative."""
    if _signed_volume(mesh) >= 0.0:
        return mesh
    return QuadMesh(mesh.nodes,
                    mesh.linear_patches[:, ::-1]
                    if len(mesh.linear_patches) else None,
                    None,
                    mesh.tri_patches[:, ::-1]
                    if len(mesh.tri_patches) else None)


def revolve_solid_mesh(profile_zr, center_xy, z_offset: float,
                       n_segments: int = 32) -> QuadMesh:
    """Watertight solid of revolution from a (z, rho) profile trace.

    The profile is treated as the cross-section polygon in the (rho, z)
    half-plane, revolved about the vertical axis through center_xy:

    - A closed trace (first point repeated at the end, e.g. an annular
      ring or a plate with a hole) revolves to a closed ring solid.
    - An open trace rho(z) is closed through the axis (filled solid),
      matching the solid interpretation used by the shadow masks.

    Vertices with rho ~ 0 collapse to single axis nodes. Face winding
    is normalized to outward. The result participates in booleans.
    """
    profile = np.asarray(profile_zr, dtype=float).reshape(-1, 2)
    if len(profile) < 2:
        raise ValueError("Revolve profile needs at least 2 points")
    n_segments = max(3, int(n_segments))
    cx, cy = (float(v) for v in center_xy)

    closed = (len(profile) >= 4 and
              np.allclose(profile[0], profile[-1], atol=1e-12))
    if closed:
        poly = profile[:-1]
    else:
        # Close through the axis: ... -> (z_last, 0) -> (z_first, 0)
        poly = list(profile)
        if abs(profile[-1, 1]) > 1e-12:
            poly.append([profile[-1, 0], 0.0])
        if abs(profile[0, 1]) > 1e-12:
            poly.append([profile[0, 0], 0.0])
        poly = np.asarray(poly)
    if len(poly) < 3:
        raise ValueError("Revolve profile is degenerate")

    theta = 2.0 * np.pi * np.arange(n_segments) / n_segments
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    nodes = []
    ring_start = []   # per polygon vertex: node index (axis) or ring start
    is_axis = []
    for z_local, rho in poly:
        rho = float(rho)
        z_g = z_offset + float(z_local)
        if abs(rho) < 1e-12:
            ring_start.append(len(nodes))
            is_axis.append(True)
            nodes.append([cx, cy, z_g])
        else:
            ring_start.append(len(nodes))
            is_axis.append(False)
            for k in range(n_segments):
                nodes.append([cx + rho * cos_t[k], cy + rho * sin_t[k], z_g])
    nodes = np.asarray(nodes)

    quads = []
    tris = []
    n_poly = len(poly)
    for i in range(n_poly):
        j = (i + 1) % n_poly
        ai, aj = is_axis[i], is_axis[j]
        si, sj = ring_start[i], ring_start[j]
        if ai and aj:
            continue  # edge along the axis: no surface
        for k in range(n_segments):
            k2 = (k + 1) % n_segments
            # Fans are the collapsed form of the strip quad
            # [si+k, si+k2, sj+k2, sj+k] and must keep its winding.
            if ai:
                tris.append([si, sj + k2, sj + k])
            elif aj:
                tris.append([si + k, si + k2, sj])
            else:
                quads.append([si + k, si + k2, sj + k2, sj + k])
    return _ensure_outward(QuadMesh(nodes, quads, None, tris))


def wire_solid_mesh(start, end, radius: float,
                    n_segments: int = 32) -> QuadMesh:
    """Watertight faceted cylinder along an arbitrary axis.

    The boolean-solid representation of the circular strut/wire: side
    quads plus triangle-fan end caps, wound outward.
    """
    if radius <= 0.0:
        raise ValueError("Wire needs a positive radius")
    a = np.asarray(start, dtype=float)
    b = np.asarray(end, dtype=float)
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-12:
        raise ValueError("Wire start and end points coincide")
    axis = axis / length
    n_segments = max(3, int(n_segments))

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, helper)) > 0.99:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(helper, axis)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    theta = 2.0 * np.pi * np.arange(n_segments) / n_segments
    ring = radius * (np.outer(np.cos(theta), u) + np.outer(np.sin(theta), v))

    nodes = list(a + ring) + list(b + ring)
    start_center = len(nodes)
    nodes.append(a)
    end_center = len(nodes)
    nodes.append(b)
    nodes = np.asarray(nodes)

    quads = []
    tris = []
    for k in range(n_segments):
        k2 = (k + 1) % n_segments
        quads.append([k, k2, n_segments + k2, n_segments + k])
        tris.append([start_center, k2, k])
        tris.append([end_center, n_segments + k, n_segments + k2])
    return _ensure_outward(QuadMesh(nodes, quads, None, tris))


#: Grid used to snap boolean-input vertices (meters). Members meant to
#: sit flush but differing by floating-point noise land on the same
#: grid point, so manifold3d merges them into one shell instead of
#: leaving two kissing closed surfaces (which TICRA MoM rejects as a
#: junction of 3+ closed surfaces after welding).
BOOLEAN_SNAP_TOL = 1e-5


def snap_mesh(mesh: QuadMesh, tol: float = BOOLEAN_SNAP_TOL) -> QuadMesh:
    """Quantize node positions to a tol grid."""
    nodes = np.round(mesh.nodes / tol) * tol
    return QuadMesh(nodes, mesh.linear_patches, mesh.curved_patches,
                    mesh.tri_patches)


def weld_mesh(mesh: QuadMesh, tol: float = BOOLEAN_SNAP_TOL) -> QuadMesh:
    """Merge coincident nodes (within tol) and drop degenerate faces."""
    keys = {}
    remap = np.empty(len(mesh.nodes), dtype=int)
    new_nodes = []
    for i, p in enumerate(mesh.nodes):
        k = tuple(np.round(p / tol).astype(np.int64))
        if k in keys:
            remap[i] = keys[k]
        else:
            keys[k] = len(new_nodes)
            remap[i] = len(new_nodes)
            new_nodes.append(p)

    def remap_faces(faces, distinct):
        out = []
        for face in faces:
            mapped = [int(remap[i]) for i in face]
            if len(set(mapped)) >= distinct:
                out.append(mapped)
        return out

    return QuadMesh(np.asarray(new_nodes),
                    remap_faces(mesh.linear_patches, 4) or None,
                    remap_faces(mesh.curved_patches, 9) or None,
                    remap_faces(mesh.tri_patches, 3) or None)


def mesh_shells(mesh: QuadMesh) -> list:
    """Split a mesh into its connected components (shells).

    Connectivity is by shared node index; run weld_mesh first when
    coincident-but-duplicated nodes should connect.
    """
    n = len(mesh.nodes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    all_faces = ([list(f) for f in mesh.linear_patches]
                 + [list(f) for f in mesh.curved_patches]
                 + [list(f) for f in mesh.tri_patches])
    for face in all_faces:
        for i in range(1, len(face)):
            union(face[0], face[i])

    roots = {}
    for face in all_faces:
        roots.setdefault(find(face[0]), []).append(face)
    if len(roots) <= 1:
        return [mesh]

    shells = []
    for faces in roots.values():
        used = sorted({i for f in faces for i in f})
        index_map = {old: new for new, old in enumerate(used)}
        quads = [[index_map[i] for i in f] for f in faces if len(f) == 4]
        curved = [[index_map[i] for i in f] for f in faces if len(f) == 9]
        tris = [[index_map[i] for i in f] for f in faces if len(f) == 3]
        shells.append(QuadMesh(mesh.nodes[used], quads or None,
                               curved or None, tris or None))
    return shells


def mesh_topology_report(mesh: QuadMesh,
                         tol: float = BOOLEAN_SNAP_TOL) -> dict:
    """Topology summary the way a MoM mesher sees it (welded by tol).

    Returns shells, junction_edges (edges with more than two incident
    faces after welding -- the configuration TICRA MoM rejects for
    closed surfaces), and patch counts.
    """
    welded = weld_mesh(mesh, tol)
    from collections import Counter
    edges = Counter()
    _, quads = welded.display_quads()
    for q in quads:
        loop = [q[0], q[1], q[2]] if q[2] == q[3] else list(q)
        for i in range(len(loop)):
            a, b = loop[i], loop[(i + 1) % len(loop)]
            if a != b:
                edges[tuple(sorted((a, b)))] += 1
    junctions = [e for e, c in edges.items() if c > 2]
    return {
        'shells': len(mesh_shells(welded)),
        'junction_edges': len(junctions),
        'junction_points': [tuple(welded.nodes[e[0]]) for e in junctions[:5]],
        'n_quads': len(welded.linear_patches),
        'n_triangles': len(welded.tri_patches),
        'open_edges': sum(1 for c in edges.values() if c == 1),
    }


def pair_triangles_into_quads(mesh: QuadMesh,
                              max_warp_deg: float = 10.0) -> QuadMesh:
    """Merge adjacent triangle pairs into quads, best pairs first.

    Boolean output is all triangles; MoM prefers fewer, better-shaped
    patches, and TICRA's 4-node patches are bilinear so a mildly
    non-planar quad (dihedral up to ``max_warp_deg`` across the removed
    diagonal) is representable exactly through its corner nodes.

    Candidate pairs are scored by planarity and squareness and taken
    greedily in score order (near-optimal on structured regions like
    facet strips and cap fans). A perfect all-quad result is not
    generally possible -- odd triangle counts in a region leave a
    remainder -- but residuals concentrate at boolean seams.
    """
    tris = mesh.tri_patches
    if len(tris) == 0:
        return mesh
    pts = mesh.nodes

    normals = np.cross(pts[tris[:, 1]] - pts[tris[:, 0]],
                       pts[tris[:, 2]] - pts[tris[:, 0]])
    norms = np.linalg.norm(normals, axis=1)
    norms[norms < 1e-30] = 1.0
    normals = normals / norms[:, np.newaxis]
    cos_warp = np.cos(np.radians(max_warp_deg))

    edge_map = {}
    for t_index, tri in enumerate(tris):
        for i in range(3):
            a, b = int(tri[i]), int(tri[(i + 1) % 3])
            edge_map.setdefault(tuple(sorted((a, b))), []).append((t_index, i))

    def quad_for(t1, i1, t2, i2):
        a = int(tris[t1][i1])
        b = int(tris[t1][(i1 + 1) % 3])
        p1 = int(tris[t1][(i1 + 2) % 3])
        p2 = int(tris[t2][(i2 + 2) % 3])
        return [p1, a, p2, b]

    candidates = []
    for owners in edge_map.values():
        if len(owners) != 2:
            continue
        (t1, i1), (t2, i2) = owners
        cos_d = float(np.dot(normals[t1], normals[t2]))
        if cos_d < cos_warp:
            continue
        quad = quad_for(t1, i1, t2, i2)
        corners = pts[quad]
        n_avg = normals[t1] + normals[t2]
        n_avg = n_avg / max(np.linalg.norm(n_avg), 1e-30)

        edges = np.array([corners[(k + 1) % 4] - corners[k]
                          for k in range(4)])
        lengths = np.linalg.norm(edges, axis=1)
        if np.min(lengths) < 1e-15:
            continue
        unit = edges / lengths[:, np.newaxis]
        convex = True
        min_corner_sin = 1.0
        for k in range(4):
            turn = np.cross(unit[k], unit[(k + 1) % 4])
            if np.dot(turn, n_avg) <= 1e-12:
                convex = False
                break
            min_corner_sin = min(min_corner_sin,
                                 float(np.linalg.norm(turn)))
        if not convex:
            continue
        # Planarity dominates; squareness breaks ties
        score = (cos_d ** 8) * min_corner_sin
        candidates.append((score, t1, t2, quad))

    candidates.sort(key=lambda c: c[0], reverse=True)
    used = np.zeros(len(tris), dtype=bool)
    quads = [list(q) for q in mesh.linear_patches]
    for score, t1, t2, quad in candidates:
        if used[t1] or used[t2]:
            continue
        used[t1] = True
        used[t2] = True
        quads.append(quad)

    remaining = [list(t) for t_index, t in enumerate(tris)
                 if not used[t_index]]
    return QuadMesh(pts, quads or None, mesh.curved_patches,
                    remaining or None)


#: Contact tolerance for cross-object junction warnings (meters).
#: Closed scatterers closer than this are fused by TICRA's MoM mesher,
#: producing junctions of 3+ closed surfaces it cannot process.
CONTACT_TOL = 1e-4


def _min_point_mesh_distance(points: np.ndarray, triangles: np.ndarray,
                             early_out: float) -> float:
    """Minimum distance from any point to a triangle mesh surface.

    Returns early once a distance below ``early_out`` is found.
    """
    best = np.inf
    v0 = triangles[:, 0]
    e1 = triangles[:, 1] - v0
    e2 = triangles[:, 2] - v0
    a = np.einsum('ij,ij->i', e1, e1)
    b = np.einsum('ij,ij->i', e1, e2)
    c = np.einsum('ij,ij->i', e2, e2)
    det = np.maximum(a * c - b * b, 1e-30)

    for p in points:
        w = p - v0
        d1 = np.einsum('ij,ij->i', e1, w)
        d2 = np.einsum('ij,ij->i', e2, w)
        # Barycentric coordinates of the unconstrained projection,
        # clamped to the triangle (approximate clamp: clip then
        # renormalize -- adequate for a contact tolerance test).
        s = np.clip((c * d1 - b * d2) / det, 0.0, 1.0)
        t = np.clip((a * d2 - b * d1) / det, 0.0, 1.0)
        over = s + t > 1.0
        if np.any(over):
            total = (s[over] + t[over])
            s[over] = s[over] / total
            t[over] = t[over] / total
        closest = v0 + s[:, np.newaxis] * e1 + t[:, np.newaxis] * e2
        dist = np.sqrt(np.min(np.sum((closest - p) ** 2, axis=1)))
        if dist < best:
            best = float(dist)
            if best < early_out:
                return best
    return best


def solids_in_contact(mesh_a: QuadMesh, mesh_b: QuadMesh,
                      tol: float = CONTACT_TOL) -> Optional[str]:
    """Detect overlap or near-touch between two closed solids.

    Returns 'overlap', 'touching', or None. Overlap is detected via
    boolean intersection volume when manifold3d is available; touch via
    node-to-surface distance in both directions.
    """
    margin = tol * 2.0
    a_min = mesh_a.nodes.min(axis=0)
    a_max = mesh_a.nodes.max(axis=0)
    b_min = mesh_b.nodes.min(axis=0)
    b_max = mesh_b.nodes.max(axis=0)
    if np.any(a_min > b_max + margin) or np.any(b_min > a_max + margin):
        return None

    if MANIFOLD_AVAILABLE:
        try:
            def to_m(mesh):
                return manifold3d.Manifold(manifold3d.Mesh(
                    vert_properties=np.ascontiguousarray(mesh.nodes,
                                                         np.float32),
                    tri_verts=np.ascontiguousarray(mesh.triangle_indices(),
                                                   np.uint32)))
            if (to_m(mesh_a) ^ to_m(mesh_b)).volume() > 1e-15:
                return 'overlap'
        except Exception:
            pass

    tris_b = mesh_b.triangles()
    if _min_point_mesh_distance(mesh_a.nodes, tris_b, tol) < tol:
        return 'touching'
    tris_a = mesh_a.triangles()
    if _min_point_mesh_distance(mesh_b.nodes, tris_a, tol) < tol:
        return 'touching'
    return None


def collect_closed_solids(shapes, all_quad: bool = False):
    """(label, mesh) for every closed solid the export will produce.

    Boolean groups yield their combined mesh (one entry per disjoint
    shell would be overkill here -- contact checking treats the group
    as one solid); every rotational copy of an ungrouped closed-solid
    shape yields its own entry. Open shapes (cylinder shells, offset
    parabolas, plates handled as reflectors) are excluded: TICRA
    supports junctions with open surfaces.
    """
    from .definitions import (
        ShapeDefinition,
        partition_boolean_groups,
    )

    solids = []
    ungrouped, groups = partition_boolean_groups(shapes)
    for name, members in groups.items():
        try:
            solids.append((f"boolean group '{name}'",
                           combine_shape_group(members, all_quad=all_quad)))
        except (RuntimeError, ValueError):
            ungrouped = ungrouped + list(members)

    for index, shape in enumerate(ungrouped):
        if shape.shape_type not in ShapeDefinition.BOOLEAN_SOLID_TYPES:
            continue
        count = max(1, shape.rotational_count)
        step_deg = 360.0 / count
        for copy_index in range(count):
            phi = (shape.phi + copy_index * step_deg) % 360.0
            label = shape.display_name()
            if count > 1:
                label += f" copy {copy_index + 1}"
            try:
                solids.append((label, build_shape_mesh(shape, phi)))
            except ValueError:
                continue
    return solids


def check_solid_contacts(named_solids, tol: float = CONTACT_TOL):
    """Pairwise contact check over (label, mesh) closed solids.

    Returns (label_a, label_b, kind) for every overlapping or touching
    pair. Any such pair will be fused by TICRA's MoM mesher into a
    junction of 3+ closed surfaces; the fix is to put both in the same
    boolean group so they merge into one closed surface.
    """
    contacts = []
    for i in range(len(named_solids)):
        for j in range(i + 1, len(named_solids)):
            kind = solids_in_contact(named_solids[i][1], named_solids[j][1],
                                     tol)
            if kind is not None:
                contacts.append((named_solids[i][0], named_solids[j][0],
                                 kind))
    return contacts


#: Self-proximity tolerance (meters): sheets of one shell closer than
#: this are fused by TICRA's MoM mesher into junctions even though the
#: mesh itself is clean.
SELF_PROXIMITY_TOL = 1e-3


def mesh_self_proximity(mesh: QuadMesh, tol: float = SELF_PROXIMITY_TOL,
                        max_reports: int = 5):
    """Find places where a mesh nearly touches itself.

    Reports nodes lying within ``tol`` of a part of the surface they
    are not topologically adjacent to (two face-rings are excluded).
    These near-flush sheets survive boolean union as a single shell but
    are welded together by TICRA's MoM mesher, producing junctions of
    3+ closed surfaces. The fix is design-side: make the members
    genuinely overlap (by several millimeters) or leave clear space.

    Returns (count, [(distance_m, (x, y, z)), ...]) with the closest
    ``max_reports`` locations.
    """
    tris_idx = mesh.triangle_indices()
    if len(tris_idx) == 0:
        return 0, []
    pts = mesh.nodes
    tris = pts[tris_idx]

    from collections import defaultdict
    neighbors = defaultdict(set)
    for tri in tris_idx:
        tri_list = tri.tolist()
        for node in tri_list:
            neighbors[node].update(tri_list)
    ring2 = {}
    for node in range(len(pts)):
        near = set(neighbors[node])
        for other in list(near):
            near.update(neighbors[other])
        ring2[node] = near

    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    a = np.einsum('ij,ij->i', e1, e1)
    b = np.einsum('ij,ij->i', e1, e2)
    c = np.einsum('ij,ij->i', e2, e2)
    det = np.maximum(a * c - b * b, 1e-30)

    centroids = tris.mean(axis=1)
    radii = np.sqrt(np.max(np.sum((tris - centroids[:, None, :]) ** 2,
                                  axis=2), axis=1))
    reach = radii + tol

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(centroids)
        max_reach = float(np.max(reach))

        def candidate_tris(p):
            return np.asarray(tree.query_ball_point(p, max_reach),
                              dtype=int)
    except ImportError:
        def candidate_tris(p):
            return np.flatnonzero(
                np.sum((centroids - p) ** 2, axis=1) <= reach ** 2)

    hits = []
    count = 0
    for node_index, p in enumerate(pts):
        cand = candidate_tris(p)
        if len(cand) == 0:
            continue
        cand = cand[[not (set(tris_idx[t].tolist()) & ring2[node_index])
                     for t in cand]]
        if len(cand) == 0:
            continue
        w = p - v0[cand]
        d1 = np.einsum('ij,ij->i', e1[cand], w)
        d2 = np.einsum('ij,ij->i', e2[cand], w)
        s = np.clip((c[cand] * d1 - b[cand] * d2) / det[cand], 0.0, 1.0)
        t = np.clip((a[cand] * d2 - b[cand] * d1) / det[cand], 0.0, 1.0)
        over = s + t > 1.0
        total = s[over] + t[over]
        s[over] = s[over] / total
        t[over] = t[over] / total
        closest = v0[cand] + s[:, np.newaxis] * e1[cand] \
            + t[:, np.newaxis] * e2[cand]
        dist = float(np.sqrt(np.min(np.sum((closest - p) ** 2, axis=1))))
        if dist < tol:
            count += 1
            hits.append((dist, tuple(float(x) for x in p)))

    hits.sort(key=lambda h: h[0])
    return count, hits[:max_reports]


def subdivide_to_quads(mesh: QuadMesh) -> QuadMesh:
    """Conforming all-quad subdivision (one barycentric step).

    Every quad splits into 4 quads and every triangle into 3 quads via
    shared edge midpoints and a face centroid, so the result is 100%
    quads, watertight, and free of degenerate patches -- at the cost of
    3-4x the patch count. Edge midpoints are shared between neighboring
    faces, keeping the mesh conforming across quad/triangle borders.
    """
    if len(mesh.curved_patches) > 0:
        raise ValueError("subdivide_to_quads does not handle curved patches")

    nodes = [list(p) for p in mesh.nodes]
    midpoint_cache = {}

    def midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        if key not in midpoint_cache:
            midpoint_cache[key] = len(nodes)
            nodes.append(list((mesh.nodes[a] + mesh.nodes[b]) / 2.0))
        return midpoint_cache[key]

    def centroid(face) -> int:
        nodes.append(list(np.mean(mesh.nodes[list(face)], axis=0)))
        return len(nodes) - 1

    quads = []
    for a, b, c, d in mesh.linear_patches:
        m_ab = midpoint(a, b)
        m_bc = midpoint(b, c)
        m_cd = midpoint(c, d)
        m_da = midpoint(d, a)
        ctr = centroid((a, b, c, d))
        quads.extend([
            [a, m_ab, ctr, m_da],
            [m_ab, b, m_bc, ctr],
            [ctr, m_bc, c, m_cd],
            [m_da, ctr, m_cd, d],
        ])
    for a, b, c in mesh.tri_patches:
        m_ab = midpoint(a, b)
        m_bc = midpoint(b, c)
        m_ca = midpoint(c, a)
        ctr = centroid((a, b, c))
        quads.extend([
            [a, m_ab, ctr, m_ca],
            [m_ab, b, m_bc, ctr],
            [ctr, m_bc, c, m_ca],
        ])
    return QuadMesh(np.asarray(nodes), quads)


#: Minimum edge length kept in boolean output (meters). The union
#: tessellation around small features (e.g. a thin wire standing on a
#: large ring face) produces sliver faces with sub-millimeter edges;
#: TICRA's mesher welds at a coarser scale, collapses them, and reports
#: junctions of 3+ closed surfaces. Collapsing them ourselves (safely,
#: with the link condition) keeps the exported mesh clean.
BOOLEAN_MIN_EDGE = 1e-3


def collapse_short_edges(mesh: QuadMesh,
                         min_edge: float = BOOLEAN_MIN_EDGE,
                         max_passes: int = 10) -> QuadMesh:
    """Collapse triangle-mesh edges shorter than ``min_edge``.

    Standard edge collapse to the midpoint, shortest edges first, with
    the link condition guarding manifoldness (a collapse is skipped
    unless the endpoints' shared neighbors are exactly the two vertices
    opposite the edge). Geometric error is bounded by min_edge.
    Operates on the triangulated form; quads/curved patches are
    tessellated first, so use before quad pairing.
    """
    tris = mesh.triangle_indices()
    if len(tris) == 0:
        return mesh
    verts = mesh.nodes.copy()
    tris = [list(t) for t in tris]

    for _ in range(max_passes):
        from collections import defaultdict
        neighbors = defaultdict(set)
        edge_faces = defaultdict(list)
        for f_index, tri in enumerate(tris):
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                neighbors[a].add(b)
                neighbors[b].add(a)
                edge_faces[tuple(sorted((a, b)))].append(f_index)

        candidates = []
        for (a, b), faces in edge_faces.items():
            length = float(np.linalg.norm(verts[a] - verts[b]))
            if length < min_edge and len(faces) == 2:
                candidates.append((length, a, b))
        # Needles/caps: faces thinner than min_edge whose edges are all
        # longer -- collapse their shortest edge anyway (error bounded
        # by that edge, which is small for such degenerate faces).
        for tri in tris:
            p = verts[tri]
            lengths = [float(np.linalg.norm(p[(i + 1) % 3] - p[i]))
                       for i in range(3)]
            longest = max(lengths)
            if longest < 1e-30:
                continue
            area = 0.5 * float(np.linalg.norm(
                np.cross(p[1] - p[0], p[2] - p[0])))
            if 2.0 * area / longest >= min_edge:
                continue
            shortest_i = int(np.argmin(lengths))
            if lengths[shortest_i] >= min_edge:
                a = tri[shortest_i]
                b = tri[(shortest_i + 1) % 3]
                if len(edge_faces[tuple(sorted((a, b)))]) == 2:
                    candidates.append((lengths[shortest_i], a, b))
        if not candidates:
            break
        candidates.sort()

        touched = set()
        collapsed = {}
        removed_faces = set()
        progress = False
        for length, a, b in candidates:
            if a in touched or b in touched:
                continue
            shared = neighbors[a] & neighbors[b]
            opposite = set()
            for f_index in edge_faces[tuple(sorted((a, b)))]:
                for v in tris[f_index]:
                    if v not in (a, b):
                        opposite.add(v)
            if shared != opposite:
                continue  # link condition: collapse would pinch the mesh
            verts[a] = 0.5 * (verts[a] + verts[b])
            collapsed[b] = a
            removed_faces.update(edge_faces[tuple(sorted((a, b)))])
            touched.add(a)
            touched.add(b)
            touched.update(shared)
            progress = True

        if not progress:
            break

        def resolve(v):
            while v in collapsed:
                v = collapsed[v]
            return v

        new_tris = []
        for f_index, tri in enumerate(tris):
            if f_index in removed_faces:
                continue
            mapped = [resolve(v) for v in tri]
            if len(set(mapped)) == 3:
                new_tris.append(mapped)
        tris = new_tris

    used = sorted({v for tri in tris for v in tri})
    index_map = {old: new for new, old in enumerate(used)}
    return QuadMesh(verts[used],
                    tri_patches=[[index_map[v] for v in tri]
                                 for tri in tris] or None)


def _remove_caps_once(verts: np.ndarray, tris: List[List[int]],
                      min_height: float):
    """One pass of cap removal: thin triangles whose edges are all long.

    For a cap (apex hanging over its long base edge at height <
    min_height), the base edge is split at the apex projection on both
    adjacent faces (keeping the mesh conforming) and the apex is
    collapsed into the new node -- removing the thin face with
    geometric error bounded by min_height.

    Returns (verts, tris, n_removed).
    """
    from collections import defaultdict

    def build_edge_faces():
        edge_faces = defaultdict(list)
        for f_index, tri in enumerate(tris):
            for i in range(3):
                a, b = tri[i], tri[(i + 1) % 3]
                edge_faces[tuple(sorted((a, b)))].append(f_index)
        return edge_faces

    removed = 0
    guard = 0
    while guard < 10000:
        guard += 1
        edge_faces = build_edge_faces()
        target = None
        for f_index, tri in enumerate(tris):
            p = verts[tri]
            lengths = [float(np.linalg.norm(p[(i + 1) % 3] - p[i]))
                       for i in range(3)]
            area = 0.5 * float(np.linalg.norm(
                np.cross(p[1] - p[0], p[2] - p[0])))
            longest_i = int(np.argmax(lengths))
            if lengths[longest_i] < 1e-30:
                continue
            height = 2.0 * area / lengths[longest_i]
            if height >= min_height:
                continue
            if min(lengths) < min_height:
                continue  # short-edge case: handled by edge collapse
            # Cap: apex opposite the longest edge
            a = tri[longest_i]
            b = tri[(longest_i + 1) % 3]
            c = tri[(longest_i + 2) % 3]
            base = verts[b] - verts[a]
            t = float(np.dot(verts[c] - verts[a], base) /
                      max(np.dot(base, base), 1e-30))
            if not (0.05 < t < 0.95):
                continue
            owners = edge_faces[tuple(sorted((a, b)))]
            if len(owners) != 2:
                continue
            target = (f_index, a, b, c, t, owners)
            break
        if target is None:
            break

        f_index, a, b, c, t, owners = target
        m_index = len(verts)
        verts = np.vstack([verts, verts[a] + t * (verts[b] - verts[a])])

        new_tris = []
        for g_index, tri in enumerate(tris):
            if g_index not in owners:
                new_tris.append(tri)
                continue
            # Split the (a, b) edge at m, preserving each face's winding
            local = list(tri)
            ia = local.index(a)
            if local[(ia + 1) % 3] == b:
                other = local[(ia + 2) % 3]
                new_tris.append([a, m_index, other])
                new_tris.append([m_index, b, other])
            else:
                other = local[(ia + 1) % 3]
                new_tris.append([a, other, m_index])
                new_tris.append([m_index, other, b])
        # Collapse the apex into the new node
        final = []
        for tri in new_tris:
            mapped = [m_index if v == c else v for v in tri]
            if len(set(mapped)) == 3:
                final.append(mapped)
        tris = final
        removed += 1
    return verts, tris, removed


def clean_boolean_mesh(mesh: QuadMesh,
                       tol: float = BOOLEAN_MIN_EDGE,
                       max_rounds: int = 4) -> QuadMesh:
    """Remove sliver-scale features from a boolean output mesh.

    Alternates short-edge collapse and cap removal until the mesh has
    no edges or face heights below ``tol`` (or no further progress).
    Geometric error is bounded by tol per feature.
    """
    current = mesh
    for _ in range(max_rounds):
        collapsed = collapse_short_edges(current, tol)
        verts, tris, n_caps = _remove_caps_once(
            collapsed.nodes, [list(t) for t in collapsed.triangle_indices()],
            tol)
        used = sorted({v for tri in tris for v in tri})
        index_map = {old: new for new, old in enumerate(used)}
        current = QuadMesh(verts[used],
                           tri_patches=[[index_map[v] for v in tri]
                                        for tri in tris] or None)
        if n_caps == 0 and len(collapsed.tri_patches) == \
                len(current.tri_patches):
            break
    return current


def combine_meshes(entries, all_quad: bool = False) -> QuadMesh:
    """Boolean-combine closed solid meshes into one triangle mesh.

    Args:
        entries: Ordered (mesh, mode) pairs with mode 'add' or
            'subtract', applied in sequence (all adds union together;
            subtracts carve from the running result).
        all_quad: When True, any triangles left after quad pairing are
            eliminated by one conforming barycentric subdivision step,
            yielding a 100% quad mesh at ~3-4x the patch count.

    Returns:
        A QuadMesh whose surface is the boolean result (quads plus
        residual triangles, or quads only with all_quad).

    Raises:
        RuntimeError: If manifold3d is not installed.
        ValueError: If no 'add' entry is present or a mesh is not a
            valid closed solid.
    """
    if not MANIFOLD_AVAILABLE:
        raise RuntimeError(
            "Boolean shape combination requires the manifold3d package "
            "(pip install manifold3d)")

    def to_manifold(mesh: QuadMesh):
        # Snap to the shared grid so members meant to sit flush merge
        # into one shell instead of leaving kissing closed surfaces.
        mesh = snap_mesh(mesh)
        verts = np.ascontiguousarray(mesh.nodes, dtype=np.float32)
        tris = np.ascontiguousarray(mesh.triangle_indices(), dtype=np.uint32)
        solid = manifold3d.Manifold(
            manifold3d.Mesh(vert_properties=verts, tri_verts=tris))
        if solid.volume() <= 0.0:
            raise ValueError("Boolean input is not a closed solid")
        return solid

    result = None
    for mesh, mode in entries:
        solid = to_manifold(mesh)
        if mode == 'subtract':
            if result is not None:
                result = result - solid
        else:
            result = solid if result is None else result + solid
    if result is None:
        raise ValueError("Boolean group needs at least one 'add' shape")

    out = result.to_mesh()
    combined = QuadMesh(np.asarray(out.vert_properties, dtype=float),
                        tri_patches=np.asarray(out.tri_verts, dtype=int))
    combined = weld_mesh(combined)
    # Remove sliver-scale tessellation (short edges AND thin caps)
    # before quad pairing -- TICRA's mesher welds at a coarser scale
    # than these features and collapses them into junctions.
    combined = clean_boolean_mesh(combined)
    paired = pair_triangles_into_quads(combined)
    if all_quad and len(paired.tri_patches) > 0:
        paired = subdivide_to_quads(paired)
    return paired


def combine_shape_group(shapes, all_quad: bool = False) -> QuadMesh:
    """Boolean-combine all rotational copies of a group's shapes.

    Shapes are processed in list order; every rotational copy of an
    'add' shape is unioned in and every copy of a 'subtract' shape is
    carved out. The result is a single mesh (the group itself has no
    rotational copies). See combine_meshes for ``all_quad``.
    """
    entries = []
    for shape in shapes:
        mode = getattr(shape, 'boolean_mode', 'add')
        count = max(1, shape.rotational_count)
        step_deg = 360.0 / count
        for copy_index in range(count):
            phi = (shape.phi + copy_index * step_deg) % 360.0
            entries.append((build_shape_mesh(shape, phi), mode))
    return combine_meshes(entries, all_quad=all_quad)


def build_shape_mesh(shape, phi_deg: float) -> QuadMesh:
    """Tessellate a mesh-based ShapeDefinition rotational copy.

    ``phi_deg`` is the shape's phi plus the rotational-copy step,
    applied as a rotation about the global z-axis (rect struts) or as
    the azimuth of the shape position (box, cylinder shell), matching
    the semantics of the analytic shadow shapes.
    """
    if shape.shape_type == 'rect_strut':
        mesh = rect_strut_mesh(shape.start_point, shape.end_point,
                               shape.strut_width, shape.strut_thickness)
        return mesh.rotated_about_z(phi_deg)

    if shape.shape_type == 'box':
        phi_rad = np.radians(phi_deg)
        center = (shape.rho * np.cos(phi_rad),
                  shape.rho * np.sin(phi_rad),
                  shape.z)
        return box_mesh(center, shape.box_dims, rotation_z_deg=phi_deg)

    if shape.shape_type == 'cylinder_shell':
        phi_rad = np.radians(phi_deg)
        center_xy = (shape.rho * np.cos(phi_rad),
                     shape.rho * np.sin(phi_rad))
        return cylinder_shell_mesh(
            center_xy,
            z_bottom=shape.z,
            height=shape.shell_height,
            radius=shape.radius,
            span_deg=shape.shell_span_deg,
            start_deg=phi_deg + shape.shell_start_deg,
            n_segments=shape.shell_segments,
        )

    if shape.shape_type == 'solid_cylinder':
        phi_rad = np.radians(phi_deg)
        center_xy = (shape.rho * np.cos(phi_rad),
                     shape.rho * np.sin(phi_rad))
        return solid_cylinder_mesh(
            center_xy,
            z_bottom=shape.z,
            height=shape.shell_height,
            radius=shape.radius,
            n_segments=shape.shell_segments,
        )

    if shape.shape_type == 'extruded_polygon':
        mesh = extruded_polygon_mesh(shape.start_point, shape.end_point,
                                     shape.cross_section)
        return mesh.rotated_about_z(phi_deg)

    # Curved shapes with native TICRA export: faceted-solid conversion
    # used for boolean groups (and anywhere a mesh is required).
    if shape.shape_type == 'strut':
        mesh = wire_solid_mesh(shape.start_point, shape.end_point,
                               shape.diameter / 2.0,
                               n_segments=shape.curve_segments)
        return mesh.rotated_about_z(phi_deg)

    if shape.shape_type == 'circular_plate':
        phi_rad = np.radians(phi_deg)
        center_xy = (shape.rho * np.cos(phi_rad),
                     shape.rho * np.sin(phi_rad))
        return revolve_solid_mesh(shape.plate_profile_zr(), center_xy,
                                  shape.z, n_segments=shape.curve_segments)

    if shape.shape_type == 'body_of_revolution':
        phi_rad = np.radians(phi_deg)
        center_xy = (shape.rho * np.cos(phi_rad),
                     shape.rho * np.sin(phi_rad))
        return revolve_solid_mesh(shape.profile_points, center_xy,
                                  shape.z, n_segments=shape.curve_segments)

    raise ValueError(f"Not a mesh-based shape type: {shape.shape_type}")


MESH_SHAPE_TYPES = ('rect_strut', 'box', 'cylinder_shell', 'solid_cylinder',
                    'extruded_polygon')

#: Shapes that may participate in boolean groups (see the data model's
#: BOOLEAN_SOLID_TYPES for the authoritative set)
BOOLEAN_SOLID_TYPES = ('rect_strut', 'box', 'solid_cylinder',
                       'extruded_polygon', 'strut', 'circular_plate',
                       'body_of_revolution')
