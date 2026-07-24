"""
Builders for the TICRA object classes needed for the UAD export case.

Each function returns a TorObject. Member names and value shapes are
copied from working .tor files: originally GRASP 10.x examples, updated
and extended against a TICRA Tools 25.0 project (Job_16.tor) which
verified:

    reflector (surfaces sequence, rim, holes, coor_sys),
    irregular_xy_grid_triangulation (file_xyz_number struct form),
    tabulated_rim_xy (minimal file_name form), tabulated_pattern,
    po_single_face_scatterer, spherical_cut, coor_sys,
    circular_struts, piecewise_linear_bor, scatterer_cluster, mom.

The frequency-list ``frequency`` class comes from GRASP 10.x files
(TICRA Tools 25 projects also use ``frequency_range``).
"""

from __future__ import annotations

from typing import Sequence as Seq

from .tor import Comment, Quantity, Ref, Sequence, Struct, Table, TorObject


def xyz_struct(x: float, y: float, z: float, unit: str = "m") -> Struct:
    return Struct([("x", Quantity(float(x), unit)),
                   ("y", Quantity(float(y), unit)),
                   ("z", Quantity(float(z), unit))])


def coor_sys(name: str,
             origin: tuple[float, float, float] | None = None,
             x_axis: tuple[float, float, float] | None = None,
             y_axis: tuple[float, float, float] | None = None,
             base: str | None = None,
             unit: str = "m") -> TorObject:
    """Cartesian coordinate system. Omitted members use GRASP defaults
    (identity orientation, zero origin, global base)."""
    members = []
    if origin is not None:
        members.append(("origin", xyz_struct(*origin, unit=unit)))
    if x_axis is not None:
        members.append(("x_axis", Struct(
            [("x", x_axis[0]), ("y", x_axis[1]), ("z", x_axis[2])])))
    if y_axis is not None:
        members.append(("y_axis", Struct(
            [("x", y_axis[0]), ("y", y_axis[1]), ("z", y_axis[2])])))
    if base is not None:
        members.append(("base", Ref(base)))
    return TorObject(name, "coor_sys", members)


def frequency(name: str, frequencies_hz: Seq[float]) -> TorObject:
    """Frequency list object. Values converted to GHz."""
    seq = Sequence(Quantity(f / 1e9, "GHz") for f in frequencies_hz)
    return TorObject(name, "frequency", [("frequency_list", seq)])


def paraboloid(name: str, focal_length_m: float) -> TorObject:
    return TorObject(name, "paraboloid",
                     [("focal_length", Quantity(focal_length_m, "m"))])


def elliptical_rim(name: str, half_axis_x_m: float,
                   half_axis_y_m: float,
                   centre_xy_m: tuple[float, float] | None = None) -> TorObject:
    """Elliptical rim, optionally centred away from the coor_sys origin.

    The ``centre`` member is the standard GRASP way to cut an offset
    aperture from a parent surface (e.g. an offset paraboloid).
    """
    members = []
    if centre_xy_m is not None:
        members.append(("centre", Struct([
            ("x", Quantity(float(centre_xy_m[0]), "m")),
            ("y", Quantity(float(centre_xy_m[1]), "m"))])))
    members.append(("half_axis", Struct([("x", Quantity(half_axis_x_m, "m")),
                                         ("y", Quantity(half_axis_y_m, "m"))])))
    return TorObject(name, "elliptical_rim", members)


def tabulated_rim_xy(name: str, file_name: str,
                     number_of_points: int | None = None,
                     unit: str | None = None) -> TorObject:
    """Rim defined by tabulated x,y points in an external .rim file.

    The minimal file_name-only form matches TICRA Tools 25 usage
    (Job_16.tor shadow_hole / primary_rim_FB); the optional members
    follow GRASP 10.x example files.
    """
    members = [("file_name", file_name)]
    if unit is not None:
        members.append(("unit", unit))
    if number_of_points is not None:
        members.append(("number_of_points", number_of_points))
    return TorObject(name, "tabulated_rim_xy", members)


def irregular_xy_grid_triangulation(name: str, file_name: str,
                                    number_line: int = 16,
                                    start_line: int = 18,
                                    interpolation: str = "linear") -> TorObject:
    """Tabulated surface from an .sfc point file.

    Member layout verified against TICRA Tools 25 (Job_16.tor), which
    reads .sfc files written by
    ribbed_surfaces.export.GRASPTriangulationExporter: the point count
    on line 16 and data starting on line 18.
    """
    return TorObject(name, "irregular_xy_grid_triangulation", [
        ("file_name", file_name),
        ("file_xyz_number", Struct([("in_file", "yes"),
                                    ("line", number_line),
                                    ("column", 1)])),
        ("file_xyz_values", Struct([("start_line", start_line),
                                    ("x_column", 1),
                                    ("y_column", 2),
                                    ("z_column", 3)])),
        ("interpolation", interpolation),
    ])


def reflector(name: str, coor_sys_name: str, surface_name: str,
              rim_name: str, hole_names: Seq[str] = ()) -> TorObject:
    """Reflector scatterer (TICRA Tools 25 verified form).

    Args:
        hole_names: Optional rim object names cut out of the surface as
            holes (e.g. strut/tower shadow regions on a primary).
    """
    members = [
        ("coor_sys", Ref(coor_sys_name)),
        ("surfaces", Sequence([Ref(surface_name)])),
        ("rim", Ref(rim_name)),
    ]
    if hole_names:
        members.append(("holes", Sequence(Ref(h) for h in hole_names)))
    return TorObject(name, "reflector", members)


def tabulated_pattern(name: str, frequency_name: str, coor_sys_name: str,
                      file_name: str, number_of_cuts: int,
                      phase_reference: tuple[float, float, float] = (0.0, 0.0, 0.0),
                      near_far: str = "far") -> TorObject:
    """Feed defined by a tabulated .cut pattern file."""
    return TorObject(name, "tabulated_pattern", [
        ("frequency", Ref(frequency_name)),
        ("coor_sys", Ref(coor_sys_name)),
        ("file_name", file_name),
        ("number_of_cuts", number_of_cuts),
        ("phase_reference", xyz_struct(*phase_reference, unit="m")),
        ("near_far", near_far),
    ])


def po_single_face_scatterer(name: str, frequency_name: str,
                             scatterer_name: str) -> TorObject:
    """PO analysis object for one reflector face (TICRA Tools 25 form)."""
    return TorObject(name, "po_single_face_scatterer", [
        ("frequency", Ref(frequency_name)),
        ("scatterer", Ref(scatterer_name)),
    ])


def circular_struts(name: str,
                    radius_m: float,
                    end_point_pairs: Seq[tuple],
                    coor_sys_name: str | None = None) -> TorObject:
    """Set of cylindrical struts sharing one wire radius.

    Verified against TICRA Tools 25 (Job_16.tor). Each entry of
    ``end_point_pairs`` is ((x1, y1, z1), (x2, y2, z2)) in meters; all
    struts in the object share ``radius_m``.
    """
    entries = Sequence()
    for p1, p2 in end_point_pairs:
        entries.append(Struct([
            ("point1_x", Quantity(float(p1[0]), "m")),
            ("point1_y", Quantity(float(p1[1]), "m")),
            ("point1_z", Quantity(float(p1[2]), "m")),
            ("point2_x", Quantity(float(p2[0]), "m")),
            ("point2_y", Quantity(float(p2[1]), "m")),
            ("point2_z", Quantity(float(p2[2]), "m")),
        ]))
    members = []
    if coor_sys_name is not None:
        members.append(("coor_sys", Ref(coor_sys_name)))
    members.extend([
        ("radius", Quantity(float(radius_m), "m")),
        ("end_points", entries),
    ])
    return TorObject(name, "circular_struts", members)


def piecewise_linear_bor(name: str,
                         nodes_z_rho: Seq[Seq[float]],
                         coor_sys_name: str | None = None) -> TorObject:
    """Body of revolution from a piecewise-linear rho(z) profile.

    Verified against TICRA Tools 25 (Job_16.tor). ``nodes_z_rho`` is a
    sequence of (z, rho) pairs in the object's coordinate system; the
    profile is revolved about that system's z-axis. Node values carry no
    units in the file (interpreted in the project's default length
    unit, meters for projects written by this package).
    """
    nodes = Table([float(z), float(rho)] for z, rho in nodes_z_rho)
    members = []
    if coor_sys_name is not None:
        members.append(("coor_sys", Ref(coor_sys_name)))
    members.append(("nodes", nodes))
    return TorObject(name, "piecewise_linear_bor", members)


def tabulated_mesh_table(name: str,
                         nodes_xyz: Seq[Seq[float]],
                         linear_patches: Seq[Seq[int]] = (),
                         curved_patches: Seq[Seq[int]] = (),
                         tri_patches: Seq[Seq[int]] = (),
                         coor_sys_name: str | None = None) -> TorObject:
    """Tabulated mesh scatterer defined inline by nodes and patches.

    Layout verified against TICRA Tools 25 (Job_16.tor). Nodes are
    (x, y, z) rows in meters (values carry no units; the project's
    default length unit applies). Patch node indices are 0-based here
    and written 1-based.

    Every patch is written with region columns ``0 -1 0.0 0.0``: the
    regions table is unused and region -1 marks the surface as solid
    PEC (dielectrics are not represented).

    linear_patches rows have 4 corner-node indices (flat quads).
    tri_patches rows have 3 node indices; each is written as a
    degenerate linear patch with the last node repeated (the GRASP
    tabulated-mesh triangle convention -- unverified against an example
    file, like curved patches below).
    curved_patches rows have 9 node indices (biquadratic patches):
    corners 1-4 counter-clockwise, mid-edge nodes 5-8 (between corners
    1-2, 2-3, 3-4, 4-1), then the center node. NOTE: the
    ``curved_patches`` member name and its node ordering follow the
    GRASP tabulated-mesh convention but no example file with curved
    patches was available -- verify the first curved export in TICRA
    Tools.
    """
    members = []
    if coor_sys_name is not None:
        members.append(("coor_sys", Ref(coor_sys_name)))
    members.append(("regions", Table()))
    members.append(("nodes", Table(
        [i + 1, float(x), float(y), float(z)]
        for i, (x, y, z) in enumerate(nodes_xyz))))
    flat_rows = [
        [0, -1, 0.0, 0.0] + [int(n) + 1 for n in patch]
        for patch in linear_patches
    ] + [
        [0, -1, 0.0, 0.0] + [int(patch[0]) + 1, int(patch[1]) + 1,
                             int(patch[2]) + 1, int(patch[2]) + 1]
        for patch in tri_patches
    ]
    if flat_rows:
        members.append(("linear_patches", Table(flat_rows)))
    if len(curved_patches) > 0:
        members.append(("curved_patches", Table(
            [0, -1, 0.0, 0.0] + [int(n) + 1 for n in patch]
            for patch in curved_patches)))
    return TorObject(name, "tabulated_mesh_table", members)


def scatterer_cluster(name: str, scatterer_names: Seq[str]) -> TorObject:
    """Group of scatterers analyzed together (e.g. by MoM)."""
    return TorObject(name, "scatterer_cluster", [
        ("scatterers", Sequence(Ref(n) for n in scatterer_names)),
    ])


def mom(name: str, frequency_name: str, scatterer_name: str) -> TorObject:
    """Method-of-moments analysis object over a scatterer (cluster)."""
    return TorObject(name, "mom", [
        ("frequency", Ref(frequency_name)),
        ("scatterer", Ref(scatterer_name)),
    ])


def spherical_cut(name: str, coor_sys_name: str, frequency_name: str,
                  file_name: str,
                  theta_start: float = -180.0, theta_end: float = 180.0,
                  theta_np: int = 721,
                  phi_start: float = 0.0, phi_end: float = 90.0,
                  phi_np: int = 3,
                  comment: str = "Field data in cuts") -> TorObject:
    return TorObject(name, "spherical_cut", [
        ("coor_sys", Ref(coor_sys_name)),
        ("theta_range", Struct([("start", theta_start),
                                ("end", theta_end),
                                ("np", theta_np)])),
        ("phi_range", Struct([("start", phi_start),
                              ("end", phi_end),
                              ("np", phi_np)])),
        ("file_name", file_name),
        ("comment", Comment(comment)),
        ("frequency", Ref(frequency_name)),
    ])
