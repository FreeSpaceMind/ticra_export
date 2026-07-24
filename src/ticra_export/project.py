"""
GRASP project assembly.

GraspProject builds the directory layout used by GRASP 10.x:

    <project_name>/
        <project_name>.gxp
        working/
            <project_name>.tor
            <project_name>.tci
            <data files: .cut, .sfc, .rim>

build_dual_reflector_project() assembles the UAD first-case export:
tabulated feed (.cut), tabulated main and sub reflector surfaces
(.sfc), tabulated main rim (.rim), a PO chain, and a spherical cut
output.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import Sequence as Seq

from . import objects as obj
from .datafiles import write_gxp, write_rim_file
from .tci import Command, TciFile
from .tor import Comment, Quantity, Ref, Sequence, Struct, TorFile


class GraspProject:
    """Container for one exportable GRASP project."""

    def __init__(self, name: str, version: str = "10.6.0"):
        self.name = str(name)
        self.version = version
        self.tor = TorFile()
        self.tci = TciFile()
        self._data_files: list[Path] = []

    def add_data_file(self, path) -> str:
        """Register an existing data file (.cut/.sfc/.rim/...) to be
        copied into the working directory. Returns the bare filename to
        use in .tor object references."""
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(p)
        self._data_files.append(p)
        return p.name

    def write(self, parent_folder, overwrite: bool = False) -> Path:
        """Create the project directory and write all files.

        Returns the project directory path.
        """
        base = Path(parent_folder) / self.name
        if base.exists() and not overwrite:
            raise FileExistsError(base)
        working = base / "working"
        working.mkdir(parents=True, exist_ok=overwrite)

        missing = self.tor.validate_refs()
        if missing:
            raise ValueError(
                f"Unresolved ref() targets in .tor: {sorted(set(missing))}")

        self.tor.write(working / f"{self.name}.tor")
        self.tci.write(working / f"{self.name}.tci")
        write_gxp(base / f"{self.name}.gxp", self.name,
                  f"working/{self.name}.tor", f"working/{self.name}.tci",
                  version=self.version)
        for src in self._data_files:
            shutil.copy2(src, working / src.name)
        return base


def build_dual_reflector_project(
        name: str,
        frequencies_hz,
        feed_cut_path,
        feed_number_of_cuts: int,
        main_sfc_path,
        sub_sfc_path,
        main_rim_points_xy: Seq[Seq[float]],
        sub_rim_points_xy: Seq[Seq[float]],
        feed_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        main_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        sub_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
        cut_theta_np: int = 721,
        rim_dir=None,
        main_hole_rims_xy: Seq[tuple[str, Seq[Seq[float]]]] = (),
        struts: Seq[tuple] = (),
        bors: Seq[tuple] = (),
        offset_parabolas: Seq[tuple] = (),
        mesh_tables: Seq[tuple] = (),
        auto_convergence: bool = True) -> GraspProject:
    """Assemble the UAD first-case export: dual reflector, tabulated
    surfaces and feed, PO analysis of sub then main, spherical cut out.

    Coordinate convention: all origins are given in the global (main
    vertex-centered) frame, consistent with UAD's .sfc convention that
    surface files are referenced to the primary vertex. The feed
    coordinate system should be placed at the feed phase center / SWE
    origin.

    Args:
        frequencies_hz: Analysis frequency or list of frequencies. Must
            match the frequencies present in the feed .cut file -- GRASP
            has no feed data for frequencies missing from the file.
        feed_cut_path: Existing .cut file (FarFieldSpherical.write_cut
            or swe.write_grasp_cut output).
        feed_number_of_cuts: Number of phi cuts in the .cut file.
        main_sfc_path / sub_sfc_path: Existing .sfc files
            (GRASPTriangulationExporter output).
        main_rim_points_xy / sub_rim_points_xy: (x, y) rim traces in
            meters, in each reflector's own xy frame.
        rim_dir: Directory in which to write the generated .rim files
            (defaults to alongside the main .sfc file).
        main_hole_rims_xy: Optional (label, points_xy) pairs; each trace
            is written as a .rim file and cut out of the main reflector
            as a hole (e.g. strut/tower shadow regions).
        struts: Optional (label, radius_m, end_point_pairs) entries;
            each becomes a circular_struts object whose end_point_pairs
            are ((x1,y1,z1), (x2,y2,z2)) tuples in the global frame.
        bors: Optional (label, nodes_z_rho, origin_xyz) entries; each
            becomes a piecewise_linear_bor revolved about a vertical
            axis at origin_xyz (a dedicated coor_sys is created when
            origin_xyz is nonzero; pass None or zeros for on-axis).
        offset_parabolas: Optional (label, focal_length_m,
            rim_centre_offset_m, aperture_radius_m, vertex_origin_xyz,
            rotation_z_deg) entries. Each becomes a reflector built
            from a paraboloid surface (vertex at vertex_origin_xyz,
            axis +z) and an elliptical rim centred at
            (rim_centre_offset_m, 0) in the paraboloid's local frame,
            with the local x-axis rotated by rotation_z_deg about z --
            the standard GRASP construction for an offset parabolic
            cut.
        mesh_tables: Optional (label, nodes_xyz, linear_patches,
            curved_patches, tri_patches) entries; each becomes a
            tabulated_mesh_table scatterer (solid PEC). Nodes are
            global-frame (x, y, z) rows in meters; patch rows are
            0-based node indices (4 for flat quads, 9 for curved
            quadratic patches, 3 for triangles written as degenerate
            quads).
        auto_convergence: When True (default), the get_currents
            commands request automatic convergence of the PO/PTD
            expansion (``auto_convergence_of_po : on``) with the
            convergence field checked on the next scatterer for the
            subreflector and on the far-field cut for the main
            reflector, so GRASP chooses the po_points itself.

    Returns:
        A GraspProject ready for .write().

    Strut/BoR objects are grouped into a ``blockage_cluster``
    scatterer_cluster but are NOT wired into the command sequence: the
    PO chain stays feed -> sub -> main. Add a MoM analysis over the
    cluster in TICRA Tools if the blockage scattering should be
    included physically (the primary's shadow holes already approximate
    the blockage for the plain PO chain).

    Serialization of all objects and commands used here is verified
    against working files (GRASP 10.x examples and a TICRA Tools 25.0
    job). The .gxp project wrapper remains GRASP 10-era and may need
    re-creation in current TICRA Tools.
    """
    proj = GraspProject(name)
    tor = proj.tor

    try:
        freq_list = [float(f) for f in frequencies_hz]
    except TypeError:
        freq_list = [float(frequencies_hz)]
    if not freq_list:
        raise ValueError("At least one frequency is required")

    # --- rim files ---------------------------------------------------
    rim_dir = Path(rim_dir) if rim_dir is not None else Path(main_sfc_path).parent
    main_rim_path = rim_dir / f"{name}_main.rim"
    sub_rim_path = rim_dir / f"{name}_sub.rim"
    write_rim_file(main_rim_path, main_rim_points_xy,
                   header=f"{name} main reflector rim")
    write_rim_file(sub_rim_path, sub_rim_points_xy,
                   header=f"{name} subreflector rim")

    hole_rims = []
    for label, points_xy in main_hole_rims_xy:
        hole_path = rim_dir / f"{name}_hole_{label}.rim"
        write_rim_file(hole_path, points_xy,
                       header=f"{name} main reflector hole: {label}")
        hole_rims.append((label, hole_path))

    # --- data files --------------------------------------------------
    feed_cut = proj.add_data_file(feed_cut_path)
    main_sfc = proj.add_data_file(main_sfc_path)
    sub_sfc = proj.add_data_file(sub_sfc_path)
    main_rim = proj.add_data_file(main_rim_path)
    sub_rim = proj.add_data_file(sub_rim_path)
    hole_files = [(label, proj.add_data_file(path))
                  for label, path in hole_rims]

    # --- coordinate systems ------------------------------------------
    tor.add(obj.coor_sys("global_coor"))
    tor.add(obj.coor_sys("main_coor", origin=main_origin_m, base="global_coor"))
    tor.add(obj.coor_sys("sub_coor", origin=sub_origin_m, base="global_coor"))
    tor.add(obj.coor_sys("feed_coor", origin=feed_origin_m, base="global_coor"))
    tor.add(obj.coor_sys("cut_coor", base="global_coor"))

    # --- frequency ----------------------------------------------------
    tor.add(obj.frequency("frequencies", freq_list))

    # --- feed ---------------------------------------------------------
    tor.add(obj.tabulated_pattern("feed", "frequencies", "feed_coor",
                                  feed_cut, feed_number_of_cuts))

    # --- surfaces, rims, reflectors ------------------------------------
    tor.add(obj.irregular_xy_grid_triangulation("main_surface", main_sfc))
    tor.add(obj.irregular_xy_grid_triangulation("sub_surface", sub_sfc))
    tor.add(obj.tabulated_rim_xy("main_rim", main_rim))
    tor.add(obj.tabulated_rim_xy("sub_rim", sub_rim))
    hole_rim_names = []
    for label, filename in hole_files:
        rim_obj_name = f"hole_{label}_rim"
        tor.add(obj.tabulated_rim_xy(rim_obj_name, filename))
        hole_rim_names.append(rim_obj_name)
    tor.add(obj.reflector("main_reflector", "main_coor",
                          "main_surface", "main_rim",
                          hole_names=hole_rim_names))

    # --- blockage scatterers (structural; not in the PO chain) ---------
    blockage_names = []
    for label, radius_m, end_point_pairs in struts:
        strut_name = f"struts_{label}"
        tor.add(obj.circular_struts(strut_name, radius_m, end_point_pairs))
        blockage_names.append(strut_name)
    for label, nodes_z_rho, origin_xyz in bors:
        bor_name = f"bor_{label}"
        coor_name = None
        if origin_xyz is not None and any(abs(v) > 1e-12 for v in origin_xyz):
            coor_name = f"{bor_name}_coor"
            tor.add(obj.coor_sys(coor_name, origin=tuple(origin_xyz),
                                 base="global_coor"))
        tor.add(obj.piecewise_linear_bor(bor_name, nodes_z_rho,
                                         coor_sys_name=coor_name))
        blockage_names.append(bor_name)
    for (label, focal_length_m, rim_centre_offset_m, aperture_radius_m,
         vertex_origin_xyz, rotation_z_deg) in offset_parabolas:
        base_name = f"offset_parabola_{label}"
        coor_name = f"{base_name}_coor"
        rot = math.radians(rotation_z_deg)
        x_axis = (math.cos(rot), math.sin(rot), 0.0)
        y_axis = (-math.sin(rot), math.cos(rot), 0.0)
        needs_axes = abs(rotation_z_deg % 360.0) > 1e-9
        tor.add(obj.coor_sys(
            coor_name,
            origin=tuple(vertex_origin_xyz),
            x_axis=x_axis if needs_axes else None,
            y_axis=y_axis if needs_axes else None,
            base="global_coor"))
        tor.add(obj.paraboloid(f"{base_name}_surface", focal_length_m))
        tor.add(obj.elliptical_rim(
            f"{base_name}_rim", aperture_radius_m, aperture_radius_m,
            centre_xy_m=(rim_centre_offset_m, 0.0)))
        tor.add(obj.reflector(base_name, coor_name,
                              f"{base_name}_surface", f"{base_name}_rim"))
        blockage_names.append(base_name)
    for label, nodes_xyz, linear_patches, curved_patches, tri_patches \
            in mesh_tables:
        mesh_name = f"mesh_{label}"
        tor.add(obj.tabulated_mesh_table(mesh_name, nodes_xyz,
                                         linear_patches=linear_patches,
                                         curved_patches=curved_patches,
                                         tri_patches=tri_patches))
        blockage_names.append(mesh_name)
    if blockage_names:
        tor.add(obj.scatterer_cluster("blockage_cluster", blockage_names))
    tor.add(obj.reflector("sub_reflector", "sub_coor",
                          "sub_surface", "sub_rim"))

    # --- PO objects -----------------------------------------------------
    tor.add(obj.po_single_face_scatterer("po_sub", "frequencies",
                                         "sub_reflector"))
    tor.add(obj.po_single_face_scatterer("po_main", "frequencies",
                                         "main_reflector"))

    # --- output ---------------------------------------------------------
    tor.add(obj.spherical_cut("far_field_cut", "cut_coor", "frequencies",
                              f"{name}_farfield.cut",
                              theta_np=cut_theta_np))

    # --- command sequence (verified against TICRA Tools 25 job) ---------
    # General practice: get_field with a single source, then add_field
    # for each remaining contribution (sub spillover and the direct
    # feed radiation).
    #
    # With auto_convergence, each get_currents carries
    # ``auto_convergence_of_po : on`` plus a convergence target -- the
    # next scatterer for the sub, the output cut for the main --
    # following the standard GRASP PO-wizard batch commands.
    sub_members = [("source", Sequence([Ref("feed")]))]
    main_members = [("source", Sequence([Ref("po_sub")]))]
    if auto_convergence:
        sub_members += [
            ("auto_convergence_of_po", True),
            ("convergence_on_scatterer", Sequence([Ref("po_main")])),
        ]
        main_members += [
            ("auto_convergence_of_po", True),
            ("convergence_on_output_grid",
             Sequence([Ref("far_field_cut")])),
        ]
    proj.tci.add(Command("po_sub", "get_currents", sub_members))
    proj.tci.add(Command("po_main", "get_currents", main_members))
    proj.tci.add(Command("far_field_cut", "get_field",
                         [("source", Sequence([Ref("po_main")]))]))
    proj.tci.add(Command("far_field_cut", "add_field",
                         [("source", Sequence([Ref("po_sub")]))]))
    proj.tci.add(Command("far_field_cut", "add_field",
                         [("source", Sequence([Ref("feed")]))]))
    return proj
