"""Tests for ticra_export.

Serialization checks compare against object blocks copied verbatim from
working GRASP 10.x .tor files (GRASPoptimization repo) and
TicraUtilities.jl test data, normalized for whitespace.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import pytest

from ticra_export import (Command, GraspProject, Ref, Sequence, TciFile,
                          TorFile, build_dual_reflector_project, objects,
                          write_rim_file)


def normalize(s: str) -> str:
    """Collapse whitespace so formatting differences don't matter."""
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s*([:,()])\s*", r"\1", s)
    return s


# --- .tor object serialization vs. verified examples --------------------

def test_paraboloid_matches_example():
    example = """single_surface  paraboloid
(
  focal_length     : 16.0 m
)"""
    assert normalize(str(objects.paraboloid("single_surface", 16.0))) == \
        normalize(example)


def test_elliptical_rim_matches_example():
    example = """single_rim  elliptical_rim
(
  half_axis        : struct(x: 20.0 m, y: 20.0 m)
)"""
    assert normalize(str(objects.elliptical_rim("single_rim", 20.0, 20.0))) == \
        normalize(example)


def test_elliptical_rim_with_centre():
    # GRASP offset-aperture construction: rim centred away from the
    # surface coor_sys origin
    example = """offset_rim  elliptical_rim
(
  centre           : struct(x: 3.0 m, y: 0.0 m),
  half_axis        : struct(x: 2.5 m, y: 2.5 m)
)"""
    got = objects.elliptical_rim("offset_rim", 2.5, 2.5, centre_xy_m=(3.0, 0.0))
    assert normalize(str(got)) == normalize(example)


def test_reflector_matches_example():
    # Verbatim from TICRA Tools 25 Job_16.tor ('sub' reflector)
    example = """sub  reflector
(
  coor_sys         : ref(sub_coor_sys),
  surfaces         : sequence(ref(sub_surface)),
  rim              : ref(sub_rim)
)"""
    got = objects.reflector("sub", "sub_coor_sys", "sub_surface", "sub_rim")
    assert normalize(str(got)) == normalize(example)


def test_empty_coor_sys_matches_example():
    example = """single_global_coor  coor_sys
(
)"""
    assert normalize(str(objects.coor_sys("single_global_coor"))) == \
        normalize(example)


def test_coor_sys_with_origin_and_axes():
    example = """single_feed_coor  coor_sys
(
  origin           : struct(x: 0.0 m, y: 0.0 m, z: 16.0 m),
  y_axis           : struct(x: 0.0, y: -1.0, z: 0.0),
  base             : ref(single_global_coor)
)"""
    got = objects.coor_sys("single_feed_coor", origin=(0.0, 0.0, 16.0),
                           y_axis=(0.0, -1.0, 0.0),
                           base="single_global_coor")
    assert normalize(str(got)) == normalize(example)


def test_po_scatterer_matches_example():
    # Verbatim from TICRA Tools 25 Job_16.tor
    example = """po_primary  po_single_face_scatterer
(
  frequency        : ref(frequency),
  scatterer        : ref(reflector)
)"""
    got = objects.po_single_face_scatterer("po_primary", "frequency",
                                           "reflector")
    assert normalize(str(got)) == normalize(example)


def test_spherical_cut_matches_example():
    example = """Feed_cut  spherical_cut
(
  coor_sys         : ref(base),
  theta_range      : struct(start: 0.0, end: 360.0, np: 201),
  phi_range        : struct(start: 0.0, end: 90.0, np: 3),
  file_name        : Feed_Data.cut,
  comment          : "Field data in cuts",
  frequency        : ref(single_frequencies)
)"""
    got = objects.spherical_cut("Feed_cut", "base", "single_frequencies",
                                "Feed_Data.cut", theta_start=0.0,
                                theta_end=360.0, theta_np=201)
    assert normalize(str(got)) == normalize(example)


def test_tabulated_rim_matches_example_structure():
    # Verbatim from TICRA Tools 25 Job_16.tor
    example = """shadow_hole  tabulated_rim_xy
(
  file_name        : shadow_region_combined_shapes_1.rim
)"""
    got = objects.tabulated_rim_xy("shadow_hole",
                                   "shadow_region_combined_shapes_1.rim")
    assert normalize(str(got)) == normalize(example)


def test_irregular_grid_matches_example():
    # Verbatim from TICRA Tools 25 Job_16.tor (fixes the
    # "Attribute must be a STRUCT: file_xyz_number" project error)
    example = """primary_surface  irregular_xy_grid_triangulation
(
  file_name        : primary_po_mesh.sfc,
  file_xyz_number  : struct(in_file: yes, line: 16, column: 1),
  file_xyz_values  : struct(start_line: 18, x_column: 1, y_column: 2, z_column: 3),
  interpolation    : linear
)"""
    got = objects.irregular_xy_grid_triangulation(
        "primary_surface", "primary_po_mesh.sfc")
    assert normalize(str(got)) == normalize(example)


def test_circular_struts_matches_example():
    # End-point structs verbatim from TICRA Tools 25 Job_16.tor
    got = objects.circular_struts(
        "circular_struts", 0.005,
        [((0.5, 0.0, 0.25), (0.5, 0.0, 0.73))])
    text = normalize(str(got))
    assert "circular_struts circular_struts" in text
    assert "radius:0.005 m" in text
    assert normalize(
        "struct(point1_x: 0.5 m, point1_y: 0.0 m, point1_z: 0.25 m, "
        "point2_x: 0.5 m, point2_y: 0.0 m, point2_z: 0.73 m)") in text


def test_piecewise_linear_bor_matches_example_structure():
    # Structure from TICRA Tools 25 Job_16.tor (kickoff_ring et al.)
    got = objects.piecewise_linear_bor(
        "kickoff_ring",
        [(0.15, 0.51), (0.15, 0.45), (0.13, 0.45), (0.13, 0.51), (0.15, 0.51)])
    text = normalize(str(got))
    assert "kickoff_ring piecewise_linear_bor" in text
    assert "nodes:table" in text
    assert "0.15 0.51 0.15 0.45 0.13 0.45 0.13 0.51 0.15 0.51" in text


def test_tabulated_mesh_table_matches_example_structure():
    # Layout from TICRA Tools 25 Job_16.tor: empty regions table, nodes
    # rows (1-based index, x, y, z), patch rows prefixed 0 -1 0.0 0.0
    # (regions unused; -1 = solid PEC)
    got = objects.tabulated_mesh_table(
        "panel", [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                  (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        linear_patches=[(0, 1, 2, 3)])
    text = normalize(str(got))
    assert "panel tabulated_mesh_table" in text
    assert "regions:table()" in text
    assert "nodes:table(1 0.0 0.0 0.0 2 1.0 0.0 0.0 3 1.0 1.0 0.0 4 0.0 1.0 0.0)" in text
    assert "linear_patches:table(0 -1 0.0 0.0 1 2 3 4)" in text


def test_tabulated_mesh_table_curved_patches():
    # 9-node curved quadratic patch: corners, mid-edges, center
    nodes = [(float(i), 0.0, 0.0) for i in range(9)]
    got = objects.tabulated_mesh_table(
        "shell", nodes, curved_patches=[(0, 2, 8, 6, 1, 5, 7, 3, 4)])
    text = normalize(str(got))
    assert "curved_patches:table(0 -1 0.0 0.0 1 3 9 7 2 6 8 4 5)" in text
    assert "linear_patches" not in text


def test_scatterer_cluster_and_mom_match_example():
    # Verbatim member layout from TICRA Tools 25 Job_16.tor
    cluster = objects.scatterer_cluster(
        "scatterer_cluster", ["circular_struts", "feed_deck"])
    assert normalize(str(cluster)) == normalize(
        """scatterer_cluster  scatterer_cluster
(
  scatterers       : sequence(ref(circular_struts),ref(feed_deck))
)""")

    mom_obj = objects.mom("mom", "frequency", "scatterer_cluster")
    assert normalize(str(mom_obj)) == normalize(
        """mom  mom
(
  frequency        : ref(frequency),
  scatterer        : ref(scatterer_cluster)
)""")


# --- ref validation ------------------------------------------------------

def test_validate_refs_finds_missing():
    tor = TorFile()
    tor.add(objects.reflector("r", "missing_coor", "missing_surf", "missing_rim"))
    assert set(tor.validate_refs()) == {"missing_coor", "missing_surf",
                                        "missing_rim"}


def test_duplicate_names_rejected():
    tor = TorFile()
    tor.add(objects.coor_sys("a"))
    with pytest.raises(ValueError):
        tor.add(objects.coor_sys("a"))


# --- .tci ------------------------------------------------------------------

def test_tci_structure():
    # Command syntax verified against TICRA Tools 25 Job_16.tci:
    # no cmd_N numbering, '&' continuations, QUIT terminator.
    tci = TciFile()
    tci.add(Command("po_sub", "get_currents",
                    [("source", Sequence([Ref("feed")]))]))
    text = str(tci)
    assert text.startswith(
        "COMMAND OBJECT po_sub get_currents ( source : sequence(ref(feed)))")
    assert "cmd_" not in text
    assert text.rstrip().endswith("QUIT")


def test_tci_multi_member_command_matches_example_style():
    # Style from Job_16.tci: members comma-separated with & continuations
    cmd = Command("mom", "get_currents", [
        ("source", Sequence([Ref("tabulated_pattern")])),
        ("field_accuracy", -50.0),
        ("auto_convergence_of_po", True),
    ])
    text = cmd.to_text()
    assert normalize(text.replace("&", " ")) == normalize(
        "COMMAND OBJECT mom get_currents ( source : "
        "sequence(ref(tabulated_pattern)), field_accuracy : -50.0, "
        "auto_convergence_of_po : on)")


def test_tci_empty_command():
    assert Command("optimisation_manager", "get_optimum").to_text() == \
        "COMMAND OBJECT optimisation_manager get_optimum ()"


# --- .rim -----------------------------------------------------------------

def test_rim_roundtrip(tmp_path):
    pts = [(1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.0, -1.0)]
    path = tmp_path / "test.rim"
    n = write_rim_file(path, pts)
    lines = path.read_text().splitlines()
    assert n == 4
    assert int(lines[1]) == 4
    x, y = (float(v) for v in lines[2].split())
    assert (x, y) == (1.0, 0.0)


# --- full project assembly --------------------------------------------------

def test_build_dual_reflector_project(tmp_path):
    import numpy as np
    # Minimal stand-in data files
    feed_cut = tmp_path / "feed.cut"
    feed_cut.write_text("dummy\n")
    main_sfc = tmp_path / "main.sfc"
    main_sfc.write_text("dummy\n")
    sub_sfc = tmp_path / "sub.sfc"
    sub_sfc.write_text("dummy\n")

    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    main_rim = np.column_stack([5.0 * np.cos(theta), 5.0 * np.sin(theta)])
    sub_rim = np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta)])

    proj = build_dual_reflector_project(
        name="test_antenna",
        frequencies_hz=[10e9, 12e9],
        feed_cut_path=feed_cut,
        feed_number_of_cuts=8,
        main_sfc_path=main_sfc,
        sub_sfc_path=sub_sfc,
        main_rim_points_xy=main_rim,
        sub_rim_points_xy=sub_rim,
        feed_origin_m=(0.0, 0.0, 1.2),
        sub_origin_m=(0.0, 0.0, 1.5),
        rim_dir=tmp_path,
        struts=[("tripod", 0.005,
                 [((0.5, 0.0, 0.25), (0.5, 0.0, 0.73)),
                  ((-0.25, 0.43, 0.25), (-0.25, 0.43, 0.73))])],
        bors=[("feed_ring", [(0.13, 0.45), (0.15, 0.51)], (0.0, 0.0, 0.0)),
              ("offset_ring", [(0.0, 0.1), (0.1, 0.1)], (0.3, 0.0, 0.2))],
        offset_parabolas=[("panel", 0.5, 0.2, 0.15, (1.0, 0.0, 0.4), 90.0)],
        mesh_tables=[("deck",
                      [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0),
                       (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
                      [(0, 1, 2, 3)], [], [(0, 1, 2)])],
    )
    base = proj.write(tmp_path / "out")

    working = base / "working"
    assert (base / "test_antenna.gxp").is_file()
    assert (working / "test_antenna.tor").is_file()
    assert (working / "test_antenna.tci").is_file()
    for fn in ["feed.cut", "main.sfc", "sub.sfc",
               "test_antenna_main.rim", "test_antenna_sub.rim"]:
        assert (working / fn).is_file(), fn

    tor_text = (working / "test_antenna.tor").read_text()
    # Frequencies in GHz, full list preserved
    assert "sequence(10.0 GHz,12.0 GHz)" in normalize(tor_text)
    # Blockage scatterers present and clustered; off-axis BoR gets a coor_sys
    assert "struts_tripod  circular_struts" in tor_text
    assert "bor_feed_ring  piecewise_linear_bor" in tor_text
    assert "bor_offset_ring_coor  coor_sys" in tor_text
    assert "blockage_cluster  scatterer_cluster" in tor_text
    # Offset parabola: rotated coor_sys, paraboloid surface, offset rim
    assert "offset_parabola_panel  reflector" in tor_text
    assert "offset_parabola_panel_surface  paraboloid" in tor_text
    assert "centre           : struct(x: 0.2 m, y: 0.0 m)" in tor_text
    assert "offset_parabola_panel_coor  coor_sys" in tor_text
    # Tabulated mesh scatterer present and clustered
    assert "mesh_deck  tabulated_mesh_table" in tor_text
    # All refs resolve
    assert proj.tor.validate_refs() == []
    # tci chain: currents sub->main, then get_field on po_main alone and
    # add_field for the sub spillover and the direct feed contribution
    tci_text = (working / "test_antenna.tci").read_text()
    assert tci_text.index("po_sub get_currents") < \
        tci_text.index("po_main get_currents") < \
        tci_text.index("far_field_cut get_field") < \
        tci_text.index("far_field_cut add_field")
    assert "get_field ( source : sequence(ref(po_main)))" in tci_text
    assert "add_field ( source : sequence(ref(po_sub)))" in tci_text
    assert "add_field ( source : sequence(ref(feed)))" in tci_text


def test_reflector_holes_member():
    r = objects.reflector("main", "coor", "surf", "rim",
                          hole_names=["hole_a_rim", "hole_b_rim"])
    text = normalize(str(r))
    assert "holes:sequence(ref(hole_a_rim),ref(hole_b_rim))" in text
    # Without holes the member is omitted entirely
    assert "holes" not in str(objects.reflector("m2", "coor", "surf", "rim"))


def test_build_dual_reflector_project_with_holes(tmp_path):
    import numpy as np
    for fn in ["feed.cut", "main.sfc", "sub.sfc"]:
        (tmp_path / fn).write_text("dummy\n")

    theta = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    main_rim = np.column_stack([5.0 * np.cos(theta), 5.0 * np.sin(theta)])
    sub_rim = np.column_stack([0.5 * np.cos(theta), 0.5 * np.sin(theta)])
    hole = np.array([[1.0, 1.0], [1.2, 1.0], [1.2, 1.2], [1.0, 1.2]])

    proj = build_dual_reflector_project(
        name="holey",
        frequencies_hz=10e9,  # scalar still accepted
        feed_cut_path=tmp_path / "feed.cut",
        feed_number_of_cuts=8,
        main_sfc_path=tmp_path / "main.sfc",
        sub_sfc_path=tmp_path / "sub.sfc",
        main_rim_points_xy=main_rim,
        sub_rim_points_xy=sub_rim,
        rim_dir=tmp_path,
        main_hole_rims_xy=[("strut1", hole), ("tower", hole + 2.0)],
    )
    base = proj.write(tmp_path / "out")

    working = base / "working"
    assert (working / "holey_hole_strut1.rim").is_file()
    assert (working / "holey_hole_tower.rim").is_file()
    assert proj.tor.validate_refs() == []
    tor_text = (working / "holey.tor").read_text()
    assert "hole_strut1_rim" in tor_text
    assert "sequence(ref(hole_strut1_rim),ref(hole_tower_rim))" in \
        normalize(tor_text)


def test_project_write_refuses_overwrite(tmp_path):
    proj = GraspProject("p")
    proj.tor.add(objects.coor_sys("global_coor"))
    proj.write(tmp_path)
    with pytest.raises(FileExistsError):
        proj.write(tmp_path)
