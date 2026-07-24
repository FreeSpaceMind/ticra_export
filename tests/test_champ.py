"""Tests for the CHAMP horn object builders and project assembler.

Expected serializations follow a CHAMP project saved by TICRA Tools
25.0 (axially corrugated horn): class names, member names, member
order, and units match that reference file.
"""

import numpy as np
import pytest

from ticra_export import champ
from ticra_export.champ import (
    AxialCorrugatedSectionDef,
    CircularWaveguideSectionDef,
    build_champ_horn_project,
    champ_horn_objects,
)


def test_frequency_range_layout():
    text = str(champ.frequency_range("frequency", 8.0e9, 15.0e9, 8))
    assert "frequency  frequency_range" in text
    assert ("frequency_range  : struct(start_frequency: 8.0 GHz, "
            "end_frequency: 15.0 GHz, number_of_frequencies: 8)") in text


def test_simple_axial_corrugated_section_layout():
    text = str(champ.simple_axial_corrugated_section(
        "axial_axial_corrugation1", 0.0154, 0.0032, 0.0092, 0.001,
        0.0058, 1))
    assert "axial_axial_corrugation1  simple_axial_corrugated_section" in text
    for member in ("input_radius", "slot_width", "slot_depth",
                   "ridge_thickness", "axial_distance",
                   "number_of_axial_slots"):
        assert member in text
    assert "input_radius     : 0.0154 m" in text
    assert "number_of_axial_slots : 1" in text


def test_circular_waveguide_section_and_port():
    sec = str(champ.circular_waveguide_section("s", 0.0154, 0.0308))
    assert "radius           : 0.0154 m" in sec
    assert "length           : 0.0308 m" in sec

    settings = str(champ.circular_waveguide_port_modal_expansion_settings(
        "exp_settings"))
    assert "m_min            : 1" in settings
    assert "n_max            : -1" in settings
    assert "m_max            : 1" in settings

    port = str(champ.circular_waveguide_port("p", 0.0154, "exp_settings"))
    assert "radius           : 0.0154 m" in port
    assert "port_expansion_settings : ref(exp_settings)" in port


def test_circular_symmetric_horn_references():
    text = str(champ.circular_symmetric_horn(
        "horn", "frequency", "p1", "p2", "sections", "mm_settings"))
    assert "frequency        : ref(frequency)" in text
    assert "port_1           : ref(p1)" in text
    assert "port_2           : ref(p2)" in text
    assert "horn_section     : ref(sections)" in text
    assert "analysis_settings : ref(mm_settings)" in text


def test_radiating_device_excitation_struct():
    text = str(champ.radiating_device("rad", "dev", "port", "mode"))
    assert "device           : ref(dev)" in text
    assert "port: ref(port)" in text
    assert "waveguide_mode: ref(mode)" in text
    assert "amplitude: 1.0" in text
    assert "phase: 0.0" in text


def test_exterior_nodes_table():
    text = str(champ.circular_symmetric_horn_exterior(
        "ext", [(0.0, 0.05), (-0.04, 0.05)]))
    assert "nodes            : table" in text
    assert "0.0  0.05" in text
    assert "-0.04  0.05" in text


def test_get_field_command_matches_champ_form():
    cmd = champ.get_field_command("cut", "rad", label="my_get_field")
    text = cmd.to_text()
    assert text.startswith("COMMAND OBJECT cut get_field")
    assert "source : sequence(ref(rad))" in text
    assert text.rstrip().endswith("my_get_field")


def test_export_reflection_coefficient_command():
    cmd = champ.export_reflection_coefficient_command(
        "dev", "port", "mode", "reflection.snp", label="exp_refl")
    text = cmd.to_text()
    assert "COMMAND OBJECT dev export_reflection_coefficient" in text
    assert ("mode_selection : struct(port: ref(port), deembed_length: 0.0 m, "
            "waveguide_mode: ref(mode))") in text
    assert "file_name : reflection.snp" in text
    assert "data_format : touchstone_db" in text
    assert text.rstrip().endswith("exp_refl")


def test_beam_parameters_command_no_members():
    text = champ.beam_parameters_command("rad").to_text()
    assert text == ("COMMAND OBJECT rad "
                    "get_radiating_device_circular_symmetric_beam_parameters"
                    " ()")


def _axial_sections(n=3):
    r1, w, t, a = 0.0154, 0.0032, 0.001, 0.0058
    return [AxialCorrugatedSectionDef(
        input_radius_m=r1 + (w + t) * i, slot_width_m=w,
        slot_depth_m=0.008, ridge_thickness_m=t, axial_distance_m=a)
        for i in range(n)]


def test_champ_horn_objects_all_refs_resolve():
    from ticra_export.tor import TorFile
    from ticra_export import objects as obj

    tor = TorFile()
    tor.add(obj.frequency("frequency", [10.0e9]))
    sections = ([CircularWaveguideSectionDef(0.0154, 0.0308)]
                + _axial_sections())
    horn = champ_horn_objects("h", "frequency", sections,
                              throat_radius_m=0.0154,
                              aperture_radius_m=0.03,
                              exterior_nodes_z_rho=[(0.0, 0.032),
                                                    (-0.03, 0.032)])
    for o in horn["objects"]:
        tor.add(o)
    assert tor.validate_refs() == []
    assert horn["radiating_device"] == "h_radiating_device"


def test_build_champ_horn_project_uniform_frequencies(tmp_path):
    sections = ([CircularWaveguideSectionDef(0.0154, 0.0308)]
                + _axial_sections())
    proj = build_champ_horn_project(
        "guppy_like",
        frequencies_hz=np.linspace(8.0e9, 15.0e9, 8),
        sections=sections,
        throat_radius_m=0.0154,
        aperture_radius_m=0.03,
        prefix="axial")
    base = proj.write(tmp_path)

    tor_text = (base / "working" / "guppy_like.tor").read_text()
    assert "frequency  frequency_range" in tor_text
    assert "number_of_frequencies: 8" in tor_text
    assert tor_text.count("simple_axial_corrugated_section") == 3
    assert "axial_circular_symmetric_horn  circular_symmetric_horn" in tor_text
    assert "axial_horn_aperture  circular_symmetric_aperture" in tor_text
    assert "axial_radiating_device  radiating_device" in tor_text

    tci_text = (base / "working" / "guppy_like.tci").read_text()
    assert "export_reflection_coefficient" in tci_text
    assert "get_field" in tci_text
    assert ("get_radiating_device_circular_symmetric_beam_parameters"
            in tci_text)
    assert tci_text.rstrip().endswith("QUIT")


def test_build_champ_horn_project_nonuniform_frequencies(tmp_path):
    proj = build_champ_horn_project(
        "nonuniform", frequencies_hz=[8.0e9, 9.0e9, 12.0e9],
        sections=_axial_sections(),
        throat_radius_m=0.0154, aperture_radius_m=0.03)
    base = proj.write(tmp_path)
    tor_text = (base / "working" / "nonuniform.tor").read_text()
    assert "frequency  frequency  " in tor_text
    assert "frequency_range" not in tor_text


def test_unsupported_section_type_raises():
    with pytest.raises(TypeError, match="Unsupported CHAMP section"):
        build_champ_horn_project(
            "bad", [10e9], sections=[object()],
            throat_radius_m=0.01, aperture_radius_m=0.02)


# --- general BoR waveguide device path --------------------------------------

def _stepped_profile():
    """Stepped corrugated-like interior: throat, slot, aperture."""
    return [
        (0.000, 0.015),
        (0.030, 0.015),   # input guide
        (0.030, 0.028),   # slot left wall (radius step: sections cannot)
        (0.034, 0.028),   # slot bottom
        (0.034, 0.020),   # slot right wall
        (0.040, 0.020),   # tooth
        (0.040, 0.045),   # aperture rim step
    ]


def test_bor_profile_layout():
    text = str(champ.bor_profile("prof", [(0.0, 0.015), (0.03, 0.02)]))
    assert "prof  piecewise_linear_bor" in text
    assert "length_unit      : m" in text
    assert "nodes            : table" in text
    assert "0.0  0.015" in text
    assert "0.03  0.02" in text


def test_general_bor_waveguide_device_layout():
    text = str(champ.general_bor_waveguide_device(
        "dev", "frequency", "prof",
        [("throat", 0.0, "z"), ("aperture", 0.04, "-z")], "mom"))
    assert "dev  general_bor_waveguide_device" in text
    assert "waveguide_geometry : ref(prof)" in text
    assert ("struct(port: ref(throat), z_position: 0.0 m, "
            "inward_normal: z)") in text
    assert ("struct(port: ref(aperture), z_position: 0.04 m, "
            "inward_normal: -z)") in text
    assert "analysis_settings : ref(mom)" in text


def test_aperture_z_displacement_and_swe():
    text = str(champ.circular_symmetric_aperture(
        "ap", "frequency", "port", "ext", "mom", z_displacement_m=0.04))
    assert "z_displacement   : 0.04 m" in text
    # member order: exterior before z_displacement before settings
    assert text.index("exterior") < text.index("z_displacement") \
        < text.index("analysis_settings")

    swe_text = str(champ.swe_output("swe", "horn.sph"))
    assert "swe  swe" in swe_text
    assert "file_name        : horn.sph" in swe_text
    assert "sphere_sample    : struct(n_phi: 24, n_theta: 360)" in swe_text


def test_champ_bor_horn_objects_resolve():
    from ticra_export.tor import TorFile
    from ticra_export import objects as obj

    profile = _stepped_profile()
    tor = TorFile()
    tor.add(obj.frequency("frequency", [10.0e9]))
    horn = champ.champ_bor_horn_objects(
        "h", "frequency", profile,
        throat_radius_m=0.015, aperture_radius_m=0.045,
        aperture_z_m=0.040,
        exterior_z_rho_m=champ.default_bor_exterior(profile, 0.002))
    for o in horn["objects"]:
        tor.add(o)
    assert tor.validate_refs() == []
    assert horn["aperture_port"] == "h_aperture_port"


def test_build_champ_bor_horn_project(tmp_path):
    proj = champ.build_champ_bor_horn_project(
        "bor_horn", np.linspace(8e9, 15e9, 8), _stepped_profile(),
        swe_file="bor_horn.sph")
    base = proj.write(tmp_path)

    tor_text = (base / "working" / "bor_horn.tor").read_text()
    assert "general_bor_waveguide_device" in tor_text
    assert tor_text.count("piecewise_linear_bor") >= 2  # interior + exterior
    assert "z_displacement" in tor_text
    assert "combined_horn_section" not in tor_text
    assert "horn_swe  swe" in tor_text
    # ports inferred from profile end points
    assert "radius           : 0.015 m" in tor_text
    assert "radius           : 0.045 m" in tor_text

    tci_text = (base / "working" / "bor_horn.tci").read_text()
    assert "export_reflection_coefficient" in tci_text
    assert "get_field" in tci_text


def test_default_bor_exterior_shape():
    ext = champ.default_bor_exterior(_stepped_profile(), 0.001)
    assert ext[0] == (0.040, 0.045)            # aperture rim
    assert ext[1][1] == pytest.approx(0.046)   # outer radius
    assert ext[-1] == (0.0, 0.015)             # throat rim
