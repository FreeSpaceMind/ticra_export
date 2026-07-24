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
