"""
CHAMP horn-analysis objects and project assemblers.

Builders for the TICRA Tools CHAMP object classes that describe a
circularly symmetric horn plus its exterior body-of-revolution MoM
aperture, along with the .tci commands that run the analysis.

PREFERRED REPRESENTATION -- general BoR waveguide device
--------------------------------------------------------
``build_champ_bor_horn_project`` / ``champ_bor_horn_objects`` describe
the horn interior as ONE piecewise_linear_bor (z, rho) wall polyline
analyzed by a ``general_bor_waveguide_device`` with circular waveguide
ports pinned at the throat (inward normal +z) and aperture (inward
normal -z) planes. The polyline may contain radius steps, grooves,
angled slots and re-entrant segments, so any BoR feed geometry is
representable -- corrugated, axially corrugated, ring-loaded, scalar,
smooth-walled. The exterior outer wall is a second
piecewise_linear_bor in the same global z coordinates, attached to a
``circular_symmetric_aperture`` whose ``z_displacement`` equals the
aperture plane position. Layout verified against a working TICRA
Tools 25.0 project (a scalar horn with angled slots,
scalar_horn_tt.tor).

Conventions for the BoR path:
- All builder inputs are meters; profiles are written with
  ``length_unit : m`` and unitless node values.
- The interior polyline runs from the throat rim to the aperture rim;
  port radii/planes default to its end points.
- The exterior polyline starts at the aperture rim and traces the
  outside of the horn body backward toward the throat plane.

LEGACY REPRESENTATION -- horn section chains
--------------------------------------------
``build_champ_horn_project`` chains circular_waveguide_section /
simple_axial_corrugated_section objects through a
combined_horn_section into a mode-matching circular_symmetric_horn
(verified against a TICRA Tools 25.0 axially corrugated project,
guppy.tor). combined_horn_section REJECTS radius steps between
consecutive sections, so stepped staircase geometries do not load;
keep this path only for radius-continuous section chains.

Reference projects use expression strings for member values; this
module writes literal numbers, which the same member slots accept.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence as Seq

from .tor import Quantity, Ref, Sequence, Struct, Table, TorObject
from .tci import Command
from .project import GraspProject

__all__ = [
    "CircularWaveguideSectionDef",
    "AxialCorrugatedSectionDef",
    "frequency_range",
    "circular_waveguide_section",
    "simple_axial_corrugated_section",
    "combined_horn_section",
    "circular_waveguide_port_modal_expansion_settings",
    "circular_waveguide_port",
    "mode_matching_settings",
    "circular_symmetric_horn",
    "circular_waveguide_mode",
    "circular_symmetric_horn_exterior",
    "mom_settings",
    "circular_symmetric_aperture",
    "bor_profile",
    "general_bor_waveguide_device",
    "swe_output",
    "combined_device",
    "radiating_device",
    "champ_spherical_cut",
    "export_reflection_coefficient_command",
    "get_field_command",
    "beam_parameters_command",
    "champ_bor_horn_objects",
    "default_bor_exterior",
    "build_champ_bor_horn_project",
    "champ_horn_objects",
    "build_champ_horn_project",
]


# ---------------------------------------------------------------------
# Section definitions (geometry input to the assembler)
# ---------------------------------------------------------------------

@dataclass
class CircularWaveguideSectionDef:
    """Uniform circular waveguide section: radius for a length [m]."""
    radius_m: float
    length_m: float


@dataclass
class AxialCorrugatedSectionDef:
    """One axially corrugated period [m], CHAMP parametrization.

    ``input_radius_m`` is the bore radius entering the section; the
    ring fin spans [input_radius, input_radius + ridge_thickness] and
    the axial groove spans out to input_radius + ridge_thickness +
    slot_width, so the next section's input radius is input_radius +
    ridge_thickness + slot_width. ``slot_depth_m`` is the groove's
    axial depth and ``axial_distance_m`` the axial advance per period.
    """
    input_radius_m: float
    slot_width_m: float
    slot_depth_m: float
    ridge_thickness_m: float
    axial_distance_m: float
    number_of_axial_slots: int = 1


# ---------------------------------------------------------------------
# .tor object builders
# ---------------------------------------------------------------------

def frequency_range(name: str, start_hz: float, end_hz: float,
                    number_of_frequencies: int) -> TorObject:
    """Uniformly spaced frequency range (CHAMP-style frequency object)."""
    return TorObject(name, "frequency_range", [
        ("frequency_range", Struct([
            ("start_frequency", Quantity(start_hz / 1e9, "GHz")),
            ("end_frequency", Quantity(end_hz / 1e9, "GHz")),
            ("number_of_frequencies", int(number_of_frequencies)),
        ])),
    ])


def circular_waveguide_section(name: str, radius_m: float,
                               length_m: float) -> TorObject:
    return TorObject(name, "circular_waveguide_section", [
        ("radius", Quantity(float(radius_m), "m")),
        ("length", Quantity(float(length_m), "m")),
    ])


def simple_axial_corrugated_section(
        name: str, input_radius_m: float, slot_width_m: float,
        slot_depth_m: float, ridge_thickness_m: float,
        axial_distance_m: float,
        number_of_axial_slots: int = 1) -> TorObject:
    return TorObject(name, "simple_axial_corrugated_section", [
        ("input_radius", Quantity(float(input_radius_m), "m")),
        ("slot_width", Quantity(float(slot_width_m), "m")),
        ("slot_depth", Quantity(float(slot_depth_m), "m")),
        ("ridge_thickness", Quantity(float(ridge_thickness_m), "m")),
        ("axial_distance", Quantity(float(axial_distance_m), "m")),
        ("number_of_axial_slots", int(number_of_axial_slots)),
    ])


def combined_horn_section(name: str, section_names: Seq[str]) -> TorObject:
    return TorObject(name, "combined_horn_section", [
        ("horn_sections", Sequence(Ref(n) for n in section_names)),
    ])


def circular_waveguide_port_modal_expansion_settings(
        name: str, m_min: int = 1, m_max: int = 1,
        n_max: int = -1) -> TorObject:
    """Port modal expansion settings; n_max -1 lets CHAMP choose."""
    return TorObject(name, "circular_waveguide_port_modal_expansion_settings", [
        ("m_min", int(m_min)),
        ("n_max", int(n_max)),
        ("m_max", int(m_max)),
    ])


def circular_waveguide_port(name: str, radius_m: float,
                            expansion_settings_name: str) -> TorObject:
    return TorObject(name, "circular_waveguide_port", [
        ("radius", Quantity(float(radius_m), "m")),
        ("port_expansion_settings", Ref(expansion_settings_name)),
    ])


def mode_matching_settings(name: str) -> TorObject:
    """Default mode-matching settings (empty member list)."""
    return TorObject(name, "mode_matching_settings", [])


def circular_symmetric_horn(name: str, frequency_name: str,
                            port_1_name: str, port_2_name: str,
                            horn_section_name: str,
                            analysis_settings_name: str) -> TorObject:
    return TorObject(name, "circular_symmetric_horn", [
        ("frequency", Ref(frequency_name)),
        ("port_1", Ref(port_1_name)),
        ("port_2", Ref(port_2_name)),
        ("horn_section", Ref(horn_section_name)),
        ("analysis_settings", Ref(analysis_settings_name)),
    ])


def circular_waveguide_mode(name: str, m_index: int = 1,
                            n_index: int = 1) -> TorObject:
    """Circular waveguide mode (default TE11 excitation: m=1, n=1)."""
    return TorObject(name, "circular_waveguide_mode", [
        ("m_index", int(m_index)),
        ("n_index", int(n_index)),
    ])


def circular_symmetric_horn_exterior(name: str,
                                     nodes_z_rho: Seq[Seq[float]]) -> TorObject:
    """Exterior outer-wall profile as (z, rho) node pairs [m]."""
    return TorObject(name, "circular_symmetric_horn_exterior", [
        ("nodes", Table([float(z), float(rho)] for z, rho in nodes_z_rho)),
    ])


def mom_settings(name: str) -> TorObject:
    """Default BoR MoM settings (empty member list)."""
    return TorObject(name, "mom_settings", [])


def circular_symmetric_aperture(name: str, frequency_name: str,
                                waveguide_port_name: str,
                                exterior_name: str,
                                analysis_settings_name: str,
                                z_displacement_m: float | None = None,
                                ) -> TorObject:
    """Exterior BoR MoM aperture fed by a circular waveguide port.

    ``exterior`` may reference either a circular_symmetric_horn_exterior
    (aperture-local coordinates, guppy style) or a piecewise_linear_bor
    profile in the horn's global coordinates combined with
    ``z_displacement_m`` set to the aperture plane's z position
    (scalar_horn_tt style; verified against TICRA Tools 25).
    """
    members = [
        ("frequency", Ref(frequency_name)),
        ("waveguide_port", Ref(waveguide_port_name)),
        ("exterior", Ref(exterior_name)),
    ]
    if z_displacement_m is not None:
        members.append(("z_displacement",
                        Quantity(float(z_displacement_m), "m")))
    members.append(("analysis_settings", Ref(analysis_settings_name)))
    return TorObject(name, "circular_symmetric_aperture", members)


def bor_profile(name: str, nodes_z_rho_m: Seq[Seq[float]],
                length_unit: str = "m") -> TorObject:
    """CHAMP piecewise_linear_bor profile with an explicit length unit.

    Nodes are (z, rho) pairs in meters (written unitless under the
    given ``length_unit`` member; pass values already scaled if using a
    unit other than meters). The polyline may step radially and double
    back in z, so any BoR wall outline (corrugations, angled slots,
    ring-loaded inserts) is representable. Verified against a TICRA
    Tools 25 project (scalar_horn_tt.tor); distinct from the GRASP
    scatterer builder in ticra_export.objects, which writes no
    length_unit member.
    """
    return TorObject(name, "piecewise_linear_bor", [
        ("length_unit", length_unit),
        ("nodes", Table([float(z), float(rho)]
                        for z, rho in nodes_z_rho_m)),
    ])


def general_bor_waveguide_device(
        name: str, frequency_name: str, geometry_name: str,
        ports: Seq[tuple],
        analysis_settings_name: str) -> TorObject:
    """BoR MoM waveguide device over a piecewise_linear_bor interior.

    ``ports`` rows are (port_name, z_position_m, inward_normal) with
    inward_normal 'z' (pointing +z into the device, throat side) or
    '-z' (aperture side). This is the universal CHAMP representation
    for stepped/corrugated BoR horn interiors -- combined_horn_section
    chains reject radius steps between sections, this device does not.
    Verified against a TICRA Tools 25 project (scalar_horn_tt.tor).
    """
    port_structs = Sequence([
        Struct([
            ("port", Ref(port_name)),
            ("z_position", Quantity(float(z_m), "m")),
            ("inward_normal", str(normal)),
        ])
        for port_name, z_m, normal in ports
    ])
    return TorObject(name, "general_bor_waveguide_device", [
        ("frequency", Ref(frequency_name)),
        ("waveguide_geometry", Ref(geometry_name)),
        ("waveguide_ports", port_structs),
        ("analysis_settings", Ref(analysis_settings_name)),
    ])


def swe_output(name: str, file_name: str, n_phi: int = 24,
               n_theta: int = 360) -> TorObject:
    """Spherical wave expansion output object (.sph file)."""
    return TorObject(name, "swe", [
        ("file_name", file_name),
        ("sphere_sample", Struct([("n_phi", int(n_phi)),
                                  ("n_theta", int(n_theta))])),
    ])


def combined_device(name: str, device_names: Seq[str]) -> TorObject:
    return TorObject(name, "combined_device", [
        ("devices", Sequence(Ref(n) for n in device_names)),
    ])


def radiating_device(name: str, device_name: str, port_name: str,
                     waveguide_mode_name: str, amplitude: float = 1.0,
                     phase_deg: float = 0.0,
                     coor_sys_name: str | None = None) -> TorObject:
    members = [
        ("device", Ref(device_name)),
        ("port_excitation", Sequence([Struct([
            ("port", Ref(port_name)),
            ("waveguide_mode", Ref(waveguide_mode_name)),
            ("amplitude", float(amplitude)),
            ("phase", float(phase_deg)),
        ])])),
    ]
    if coor_sys_name is not None:
        members.append(("coor_sys", Ref(coor_sys_name)))
    return TorObject(name, "radiating_device", members)


def champ_spherical_cut(name: str, frequency_name: str,
                        theta_start: float = -180.0, theta_end: float = 180.0,
                        theta_np: int = 761,
                        phi_start: float = 0.0, phi_end: float = 90.0,
                        phi_np: int = 3,
                        coor_sys_name: str | None = None) -> TorObject:
    """Spherical cut for a CHAMP radiating device (no output file)."""
    members = []
    if coor_sys_name is not None:
        members.append(("coor_sys", Ref(coor_sys_name)))
    members.extend([
        ("theta_range", Struct([("start", float(theta_start)),
                                ("end", float(theta_end)),
                                ("np", int(theta_np))])),
        ("phi_range", Struct([("start", float(phi_start)),
                              ("end", float(phi_end)),
                              ("np", int(phi_np))])),
        ("frequency", Ref(frequency_name)),
    ])
    return TorObject(name, "spherical_cut", members)


# ---------------------------------------------------------------------
# .tci command builders
# ---------------------------------------------------------------------

def export_reflection_coefficient_command(
        combined_device_name: str, port_name: str,
        waveguide_mode_name: str, file_name: str = "reflection.snp",
        label: str | None = None) -> Command:
    return Command(combined_device_name, "export_reflection_coefficient", [
        ("mode_selection", Struct([
            ("port", Ref(port_name)),
            ("deembed_length", Quantity(0.0, "m")),
            ("waveguide_mode", Ref(waveguide_mode_name)),
        ])),
        ("file_name", file_name),
        ("data_format", "touchstone_db"),
    ], label=label)


def get_field_command(cut_name: str, radiating_device_name: str,
                      label: str | None = None) -> Command:
    return Command(cut_name, "get_field", [
        ("source", Sequence([Ref(radiating_device_name)])),
    ], label=label)


def beam_parameters_command(radiating_device_name: str) -> Command:
    return Command(radiating_device_name,
                   "get_radiating_device_circular_symmetric_beam_parameters")


# ---------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------

def _build_section_objects(prefix: str, sections) -> list:
    objs = []
    for i, sec in enumerate(sections, start=1):
        if isinstance(sec, CircularWaveguideSectionDef):
            objs.append(circular_waveguide_section(
                f"{prefix}_section{i}", sec.radius_m, sec.length_m))
        elif isinstance(sec, AxialCorrugatedSectionDef):
            objs.append(simple_axial_corrugated_section(
                f"{prefix}_section{i}", sec.input_radius_m, sec.slot_width_m,
                sec.slot_depth_m, sec.ridge_thickness_m, sec.axial_distance_m,
                sec.number_of_axial_slots))
        else:
            raise TypeError(f"Unsupported CHAMP section type: {type(sec)!r}")
    return objs


def default_exterior_nodes(aperture_radius_m: float, wall_thickness_m: float,
                           back_length_m: float) -> list:
    """Simple cylindrical outer shell from the aperture plane backward.

    Matches the reference project's style: the aperture plane is z = 0
    and the outer wall (aperture radius plus wall thickness) extends
    ``back_length_m`` in -z.
    """
    outer = float(aperture_radius_m) + float(wall_thickness_m)
    return [(0.0, outer), (-abs(float(back_length_m)), outer)]


def champ_bor_horn_objects(prefix: str, frequency_name: str,
                           profile_z_rho_m: Seq[Seq[float]],
                           throat_radius_m: float,
                           aperture_radius_m: float,
                           aperture_z_m: float,
                           exterior_z_rho_m: Seq[Seq[float]],
                           throat_z_m: float = 0.0,
                           excitation_m_index: int = 1,
                           excitation_n_index: int = 1,
                           coor_sys_name: str | None = None) -> dict:
    """All .tor objects for a general-BoR-waveguide horn device chain.

    This is the universal CHAMP feed representation: the interior wall
    is one piecewise_linear_bor (z, rho) polyline -- steps, grooves,
    angled slots and re-entrant features are all allowed -- analyzed by
    a general_bor_waveguide_device with circular waveguide ports pinned
    at the throat (inward normal +z) and aperture (inward normal -z)
    planes. The exterior outer-wall profile is a second
    piecewise_linear_bor in the SAME global z coordinates, attached to
    a circular_symmetric_aperture whose z_displacement equals the
    aperture plane position. Layout verified against a working TICRA
    Tools 25 project (scalar_horn_tt.tor).

    Args:
        profile_z_rho_m: Interior wall polyline from the throat rim at
            (throat_z_m, throat_radius_m) to the aperture rim at
            (aperture_z_m, aperture_radius_m), meters.
        exterior_z_rho_m: Outer-wall polyline starting at the aperture
            rim and tracing the outside of the horn body backward,
            global z coordinates, meters.

    Returns a dict with 'objects' plus key object names
    ('radiating_device', 'combined_device', 'throat_port',
    'aperture_port', 'waveguide_mode').
    """
    p = prefix
    objs = [
        circular_waveguide_port_modal_expansion_settings(
            f"{p}_throat_port_expansion_settings"),
        circular_waveguide_port(f"{p}_throat_port", throat_radius_m,
                                f"{p}_throat_port_expansion_settings"),
        circular_waveguide_port_modal_expansion_settings(
            f"{p}_aperture_port_expansion_settings"),
        circular_waveguide_port(f"{p}_aperture_port", aperture_radius_m,
                                f"{p}_aperture_port_expansion_settings"),
        mom_settings(f"{p}_MoM_settings"),
        bor_profile(f"{p}_interior_profile", profile_z_rho_m),
        general_bor_waveguide_device(
            f"{p}_horn_interior", frequency_name,
            f"{p}_interior_profile",
            [(f"{p}_throat_port", throat_z_m, "z"),
             (f"{p}_aperture_port", aperture_z_m, "-z")],
            f"{p}_MoM_settings"),
        bor_profile(f"{p}_exterior_profile", exterior_z_rho_m),
        circular_symmetric_aperture(
            f"{p}_horn_aperture", frequency_name, f"{p}_aperture_port",
            f"{p}_exterior_profile", f"{p}_MoM_settings",
            z_displacement_m=aperture_z_m),
        combined_device(f"{p}_combined_device",
                        [f"{p}_horn_aperture", f"{p}_horn_interior"]),
        circular_waveguide_mode(f"{p}_circular_waveguide_mode",
                                m_index=excitation_m_index,
                                n_index=excitation_n_index),
        radiating_device(f"{p}_radiating_device", f"{p}_combined_device",
                         f"{p}_throat_port",
                         f"{p}_circular_waveguide_mode",
                         coor_sys_name=coor_sys_name),
    ]
    return {
        "objects": objs,
        "radiating_device": f"{p}_radiating_device",
        "combined_device": f"{p}_combined_device",
        "throat_port": f"{p}_throat_port",
        "aperture_port": f"{p}_aperture_port",
        "waveguide_mode": f"{p}_circular_waveguide_mode",
    }


def default_bor_exterior(profile_z_rho_m: Seq[Seq[float]],
                         wall_thickness_m: float) -> list:
    """Simple exterior outer-wall polyline for a BoR horn interior.

    Traces from the aperture rim radially out by the wall thickness,
    straight back to the throat plane at constant outer radius, then
    down to the throat rim.
    """
    profile = [(float(z), float(r)) for z, r in profile_z_rho_m]
    z0, r0 = profile[0]
    z_ap, r_ap = profile[-1]
    outer = max(r for _, r in profile) + float(wall_thickness_m)
    return [(z_ap, r_ap), (z_ap, outer), (z0, outer), (z0, r0)]


def build_champ_bor_horn_project(
        name: str,
        frequencies_hz,
        profile_z_rho_m: Seq[Seq[float]],
        throat_radius_m: float | None = None,
        aperture_radius_m: float | None = None,
        aperture_z_m: float | None = None,
        exterior_z_rho_m: Seq[Seq[float]] | None = None,
        exterior_wall_thickness_m: float | None = None,
        prefix: str = "horn",
        cut_theta_np: int = 761,
        cut_phi_np: int = 3,
        reflection_file: str = "reflection.snp",
        swe_file: str | None = None) -> GraspProject:
    """Standalone CHAMP project for any BoR horn interior profile.

    The universal replacement for the section-based assembler: pass the
    interior wall polyline (throat rim to aperture rim, meters) and the
    port geometry is inferred from its end points unless given
    explicitly. Includes the exterior BoR MoM aperture, a TE11-excited
    radiating device, a spherical cut, reflection-coefficient export,
    beam parameters, and (optionally) an SWE .sph output object.
    """
    profile = [(float(z), float(r)) for z, r in profile_z_rho_m]
    if len(profile) < 2:
        raise ValueError("Interior profile needs at least 2 nodes")
    throat_z, throat_r = profile[0]
    aperture_z_end, aperture_r_end = profile[-1]
    if throat_radius_m is None:
        throat_radius_m = throat_r
    if aperture_radius_m is None:
        aperture_radius_m = aperture_r_end
    if aperture_z_m is None:
        aperture_z_m = aperture_z_end
    if exterior_z_rho_m is None:
        wall = (float(exterior_wall_thickness_m)
                if exterior_wall_thickness_m is not None
                else max(0.002, 0.05 * float(aperture_radius_m)))
        exterior_z_rho_m = default_bor_exterior(profile, wall)

    proj = GraspProject(name)
    tor = proj.tor
    freq_name = _add_frequency_object(tor, frequencies_hz)

    horn = champ_bor_horn_objects(
        prefix, freq_name, profile, throat_radius_m, aperture_radius_m,
        aperture_z_m, exterior_z_rho_m, throat_z_m=throat_z)
    for o in horn["objects"]:
        tor.add(o)

    cut_name = f"{prefix}_cut"
    tor.add(champ_spherical_cut(cut_name, freq_name,
                                theta_np=cut_theta_np, phi_np=cut_phi_np))
    if swe_file is not None:
        tor.add(swe_output(f"{prefix}_swe", swe_file))

    proj.tci.add(export_reflection_coefficient_command(
        horn["combined_device"], horn["throat_port"],
        horn["waveguide_mode"], reflection_file,
        label=f"{prefix}_export_reflection_coefficient"))
    proj.tci.add(get_field_command(cut_name, horn["radiating_device"],
                                   label=f"{prefix}_get_field"))
    proj.tci.add(beam_parameters_command(horn["radiating_device"]))
    return proj


def _add_frequency_object(tor, frequencies_hz) -> str:
    """Add a frequency_range (uniform) or frequency list object."""
    from . import objects as obj

    freqs = [float(f) for f in (frequencies_hz
                                if hasattr(frequencies_hz, "__len__")
                                else [frequencies_hz])]
    if not freqs:
        raise ValueError("At least one frequency is required")
    freqs = sorted(freqs)
    name = "frequency"
    if len(freqs) >= 2:
        steps = [freqs[i + 1] - freqs[i] for i in range(len(freqs) - 1)]
        uniform = max(steps) - min(steps) <= 1e-6 * freqs[-1]
    else:
        uniform = True
    if uniform and len(freqs) >= 2:
        tor.add(frequency_range(name, freqs[0], freqs[-1], len(freqs)))
    else:
        tor.add(obj.frequency(name, freqs))
    return name


def champ_horn_objects(prefix: str, frequency_name: str, sections,
                       throat_radius_m: float, aperture_radius_m: float,
                       exterior_nodes_z_rho: Seq[Seq[float]],
                       excitation_m_index: int = 1,
                       excitation_n_index: int = 1) -> dict:
    """All .tor objects for one section-based CHAMP horn device chain.

    DEPRECATED for stepped geometries: combined_horn_section rejects
    radius steps between consecutive sections, so a
    circular_waveguide_section staircase does not load. Use
    champ_bor_horn_objects / build_champ_bor_horn_project instead;
    this chain remains valid only for horns whose sections are
    radius-continuous (e.g. purely axial corrugation chains).

    Returns a dict with 'objects' (ordered TorObject list referencing
    ``frequency_name``, which must exist in the enclosing .tor) plus
    the names of the key objects: 'radiating_device', 'combined_device',
    'throat_port', 'waveguide_mode', and 'cut' is NOT included here --
    output objects/commands are the caller's choice.
    """
    p = prefix
    objs = _build_section_objects(p, sections)
    section_names = [o.display_name for o in objs]

    objs.append(circular_waveguide_port_modal_expansion_settings(
        f"{p}_throat_port_expansion_settings"))
    objs.append(circular_waveguide_port(
        f"{p}_throat_port", throat_radius_m,
        f"{p}_throat_port_expansion_settings"))
    objs.append(circular_waveguide_port_modal_expansion_settings(
        f"{p}_aperture_port_expansion_settings"))
    objs.append(circular_waveguide_port(
        f"{p}_aperture_port", aperture_radius_m,
        f"{p}_aperture_port_expansion_settings"))
    objs.append(combined_horn_section(f"{p}_combined_horn_section",
                                      section_names))
    objs.append(mode_matching_settings(f"{p}_mode_matching_settings"))
    objs.append(circular_symmetric_horn(
        f"{p}_circular_symmetric_horn", frequency_name,
        f"{p}_throat_port", f"{p}_aperture_port",
        f"{p}_combined_horn_section", f"{p}_mode_matching_settings"))
    objs.append(circular_waveguide_mode(
        f"{p}_circular_waveguide_mode",
        m_index=excitation_m_index, n_index=excitation_n_index))
    objs.append(circular_symmetric_horn_exterior(
        f"{p}_circular_symmetric_horn_exterior", exterior_nodes_z_rho))
    objs.append(mom_settings(f"{p}_MoM_settings"))
    objs.append(circular_symmetric_aperture(
        f"{p}_horn_aperture", frequency_name, f"{p}_aperture_port",
        f"{p}_circular_symmetric_horn_exterior", f"{p}_MoM_settings"))
    objs.append(combined_device(
        f"{p}_combined_device",
        [f"{p}_circular_symmetric_horn", f"{p}_horn_aperture"]))
    objs.append(radiating_device(
        f"{p}_radiating_device", f"{p}_combined_device",
        f"{p}_throat_port", f"{p}_circular_waveguide_mode"))

    return {
        "objects": objs,
        "radiating_device": f"{p}_radiating_device",
        "combined_device": f"{p}_combined_device",
        "throat_port": f"{p}_throat_port",
        "waveguide_mode": f"{p}_circular_waveguide_mode",
    }


def build_champ_horn_project(
        name: str,
        frequencies_hz,
        sections,
        throat_radius_m: float,
        aperture_radius_m: float,
        exterior_nodes_z_rho: Seq[Seq[float]] | None = None,
        exterior_wall_thickness_m: float | None = None,
        exterior_back_length_m: float | None = None,
        prefix: str = "horn",
        cut_theta_np: int = 761,
        cut_phi_np: int = 3,
        reflection_file: str = "reflection.snp") -> GraspProject:
    """Standalone section-based CHAMP project analyzing one horn.

    DEPRECATED for stepped geometries: combined_horn_section rejects
    radius steps between consecutive sections. Use
    build_champ_bor_horn_project (piecewise_linear_bor interior +
    general_bor_waveguide_device), which handles any BoR profile.

    Args:
        frequencies_hz: Analysis frequencies. A uniformly spaced list
            becomes a frequency_range object; otherwise a frequency
            list object is written.
        sections: Ordered CircularWaveguideSectionDef /
            AxialCorrugatedSectionDef entries from throat to aperture,
            radius-continuous between consecutive sections.
        throat_radius_m / aperture_radius_m: Port radii. The aperture
            port radius must equal the open cross-section radius at the
            horn mouth.
        exterior_nodes_z_rho: Outer-wall (z, rho) profile with the
            aperture plane at z = 0. Defaults to a cylindrical shell
            (see default_exterior_nodes) using
            ``exterior_wall_thickness_m`` (default: aperture radius /
            20) and ``exterior_back_length_m`` (default: aperture
            radius).
    """
    proj = GraspProject(name)
    tor = proj.tor
    freq_name = _add_frequency_object(tor, frequencies_hz)

    if exterior_nodes_z_rho is None:
        wall = (float(exterior_wall_thickness_m)
                if exterior_wall_thickness_m is not None
                else float(aperture_radius_m) / 20.0)
        back = (float(exterior_back_length_m)
                if exterior_back_length_m is not None
                else float(aperture_radius_m))
        exterior_nodes_z_rho = default_exterior_nodes(
            aperture_radius_m, wall, back)

    horn = champ_horn_objects(prefix, freq_name, sections,
                              throat_radius_m, aperture_radius_m,
                              exterior_nodes_z_rho)
    for o in horn["objects"]:
        tor.add(o)

    cut_name = f"{prefix}_cut"
    tor.add(champ_spherical_cut(cut_name, freq_name,
                                theta_np=cut_theta_np, phi_np=cut_phi_np))

    proj.tci.add(export_reflection_coefficient_command(
        horn["combined_device"], horn["throat_port"],
        horn["waveguide_mode"], reflection_file,
        label=f"{prefix}_export_reflection_coefficient"))
    proj.tci.add(get_field_command(cut_name, horn["radiating_device"],
                                   label=f"{prefix}_get_field"))
    proj.tci.add(beam_parameters_command(horn["radiating_device"]))

    return proj
