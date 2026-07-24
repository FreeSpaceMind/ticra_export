"""
Parametric shape definitions for TICRA-exportable structures.

A ShapeDefinition stores one editable shape (plate, strut, body of
revolution, box, ...) with its placement, rotational symmetry, and
boolean-group membership. Shapes tessellate to quad meshes via
mesh_shapes.build_shape_mesh and export to TICRA tabulated meshes,
piecewise-linear BoRs, or circular struts.

The dict serialization (to_dict/from_dict) is stable, so shape-library
JSON files remain interchangeable between applications built on this
package.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class ShapeDefinition:
    """
    Shape definition for a structure object before symmetry expansion.

    The structure is intentionally generic so new shape types can be added
    without redesigning the container/editor API.
    """

    SUPPORTED_TYPES = {'circular_plate', 'offset_parabola', 'strut',
                       'body_of_revolution', 'rect_strut', 'box',
                       'cylinder_shell', 'solid_cylinder',
                       'extruded_polygon'}

    #: Shape types tessellated to quad meshes (see shapes.mesh_shapes)
    MESH_TYPES = {'rect_strut', 'box', 'cylinder_shell', 'solid_cylinder',
                  'extruded_polygon'}

    #: Shapes that may participate in boolean groups. Mesh types are
    #: natively closed solids; strut/plate/BoR shapes are converted to
    #: faceted solids (curve_segments edges) when grouped.
    BOOLEAN_SOLID_TYPES = {'rect_strut', 'box', 'solid_cylinder',
                           'extruded_polygon', 'strut', 'circular_plate',
                           'body_of_revolution'}

    DEFAULT_STRUT_START = (1.0, 0.0, -0.5)
    DEFAULT_STRUT_END = (0.1, 0.0, 0.4)
    DEFAULT_STRUT_DIAMETER = 0.02
    DEFAULT_BOR_PROFILE = ((-0.1, 0.05), (0.1, 0.05))
    DEFAULT_STRUT_WIDTH = 0.05
    DEFAULT_STRUT_THICKNESS = 0.02
    DEFAULT_BOX_DIMS = (0.2, 0.2, 0.2)
    DEFAULT_SHELL_HEIGHT = 0.2
    DEFAULT_SHELL_SPAN_DEG = 360.0
    DEFAULT_SHELL_SEGMENTS = 8
    DEFAULT_CROSS_SECTION = ((-0.025, -0.01), (0.025, -0.01),
                             (0.025, 0.01), (-0.025, 0.01))
    DEFAULT_PLATE_THICKNESS = 0.01
    DEFAULT_CURVE_SEGMENTS = 32

    def __init__(
        self,
        shape_type: str = 'circular_plate',
        rho: float = 0.0,
        phi: float = 0.0,
        z: float = 0.0,
        parameters: Optional[Dict[str, float]] = None,
        rotational_count: int = 1,
        include_in_shadow_rim: bool = True,
        boolean_group: str = '',
        boolean_mode: str = 'add',
    ):
        self.shape_type: str = shape_type if shape_type in self.SUPPORTED_TYPES else 'circular_plate'
        self.rho: float = rho
        self.phi: float = phi
        self.z: float = z
        self.parameters: Dict[str, float] = dict(parameters or {})
        self.rotational_count: int = max(1, int(rotational_count))
        # When False the shape is skipped in projected shadow-rim
        # generation (.rim export and GRASP shadow holes) but still
        # shadows PO points, previews, and exports as a scatterer.
        # Useful for thin wires whose projected rims degenerate.
        self.include_in_shadow_rim: bool = bool(include_in_shadow_rim)
        # Closed-solid shapes sharing a nonempty boolean_group are
        # boolean-combined (in list order) into one mesh for shadow
        # masks, previews, and export. Mode 'add' unions the shape in;
        # 'subtract' carves it out. Requires the manifold3d package.
        self.boolean_group: str = str(boolean_group or '')
        self.boolean_mode: str = (boolean_mode
                                  if boolean_mode in ('add', 'subtract')
                                  else 'add')

    @property
    def radius(self) -> float:
        """Convenience accessor for circular-plate radius."""
        return float(self.parameters.get('radius', 0.1))

    @radius.setter
    def radius(self, value: float):
        """Convenience accessor for circular-plate radius."""
        self.parameters['radius'] = float(value)

    @property
    def diameter(self) -> float:
        """Convenience accessor for diameter-based shapes."""
        if self.shape_type == 'circular_plate':
            return 2.0 * self.radius
        return float(self.parameters.get('diameter', 0.1))

    @diameter.setter
    def diameter(self, value: float):
        """Convenience accessor for diameter-based shapes."""
        if self.shape_type == 'circular_plate':
            self.radius = float(value) / 2.0
        else:
            self.parameters['diameter'] = float(value)

    @property
    def f_over_d(self) -> float:
        """Convenience accessor for offset-parabola focal ratio."""
        return float(self.parameters.get('f_over_d', 0.6))

    @f_over_d.setter
    def f_over_d(self, value: float):
        """Convenience accessor for offset-parabola focal ratio."""
        self.parameters['f_over_d'] = float(value)

    @property
    def offset(self) -> float:
        """Convenience accessor for offset-parabola lateral offset."""
        return float(self.parameters.get('offset', 0.0))

    @offset.setter
    def offset(self, value: float):
        """Convenience accessor for offset-parabola lateral offset."""
        self.parameters['offset'] = float(value)

    @property
    def start_point(self) -> List[float]:
        """Strut start point [x, y, z] in the global frame (meters)."""
        point = self.parameters.get('start_point', self.DEFAULT_STRUT_START)
        return [float(v) for v in point]

    @start_point.setter
    def start_point(self, value):
        self.parameters['start_point'] = [float(v) for v in value]

    @property
    def end_point(self) -> List[float]:
        """Strut end point [x, y, z] in the global frame (meters)."""
        point = self.parameters.get('end_point', self.DEFAULT_STRUT_END)
        return [float(v) for v in point]

    @end_point.setter
    def end_point(self, value):
        self.parameters['end_point'] = [float(v) for v in value]

    @property
    def profile_points(self) -> List[List[float]]:
        """Body-of-revolution profile as [z, rho] pairs (meters).

        z is relative to the shape's z position; rho is the radius about
        the vertical axis through the shape's (rho, phi) location. Matches
        TICRA's BoR definition (piecewise-linear rho(z) about the local
        z-axis).
        """
        profile = self.parameters.get('profile', self.DEFAULT_BOR_PROFILE)
        return [[float(p[0]), float(p[1])] for p in profile]

    @profile_points.setter
    def profile_points(self, value):
        self.parameters['profile'] = [[float(p[0]), float(p[1])] for p in value]

    @property
    def strut_width(self) -> float:
        """Rectangular strut cross-section width (meters)."""
        return float(self.parameters.get('width', self.DEFAULT_STRUT_WIDTH))

    @strut_width.setter
    def strut_width(self, value: float):
        self.parameters['width'] = float(value)

    @property
    def strut_thickness(self) -> float:
        """Rectangular strut cross-section thickness (meters)."""
        return float(self.parameters.get('thickness',
                                         self.DEFAULT_STRUT_THICKNESS))

    @strut_thickness.setter
    def strut_thickness(self, value: float):
        self.parameters['thickness'] = float(value)

    @property
    def box_dims(self) -> List[float]:
        """Box dimensions [length, width, height] (meters).

        Length is along the (phi-rotated) local x-axis, width along
        local y, height along z.
        """
        dims = self.parameters.get('dims', self.DEFAULT_BOX_DIMS)
        return [float(v) for v in dims]

    @box_dims.setter
    def box_dims(self, value):
        self.parameters['dims'] = [float(v) for v in value]

    @property
    def shell_height(self) -> float:
        """Cylinder shell height (meters), extending +z from shape z."""
        return float(self.parameters.get('height', self.DEFAULT_SHELL_HEIGHT))

    @shell_height.setter
    def shell_height(self, value: float):
        self.parameters['height'] = float(value)

    @property
    def shell_span_deg(self) -> float:
        """Cylinder shell azimuthal span in degrees (360 = full ring)."""
        return float(self.parameters.get('span_deg',
                                         self.DEFAULT_SHELL_SPAN_DEG))

    @shell_span_deg.setter
    def shell_span_deg(self, value: float):
        self.parameters['span_deg'] = float(value)

    @property
    def shell_start_deg(self) -> float:
        """Cylinder shell start azimuth relative to the shape phi."""
        return float(self.parameters.get('start_deg', 0.0))

    @shell_start_deg.setter
    def shell_start_deg(self, value: float):
        self.parameters['start_deg'] = float(value)

    @property
    def plate_thickness(self) -> float:
        """Circular-plate axial thickness (meters), centred on shape z."""
        return float(self.parameters.get('thickness',
                                         self.DEFAULT_PLATE_THICKNESS))

    @plate_thickness.setter
    def plate_thickness(self, value: float):
        self.parameters['thickness'] = float(value)

    @property
    def hole_radius(self) -> float:
        """Circular-plate central hole radius (meters, 0 = no hole)."""
        return float(self.parameters.get('hole_radius', 0.0))

    @hole_radius.setter
    def hole_radius(self, value: float):
        self.parameters['hole_radius'] = float(value)

    @property
    def curve_segments(self) -> int:
        """Edges used to facet curved shapes into boolean solids."""
        return int(self.parameters.get('segments',
                                       self.DEFAULT_CURVE_SEGMENTS))

    @curve_segments.setter
    def curve_segments(self, value: int):
        self.parameters['segments'] = int(value)

    def plate_profile_zr(self) -> List[List[float]]:
        """Auto-generated BoR profile trace for a circular plate.

        A closed (z, rho) loop relative to the shape z (mid-plane),
        annular when hole_radius > 0, filled to the axis otherwise --
        the same trace style as hand-built piecewise_linear_bor decks.
        """
        half = max(self.plate_thickness, 1e-6) / 2.0
        outer = max(self.radius, 0.0)
        inner = min(max(self.hole_radius, 0.0), outer)
        return [
            [half, outer],
            [half, inner],
            [-half, inner],
            [-half, outer],
            [half, outer],
        ]

    @property
    def cross_section(self) -> List[List[float]]:
        """Extruded-polygon cross-section as [u, v] pairs (meters).

        u is the horizontal axis perpendicular to the sweep direction,
        v completes the right-handed frame (same convention as the
        rectangular strut's width/thickness axes).
        """
        points = self.parameters.get('cross_section',
                                     self.DEFAULT_CROSS_SECTION)
        return [[float(p[0]), float(p[1])] for p in points]

    @cross_section.setter
    def cross_section(self, value):
        self.parameters['cross_section'] = [
            [float(p[0]), float(p[1])] for p in value]

    @property
    def shell_segments(self) -> int:
        """Curved patches around the cylinder shell circumference."""
        return int(self.parameters.get('segments',
                                       self.DEFAULT_SHELL_SEGMENTS))

    @shell_segments.setter
    def shell_segments(self, value: int):
        self.parameters['segments'] = int(value)

    def display_name(self) -> str:
        """Human-readable name for UI summaries."""
        names = {
            'circular_plate': 'Circular Plate',
            'offset_parabola': 'Offset Parabola',
            'strut': 'Strut / Wire',
            'body_of_revolution': 'Body of Revolution',
            'rect_strut': 'Rectangular Strut',
            'box': 'Box',
            'cylinder_shell': 'Cylinder Shell',
            'solid_cylinder': 'Solid Cylinder',
            'extruded_polygon': 'Extruded Cross-Section',
        }
        return names.get(self.shape_type, self.shape_type.replace('_', ' ').title())

    def in_boolean_group(self) -> bool:
        """True when this shape participates in a boolean group."""
        return bool(self.boolean_group) and \
            self.shape_type in self.BOOLEAN_SOLID_TYPES

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            'shape_type': self.shape_type,
            'rho': self.rho,
            'phi': self.phi,
            'z': self.z,
            'parameters': dict(self.parameters),
            'rotational_count': self.rotational_count,
            'include_in_shadow_rim': self.include_in_shadow_rim,
            'boolean_group': self.boolean_group,
            'boolean_mode': self.boolean_mode,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ShapeDefinition':
        """Restore shape definition from dictionary."""
        parameters = data.get('parameters')
        if parameters is None:
            # Backward compatibility with legacy circular-plate dicts.
            parameters = {'radius': data.get('radius', 0.1)}

        return ShapeDefinition(
            shape_type=data.get('shape_type', 'circular_plate'),
            rho=float(data.get('rho', 0.0)),
            phi=float(data.get('phi', 0.0)),
            z=float(data.get('z', 0.0)),
            parameters=parameters,
            rotational_count=int(data.get('rotational_count', 1)),
            include_in_shadow_rim=bool(data.get('include_in_shadow_rim', True)),
            boolean_group=str(data.get('boolean_group', '') or ''),
            boolean_mode=str(data.get('boolean_mode', 'add')),
        )


#: Backward-compatible alias for callers using the class's previous name.
ShadowShapeDefinition = ShapeDefinition


def partition_boolean_groups(shapes):
    """Split shapes into ungrouped shapes and ordered boolean groups.

    Returns (ungrouped, groups) where groups maps group name to the
    member shapes in list order. Shapes with a group name but a
    non-solid type are treated as ungrouped.
    """
    ungrouped = []
    groups: Dict[str, List[ShapeDefinition]] = {}
    for shape in shapes:
        group = getattr(shape, 'boolean_group', '') or ''
        if group and shape.shape_type in \
                ShapeDefinition.BOOLEAN_SOLID_TYPES:
            groups.setdefault(group, []).append(shape)
        else:
            ungrouped.append(shape)
    # A one-member group is just the shape itself (subtracting from
    # nothing is meaningless); keep it in the normal per-shape path.
    for name in list(groups):
        if len(groups[name]) == 1:
            ungrouped.append(groups.pop(name)[0])
    return ungrouped, groups
