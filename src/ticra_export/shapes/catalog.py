"""
Declarative catalog of structure shape types.

Single source of truth for what shape types exist and which parameters
each one takes. The Structures editor builds its parameter form
directly from these specs, so adding a new shape type is:

  1. a geometry function (core.mesh_shapes for mesh-based shapes, or a
     dedicated mask in core.custom_shadow for analytic ones),
  2. a ShapeSpec entry here,
  3. accessors/defaults on ShapeDefinition if convenient.

No Qt imports here -- pure data, testable headless.

Field kinds:
  'float'   one value           -> parameters[key] = float
  'int'     one integer         -> parameters[key] = int
  'point3'  x/y/z triple        -> parameters[key] = [x, y, z]
  'table2'  editable 2-col rows -> parameters[key] = [[a, b], ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np

from .definitions import ShapeDefinition


@dataclass(frozen=True)
class ParamField:
    key: str
    label: str
    kind: str
    default: object
    minimum: float = -1000.0
    maximum: float = 1000.0
    decimals: int = 4
    step: float = 0.05
    columns: Tuple[str, str] = ("A", "B")
    min_rows: int = 2


@dataclass(frozen=True)
class ShapeSpec:
    type_id: str
    display_name: str
    fields: Tuple[ParamField, ...]
    #: Whether the shape is placed by rho/z (endpoint-based shapes are not)
    uses_position: bool = True
    description: str = ''
    #: Compact geometry summary for the shapes table
    size_summary: Optional[Callable] = None

    @property
    def mesh_based(self) -> bool:
        return self.type_id in ShapeDefinition.MESH_TYPES

    @property
    def boolean_capable(self) -> bool:
        return self.type_id in ShapeDefinition.BOOLEAN_SOLID_TYPES


def _strut_length(shape) -> float:
    return float(np.linalg.norm(
        np.subtract(shape.end_point, shape.start_point)))


SHAPE_SPECS = {}


def _register(spec: ShapeSpec):
    SHAPE_SPECS[spec.type_id] = spec


_register(ShapeSpec(
    type_id='circular_plate',
    display_name='Circular Plate',
    fields=(
        ParamField('radius', 'Radius (m)', 'float', 0.1,
                   minimum=0.001, maximum=100.0, step=0.01),
        ParamField('thickness', 'Thickness (m)', 'float',
                   ShapeDefinition.DEFAULT_PLATE_THICKNESS,
                   minimum=0.0001, maximum=10.0, step=0.005),
        ParamField('hole_radius', 'Hole Radius (m)', 'float', 0.0,
                   minimum=0.0, maximum=100.0, step=0.01),
        ParamField('segments', 'Curve Segments', 'int',
                   ShapeDefinition.DEFAULT_CURVE_SEGMENTS,
                   minimum=3, maximum=256),
    ),
    description='Thick disc (optional central hole) centred on the '
                'shape z; exports as an auto-generated '
                'piecewise_linear_bor. Curve segments facet it into a '
                'solid for boolean groups.',
    size_summary=lambda s: (
        f"R={s.radius:.4f} m, t={s.plate_thickness:.4f} m"
        + (f", hole={s.hole_radius:.4f} m" if s.hole_radius > 0 else "")),
))

_register(ShapeSpec(
    type_id='offset_parabola',
    display_name='Offset Parabola',
    fields=(
        ParamField('diameter', 'Diameter (m)', 'float', 0.3,
                   minimum=0.001, maximum=100.0, step=0.01),
        ParamField('f_over_d', 'F/D', 'float', 0.6,
                   minimum=0.05, maximum=10.0, decimals=3, step=0.05),
        ParamField('offset', 'Offset (m)', 'float', 0.05,
                   minimum=0.0, maximum=100.0, step=0.01),
    ),
    description='Circular cut of a parent paraboloid; exports as a '
                'paraboloid reflector with an offset elliptical rim.',
    size_summary=lambda s: (f"D={s.diameter:.4f} m, F/D={s.f_over_d:.3f}, "
                            f"off={s.offset:.4f} m"),
))

_register(ShapeSpec(
    type_id='strut',
    display_name='Strut / Wire',
    fields=(
        ParamField('start_point', 'Start X/Y/Z (m)', 'point3',
                   list(ShapeDefinition.DEFAULT_STRUT_START)),
        ParamField('end_point', 'End X/Y/Z (m)', 'point3',
                   list(ShapeDefinition.DEFAULT_STRUT_END)),
        ParamField('diameter', 'Diameter (m)', 'float',
                   ShapeDefinition.DEFAULT_STRUT_DIAMETER,
                   minimum=0.001, maximum=10.0, step=0.005),
        ParamField('segments', 'Curve Segments', 'int',
                   ShapeDefinition.DEFAULT_CURVE_SEGMENTS,
                   minimum=3, maximum=256),
    ),
    uses_position=False,
    description='Straight circular wire between two points; exports as '
                'a TICRA circular_struts entry. Curve segments facet it '
                'into a solid for boolean groups.',
    size_summary=lambda s: (f"D={s.diameter:.4f} m, "
                            f"L={_strut_length(s):.4f} m"),
))

_register(ShapeSpec(
    type_id='body_of_revolution',
    display_name='Body of Revolution',
    fields=(
        ParamField('profile', 'Profile Points (Z rel. to shape Z)', 'table2',
                   [list(p) for p in ShapeDefinition.DEFAULT_BOR_PROFILE],
                   columns=('Z (m)', 'Rho (m)')),
        ParamField('segments', 'Curve Segments', 'int',
                   ShapeDefinition.DEFAULT_CURVE_SEGMENTS,
                   minimum=3, maximum=256),
    ),
    description='Z/Rho profile revolved about a vertical axis; exports '
                'as a TICRA piecewise_linear_bor. Repeat the first point '
                'at the end to close the trace (e.g. an annular ring). '
                'Curve segments facet it into a solid for boolean groups.',
    size_summary=lambda s: (
        f"{len(s.profile_points)} pts, "
        f"Rmax={max((p[1] for p in s.profile_points), default=0.0):.4f} m"),
))

_register(ShapeSpec(
    type_id='rect_strut',
    display_name='Rectangular Strut',
    fields=(
        ParamField('start_point', 'Start X/Y/Z (m)', 'point3',
                   list(ShapeDefinition.DEFAULT_STRUT_START)),
        ParamField('end_point', 'End X/Y/Z (m)', 'point3',
                   list(ShapeDefinition.DEFAULT_STRUT_END)),
        ParamField('width', 'Width (m)', 'float',
                   ShapeDefinition.DEFAULT_STRUT_WIDTH,
                   minimum=0.001, maximum=10.0, step=0.005),
        ParamField('thickness', 'Thickness (m)', 'float',
                   ShapeDefinition.DEFAULT_STRUT_THICKNESS,
                   minimum=0.001, maximum=10.0, step=0.005),
    ),
    uses_position=False,
    description='Closed rectangular-section beam between two points '
                '(boolean-capable solid).',
    size_summary=lambda s: (f"{s.strut_width:.4f}x{s.strut_thickness:.4f} m, "
                            f"L={_strut_length(s):.4f} m"),
))

_register(ShapeSpec(
    type_id='box',
    display_name='Box',
    fields=(
        ParamField('dims', 'L/W/H (m)', 'point3',
                   list(ShapeDefinition.DEFAULT_BOX_DIMS)),
    ),
    description='Closed box at the shape position, length along the '
                'phi-rotated x-axis (boolean-capable solid).',
    size_summary=lambda s: "x".join(f"{v:.3f}" for v in s.box_dims) + " m",
))

_register(ShapeSpec(
    type_id='cylinder_shell',
    display_name='Cylinder Shell',
    fields=(
        ParamField('radius', 'Radius (m)', 'float', 0.1,
                   minimum=0.001, maximum=100.0, step=0.01),
        ParamField('height', 'Height (m)', 'float',
                   ShapeDefinition.DEFAULT_SHELL_HEIGHT,
                   minimum=0.001, maximum=100.0, step=0.05),
        ParamField('span_deg', 'Span (deg)', 'float',
                   ShapeDefinition.DEFAULT_SHELL_SPAN_DEG,
                   minimum=1.0, maximum=360.0, decimals=1, step=15.0),
        ParamField('segments', 'Segments', 'int',
                   ShapeDefinition.DEFAULT_SHELL_SEGMENTS,
                   minimum=1, maximum=128),
    ),
    description='Open cylindrical sheet with true curved quadratic '
                'patches (not boolean-capable: not a closed solid).',
    size_summary=lambda s: (f"R={s.radius:.4f} m, H={s.shell_height:.4f} m, "
                            f"{s.shell_span_deg:.0f} deg"),
))

_register(ShapeSpec(
    type_id='solid_cylinder',
    display_name='Solid Cylinder',
    fields=(
        ParamField('radius', 'Radius (m)', 'float', 0.1,
                   minimum=0.001, maximum=100.0, step=0.01),
        ParamField('height', 'Height (m)', 'float',
                   ShapeDefinition.DEFAULT_SHELL_HEIGHT,
                   minimum=0.001, maximum=100.0, step=0.05),
        ParamField('segments', 'Segments', 'int',
                   ShapeDefinition.DEFAULT_SHELL_SEGMENTS,
                   minimum=3, maximum=128),
    ),
    description='Closed faceted cylinder extending +z from the shape '
                'position (boolean-capable solid).',
    size_summary=lambda s: (f"R={s.radius:.4f} m, "
                            f"H={s.shell_height:.4f} m"),
))

_register(ShapeSpec(
    type_id='extruded_polygon',
    display_name='Extruded Cross-Section',
    fields=(
        ParamField('start_point', 'Start X/Y/Z (m)', 'point3',
                   list(ShapeDefinition.DEFAULT_STRUT_START)),
        ParamField('end_point', 'End X/Y/Z (m)', 'point3',
                   list(ShapeDefinition.DEFAULT_STRUT_END)),
        ParamField('cross_section', 'Cross-Section U/V (m)', 'table2',
                   [list(p) for p in
                    ShapeDefinition.DEFAULT_CROSS_SECTION],
                   columns=('U (m)', 'V (m)'), min_rows=3),
    ),
    uses_position=False,
    description='Arbitrary polygon cross-section swept between two '
                'points; U is the horizontal axis perpendicular to the '
                'sweep, V completes the frame (boolean-capable solid).',
    size_summary=lambda s: (f"{len(s.cross_section)}-gon, "
                            f"L={_strut_length(s):.4f} m"),
))


def get_spec(type_id: str) -> ShapeSpec:
    return SHAPE_SPECS[type_id]


def size_summary(shape) -> str:
    """Compact geometry text for the shapes table."""
    spec = SHAPE_SPECS.get(shape.shape_type)
    if spec is None or spec.size_summary is None:
        return "-"
    try:
        return spec.size_summary(shape)
    except Exception:
        return "-"
