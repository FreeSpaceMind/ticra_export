"""Tests for ticra_export.shapes: definitions, meshing, booleans, I/O."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import numpy as np
import pytest

from ticra_export import objects
from ticra_export.shapes import (BOOLEAN_SOLID_TYPES, MESH_SHAPE_TYPES,
                                 SHAPE_SPECS, ShadowShapeDefinition,
                                 ShapeDefinition, build_shape_mesh,
                                 combine_shape_group, get_spec,
                                 load_shape_library, mesh_self_proximity,
                                 mesh_topology_report,
                                 partition_boolean_groups,
                                 save_shape_library, size_summary,
                                 validate_tor_file)
from ticra_export.shapes.mesh_shapes import _signed_volume
from ticra_export.tor import TorFile

try:
    import manifold3d  # noqa: F401
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False

needs_manifold = pytest.mark.skipif(not HAS_MANIFOLD,
                                    reason="manifold3d not installed")


# --- ShapeDefinition ------------------------------------------------------

def test_shape_definition_dict_roundtrip():
    shape = ShapeDefinition('strut', rho=1.0, phi=30.0, z=-0.2,
                            parameters={'start_point': [0.1, 0.0, 0.0],
                                        'end_point': [0.1, 0.0, 1.0],
                                        'diameter': 0.008, 'segments': 16},
                            rotational_count=3,
                            include_in_shadow_rim=False,
                            boolean_group='tower', boolean_mode='add')
    restored = ShapeDefinition.from_dict(shape.to_dict())
    assert restored.to_dict() == shape.to_dict()
    assert restored.start_point == [0.1, 0.0, 0.0]
    assert restored.curve_segments == 16
    assert restored.include_in_shadow_rim is False


def test_legacy_alias_is_same_class():
    assert ShadowShapeDefinition is ShapeDefinition


def test_unknown_type_falls_back_to_plate():
    assert ShapeDefinition('warp_drive').shape_type == 'circular_plate'


def test_plate_profile_annular_when_holed():
    plate = ShapeDefinition('circular_plate',
                            parameters={'radius': 0.2, 'thickness': 0.02,
                                        'hole_radius': 0.05})
    profile = plate.plate_profile_zr()
    assert profile[0] == profile[-1]          # closed trace
    rhos = {p[1] for p in profile}
    assert 0.05 in rhos and 0.2 in rhos and 0.0 not in rhos


def test_partition_demotes_single_member_groups():
    a = ShapeDefinition('box', boolean_group='g1')
    b = ShapeDefinition('box', boolean_group='g1')
    lonely = ShapeDefinition('box', boolean_group='g2')
    open_shape = ShapeDefinition('cylinder_shell', boolean_group='g1')
    ungrouped, groups = partition_boolean_groups([a, b, lonely, open_shape])
    assert list(groups) == ['g1'] and groups['g1'] == [a, b]
    assert lonely in ungrouped and open_shape in ungrouped


# --- Shape meshing --------------------------------------------------------

CLOSED_SOLID_SHAPES = {
    'box': {'dims': [0.2, 0.3, 0.1]},
    'rect_strut': {'start_point': [0.0, 0.0, 0.0],
                   'end_point': [0.1, 0.1, 0.5],
                   'width': 0.05, 'thickness': 0.02},
    'solid_cylinder': {'radius': 0.1, 'height': 0.3, 'segments': 16},
    'extruded_polygon': {'start_point': [0.0, 0.0, 0.0],
                         'end_point': [0.0, 0.0, 0.4],
                         'cross_section': [[-0.03, -0.01], [0.03, -0.01],
                                           [0.03, 0.01], [-0.03, 0.01]]},
    'strut': {'start_point': [0.0, 0.0, 0.0], 'end_point': [0.0, 0.1, 0.6],
              'diameter': 0.01, 'segments': 16},
    'circular_plate': {'radius': 0.15, 'thickness': 0.01,
                       'hole_radius': 0.04, 'segments': 24},
    'body_of_revolution': {'profile': [[-0.1, 0.08], [-0.1, 0.0],
                                       [-0.2, 0.0], [-0.2, 0.08],
                                       [-0.1, 0.08]],
                           'segments': 20},
}


@pytest.mark.parametrize('shape_type', sorted(CLOSED_SOLID_SHAPES))
def test_solid_shapes_mesh_closed_and_outward(shape_type):
    shape = ShapeDefinition(shape_type,
                            parameters=CLOSED_SOLID_SHAPES[shape_type])
    mesh = build_shape_mesh(shape, 0.0)
    report = mesh_topology_report(mesh)
    assert report['open_edges'] == 0, shape_type
    assert report['shells'] == 1, shape_type
    assert report['junction_edges'] == 0, shape_type
    assert _signed_volume(mesh) > 0.0, shape_type


def test_cylinder_shell_is_open_curved_surface():
    shape = ShapeDefinition('cylinder_shell',
                            parameters={'radius': 0.2, 'height': 0.3,
                                        'span_deg': 180.0, 'segments': 8})
    mesh = build_shape_mesh(shape, 0.0)
    assert len(mesh.curved_patches) > 0
    assert mesh_topology_report(mesh)['open_edges'] > 0


def test_type_tuples_match_definition_sets():
    assert set(MESH_SHAPE_TYPES) == ShapeDefinition.MESH_TYPES
    assert set(BOOLEAN_SOLID_TYPES) == ShapeDefinition.BOOLEAN_SOLID_TYPES


# --- Boolean combination --------------------------------------------------

def _tower_shapes():
    return [
        ShapeDefinition('solid_cylinder', boolean_group='g',
                        parameters={'radius': 0.08, 'height': 0.4,
                                    'segments': 24}),
        ShapeDefinition('box', boolean_group='g', z=0.15,
                        parameters={'dims': [0.3, 0.05, 0.05]}),
    ]


@needs_manifold
def test_union_yields_single_clean_shell():
    mesh = combine_shape_group(_tower_shapes())
    report = mesh_topology_report(mesh)
    assert report['shells'] == 1
    assert report['open_edges'] == 0
    assert report['junction_edges'] == 0
    close_nodes, reports = mesh_self_proximity(mesh)
    assert close_nodes == 0, reports


@needs_manifold
def test_subtract_removes_volume():
    add_only = combine_shape_group(_tower_shapes())
    shapes = _tower_shapes()
    shapes[1].boolean_mode = 'subtract'
    carved = combine_shape_group(shapes)
    assert 0.0 < _signed_volume(carved) < _signed_volume(add_only)


@needs_manifold
def test_all_quad_mode_has_no_triangles():
    mesh = combine_shape_group(_tower_shapes(), all_quad=True)
    report = mesh_topology_report(mesh)
    assert report['n_triangles'] == 0
    assert report['open_edges'] == 0
    assert report['shells'] == 1


# --- Shape library JSON I/O -----------------------------------------------

def test_shape_library_roundtrip(tmp_path):
    shapes = [ShapeDefinition('strut', boolean_group='t',
                              parameters={'diameter': 0.01}),
              ShapeDefinition('body_of_revolution',
                              parameters={'profile': [[0.0, 0.1],
                                                      [0.2, 0.1]]})]
    path = tmp_path / 'shapes.json'
    assert save_shape_library(path, shapes) == 2
    loaded, skipped = load_shape_library(path)
    assert skipped == []
    assert [s.to_dict() for s in loaded] == [s.to_dict() for s in shapes]


def test_shape_library_accepts_prefixed_format_name(tmp_path):
    import json
    path = tmp_path / 'legacy.json'
    payload = {'format': 'someapp_structure_shapes', 'version': 1,
               'shapes': [ShapeDefinition('box').to_dict()]}
    path.write_text(json.dumps(payload))
    loaded, skipped = load_shape_library(path)
    assert len(loaded) == 1 and skipped == []


def test_shape_library_rejects_unrelated_json(tmp_path):
    import json
    path = tmp_path / 'other.json'
    path.write_text(json.dumps({'format': 'not_shapes', 'shapes': []}))
    with pytest.raises(ValueError):
        load_shape_library(path)


# --- Catalog --------------------------------------------------------------

def test_every_supported_type_has_a_spec():
    for type_id in ShapeDefinition.SUPPORTED_TYPES:
        spec = get_spec(type_id)
        assert spec is not None and spec in SHAPE_SPECS.values()
        assert isinstance(size_summary(ShapeDefinition(type_id)), str)


# --- .tor export + validator round trip ------------------------------------

@needs_manifold
def test_combined_mesh_survives_tor_validation(tmp_path):
    mesh = combine_shape_group(_tower_shapes())
    tor = TorFile()
    tor.add(objects.coor_sys('global_coor'))
    tor.add(objects.tabulated_mesh_table(
        'tower_mesh', mesh.nodes, mesh.linear_patches,
        mesh.curved_patches, mesh.tri_patches, 'global_coor'))
    path = tmp_path / 'tower.tor'
    path.write_text(str(tor))

    messages = []
    issues = validate_tor_file(str(path),
                               report=lambda *a: messages.append(a))
    assert issues == 0, messages
