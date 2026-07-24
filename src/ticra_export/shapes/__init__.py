"""
Parametric structure shapes convertible to TICRA tabulated meshes.

Workflow: define shapes (ShapeDefinition), tessellate them to quad
meshes (build_shape_mesh / combine_shape_group for boolean groups),
then hand the node/patch arrays to ticra_export.objects
.tabulated_mesh_table for the .tor file. tor_mesh_validator checks an
exported .tor for the junction/degeneracy problems TICRA's MoM solver
rejects.
"""

from .definitions import (ShadowShapeDefinition, ShapeDefinition,
                          partition_boolean_groups)
from .mesh_shapes import (BOOLEAN_SOLID_TYPES, MESH_SHAPE_TYPES, QuadMesh,
                          build_shape_mesh, check_solid_contacts,
                          clean_boolean_mesh, collect_closed_solids,
                          combine_meshes, combine_shape_group,
                          mesh_self_proximity, mesh_shells,
                          mesh_topology_report, pair_triangles_into_quads,
                          subdivide_to_quads, weld_mesh)
from .catalog import SHAPE_SPECS, ParamField, ShapeSpec, get_spec, size_summary
from .library import (FORMAT_NAME, FORMAT_VERSION, load_shape_library,
                      save_shape_library)
from .tor_mesh_validator import validate_tor_file

__all__ = [
    'ShapeDefinition', 'ShadowShapeDefinition', 'partition_boolean_groups',
    'QuadMesh', 'build_shape_mesh', 'combine_meshes', 'combine_shape_group',
    'clean_boolean_mesh', 'weld_mesh', 'mesh_shells', 'mesh_topology_report',
    'mesh_self_proximity', 'pair_triangles_into_quads', 'subdivide_to_quads',
    'collect_closed_solids', 'check_solid_contacts',
    'MESH_SHAPE_TYPES', 'BOOLEAN_SOLID_TYPES',
    'SHAPE_SPECS', 'ParamField', 'ShapeSpec', 'get_spec', 'size_summary',
    'save_shape_library', 'load_shape_library', 'FORMAT_NAME',
    'FORMAT_VERSION', 'validate_tor_file',
]
