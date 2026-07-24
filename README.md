# ticra_export

Generate TICRA Tools / GRASP projects directly from Python, and convert
parametric structure shapes into TICRA tabulated meshes.

The package has two halves:

- **`ticra_export`** — writers for a complete GRASP project directory
  (`.gxp`, `.tor`, `.tci`) from existing data files (`.cut` feed
  patterns, `.sfc` triangulated surfaces, `.rim` tabulated rims), plus
  builders for the individual `.tor` object classes.
- **`ticra_export.shapes`** — a parametric shape system (plates, wire
  and rectangular struts, bodies of revolution, boxes, cylinder shells,
  extruded cross-sections) that tessellates to quad meshes, supports
  boolean union/subtract combination via `manifold3d`, and exports as
  `tabulated_mesh_table`, `piecewise_linear_bor`, or `circular_struts`
  objects. Includes mesh-hygiene passes (weld, sliver-edge collapse,
  cap removal, triangle→quad pairing, all-quad subdivision) tuned to
  avoid the junction/degeneracy problems TICRA's MoM solver rejects,
  and a `.tor` validator that diagnoses them.

This package writes files; it does not run GRASP and does not depend on
a GRASP installation.

## Install

```
pip install -e .           # writers only (numpy)
pip install -e .[full]     # + manifold3d (boolean CSG) and scipy (fast proximity checks)
```

## Usage: full project export

```python
from ticra_export import build_dual_reflector_project

project = build_dual_reflector_project(
    name="my_antenna",
    frequencies_hz=[12.25e9, 12.5e9, 12.75e9],  # must match the .cut file
    feed_cut_path="feed.cut",
    feed_number_of_cuts=8,
    main_sfc_path="main_deformed.sfc",
    sub_sfc_path="sub_shaped.sfc",
    main_rim_points_xy=main_rim,       # (N, 2) arrays, meters
    sub_rim_points_xy=sub_rim,
    feed_origin_m=(0.0, 0.0, 1.2),     # feed phase center, global frame
    sub_origin_m=(0.0, 0.0, 1.5),
)
project.write("/path/to/output")
```

This produces:

```
my_antenna/
    my_antenna.gxp
    working/
        my_antenna.tor
        my_antenna.tci
        feed.cut
        main_deformed.sfc
        sub_shaped.sfc
        my_antenna_main.rim
        my_antenna_sub.rim
```

Blockage structures (struts, BoRs, offset parabolas, tabulated meshes)
are passed via the `struts`, `bors`, `offset_parabolas`, and
`mesh_tables` arguments and grouped in a `scatterer_cluster`.

Lower-level building blocks (`TorFile`, `TorObject`, `TciFile`,
`Command`, `GraspProject`, `ticra_export.objects`) are available for
assembling other configurations.

## Usage: shapes to tabulated meshes

```python
from ticra_export.shapes import (ShapeDefinition, build_shape_mesh,
                                 combine_shape_group, mesh_topology_report)
from ticra_export import objects

tube = ShapeDefinition('body_of_revolution', boolean_group='tower',
                       parameters={'profile': [[-0.5, 0.3], [-0.5, 0.0],
                                               [-0.55, 0.0], [-0.55, 0.3],
                                               [-0.5, 0.3]],
                                   'segments': 32})
leg = ShapeDefinition('rect_strut', boolean_group='tower',
                      parameters={'start_point': [0.29, 0.0, -0.53],
                                  'end_point': [0.16, 0.0, 0.02],
                                  'width': 0.15, 'thickness': 0.02})

mesh = combine_shape_group([tube, leg])        # manifold3d union + cleanup
print(mesh_topology_report(mesh))              # shells / junctions / open edges

tor_obj = objects.tabulated_mesh_table(
    'tower_mesh', mesh.nodes, mesh.linear_patches,
    mesh.curved_patches, mesh.tri_patches, 'global_coor')
```

Shape lists round-trip to JSON (`save_shape_library` /
`load_shape_library`, format `structure_shapes`; files written with an
application-prefixed format name are accepted too), and
`ticra_export.shapes.catalog` provides a declarative parameter catalog
(`SHAPE_SPECS`) for building editors on top of the shape system.

Validate an exported `.tor` for MoM-breaking topology (junctions of 3+
closed surfaces, sliver edges, self-proximity, unintended contacts):

```
python -m ticra_export.shapes.tor_mesh_validator project.tor
```

## Format provenance and verification status

Verified against working files — GRASP 10.x `.tor` examples,
TicraUtilities.jl test data, and a TICRA Tools 25.0 project:

- `.tor` object syntax and member names for all supported classes,
  including `reflector` (`surfaces` sequence + `holes`),
  `irregular_xy_grid_triangulation` (struct-valued `file_xyz_number`),
  `circular_struts`, `piecewise_linear_bor`, `scatterer_cluster`, `mom`
- `.tci` command syntax (`COMMAND OBJECT ... get_currents/get_field/
  add_field`, `&` continuations, `QUIT` terminator, no command numbering)
- `.rim` file format (header line, count line, x/y pairs)
- `.gxp` project wrapper (accepted by TICRA Tools 25.0)

`tabulated_mesh_table` is verified for the empty `regions` table, the
`nodes` table, and `linear_patches` rows (patch region -1 = solid PEC;
dielectrics are not represented).

Not verified:

- The `curved_patches` member (9-node biquadratic patches: corners 1-4
  counter-clockwise, mid-edge nodes 5-8, center 9) follows the GRASP
  tabulated-mesh convention, but no example file with curved patches was
  available — verify the first curved export in TICRA Tools
- Triangles written as degenerate quads (last node repeated)
- The `centre` member of `elliptical_rim` (standard GRASP offset-aperture
  idiom; the example files only contained origin-centred rims)

## Coordinate convention

All origins are specified in the global frame, taken to be the main
reflector vertex-centered frame, matching the `.sfc` export convention
(surface files referenced to the primary vertex). The feed coordinate
system should be placed at the feed phase center (the SWE origin).

## Roadmap

- CHAMP project export (feed/horn designs as BoR objects) for use with
  the `antenna_feeds` package
- Verification of curved-patch and degenerate-quad serializations

## Tests

```
python -m pytest tests/
```

Serialization tests compare generated objects against blocks copied
verbatim from working GRASP `.tor` files.
