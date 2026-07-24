"""
Validate the mesh topology of an exported .tor file.

Parses tabulated_mesh_table, piecewise_linear_bor, and circular_struts
objects out of a .tor file, rebuilds them as solids, and reports the
things TICRA's MoM mesher rejects:

- edges shared by more than two patches (junctions) inside any mesh,
  checked at several weld tolerances,
- multiple closed shells inside one mesh object,
- open (boundary) edges in a mesh that should be closed,
- degenerate quad rows (triangles written with a repeated node),
- pairwise contact/overlap between all closed solids in the file.

Run on the exact file TICRA fails on:

    python -m ticra_export.shapes.tor_mesh_validator project.tor

Only numpy is required; contact overlap detection additionally uses
manifold3d when installed.
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

from .mesh_shapes import (
    QuadMesh,
    check_solid_contacts,
    mesh_self_proximity,
    mesh_topology_report,
    revolve_solid_mesh,
    wire_solid_mesh,
)

_OBJECT_RE = re.compile(r'^(\S+)\s+(\S+)\s*$')


def _extract_objects(text: str) -> List[Tuple[str, str, str]]:
    """(name, class, body) for each object block in a .tor file."""
    objects = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _OBJECT_RE.match(lines[i].strip()) if lines[i].strip() else None
        header = None
        if match and not lines[i].lstrip().startswith(('//', '/*')):
            header = (match.group(1), match.group(2))
        if header and i + 1 < len(lines) and lines[i + 1].strip().startswith('('):
            depth = 0
            body_lines = []
            j = i + 1
            while j < len(lines):
                depth += lines[j].count('(') - lines[j].count(')')
                body_lines.append(lines[j])
                j += 1
                if depth <= 0:
                    break
            objects.append((header[0], header[1], "\n".join(body_lines)))
            i = j
        else:
            i += 1
    return objects


def _extract_member(body: str, member: str) -> Optional[str]:
    """Raw text of one member value (balanced-paren aware)."""
    pattern = re.compile(rf'\b{member}\s*:', re.IGNORECASE)
    match = pattern.search(body)
    if not match:
        return None
    start = match.end()
    depth = 0
    for k in range(start, len(body)):
        ch = body[k]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
            if depth < 0:
                return body[start:k]
        elif ch == ',' and depth == 0:
            return body[start:k]
    return body[start:]


def _table_rows(raw: Optional[str]) -> List[List[float]]:
    """Numeric rows of a table(...) value; rows with expressions are
    skipped (reported by the caller via row count mismatch)."""
    if raw is None:
        return []
    inner = raw.strip()
    if inner.startswith('table'):
        inner = inner[len('table'):].strip()
    if inner.startswith('('):
        inner = inner[1:]
    if inner.endswith(')'):
        inner = inner[:-1]
    rows = []
    for line in inner.splitlines():
        line = line.strip()
        if not line or line in ('(', ')'):
            continue
        parts = line.replace(',', ' ').split()
        try:
            rows.append([float(p) for p in parts])
        except ValueError:
            continue
    return rows


def parse_tabulated_mesh(body: str) -> Tuple[QuadMesh, int]:
    """QuadMesh plus degenerate-quad row count from an object body."""
    node_rows = _table_rows(_extract_member(body, 'nodes'))
    nodes = np.array([row[1:4] for row in node_rows if len(row) >= 4])

    quads = []
    tris = []
    degenerate = 0
    for row in _table_rows(_extract_member(body, 'linear_patches')):
        if len(row) < 8:
            continue
        idx = [int(v) - 1 for v in row[4:8]]
        if idx[2] == idx[3] or len(set(idx)) == 3:
            tris.append(sorted(set(idx), key=idx.index)[:3])
            degenerate += 1
        else:
            quads.append(idx)
    curved = []
    for row in _table_rows(_extract_member(body, 'curved_patches')):
        if len(row) >= 13:
            curved.append([int(v) - 1 for v in row[4:13]])
    return QuadMesh(nodes, quads or None, curved or None, tris or None), \
        degenerate


def parse_piecewise_linear_bor(name: str, body: str,
                               coor_origins: Dict[str, Tuple[float, float, float]],
                               coor_ref: Optional[str]) -> Optional[QuadMesh]:
    rows = _table_rows(_extract_member(body, 'nodes'))
    profile = [row[:2] for row in rows if len(row) >= 2]
    if len(profile) < 2:
        return None
    origin = coor_origins.get(coor_ref or '', (0.0, 0.0, 0.0))
    return revolve_solid_mesh(profile, origin[:2], origin[2], 32)


def parse_circular_struts(body: str) -> List[QuadMesh]:
    radius_text = _extract_member(body, 'radius') or '0'
    match = re.search(r'([-+0-9.eE]+)\s*(mm|cm|m)?', radius_text.strip())
    radius = 0.0
    if match:
        radius = float(match.group(1))
        if match.group(2) == 'mm':
            radius /= 1000.0
        elif match.group(2) == 'cm':
            radius /= 100.0
    struts = []
    raw = _extract_member(body, 'end_points') or ''
    for struct_text in re.findall(r'struct\(([^)]*)\)', raw):
        values = {}
        for key, num, unit in re.findall(
                r'(point\d_[xyz])\s*:\s*([-+0-9.eE]+)\s*(mm|cm|m)?',
                struct_text):
            value = float(num)
            if unit == 'mm':
                value /= 1000.0
            elif unit == 'cm':
                value /= 100.0
            values[key] = value
        if len(values) == 6 and radius > 0:
            p1 = (values['point1_x'], values['point1_y'], values['point1_z'])
            p2 = (values['point2_x'], values['point2_y'], values['point2_z'])
            try:
                struts.append(wire_solid_mesh(p1, p2, radius, 32))
            except ValueError:
                continue
    return struts


def _coor_origins(objects) -> Dict[str, Tuple[float, float, float]]:
    origins = {}
    for name, class_name, body in objects:
        if class_name != 'coor_sys':
            continue
        raw = _extract_member(body, 'origin')
        origin = [0.0, 0.0, 0.0]
        if raw:
            for axis_i, axis in enumerate('xyz'):
                m = re.search(rf'\b{axis}\s*:\s*([-+0-9.eE]+)\s*(mm|cm|m)?',
                              raw)
                if m:
                    value = float(m.group(1))
                    if m.group(2) == 'mm':
                        value /= 1000.0
                    elif m.group(2) == 'cm':
                        value /= 100.0
                    origin[axis_i] = value
        origins[name] = tuple(origin)
    return origins


def validate_tor_file(path, report=print) -> int:
    """Analyze a .tor file; returns the number of problems found."""
    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    objects = _extract_objects(text)
    origins = _coor_origins(objects)

    problems = 0
    solids = []
    for name, class_name, body in objects:
        if class_name == 'tabulated_mesh_table':
            mesh, degenerate = parse_tabulated_mesh(body)
            all_faces = ([i for f in mesh.linear_patches for i in f]
                         + [i for f in mesh.curved_patches for i in f]
                         + [i for f in mesh.tri_patches for i in f])
            if mesh.patch_count == 0 or len(mesh.nodes) == 0 or \
                    (all_faces and max(all_faces) >= len(mesh.nodes)):
                report(f"[{name}] nodes/patches not fully numeric "
                       "(expression-based mesh?) -- skipped")
                continue
            report(f"[{name}] {len(mesh.linear_patches)} quads, "
                   f"{len(mesh.tri_patches)} triangles"
                   + (f" ({degenerate} written as degenerate quads)"
                      if degenerate else "")
                   + f", {len(mesh.curved_patches)} curved")
            for tol in (1e-6, 1e-5, 1e-4):
                topo = mesh_topology_report(mesh, tol)
                issues = []
                if topo['junction_edges'] > 0:
                    issues.append(f"{topo['junction_edges']} junction "
                                  f"edge(s) near {topo['junction_points'][:2]}")
                if topo['shells'] > 1:
                    issues.append(f"{topo['shells']} shells in one object")
                if topo['open_edges'] > 0:
                    issues.append(f"{topo['open_edges']} open edge(s)")
                if issues:
                    problems += 1
                    report(f"[{name}] PROBLEM at weld tol {tol:g}: "
                           + "; ".join(issues))
            count, hits = mesh_self_proximity(mesh)
            if count > 0:
                problems += 1
                spots = ", ".join(
                    f"{d * 1000:.2f}mm at ({p[0]:.4f}, {p[1]:.4f}, "
                    f"{p[2]:.4f})" for d, p in hits[:3])
                report(f"[{name}] PROBLEM: surface nearly touches itself "
                       f"at {count} node(s) (closest {spots}). TICRA MoM "
                       "welds these sheets together into junctions of 3+ "
                       "closed surfaces. Make the members genuinely "
                       "overlap by several millimeters (or leave clear "
                       "space) at these locations.")
            solids.append((name, mesh))
        elif class_name == 'piecewise_linear_bor':
            coor = None
            coor_raw = _extract_member(body, 'coor_sys')
            if coor_raw:
                m = re.search(r'ref\((\w+)\)', coor_raw)
                coor = m.group(1) if m else None
            mesh = parse_piecewise_linear_bor(name, body, origins, coor)
            if mesh is not None:
                solids.append((name, mesh))
                report(f"[{name}] BoR reconstructed for contact checking")
        elif class_name == 'circular_struts':
            for k, mesh in enumerate(parse_circular_struts(body)):
                solids.append((f"{name}[{k}]", mesh))
            report(f"[{name}] struts reconstructed for contact checking")

    contacts = check_solid_contacts(solids)
    for name_a, name_b, kind in contacts:
        problems += 1
        report(f"PROBLEM: {name_a} and {name_b} are {kind} -- TICRA MoM "
               "rejects junctions of 3+ closed surfaces. Merge them into "
               "one boolean group.")

    if problems == 0:
        report("No junction/shell/contact problems found "
               f"({len(solids)} closed solid(s) checked).")
    return problems


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage: python -m ticra_export.shapes."
              "tor_mesh_validator <file.tor>")
        sys.exit(2)
    sys.exit(1 if validate_tor_file(sys.argv[1]) else 0)
