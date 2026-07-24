"""
Portable structure-shape setups: save/load the Structures panel's shape
list as a standalone JSON file, so a modeled assembly (decks, struts,
boolean groups, ...) can be moved between projects.

The file carries exactly the per-shape dictionaries used by project
serialization, so anything a project can hold round-trips: geometry
parameters, rotational copies, boolean group/mode, and the shadow-rim
inclusion flag.
"""

from __future__ import annotations

import json
from typing import List, Tuple

from .definitions import ShapeDefinition

FORMAT_NAME = 'structure_shapes'
FORMAT_VERSION = 1


def save_shape_library(path, shapes) -> int:
    """Write shapes to a JSON shape-library file.

    Args:
        path: Output file path.
        shapes: Iterable of ShapeDefinition.

    Returns:
        Number of shapes written.
    """
    serialized = [shape.to_dict() for shape in shapes]
    payload = {
        'format': FORMAT_NAME,
        'version': FORMAT_VERSION,
        'shapes': serialized,
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return len(serialized)


def load_shape_library(path) -> Tuple[List[ShapeDefinition], List[str]]:
    """Read shapes from a JSON shape-library file.

    Accepts the wrapped format written by save_shape_library and, for
    convenience, a bare JSON list of shape dictionaries.

    Returns:
        (shapes, skipped) where skipped lists human-readable reasons
        for entries that could not be imported.

    Raises:
        ValueError: If the file is not a shape library at all.
    """
    with open(path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Not a valid JSON file: {exc}") from exc

    if isinstance(data, list):
        raw_shapes = data
    elif isinstance(data, dict):
        # Accept prefixed variants of the format name (earlier tools
        # wrote an application-prefixed identifier for the same layout).
        fmt = data.get('format')
        if not (isinstance(fmt, str) and fmt.endswith(FORMAT_NAME)):
            raise ValueError(
                "Not a structure shape library file "
                f"(expected format '{FORMAT_NAME}')")
        raw_shapes = data.get('shapes', [])
        if not isinstance(raw_shapes, list):
            raise ValueError("Shape library 'shapes' entry is not a list")
    else:
        raise ValueError("Not a structure shape library file")

    shapes: List[ShapeDefinition] = []
    skipped: List[str] = []
    for index, entry in enumerate(raw_shapes):
        if not isinstance(entry, dict):
            skipped.append(f"entry {index}: not a shape dictionary")
            continue
        shape_type = entry.get('shape_type')
        if shape_type not in ShapeDefinition.SUPPORTED_TYPES:
            skipped.append(f"entry {index}: unsupported shape type "
                           f"'{shape_type}'")
            continue
        try:
            shapes.append(ShapeDefinition.from_dict(entry))
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            skipped.append(f"entry {index}: {exc}")
    return shapes, skipped
