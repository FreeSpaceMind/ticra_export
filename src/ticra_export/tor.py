"""
TICRA Object Repository (.tor) file model and serializer.

The .tor format is a plain-text object repository. Each object is:

    display_name  class_name
    (
      member_name      : value,
      ...
    )

Values may be:
  - Quantity: number with optional unit, e.g. ``16.0 m``, ``50.0 MHz``
  - Ref: reference to another object, e.g. ``ref(single_global_coor)``
  - Struct: named members, e.g. ``struct(x: 0.0 m, y: 0.0 m, z: 16.0 m)``
  - Sequence: ordered values, e.g. ``sequence(20.0 MHz, 50.0 MHz)``
  - Quoted comment strings and bare words (filenames, enums like ``far``)

Syntax verified against working GRASP 10.x .tor files (GRASPoptimization
repo) and TicraUtilities.jl test data. TICRA has maintained .tor
compatibility across versions; this serializer targets that common syntax.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Mapping, Union


class Quantity:
    """A number with an optional unit, e.g. Quantity(16.0, 'm')."""

    def __init__(self, value: float, unit: str | None = None):
        self.value = value
        self.unit = unit

    def __str__(self) -> str:
        if self.unit:
            return f"{format_number(self.value)} {self.unit}"
        return format_number(self.value)

    def __repr__(self) -> str:
        return f"Quantity({self.value!r}, {self.unit!r})"


class Ref:
    """A reference to another .tor object by display name."""

    def __init__(self, name: str):
        self.name = str(name)

    def __str__(self) -> str:
        return f"ref({self.name})"

    def __repr__(self) -> str:
        return f"Ref({self.name!r})"


class Struct(OrderedDict):
    """A struct(...) value with named members."""

    def __str__(self) -> str:
        inner = ", ".join(f"{k}: {serialize_value(v)}" for k, v in self.items())
        return f"struct({inner})"


class Sequence(list):
    """A sequence(...) value with ordered elements."""

    def __str__(self) -> str:
        inner = ",".join(serialize_value(v) for v in self)
        return f"sequence({inner})"


class Comment(str):
    """A double-quoted string value."""

    def __str__(self) -> str:
        return f'"{str.__str__(self)}"'


class Table(list):
    """A table(...) value: rows of whitespace-separated values.

    Each element is one row (an iterable of values). Serialized in the
    layout used by TICRA Tools, e.g. piecewise_linear_bor nodes:

        nodes            : table
          (
          0.15  0.51
          0.15  0.45
          )
    """

    def __str__(self) -> str:
        lines = ["table", "    ("]
        for row in self:
            cells = "  ".join(serialize_value(v) for v in row)
            lines.append(f"    {cells}  ")
        lines.append("    )")
        return "\n".join(lines)


def format_number(x: float) -> str:
    """Format a number the way GRASP examples do: always with a decimal
    point for floats, plain for integers."""
    if isinstance(x, bool):
        raise TypeError("bool is not a valid .tor number")
    if isinstance(x, int):
        return str(x)
    s = repr(float(x))
    return s


def serialize_value(v: Any) -> str:
    """Serialize a member value to its .tor text form."""
    if isinstance(v, (Quantity, Ref, Struct, Sequence, Comment, Table)):
        return str(v)
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, (int, float)):
        return format_number(v)
    if isinstance(v, str):
        # Bare word: filename, enum value (far/near, on/off), etc.
        return v
    raise TypeError(f"Cannot serialize value of type {type(v).__name__}: {v!r}")


class TorObject:
    """A single object in the repository.

    Args:
        display_name: Object name used by ref(...) elsewhere.
        class_name: TICRA class, e.g. 'reflector', 'coor_sys'.
        members: Mapping of member name -> value. Order is preserved.
    """

    def __init__(self, display_name: str, class_name: str,
                 members: Mapping[str, Any] | None = None):
        self.display_name = str(display_name)
        self.class_name = str(class_name)
        self.members: "OrderedDict[str, Any]" = OrderedDict(members or {})

    def __str__(self) -> str:
        lines = [f"{self.display_name}  {self.class_name}  ", "("]
        items = list(self.members.items())
        for i, (k, v) in enumerate(items):
            comma = "," if i < len(items) - 1 else ""
            lines.append(f"  {k:<16} : {serialize_value(v)}{comma}")
        lines.append(")")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (f"TorObject({self.display_name!r}, {self.class_name!r}, "
                f"{dict(self.members)!r})")


class TorFile:
    """An ordered collection of TorObjects, writable to a .tor file."""

    def __init__(self, objects: Iterable[TorObject] = ()):
        self._objects: "OrderedDict[str, TorObject]" = OrderedDict()
        for obj in objects:
            self.add(obj)

    def add(self, obj: TorObject) -> TorObject:
        """Add an object. Raises ValueError on duplicate display names."""
        if obj.display_name in self._objects:
            raise ValueError(
                f"Duplicate object name in .tor file: {obj.display_name}")
        self._objects[obj.display_name] = obj
        return obj

    def __getitem__(self, name: str) -> TorObject:
        return self._objects[name]

    def __contains__(self, name: str) -> bool:
        return name in self._objects

    def __iter__(self):
        return iter(self._objects.values())

    def __len__(self) -> int:
        return len(self._objects)

    def validate_refs(self) -> list[str]:
        """Return a list of ref() targets that are not defined in this file."""
        missing = []

        def check(v):
            if isinstance(v, Ref) and v.name not in self._objects:
                missing.append(v.name)
            elif isinstance(v, Struct):
                for x in v.values():
                    check(x)
            elif isinstance(v, Sequence):
                for x in v:
                    check(x)
            elif isinstance(v, Table):
                for row in v:
                    for x in row:
                        check(x)

        for obj in self:
            for v in obj.members.values():
                check(v)
        return missing

    def __str__(self) -> str:
        return "\n \n".join(str(obj) for obj in self) + "\n"

    def write(self, path) -> None:
        with open(path, "w", newline="\n") as f:
            f.write(str(self))
