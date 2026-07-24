"""
TICRA Command Interface (.tci) file model and serializer.

Syntax verified against a TICRA Tools 25.0 job file (Job_16.tci):

    COMMAND OBJECT target_name command_name ( member : value,  &
    member : value)

    QUIT

Lines within a command are continued with a trailing '&', a command
with no members serializes as '()', and the file ends with QUIT. The
command names used by the project builder ('get_currents' on PO/MoM
objects with a 'source' sequence, 'get_field' on cut/grid objects) are
verified against the same file.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Iterable, Mapping

from .tor import serialize_value


class Command:
    """A single COMMAND OBJECT statement.

    ``label`` is the optional trailing command label written after the
    closing parenthesis (CHAMP job files label commands this way, e.g.
    ``... ) axial_get_field``).
    """

    def __init__(self, target_name: str, command_name: str,
                 members: Mapping[str, Any] | None = None,
                 label: str | None = None):
        self.target_name = str(target_name)
        self.command_name = str(command_name)
        self.members: "OrderedDict[str, Any]" = OrderedDict(members or {})
        self.label = label

    def to_text(self) -> str:
        header = f"COMMAND OBJECT {self.target_name} {self.command_name}"
        suffix = f" {self.label}" if self.label else ""
        parts = [f"{k} : {serialize_value(v)}" for k, v in self.members.items()]
        if not parts:
            return f"{header} (){suffix}"

        lines = []
        for i, part in enumerate(parts):
            terminator = f"){suffix}" if i == len(parts) - 1 else ","
            prefix = f"{header} ( " if i == 0 else "  "
            lines.append(f"{prefix}{part}{terminator}")
        return " &\n".join(lines)


class TciFile:
    """An ordered list of Commands, writable to a .tci file."""

    def __init__(self, commands: Iterable[Command] = ()):
        self.commands: list[Command] = list(commands)

    def add(self, command: Command) -> Command:
        self.commands.append(command)
        return command

    def __str__(self) -> str:
        blocks = [cmd.to_text() for cmd in self.commands]
        blocks.append("QUIT")
        return "\n\n".join(blocks) + "\n"

    def write(self, path) -> None:
        with open(path, "w", newline="\n") as f:
            f.write(str(self))
