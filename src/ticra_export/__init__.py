"""
ticra_export: Generate TICRA GRASP project files from Python.

Provides a .tor/.tci object model and serializers, auxiliary data-file
writers (.rim, .gxp), and a project assembler that builds a complete
GRASP project directory from existing .cut and .sfc files.
"""

__version__ = "0.1.0"

from .tor import Comment, Quantity, Ref, Sequence, Struct, TorFile, TorObject
from .tci import Command, TciFile
from .datafiles import write_gxp, write_rim_file
from .project import GraspProject, build_dual_reflector_project
from . import objects
from . import champ

__all__ = [
    "Comment", "Quantity", "Ref", "Sequence", "Struct", "TorFile",
    "TorObject", "Command", "TciFile", "write_gxp", "write_rim_file",
    "GraspProject", "build_dual_reflector_project", "objects", "champ",
]
