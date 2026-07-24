"""
Writers for TICRA auxiliary data files.

.rim format verified against TicraUtilities.jl test data
(h_9m_rim.rim): one free-text header line, one line with the point
count, then whitespace-separated x y pairs.

.gxp project file format is from pygrasp (GRASP 10 era). The modern
TICRA Tools project wrapper may differ -- flagged for verification.
"""

from __future__ import annotations

import os
from typing import Iterable, Sequence


def write_rim_file(path, points_xy: Sequence[Sequence[float]],
                   header: str = "Rim exported by ticra_export") -> int:
    """Write a tabulated rim (.rim) file.

    Args:
        path: Output file path.
        points_xy: Iterable of (x, y) pairs tracing the rim in order.
            Do not repeat the first point at the end.
        header: Free-text first line.

    Returns:
        Number of points written (needed for the tabulated_rim_xy
        object's number_of_points member).
    """
    pts = [(float(p[0]), float(p[1])) for p in points_xy]
    if len(pts) < 3:
        raise ValueError(f"Rim needs at least 3 points, got {len(pts)}")
    with open(path, "w", newline="\n") as f:
        f.write(header + "\n")
        f.write(f"{len(pts)}\n")
        for x, y in pts:
            f.write(f"{x:.10e}\t{y:.10e}\n")
    return len(pts)


def write_gxp(path, project_name: str, tor_relpath: str, tci_relpath: str,
              version: str = "10.6.0",
              default_units: str = "GHz m S/m 1") -> None:
    """Write a GRASP project (.gxp) file.

    WARNING: format taken from pygrasp and matches GRASP 10.x. Not
    verified against current TICRA Tools; the file may need adjustment
    or manual re-creation there.
    """
    lines = [
        "[Comment]",
        f"{project_name} exported by ticra_export",
        "[TOR file]",
        tor_relpath,
        "[Auxiliary TOR files]",
        "[TCI file]",
        tci_relpath,
        "[Default units]",
        default_units,
        "[Project setup]",
        "<!DOCTYPE Project>",
        f'<Project version="{version}" application="GRASP">',
        " <Results/>",
        " <ResultWindows/>",
        " <WizardData/>",
        " <view_configuration/>",
        "</Project>",
    ]
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
