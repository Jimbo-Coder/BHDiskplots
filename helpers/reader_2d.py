"""CarpetIOASCII 2D slice reader for BHDisk plots."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
import fcntl
import hashlib
import json
import numpy as np
import os
import re
import shutil
import subprocess
import tempfile
import time
from io import BytesIO
from pathlib import Path as StdPath

from config import CACHE_ROOT

# Large CarpetIOASCII 2D slice reader.
# This indexes byte ranges for one iteration, then reads only selected
# reflevel/timelevel/map/component blocks.

ITERATION_RE = re.compile(rb"^# iteration\s+(\d+)")
REFLEVEL_RE = re.compile(
    rb"^# refinement level\s+(\d+)\s+"
    rb"multigrid level\s+(\d+)\s+"
    rb"map\s+(\d+)\s+"
    rb"component\s+(\d+)\s+"
    rb"time level\s+(\d+)"
)

TWO_D_INDEX_CACHE_ROOT = StdPath(CACHE_ROOT / "2d_indices")
TWO_D_INDEX_CACHE_VERSION = 1
TWO_D_INDEX_SAMPLE_BYTES = 4096
TWO_D_INDEX_SAMPLE_COUNT = 9

@dataclass(frozen=True)
class GridComponent:
    it: int
    ref_level: int
    component: int
    bytestart: int
    bytestop: int
    line_count: int
    multigrid_level: int = 0
    map_id: int = 0
    time_level: int = 0
    x_min: float = np.nan
    x_max: float = np.nan
    y_min: float = np.nan
    y_max: float = np.nan
    z_min: float = np.nan
    z_max: float = np.nan
    dx: float = np.nan
    dy: float = np.nan
    dz: float = np.nan

    @property
    def has_data(self):
        return self.line_count > 0 and self.bytestop > self.bytestart

    @property
    def byte_span(self):
        return self.bytestop - self.bytestart


@dataclass(frozen=True)
class TwoDFilePosition:
    """Byte position carrying the source file for a virtual merged timeline."""
    filepath: StdPath
    start_byte: int

@dataclass
class Slice2D:
    filepath: object
    variable: str
    plane: str
    iteration: int
    ref_level: object
    components: list
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    data: np.ndarray
    time: np.ndarray
    component: np.ndarray
    rl: np.ndarray

    @property
    def axis_names(self):
        return {"xy": ("x", "y"), "xz": ("x", "z"), "yz": ("y", "z")}[self.plane]

    @property
    def a(self):
        return getattr(self, self.axis_names[0])

    @property
    def b(self):
        return getattr(self, self.axis_names[1])

    def to_grid(self):
        """Put point data on the regular Carpet grid for this reflevel.

        Missing points are NaN. If components overlap, the later file value wins.
        This is intended for one reflevel; for all-level data use
        ``to_composite_grid`` or ``iter_level_grids``.
        """
        a_unique = np.unique(self.a)
        b_unique = np.unique(self.b)
        grid = np.full((b_unique.size, a_unique.size), np.nan)
        ia = np.searchsorted(a_unique, self.a)
        ib = np.searchsorted(b_unique, self.b)
        grid[ib, ia] = self.data
        return a_unique, b_unique, grid

    def composite_points(self, remove_covered=True, atol=None):
        """Return AMR-composited point arrays.

        ``remove_covered=True`` removes lower-reflevel points whose coordinates
        lie inside a finer component's bounding box. This handles overlapping
        grids even when the coarse and fine points are not identical. Remaining
        near-duplicate coordinates are resolved by keeping the highest reflevel.
        """
        if atol is None:
            atol = self._coordinate_tolerance()

        keep = np.ones(self.data.size, dtype=bool)
        if remove_covered:
            for fine_rl in sorted(np.unique(self.rl), reverse=True):
                fine_mask = self.rl == fine_rl
                for component in np.unique(self.component[fine_mask]):
                    patch = fine_mask & (self.component == component)
                    amin = np.min(self.a[patch]) - atol
                    amax = np.max(self.a[patch]) + atol
                    bmin = np.min(self.b[patch]) - atol
                    bmax = np.max(self.b[patch]) + atol
                    covered_coarse = (
                        (self.rl < fine_rl) & keep &
                        (self.a >= amin) & (self.a <= amax) &
                        (self.b >= bmin) & (self.b <= bmax)
                    )
                    keep[covered_coarse] = False

        surviving = np.nonzero(keep)[0]
        a = self.a[surviving]
        b = self.b[surviving]
        rl = self.rl[surviving]

        # Resolve coordinate duplicates to tolerance. Sorting by rl makes the
        # highest reflevel the last entry for each near-coordinate group.
        order = np.lexsort((rl, b, a))
        a_sorted = a[order]
        b_sorted = b[order]
        keep_sorted = np.ones(order.size, dtype=bool)
        same_as_next = (np.abs(a_sorted[:-1] - a_sorted[1:]) <= atol) & (np.abs(b_sorted[:-1] - b_sorted[1:]) <= atol)
        keep_sorted[:-1][same_as_next] = False
        selected = surviving[order[keep_sorted]]
        return self.a[selected], self.b[selected], self.data[selected], self.rl[selected]

    def to_composite_grid(self, remove_covered=True, atol=None):
        """Grid all reflevels after AMR overlap cleanup.

        Coarser points covered by finer patch bounding boxes are removed first,
        then near-duplicate coordinates keep the finest value.
        """
        a, b, data, _ = self.composite_points(remove_covered=remove_covered, atol=atol)
        a_unique = np.unique(a)
        b_unique = np.unique(b)
        grid = np.full((b_unique.size, a_unique.size), np.nan)
        ia = np.searchsorted(a_unique, a)
        ib = np.searchsorted(b_unique, b)
        grid[ib, ia] = data
        return a_unique, b_unique, grid

    def _coordinate_tolerance(self):
        """Tolerance for matching coordinates printed with slight ASCII roundoff.

        The tolerance is tied to the smallest observed grid spacing, so values
        like 3.125, 3.125000, and 3.124999 collapse together, while neighboring
        grid points remain distinct.
        """
        spacings = []
        for arr in (self.a, self.b):
            unique = np.unique(arr)
            if unique.size > 1:
                diffs = np.diff(unique)
                positive = diffs[diffs > 0]
                if positive.size:
                    # Use a typical spacing, not the minimum; near-duplicate
                    # ASCII coordinates would otherwise define the tolerance.
                    spacings.append(np.median(positive))
        scale = max(np.max(np.abs(self.a)), np.max(np.abs(self.b)), 1.0)
        if spacings:
            return max(min(spacings) * 1.0e-2, scale * 1.0e-12)
        return scale * 1.0e-12

    def iter_level_grids(self):
        """Yield ``(rl, a, b, grid)`` from coarse to fine."""
        for rl in np.unique(self.rl):
            mask = self.rl == rl
            sub = Slice2D(
                filepath=self.filepath, variable=self.variable, plane=self.plane,
                iteration=self.iteration, ref_level=int(rl),
                components=sorted(set(self.component[mask])),
                x=self.x[mask], y=self.y[mask], z=self.z[mask],
                data=self.data[mask], time=self.time[mask],
                component=self.component[mask], rl=self.rl[mask],
            )
            a, b, grid = sub.to_grid()
            yield int(rl), a, b, grid

    def iter_level_component_grids(self):
        """Yield component-resolved AMR grids to avoid stitching artifacts.

        Plotting all components at one level together can create large NaN-backed
        rectangles at patch boundaries. Keeping components separate avoids giant
        flat square fill artifacts from missing points between patches.
        """
        for rl in np.unique(self.rl):
            level_mask = self.rl == rl
            for component in sorted(np.unique(self.component[level_mask])):
                component_mask = level_mask & (self.component == component)
                sub = Slice2D(
                    filepath=self.filepath,
                    variable=self.variable,
                    plane=self.plane,
                    iteration=self.iteration,
                    ref_level=int(rl),
                    components=sorted(set(self.component[component_mask])),
                    x=self.x[component_mask],
                    y=self.y[component_mask],
                    z=self.z[component_mask],
                    data=self.data[component_mask],
                    time=self.time[component_mask],
                    component=self.component[component_mask],
                    rl=self.rl[component_mask],
                )
                a, b, grid = sub.to_grid()
                yield int(rl), int(component), a, b, grid

    def to_uniform_grid(self, xlim, ylim, shape):
        """Nearest-neighbor AMR resampling with finer components taking priority."""
        nx, ny = (int(value) for value in shape)
        if nx < 2 or ny < 2:
            raise ValueError("Uniform 2D grid dimensions must both be at least 2")
        a_target = np.linspace(float(xlim[0]), float(xlim[1]), nx)
        b_target = np.linspace(float(ylim[0]), float(ylim[1]), ny)
        uniform = np.full((ny, nx), np.nan)

        for _, _, a, b, grid in self.iter_level_component_grids():
            if a.size == 0 or b.size == 0:
                continue
            da = _typical_spacing(a)
            db = _typical_spacing(b)
            a_mask = (a_target >= a[0] - 0.5 * da) & (a_target <= a[-1] + 0.5 * da)
            b_mask = (b_target >= b[0] - 0.5 * db) & (b_target <= b[-1] + 0.5 * db)
            if not np.any(a_mask) or not np.any(b_mask):
                continue
            target_a_indices = np.flatnonzero(a_mask)
            target_b_indices = np.flatnonzero(b_mask)
            source_a_indices = _nearest_coordinate_indices(a, a_target[a_mask])
            source_b_indices = _nearest_coordinate_indices(b, b_target[b_mask])
            sampled = grid[np.ix_(source_b_indices, source_a_indices)]
            destination = uniform[np.ix_(target_b_indices, target_a_indices)]
            finite = np.isfinite(sampled)
            destination[finite] = sampled[finite]
            uniform[np.ix_(target_b_indices, target_a_indices)] = destination
        return a_target, b_target, uniform


def _normalize_plane(plane):
    plane = str(plane).lower()
    if plane not in {"xy", "xz", "yz"}:
        raise ValueError("plane must be 'xy', 'xz', or 'yz'")
    return plane


def _is_data_line(line):
    stripped = line.lstrip()
    if not stripped or stripped.startswith(b"#"):
        return False
    return stripped[:1].isdigit() or stripped[:1] in {b"+", b"-", b"."}


def _indexed_grid_point(line):
    """Return Carpet grid indices and coordinates from one numeric ASCII row."""
    if line is None:
        return None
    parts = line.split()
    if len(parts) < 12:
        return None
    try:
        indices = tuple(int(parts[index]) for index in (5, 6, 7))
        coordinates = tuple(float(parts[index]) for index in (9, 10, 11))
    except ValueError:
        return None
    return indices, coordinates


def _block_geometry(first_point, last_point):
    if first_point is None or last_point is None:
        return (np.nan,) * 9
    first_indices, first_coordinates = first_point
    last_indices, last_coordinates = last_point
    bounds = []
    spacings = []
    for first_index, last_index, first_coordinate, last_coordinate in zip(
        first_indices,
        last_indices,
        first_coordinates,
        last_coordinates,
    ):
        bounds.extend((min(first_coordinate, last_coordinate), max(first_coordinate, last_coordinate)))
        index_span = abs(last_index - first_index)
        spacings.append(abs(last_coordinate - first_coordinate) / index_span if index_span else np.nan)
    return (*bounds, *spacings)


def _typical_spacing(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.size < 2:
        return 0.0
    differences = np.diff(coordinates)
    positive = differences[differences > 0.0]
    return float(np.median(positive)) if positive.size else 0.0


def _nearest_coordinate_indices(source, target):
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    right = np.searchsorted(source, target, side="left")
    right = np.clip(right, 0, source.size - 1)
    left = np.clip(right - 1, 0, source.size - 1)
    choose_left = np.abs(target - source[left]) <= np.abs(source[right] - target)
    return np.where(choose_left, left, right)


def index_2d_ascii(filepath, iteration=None, stop_after_iteration=True, start_byte=None):
    """Return GridComponent byte ranges without storing the ASCII data lines.

    If iteration is provided, scanning stops once the file moves past that
    iteration. That is the important part for huge appended ASCII files.
    """
    filepath = StdPath(str(filepath))
    grids = {}
    current_it = None
    current_ref = None
    open_block = None
    saw_requested_it = False

    def close_block(stop_byte):
        nonlocal open_block
        if open_block is None:
            return
        last_point = _indexed_grid_point(open_block["last_data_line"])
        geometry = _block_geometry(open_block["first_point"], last_point)
        grid = GridComponent(
            it=open_block["it"],
            ref_level=open_block["ref_level"],
            component=open_block["component"],
            bytestart=open_block["bytestart"],
            bytestop=stop_byte,
            line_count=open_block["line_count"],
            multigrid_level=open_block["multigrid_level"],
            map_id=open_block["map_id"],
            time_level=open_block["time_level"],
            x_min=geometry[0],
            x_max=geometry[1],
            y_min=geometry[2],
            y_max=geometry[3],
            z_min=geometry[4],
            z_max=geometry[5],
            dx=geometry[6],
            dy=geometry[7],
            dz=geometry[8],
        )
        # Keep distinct blocks that share a refinement level + component but
        # differ in multigrid/map/time level. Overwriting with an abbreviated
        # key drops valid data and breaks the merged slice.
        grids[(grid.it, grid.ref_level, grid.multigrid_level, grid.map_id, grid.time_level, grid.component)] = grid
        open_block = None

    with open(filepath, "rb") as f:
        if start_byte is not None:
            f.seek(int(start_byte))
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                close_block(pos)
                break

            it_match = ITERATION_RE.match(line)
            if it_match:
                close_block(pos)
                next_it = int(it_match.group(1))
                if iteration is not None and stop_after_iteration:
                    if saw_requested_it and next_it != iteration:
                        break
                    if not saw_requested_it and next_it > iteration:
                        break
                current_it = next_it
                current_ref = None
                if current_it == iteration:
                    saw_requested_it = True
                continue

            ref_match = REFLEVEL_RE.match(line)
            if ref_match:
                close_block(pos)
                # reflevel, multigrid level, map, component, timelevel
                current_ref = tuple(int(part) for part in ref_match.groups())
                continue

            if line.startswith(b"# column format:"):
                if current_it is None or current_ref is None:
                    continue
                ref_level, multigrid_level, map_id, component, time_level = current_ref
                open_block = {
                    "it": current_it,
                    "ref_level": ref_level,
                    "multigrid_level": multigrid_level,
                    "map_id": map_id,
                    "component": component,
                    "time_level": time_level,
                    "bytestart": f.tell(),
                    "line_count": 0,
                    "first_point": None,
                    "last_data_line": None,
                }
                continue

            if open_block is not None and _is_data_line(line):
                open_block["line_count"] += 1
                if open_block["first_point"] is None:
                    open_block["first_point"] = _indexed_grid_point(line)
                open_block["last_data_line"] = line

    if iteration is None:
        return grids
    return {key: grid for key, grid in grids.items() if grid.it == iteration}


def first_2d_iteration(filepath):
    """Return the first iteration header in a 2D ASCII file."""
    info = first_2d_iteration_info(filepath)
    return None if info is None else info[0]


def first_2d_iteration_info(filepath):
    """Return ``(iteration, byte_offset)`` for the first iteration header."""
    filepath = StdPath(str(filepath))
    with open(filepath, "rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                return None
            match = ITERATION_RE.match(line)
            if match:
                return int(match.group(1)), pos


def last_2d_iteration(filepath):
    """Return the last iteration header in a 2D ASCII file."""
    info = last_2d_iteration_info(filepath)
    return None if info is None else info[0]


def last_2d_iteration_info(filepath, chunk_size=1024 * 1024):
    """Return ``(iteration, byte_offset)`` for the last iteration header.

    This reads backward from EOF in chunks, so ``iteration=-1`` does not scan
    the whole appended ASCII file from the beginning.
    """
    filepath = StdPath(str(filepath))
    with open(filepath, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        end = file_size
        suffix = b""
        while end > 0:
            start = max(0, end - chunk_size)
            f.seek(start)
            chunk = f.read(end - start)
            data = chunk + suffix
            lines = data.splitlines(keepends=True)
            if start > 0 and lines:
                suffix = lines[0]
                search_lines = lines[1:]
                offset = start + len(suffix)
            else:
                suffix = b""
                search_lines = lines
                offset = start
            positions = []
            pos = offset
            for line in search_lines:
                positions.append(pos)
                pos += len(line)
            for line, line_pos in zip(reversed(search_lines), reversed(positions)):
                match = ITERATION_RE.match(line)
                if match:
                    return int(match.group(1)), line_pos
            end = start
    return None


def previous_2d_iteration_info(filepath, before_byte, chunk_size=1024 * 1024):
    """Return ``(iteration, byte_offset)`` for the header before ``before_byte``."""
    filepath = StdPath(str(filepath))
    with open(filepath, "rb") as f:
        f.seek(0, 2)
        end = min(int(before_byte), f.tell())
        suffix = b""
        while end > 0:
            start = max(0, end - chunk_size)
            f.seek(start)
            chunk = f.read(end - start)
            data = chunk + suffix
            lines = data.splitlines(keepends=True)
            if start > 0 and lines:
                suffix = lines[0]
                search_lines = lines[1:]
                offset = start + len(suffix)
            else:
                suffix = b""
                search_lines = lines
                offset = start
            positions = []
            pos = offset
            for line in search_lines:
                positions.append(pos)
                pos += len(line)
            for line, line_pos in zip(reversed(search_lines), reversed(positions)):
                match = ITERATION_RE.match(line)
                if match and line_pos < before_byte:
                    return int(match.group(1)), line_pos
            end = start
    return None


def first_header_for_2d_iteration_candidate(filepath, iteration, header_byte):
    """Return the earliest header byte for a candidate iteration.

    CarpetIOASCII files can repeat the same iteration header for multiple
    patches/reflevels. Starting at the last repeated header only sees a suffix
    of that iteration, so collapse each candidate iteration to its first header.
    """
    earliest = int(header_byte)
    previous = previous_2d_iteration_info(filepath, earliest)
    while previous is not None and previous[0] == iteration:
        earliest = previous[1]
        previous = previous_2d_iteration_info(filepath, earliest)
    return iteration, earliest


def _next_2d_iteration_info(f, start_byte, stop_byte=None):
    """Return the next ``(iteration, byte_offset)`` at or after ``start_byte``."""
    f.seek(max(0, int(start_byte)))
    if start_byte > 0:
        f.readline()
    while True:
        pos = f.tell()
        if stop_byte is not None and pos > stop_byte:
            return None
        line = f.readline()
        if not line:
            return None
        match = ITERATION_RE.match(line)
        if match:
            return int(match.group(1)), pos


def _scan_2d_iteration_range(filepath, target_iteration, start_byte, stop_byte=None):
    """Scan forward for ``target_iteration`` between byte offsets."""
    with open(filepath, "rb") as f:
        f.seek(int(start_byte))
        while True:
            pos = f.tell()
            if stop_byte is not None and pos > stop_byte:
                return None
            line = f.readline()
            if not line:
                return None
            match = ITERATION_RE.match(line)
            if not match:
                continue
            it = int(match.group(1))
            if it == target_iteration:
                return it, pos
            if it > target_iteration:
                return None


def _scan_nearest_2d_iteration_info(filepath, target_iteration, start_byte, stop_byte=None):
    """Scan a bounded byte range and return the closest iteration header."""
    best = None
    with open(filepath, "rb") as f:
        f.seek(int(start_byte))
        while True:
            pos = f.tell()
            if stop_byte is not None and pos > stop_byte:
                return best
            line = f.readline()
            if not line:
                return best
            match = ITERATION_RE.match(line)
            if not match:
                continue
            it = int(match.group(1))
            if best is None or abs(it - target_iteration) < abs(best[0] - target_iteration):
                best = (it, pos)
            if it >= target_iteration:
                return best


def find_2d_iteration_info(filepath, iteration, linear_threshold=64 * 1024 * 1024):
    """Return ``(iteration, byte_offset)`` for a requested iteration.

    ``iteration=-1`` returns the last header via a reverse EOF scan. Other
    iterations are found by binary search over byte offsets, followed by a
    bounded forward scan. Returns ``None`` if the iteration is outside the file
    range or not present.
    """
    filepath = StdPath(str(filepath))
    if iteration is None:
        return None
    iteration = int(iteration)
    if iteration == -1:
        return last_2d_iteration_info(filepath)

    first = first_2d_iteration_info(filepath)
    if first is None:
        return None
    first_it, first_pos = first
    if iteration == first_it:
        return first

    last = last_2d_iteration_info(filepath)
    if last is None:
        return None
    last_it, last_pos = last
    if iteration == last_it:
        return last
    if iteration < first_it or iteration > last_it:
        return None

    lo_it, lo_pos = first_it, first_pos
    hi_it, hi_pos = last_it, last_pos
    with open(filepath, "rb") as f:
        while hi_pos - lo_pos > linear_threshold:
            mid = (lo_pos + hi_pos) // 2
            probe = _next_2d_iteration_info(f, mid, stop_byte=hi_pos)
            if probe is None:
                hi_pos = mid
                continue
            probe_it, probe_pos = probe
            if probe_it == iteration:
                return probe
            if probe_it < iteration:
                if probe_pos <= lo_pos:
                    break
                lo_it, lo_pos = probe_it, probe_pos
            else:
                if probe_pos >= hi_pos:
                    break
                hi_it, hi_pos = probe_it, probe_pos

    return _scan_2d_iteration_range(filepath, iteration, lo_pos, stop_byte=hi_pos)


def nearest_2d_iteration_info(filepath, iteration, linear_threshold=64 * 1024 * 1024):
    """Return ``(iteration, byte_offset)`` for the header nearest ``iteration``."""
    filepath = StdPath(str(filepath))
    if iteration is None:
        return None
    iteration = int(iteration)
    exact = find_2d_iteration_info(filepath, iteration, linear_threshold=linear_threshold)
    if exact is not None:
        return exact

    first = first_2d_iteration_info(filepath)
    if first is None:
        return None
    first_it, first_pos = first
    if iteration <= first_it:
        return first

    last = last_2d_iteration_info(filepath)
    if last is None:
        return None
    last_it, last_pos = last
    if iteration >= last_it:
        return last

    lo_it, lo_pos = first_it, first_pos
    hi_it, hi_pos = last_it, last_pos
    with open(filepath, "rb") as f:
        while hi_pos - lo_pos > linear_threshold:
            mid = (lo_pos + hi_pos) // 2
            probe = _next_2d_iteration_info(f, mid, stop_byte=hi_pos)
            if probe is None:
                hi_pos = mid
                continue
            probe_it, probe_pos = probe
            if probe_it < iteration:
                if probe_pos <= lo_pos:
                    break
                lo_it, lo_pos = probe_it, probe_pos
            else:
                if probe_pos >= hi_pos:
                    break
                hi_it, hi_pos = probe_it, probe_pos

    return _scan_nearest_2d_iteration_info(filepath, iteration, lo_pos, stop_byte=hi_pos)


def nearest_2d_iteration(filepath, iteration, linear_threshold=64 * 1024 * 1024):
    """Return the available 2D iteration nearest the requested value."""
    info = nearest_2d_iteration_info(filepath, iteration, linear_threshold=linear_threshold)
    return None if info is None else info[0]


def _time_from_blocks(filepath, blocks):
    """Return the CarpetIOASCII time from the first numeric row in ``blocks``."""
    with open(filepath, "rb") as f:
        for block in sorted(blocks, key=lambda g: (g.ref_level, g.multigrid_level, g.map_id, g.time_level, g.component)):
            if not block.has_data:
                continue
            f.seek(block.bytestart)
            while f.tell() < block.bytestop:
                line = f.readline()
                if not line:
                    break
                if not _is_data_line(line):
                    continue
                parts = line.split()
                if len(parts) >= 9:
                    return float(parts[8])
    return None


def _iteration_time_info_from_candidate(filepath, iteration, header_byte, ref_level):
    """Return ``(iteration, time, first_header_byte)`` for a plottable candidate."""
    iteration, start_byte = first_header_for_2d_iteration_candidate(filepath, iteration, header_byte)
    grids = index_2d_ascii(filepath, iteration=iteration, start_byte=start_byte)
    blocks = [grid for grid in grids.values() if grid.has_data]
    if not blocks:
        return None
    try:
        _, selected_ref_levels = _resolve_reflevels(blocks, ref_level)
    except ValueError:
        return None
    if selected_ref_levels is not None:
        blocks = [grid for grid in blocks if grid.ref_level in selected_ref_levels]
    if not blocks:
        return None
    slice_time = _time_from_blocks(filepath, blocks)
    if slice_time is None:
        return None
    return iteration, slice_time, start_byte


def first_2d_iteration_time_info(filepath, ref_level="finest"):
    """Return ``(iteration, time, byte_offset)`` for the first plottable slice."""
    filepath = StdPath(str(filepath))
    info = first_2d_iteration_info(filepath)
    seen_iterations = set()
    with open(filepath, "rb") as f:
        while info is not None:
            iteration, header_byte = info
            if iteration not in seen_iterations:
                seen_iterations.add(iteration)
                result = _iteration_time_info_from_candidate(filepath, iteration, header_byte, ref_level)
                if result is not None:
                    return result
            info = _next_2d_iteration_info(f, header_byte + 1)
    return None


def last_2d_iteration_time_info(filepath, ref_level="finest"):
    """Return ``(iteration, time, byte_offset)`` for the last plottable slice."""
    filepath = StdPath(str(filepath))
    info = last_2d_iteration_info(filepath)
    seen_iterations = set()
    while info is not None:
        iteration, header_byte = info
        if iteration not in seen_iterations:
            seen_iterations.add(iteration)
            result = _iteration_time_info_from_candidate(filepath, iteration, header_byte, ref_level)
            if result is not None:
                return result
        info = previous_2d_iteration_info(filepath, header_byte)
    return None


def _previous_distinct_2d_iteration_info(filepath, before_byte, current_iteration):
    info = previous_2d_iteration_info(filepath, before_byte)
    while info is not None and info[0] == current_iteration:
        info = previous_2d_iteration_info(filepath, info[1])
    return info


def _next_distinct_2d_iteration_info(filepath, after_byte, current_iteration):
    with open(filepath, "rb") as f:
        info = _next_2d_iteration_info(f, after_byte)
        while info is not None and info[0] == current_iteration:
            info = _next_2d_iteration_info(f, info[1] + 1)
        return info


def _advance_distinct_2d_iteration_info(filepath, info, direction):
    """Step to the previous or next distinct iteration header from ``info``."""
    iteration, header_byte = int(info[0]), int(info[-1])
    if direction < 0:
        return _previous_distinct_2d_iteration_info(filepath, header_byte, iteration)
    return _next_distinct_2d_iteration_info(filepath, header_byte + 1, iteration)


def _nearest_valid_time_side(filepath, seed_info, target_time, ref_level, direction):
    """Find a valid plottable slice on one side of ``target_time``."""
    info = seed_info
    seen_iterations = set()
    while info is not None:
        iteration, header_byte = int(info[0]), int(info[-1])
        if iteration in seen_iterations:
            info = _advance_distinct_2d_iteration_info(filepath, info, direction)
            continue
        seen_iterations.add(iteration)

        result = _iteration_time_info_from_candidate(filepath, iteration, header_byte, ref_level)
        if result is not None:
            slice_time = result[1]
            if direction < 0 and slice_time <= target_time:
                return result
            if direction > 0 and slice_time >= target_time:
                return result
        info = _advance_distinct_2d_iteration_info(filepath, info, direction)
    return None


def nearest_2d_iteration_time_info(filepath, target_time, ref_level="finest", first=None, last=None):
    """Return ``(iteration, time, byte_offset)`` for the slice nearest ``target_time``."""
    filepath = StdPath(str(filepath))
    if first is None:
        first = first_2d_iteration_time_info(filepath, ref_level=ref_level)
    if first is None:
        return None
    if last is None:
        last = last_2d_iteration_time_info(filepath, ref_level=ref_level)
    if last is None:
        return None

    first_it, first_time, _ = first
    last_it, last_time, _ = last
    target_time = float(target_time)
    if target_time <= first_time:
        return first
    if target_time >= last_time:
        return last
    if last_time == first_time or last_it == first_it:
        return min((first, last), key=lambda item: abs(item[1] - target_time))

    frac = (target_time - first_time) / (last_time - first_time)
    target_iteration = int(round(first_it + frac * (last_it - first_it)))
    seed = nearest_2d_iteration_info(filepath, target_iteration)
    if seed is None:
        seed = (first_it, first[-1])

    candidates = [
        _nearest_valid_time_side(filepath, seed, target_time, ref_level, direction=-1),
        _nearest_valid_time_side(filepath, seed, target_time, ref_level, direction=1),
    ]
    candidates = [candidate for candidate in candidates if candidate is not None]
    if not candidates:
        candidates = [first, last]
    return min(candidates, key=lambda item: abs(item[1] - target_time))


def nearest_2d_iteration_to_time(filepath, target_time, ref_level="finest"):
    """Return the available 2D iteration whose data time is nearest ``target_time``."""
    info = nearest_2d_iteration_time_info(filepath, target_time, ref_level=ref_level)
    return None if info is None else info[0]


def _resolve_reflevels_from_available(available, ref_level):
    available = sorted({int(level) for level in available})
    if not available:
        raise ValueError("No refinement levels are available")

    def resolve_one(level):
        if level is None or level == "finest":
            return available[-1]
        level = int(level)
        if level < 0:
            try:
                return available[level]
            except IndexError:
                raise ValueError(f"Requested ref_level {level}, but available levels are {available}")
        return level

    if ref_level == "all":
        return "all", None

    if isinstance(ref_level, (list, tuple, set, np.ndarray)):
        selected = sorted({resolve_one(level) for level in ref_level})
        missing = [level for level in selected if level not in available]
        if missing:
            raise ValueError(f"Requested reflevels {missing}, but available levels are {available}")
        label = selected[0] if len(selected) == 1 else selected
        return label, set(selected)

    selected = resolve_one(ref_level)
    if selected not in available:
        raise ValueError(f"Requested ref_level {selected}, but available levels are {available}")
    return selected, {selected}


def _valid_2d_scan_entry(iteration, entry, ref_level):
    if entry["time"] is None:
        return None
    try:
        _resolve_reflevels_from_available(entry["refs_with_data"], ref_level)
    except ValueError:
        return None
    return int(iteration), float(entry["time"]), int(entry["first_header_byte"])


def _scan_valid_2d_iteration_time_infos_linewise(
    filepath,
    ref_level="finest",
    start_byte=0,
    stop_byte=None,
):
    """Scan one source range for valid iteration metadata."""
    filepath = StdPath(str(filepath))
    current_it = None
    current_ref_level = None
    current_entry = None
    open_block = None

    def entry_for(iteration, header_byte):
        nonlocal current_entry
        if current_entry is None:
            current_entry = {
                "first_header_byte": int(header_byte),
                "time": None,
                "refs_with_data": set(),
            }
        else:
            current_entry["first_header_byte"] = min(current_entry["first_header_byte"], int(header_byte))
        return current_entry

    def close_block():
        nonlocal open_block
        if open_block is None:
            return
        if open_block["has_data"]:
            entry = entry_for(open_block["iteration"], open_block["header_byte"])
            entry["refs_with_data"].add(open_block["ref_level"])
            if entry["time"] is None:
                entry["time"] = open_block["time"]
        open_block = None

    with open(filepath, "rb") as f:
        f.seek(max(0, int(start_byte)))
        while True:
            pos = f.tell()
            if stop_byte is not None and pos >= int(stop_byte):
                close_block()
                break
            line = f.readline()
            if not line:
                close_block()
                break

            it_match = ITERATION_RE.match(line)
            if it_match:
                close_block()
                next_it = int(it_match.group(1))
                if current_it is not None and next_it != current_it:
                    result = _valid_2d_scan_entry(current_it, current_entry, ref_level)
                    if result is not None:
                        yield result
                    current_entry = None
                current_it = next_it
                current_ref_level = None
                entry_for(current_it, pos)
                continue

            ref_match = REFLEVEL_RE.match(line)
            if ref_match:
                close_block()
                current_ref_level = int(ref_match.group(1))
                continue

            if line.startswith(b"# column format:"):
                if current_it is None or current_ref_level is None:
                    continue
                open_block = {
                    "iteration": current_it,
                    "header_byte": current_entry["first_header_byte"],
                    "ref_level": current_ref_level,
                    "has_data": False,
                    "time": None,
                }
                continue

            if open_block is not None and _is_data_line(line):
                open_block["has_data"] = True
                if open_block["time"] is None:
                    parts = line.split()
                    if len(parts) >= 9:
                        open_block["time"] = float(parts[8])

    if current_it is not None:
        result = _valid_2d_scan_entry(current_it, current_entry, ref_level)
        if result is not None:
            yield result


def _data_time_after_column_header(source, header_byte, stop_byte):
    source.seek(int(header_byte))
    source.readline()
    while source.tell() < int(stop_byte):
        line = source.readline()
        if not line:
            return None
        if ITERATION_RE.match(line) or REFLEVEL_RE.match(line):
            return None
        if _is_data_line(line):
            parts = line.split()
            return float(parts[8]) if len(parts) >= 9 else None
    return None


def _scan_valid_2d_iteration_time_infos_external(
    filepath,
    ref_level="finest",
    start_byte=0,
    stop_byte=None,
):
    """Use a native text scanner to locate sparse Carpet headers."""
    executable = shutil.which("rg") or shutil.which("grep")
    if executable is None:
        raise FileNotFoundError("neither ripgrep nor grep is available")

    filepath = StdPath(str(filepath))
    source_size = filepath.stat().st_size
    start = max(0, int(start_byte))
    stop = source_size if stop_byte is None else min(source_size, int(stop_byte))
    if stop <= start:
        return

    if StdPath(executable).name == "rg":
        command = [
            executable,
            "--text",
            "--byte-offset",
            "--no-heading",
            "--no-line-number",
            "--color=never",
            r"^# (iteration|refinement level|column format:)",
        ]
    else:
        command = [
            executable,
            "-a",
            "-b",
            "-E",
            r"^# (iteration|refinement level|column format:)",
        ]
    input_file = None
    if start:
        input_file = open(filepath, "rb")
        input_file.seek(start)
        command.append("-")
    else:
        command.append(str(filepath))

    entries = []
    current_it = None
    current_ref_level = None
    current_entry = None
    process = None
    try:
        with open(filepath, "rb") as source:
            process = subprocess.Popen(
                command,
                stdin=input_file,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={**os.environ, "LC_ALL": "C"},
            )
            for output_line in process.stdout:
                offset_text, separator, line = output_line.partition(b":")
                if not separator:
                    continue
                position = start + int(offset_text)
                if position >= stop:
                    process.terminate()
                    break
                line = line.rstrip(b"\r\n")

                it_match = ITERATION_RE.match(line)
                if it_match:
                    next_it = int(it_match.group(1))
                    if current_it is not None and next_it != current_it:
                        result = _valid_2d_scan_entry(
                            current_it,
                            current_entry,
                            ref_level,
                        )
                        if result is not None:
                            entries.append(result)
                        current_entry = None
                    current_it = next_it
                    current_ref_level = None
                    if current_entry is None:
                        current_entry = {
                            "first_header_byte": position,
                            "time": None,
                            "refs_with_data": set(),
                        }
                    else:
                        current_entry["first_header_byte"] = min(
                            current_entry["first_header_byte"],
                            position,
                        )
                    continue

                ref_match = REFLEVEL_RE.match(line)
                if ref_match:
                    current_ref_level = int(ref_match.group(1))
                    continue

                if (
                    line.startswith(b"# column format:")
                    and current_entry is not None
                    and current_ref_level is not None
                    and current_ref_level not in current_entry["refs_with_data"]
                ):
                    block_time = _data_time_after_column_header(
                        source,
                        position,
                        stop,
                    )
                    if block_time is not None:
                        current_entry["refs_with_data"].add(current_ref_level)
                        if current_entry["time"] is None:
                            current_entry["time"] = block_time

            if current_it is not None:
                result = _valid_2d_scan_entry(
                    current_it,
                    current_entry,
                    ref_level,
                )
                if result is not None:
                    entries.append(result)

            _, stderr = process.communicate()
            if process.returncode not in (0, 1, -15):
                raise OSError(
                    f"header scan failed with code {process.returncode}: "
                    f"{stderr.decode(errors='replace').strip()}"
                )
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            process.wait()
        if input_file is not None:
            input_file.close()

    yield from entries


def _scan_valid_2d_iteration_time_infos(
    filepath,
    ref_level="finest",
    start_byte=0,
    stop_byte=None,
):
    """Scan valid iteration metadata without parsing numeric array rows."""
    try:
        yield from _scan_valid_2d_iteration_time_infos_external(
            filepath,
            ref_level=ref_level,
            start_byte=start_byte,
            stop_byte=stop_byte,
        )
    except (OSError, ValueError, BufferError):
        yield from _scan_valid_2d_iteration_time_infos_linewise(
            filepath,
            ref_level=ref_level,
            start_byte=start_byte,
            stop_byte=stop_byte,
        )


def clean_2d_restart_timeline(infos):
    """Remove checkpoint-overlap branches while preserving file-order precedence.

    Appended CarpetIOASCII output can rewind slightly after a checkpoint
    restart. When that happens, the later branch is authoritative from its
    first time onward, even if neither times nor iterations exactly match the
    old branch.
    """
    timeline = []
    for info in infos:
        iteration, time_value, start_byte = info
        iteration = int(iteration)
        time_value = float(time_value)
        current = (iteration, time_value, start_byte)
        if not np.isfinite(time_value):
            continue

        if timeline:
            previous_time = timeline[-1][1]
            tolerance = 1.0e-12 * max(1.0, abs(previous_time), abs(time_value))
            if time_value <= previous_time + tolerance:
                # Keep the later appended branch through a non-exact overlap.
                timeline = [old for old in timeline if old[1] < time_value - tolerance]

        # A repeated iteration can carry a slightly shifted restart time. The
        # later occurrence is the one that should be rendered.
        timeline = [old for old in timeline if old[0] != iteration]
        timeline.append(current)

    return timeline


def _existing_2d_paths(paths):
    return [StdPath(path) for path in paths if StdPath(path).exists()]


def _located_2d_info(filepath, info):
    if info is None:
        return None
    return int(info[0]), float(info[1]), TwoDFilePosition(StdPath(filepath), int(info[2]))


def _ref_level_cache_key(ref_level):
    if isinstance(ref_level, (list, tuple, set, np.ndarray)):
        return "levels", tuple(sorted({int(level) for level in ref_level}))
    if ref_level is None:
        return "selector", None
    return "selector", str(ref_level)


def _ref_level_from_cache_key(cache_key):
    kind, value = cache_key
    return list(value) if kind == "levels" else value


def _json_ref_level_key(ref_level_key):
    kind, value = ref_level_key
    if isinstance(value, tuple):
        value = list(value)
    return [kind, value]


def _2d_index_cache_path(filepath, ref_level):
    filepath = StdPath(str(filepath)).resolve()
    key = {
        "source": str(filepath),
        "ref_level": _json_ref_level_key(_ref_level_cache_key(ref_level)),
    }
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", filepath.name)[:80]
    return TWO_D_INDEX_CACHE_ROOT / f"{stem}.{digest}.json"


def _sampled_prefix_digest(filepath, prefix_size):
    """Fingerprint sparse pieces of an already indexed source prefix."""
    prefix_size = max(0, int(prefix_size))
    digest = hashlib.sha256()
    digest.update(str(prefix_size).encode("ascii"))
    if prefix_size == 0:
        return digest.hexdigest()

    sample_size = min(TWO_D_INDEX_SAMPLE_BYTES, prefix_size)
    max_start = max(0, prefix_size - sample_size)
    starts = np.linspace(
        0,
        max_start,
        num=min(TWO_D_INDEX_SAMPLE_COUNT, max_start + 1),
        dtype=np.int64,
    )
    with open(filepath, "rb") as source:
        for start in np.unique(starts):
            source.seek(int(start))
            payload = source.read(sample_size)
            digest.update(int(start).to_bytes(8, "little", signed=False))
            digest.update(payload)
    return digest.hexdigest()


def _load_2d_index_record(cache_path):
    try:
        with open(cache_path, "r", encoding="utf-8") as cache_file:
            record = json.load(cache_file)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return None
    if record.get("version") != TWO_D_INDEX_CACHE_VERSION:
        return None
    return record


def _write_2d_index_record(cache_path, record):
    cache_path.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
    try:
        cache_path.parent.chmod(0o2775)
    except OSError:
        pass
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=cache_path.parent,
            prefix=f".{cache_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as cache_file:
            temporary_path = StdPath(cache_file.name)
            json.dump(record, cache_file, separators=(",", ":"))
            cache_file.flush()
            os.fsync(cache_file.fileno())
            os.fchmod(cache_file.fileno(), 0o664)
        os.replace(temporary_path, cache_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _raw_infos_from_record(record):
    infos = record.get("entries", [])
    if not isinstance(infos, list):
        return None
    try:
        return [
            (int(iteration), float(time_value), int(start_byte))
            for iteration, time_value, start_byte in infos
        ]
    except (TypeError, ValueError, OverflowError):
        return None


def _build_or_extend_2d_index(filepath, ref_level, cache_path):
    filepath = StdPath(str(filepath)).resolve()
    stat = filepath.stat()
    source_size = int(stat.st_size)
    source_mtime_ns = int(stat.st_mtime_ns)
    record = _load_2d_index_record(cache_path)
    raw_infos = []
    scan_start = 0

    expected_ref_level = _json_ref_level_key(_ref_level_cache_key(ref_level))
    if (
        record is not None
        and record.get("source") == str(filepath)
        and record.get("ref_level") == expected_ref_level
    ):
        cached_size = int(record.get("source_size", -1))
        cached_mtime_ns = int(record.get("source_mtime_ns", -1))
        cached_infos = _raw_infos_from_record(record)
        if (
            cached_infos is not None
            and source_size == cached_size
            and source_mtime_ns == cached_mtime_ns
        ):
            return cached_infos
        if cached_infos is not None and source_size > cached_size >= 0:
            expected = record.get("prefix_digest")
            actual = _sampled_prefix_digest(filepath, cached_size)
            if expected == actual:
                raw_infos = cached_infos
                if raw_infos:
                    # Re-read the last iteration because it may have been
                    # incomplete when the previous index was written.
                    scan_start = int(raw_infos[-1][2])
                    raw_infos = [info for info in raw_infos if info[2] < scan_start]

    action = "updating" if scan_start else "building"
    print(
        f"{action} 2D index for {filepath} "
        f"({source_size / (1024 ** 3):.2f} GiB)",
        flush=True,
    )
    suffix_infos = list(
        _scan_valid_2d_iteration_time_infos(
            filepath,
            ref_level=ref_level,
            start_byte=scan_start,
            stop_byte=source_size,
        )
    )
    raw_infos.extend(suffix_infos)
    updated = {
        "version": TWO_D_INDEX_CACHE_VERSION,
        "source": str(filepath),
        "source_size": source_size,
        "source_mtime_ns": source_mtime_ns,
        "source_device": int(stat.st_dev),
        "source_inode": int(stat.st_ino),
        "ref_level": expected_ref_level,
        "prefix_digest": _sampled_prefix_digest(filepath, source_size),
        "entries": raw_infos,
    }
    _write_2d_index_record(cache_path, updated)
    return raw_infos


@lru_cache(maxsize=128)
def _cached_valid_2d_iteration_time_infos(path_token, ref_level_key, cache_root):
    del cache_root  # Included in the key when tests or users redirect the cache.
    filepath = StdPath(path_token[0])
    ref_level = _ref_level_from_cache_key(ref_level_key)
    cache_path = _2d_index_cache_path(filepath, ref_level)
    try:
        cache_path.parent.mkdir(mode=0o2775, parents=True, exist_ok=True)
        try:
            cache_path.parent.chmod(0o2775)
        except OSError:
            pass
        lock_path = cache_path.with_suffix(cache_path.suffix + ".lock")
        with open(lock_path, "a+b") as lock_file:
            try:
                os.fchmod(lock_file.fileno(), 0o664)
            except OSError:
                pass
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            raw_infos = _build_or_extend_2d_index(
                filepath,
                ref_level,
                cache_path,
            )
    except OSError as error:
        print(
            f"warning: could not use 2D index {cache_path}: {error}; "
            "scanning source directly"
        )
        raw_infos = list(
            _scan_valid_2d_iteration_time_infos(filepath, ref_level=ref_level)
        )
    return tuple(clean_2d_restart_timeline(raw_infos))


def valid_2d_iteration_time_infos(filepath, ref_level="finest"):
    """Return the cached valid iteration timeline for one 2D ASCII source."""
    infos = _cached_valid_2d_iteration_time_infos(
        _2d_file_cache_token(filepath),
        _ref_level_cache_key(ref_level),
        str(TWO_D_INDEX_CACHE_ROOT),
    )
    return list(infos)


def iter_valid_2d_iteration_time_infos(filepath, ref_level="finest"):
    """Yield the cached valid iteration timeline for one source."""
    yield from valid_2d_iteration_time_infos(filepath, ref_level=ref_level)


def _2d_file_cache_token(filepath):
    filepath = StdPath(str(filepath))
    stat = filepath.stat()
    return (
        str(filepath),
        int(stat.st_dev),
        int(stat.st_ino),
        int(stat.st_size),
        int(stat.st_mtime_ns),
    )


@lru_cache(maxsize=128)
def _cached_2d_file_bounds(path_token, ref_level_key):
    filepath = StdPath(path_token[0])
    ref_level = _ref_level_from_cache_key(ref_level_key)
    first = first_2d_iteration_time_info(filepath, ref_level=ref_level)
    last = last_2d_iteration_time_info(filepath, ref_level=ref_level)
    return first, last


def _2d_file_bounds(filepath, ref_level):
    """Return cached valid bounds, invalidated when file metadata changes."""
    if _2d_index_cache_path(filepath, ref_level).exists():
        infos = valid_2d_iteration_time_infos(filepath, ref_level=ref_level)
        if infos:
            return infos[0], infos[-1]
    return _cached_2d_file_bounds(
        _2d_file_cache_token(filepath),
        _ref_level_cache_key(ref_level),
    )


def _authoritative_2d_bounds(paths, ref_level):
    bounds = []
    for filepath in _existing_2d_paths(paths):
        first, last = _2d_file_bounds(filepath, ref_level)
        if first is not None and last is not None:
            bounds.append((filepath, first, last))
    return bounds


def first_composite_2d_iteration_time_info(
    authoritative_paths,
    supplemental_paths=(),
    ref_level="finest",
):
    """Return the first valid slice across restart and fill-only 2D sources."""
    candidates = []
    bounds = _authoritative_2d_bounds(authoritative_paths, ref_level)
    for index, (filepath, first, _) in enumerate(bounds):
        next_start = bounds[index + 1][1][1] if index + 1 < len(bounds) else np.inf
        if first[1] < next_start:
            candidates.append((_located_2d_info(filepath, first), 0))
    for filepath in _existing_2d_paths(supplemental_paths):
        info, _ = _2d_file_bounds(filepath, ref_level)
        if info is not None:
            candidates.append((_located_2d_info(filepath, info), 1))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (item[0][1], item[1]))[0]


def last_composite_2d_iteration_time_info(
    authoritative_paths,
    supplemental_paths=(),
    ref_level="finest",
):
    """Return the last valid slice across restart and fill-only 2D sources."""
    candidates = []
    bounds = _authoritative_2d_bounds(authoritative_paths, ref_level)
    if bounds:
        filepath, _, last = bounds[-1]
        candidates.append((_located_2d_info(filepath, last), 0))
    for filepath in _existing_2d_paths(supplemental_paths):
        _, info = _2d_file_bounds(filepath, ref_level)
        if info is not None:
            candidates.append((_located_2d_info(filepath, info), 1))
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item[0][1], -item[1]))[0]


def nearest_composite_2d_iteration_time_info(
    authoritative_paths,
    target_time,
    supplemental_paths=(),
    ref_level="finest",
):
    """Find the nearest valid slice while respecting ordered restart cutovers."""
    target_time = float(target_time)
    candidates = []
    bounds = _authoritative_2d_bounds(authoritative_paths, ref_level)
    for index, (filepath, first, last) in enumerate(bounds):
        start = float(first[1])
        stop = float(bounds[index + 1][1][1]) if index + 1 < len(bounds) else np.inf
        if start >= stop:
            continue
        query = max(target_time, start)
        if np.isfinite(stop):
            query = min(query, np.nextafter(stop, -np.inf))
        if _2d_index_cache_path(filepath, ref_level).exists():
            indexed = valid_2d_iteration_time_infos(filepath, ref_level=ref_level)
            info = min(indexed, key=lambda item: abs(item[1] - query)) if indexed else None
        else:
            info = nearest_2d_iteration_time_info(
                filepath,
                query,
                ref_level=ref_level,
                first=first,
                last=last,
            )
        if info is not None and start <= info[1] < stop:
            # Later authoritative sources win an exact-distance tie.
            candidates.append((_located_2d_info(filepath, info), -index))
    for index, filepath in enumerate(_existing_2d_paths(supplemental_paths)):
        first, last = _2d_file_bounds(filepath, ref_level)
        if _2d_index_cache_path(filepath, ref_level).exists():
            indexed = valid_2d_iteration_time_infos(filepath, ref_level=ref_level)
            info = min(indexed, key=lambda item: abs(item[1] - target_time)) if indexed else None
        else:
            info = nearest_2d_iteration_time_info(
                filepath,
                target_time,
                ref_level=ref_level,
                first=first,
                last=last,
            )
        if info is not None:
            candidates.append((_located_2d_info(filepath, info), len(bounds) + index + 1))
    if not candidates:
        return None
    return min(candidates, key=lambda item: (abs(item[0][1] - target_time), item[1]))[0]


def valid_composite_2d_iteration_time_infos(
    authoritative_paths,
    supplemental_paths=(),
    ref_level="finest",
):
    """Scan and merge a virtual 2D timeline; supplemental data only fills gaps."""
    authoritative = {}
    paths = _existing_2d_paths(authoritative_paths)
    starts = []
    source_infos = []
    for filepath in paths:
        infos = valid_2d_iteration_time_infos(filepath, ref_level=ref_level)
        if infos:
            starts.append(float(infos[0][1]))
            source_infos.append((filepath, infos))

    for index, (filepath, infos) in enumerate(source_infos):
        stop = starts[index + 1] if index + 1 < len(starts) else np.inf
        for info in infos:
            if info[1] < stop:
                authoritative[int(info[0])] = _located_2d_info(filepath, info)

    merged = dict(authoritative)
    for filepath in _existing_2d_paths(supplemental_paths):
        for info in valid_2d_iteration_time_infos(filepath, ref_level=ref_level):
            merged.setdefault(int(info[0]), _located_2d_info(filepath, info))
    return sorted(merged.values(), key=lambda info: (info[1], info[0]))


def iter_valid_composite_2d_iteration_time_infos(
    authoritative_paths,
    supplemental_paths=(),
    ref_level="finest",
):
    yield from valid_composite_2d_iteration_time_infos(
        authoritative_paths,
        supplemental_paths=supplemental_paths,
        ref_level=ref_level,
    )


def print_grids(grids, max_refs=10, max_comps=10):
    """Print a compact summary of indexed CarpetIOASCII blocks."""
    grouped = defaultdict(lambda: defaultdict(list))
    for grid in grids.values():
        grouped[grid.it][grid.ref_level].append(grid)

    for it in sorted(grouped):
        refs = sorted(grouped[it])
        byte_start = min(g.bytestart for ref in refs for g in grouped[it][ref])
        byte_stop = max(g.bytestop for ref in refs for g in grouped[it][ref])
        print(f"\nIteration {it}, refs {len(refs)}, bytes {byte_start}-{byte_stop} ({byte_stop - byte_start})")
        for rl in refs[:max_refs]:
            comps = sorted(grouped[it][rl], key=lambda g: g.component)
            nonzero = [g for g in comps if g.has_data]
            print(f"  Ref level {rl}: {len(comps)} components, {len(nonzero)} with data")
            print("  Component  |          Bytes        | Span     | Lines | Has_Data")
            print("   ---------------------------------------------------------------------")
            for g in comps[:max_comps]:
                print(f"    comp {g.component:>3} | {g.bytestart:>10}-{g.bytestop:<10} "
                      f"| {g.byte_span:<8} | {g.line_count:<5} | {g.has_data}")
            if len(comps) > max_comps:
                print(f"    ... ({len(comps) - max_comps} more)")


def read_2d_file(
    filepath,
    variable,
    plane,
    iteration=0,
    ref_level="finest",
    components="all",
    start_byte=None,
    required_ref_levels=None,
    region=None,
    selection_grid_shape=None,
):
    """Read a 2D ASCII slice from an explicit ``variable.plane.asc`` file.

    ``iteration=-1`` selects the last iteration in the file that has data.

    ``ref_level`` can be:
    - ``"finest"`` or ``-1`` for the finest level.
    - ``-2`` for the next-finest level, etc.
    - an integer level such as ``7``.
    - a list/tuple such as ``[-1, -2]`` for the finest two levels.
    - ``"all"`` for every level.

    ``required_ref_levels`` controls iteration validity independently of the
    levels that are loaded. When ``region`` is provided, indexed component
    extents are used to read only the finest-first blocks needed to cover that
    physical window.
    """
    plane = _normalize_plane(plane)
    filepath = StdPath(str(filepath))
    requested_iteration = iteration
    start_byte = None if start_byte is None else int(start_byte)
    grids = None
    blocks = []
    if iteration == -1:
        iteration_info = last_2d_iteration_info(filepath)
        last_reflevel_error = None
        checked_iterations = 0
        seen_iterations = set()
        search_t0 = time.perf_counter()
        while iteration_info is not None:
            iter_t0 = time.perf_counter()
            iteration, start_byte = iteration_info
            if iteration in seen_iterations:
                iteration_info = previous_2d_iteration_info(filepath, start_byte)
                continue
            seen_iterations.add(iteration)
            iteration, start_byte = first_header_for_2d_iteration_candidate(filepath, iteration, start_byte)
            checked_iterations += 1
            grids = index_2d_ascii(filepath, iteration=iteration, start_byte=start_byte)
            blocks = [grid for grid in grids.values() if grid.has_data]
            available_reflevels = sorted({grid.ref_level for grid in blocks})
            should_print = checked_iterations <= 10 or checked_iterations % 25 == 0
            if should_print:
                print(
                    f"{filepath.name}: candidate {checked_iterations}, iteration {iteration}, "
                    f"reflevels={available_reflevels}, data_blocks={len(blocks)}, "
                    f"candidate_time={time.perf_counter() - iter_t0:.2f}s, "
                    f"elapsed={time.perf_counter() - search_t0:.2f}s"
                )
            if blocks:
                try:
                    _resolve_reflevels(blocks, required_ref_levels if required_ref_levels is not None else ref_level)
                    print(
                        f"{filepath.name}: selected iteration {iteration} after checking "
                        f"{checked_iterations} candidate iteration(s) in {time.perf_counter() - search_t0:.2f}s"
                    )
                    break
                except ValueError as err:
                    last_reflevel_error = err
                    if should_print:
                        print(f"{filepath.name}: skipping iteration {iteration}: {err}")
            iteration_info = previous_2d_iteration_info(filepath, start_byte)
        if not blocks or iteration_info is None:
            first_it = first_2d_iteration(filepath)
            last_it = last_2d_iteration(filepath)
            range_msg = f" available iterations run from {first_it} to {last_it}." if first_it is not None else ""
            detail = f" Last reflevel error: {last_reflevel_error}" if last_reflevel_error is not None else ""
            raise ValueError(f"No complete data found for any iteration in {filepath}.{range_msg}{detail}")
    else:
        if iteration is not None:
            if start_byte is None:
                iteration_info = find_2d_iteration_info(filepath, iteration)
                if iteration_info is None:
                    first_it = first_2d_iteration(filepath)
                    last_it = last_2d_iteration(filepath)
                    range_msg = f" available iterations run from {first_it} to {last_it}." if first_it is not None else ""
                    raise ValueError(f"Iteration {requested_iteration} not found in {filepath}.{range_msg}")
                iteration, start_byte = iteration_info
            else:
                iteration = int(iteration)
            iteration, start_byte = first_header_for_2d_iteration_candidate(filepath, iteration, start_byte)
        grids = index_2d_ascii(filepath, iteration=iteration, start_byte=start_byte)
        blocks = [grid for grid in grids.values() if grid.has_data]
    if not blocks:
        first_it = first_2d_iteration(filepath)
        last_it = last_2d_iteration(filepath)
        range_msg = f" available iterations run from {first_it} to {last_it}." if first_it is not None else ""
        raise ValueError(f"No data found for iteration {requested_iteration} (resolved to {iteration}) in {filepath}.{range_msg}")

    if required_ref_levels is not None:
        _resolve_reflevels(blocks, required_ref_levels)

    selected_ref_level, selected_ref_levels = _resolve_reflevels(blocks, ref_level)
    if selected_ref_levels is not None:
        blocks = [grid for grid in blocks if grid.ref_level in selected_ref_levels]

    if components != "all":
        component_set = {int(component) for component in components}
        blocks = [grid for grid in blocks if grid.component in component_set]

    if region is not None:
        blocks = _select_blocks_for_region(
            blocks,
            plane,
            region,
            selection_grid_shape,
        )

    if not blocks:
        raise ValueError(f"No blocks left after selecting ref_level={ref_level}, components={components}")

    raw = _read_2d_blocks(filepath, blocks)
    return Slice2D(
        filepath=filepath,
        variable=variable,
        plane=plane,
        iteration=iteration,
        ref_level=selected_ref_level,
        components=sorted({grid.component for grid in blocks}),
        x=raw[:, 4],
        y=raw[:, 5],
        z=raw[:, 6],
        data=raw[:, 7],
        time=raw[:, 3],
        component=raw[:, 2].astype(int),
        rl=raw[:, 1].astype(int),
    )


def _resolve_reflevels(blocks, ref_level):
    available = sorted({grid.ref_level for grid in blocks})
    return _resolve_reflevels_from_available(available, ref_level)


def _block_plane_extent(block, plane):
    names = {
        "xy": ("x_min", "x_max", "dx", "y_min", "y_max", "dy"),
        "xz": ("x_min", "x_max", "dx", "z_min", "z_max", "dz"),
        "yz": ("y_min", "y_max", "dy", "z_min", "z_max", "dz"),
    }[plane]
    amin, amax, da, bmin, bmax, db = (float(getattr(block, name)) for name in names)
    if not np.all(np.isfinite((amin, amax, bmin, bmax))):
        return None
    da = da if np.isfinite(da) else 0.0
    db = db if np.isfinite(db) else 0.0
    return amin - 0.5 * da, amax + 0.5 * da, bmin - 0.5 * db, bmax + 0.5 * db


def _select_blocks_for_region(blocks, plane, region, grid_shape=None):
    """Choose the minimal finest-first set of blocks covering a target region."""
    xlim, ylim = region
    nx, ny = (256, 256) if grid_shape is None else tuple(int(value) for value in grid_shape)
    if nx < 2 or ny < 2:
        raise ValueError("Selection grid dimensions must both be at least 2")
    a_target = np.linspace(float(xlim[0]), float(xlim[1]), nx)
    b_target = np.linspace(float(ylim[0]), float(ylim[1]), ny)
    uncovered = np.ones((ny, nx), dtype=bool)
    selected = []
    ordering = sorted(
        blocks,
        key=lambda block: (
            -block.ref_level,
            block.multigrid_level,
            block.map_id,
            block.time_level,
            block.component,
        ),
    )
    for block in ordering:
        extent = _block_plane_extent(block, plane)
        if extent is None:
            # Unknown geometry is safer to load than to silently omit.
            selected.append(block)
            continue
        amin, amax, bmin, bmax = extent
        a_indices = np.flatnonzero((a_target >= amin) & (a_target <= amax))
        b_indices = np.flatnonzero((b_target >= bmin) & (b_target <= bmax))
        if a_indices.size == 0 or b_indices.size == 0:
            continue
        a_slice = slice(int(a_indices[0]), int(a_indices[-1]) + 1)
        b_slice = slice(int(b_indices[0]), int(b_indices[-1]) + 1)
        coverage = uncovered[b_slice, a_slice]
        if not np.any(coverage):
            continue
        selected.append(block)
        coverage[:] = False
        if not np.any(uncovered):
            break
    return selected


def _read_2d_blocks(filepath, blocks):
    # File comment columns are 1-based:
    # 1:it 2:tl 3:rl 4:c 5:ml 6:ix 7:iy 8:iz 9:time 10:x 11:y 12:z 13:data
    # This keeps it, rl, component, time, x, y, z, data.
    usecols = (0, 2, 3, 8, 9, 10, 11, 12)
    chunks = []
    with open(filepath, "rb") as f:
        for block in sorted(blocks, key=lambda g: (g.ref_level, g.multigrid_level, g.map_id, g.time_level, g.component)):
            f.seek(block.bytestart)
            payload = f.read(block.bytestop - block.bytestart)
            if not payload.strip():
                continue
            chunk = np.loadtxt(BytesIO(payload), usecols=usecols, ndmin=2)
            if chunk.size:
                chunks.append(chunk)
    if not chunks:
        raise ValueError("Selected blocks had no numeric data")
    return np.vstack(chunks)


def plot_2d_slice(slice2d, ax=None, composite=False, **pcolormesh_kwargs):
    """Plot one Slice2D using its native grid or AMR composite grid."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    if composite:
        a, b, grid = slice2d.to_composite_grid()
    else:
        a, b, grid = slice2d.to_grid()
    mesh = ax.pcolormesh(a, b, np.ma.masked_invalid(grid), shading="auto", **pcolormesh_kwargs)
    ax.set_xlabel(slice2d.axis_names[0])
    ax.set_ylabel(slice2d.axis_names[1])
    ax.set_aspect("equal", adjustable="box")
    return mesh


def plot_2d_reflevels(slice2d, ax=None, draw_boxes=False, **pcolormesh_kwargs):
    """Plot all refinement levels coarse-to-fine with per-component grids."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    mesh = None
    level_extents = []
    sparse_mesh_kwargs = dict(pcolormesh_kwargs)
    sparse_mesh_kwargs.pop("shading", None)
    for rl, component, a, b, grid in slice2d.iter_level_component_grids():
        masked = np.ma.masked_invalid(grid)
        if masked.count() == 0:
            continue
        finite_mask = np.isfinite(masked)
        row_has_data = np.any(finite_mask, axis=1)
        col_has_data = np.any(finite_mask, axis=0)
        if not row_has_data.any() or not col_has_data.any():
            continue
        trimmed_a = a[col_has_data]
        trimmed_b = b[row_has_data]
        trimmed_grid = masked[np.ix_(row_has_data, col_has_data)]
        trimmed_finite = np.isfinite(trimmed_grid)
        coverage = float(trimmed_finite.sum()) / trimmed_finite.size if trimmed_finite.size else 0.0

        if coverage < 0.30 and trimmed_finite.sum() >= 6:
            y2d, x2d = np.meshgrid(trimmed_b, trimmed_a, indexing="ij")
            mesh = ax.tripcolor(
                x2d[trimmed_finite],
                y2d[trimmed_finite],
                np.asarray(trimmed_grid)[trimmed_finite],
                shading="gouraud",
                **sparse_mesh_kwargs,
            )
        else:
            mesh = ax.pcolormesh(trimmed_a, trimmed_b, trimmed_grid, shading="auto", **pcolormesh_kwargs)
        finite_a = trimmed_a if trimmed_a.size else np.array([])
        finite_b = trimmed_b if trimmed_b.size else np.array([])
        if finite_a.size:
            level_extents.append(
                (
                    float(np.min(finite_a)),
                    float(np.max(finite_a)),
                    float(np.min(finite_b)),
                    float(np.max(finite_b)),
                )
            )
    if draw_boxes:
        for amin, amax, bmin, bmax in level_extents:
            rect = plt.Rectangle(
                (amin, bmin),
                amax - amin,
                bmax - bmin,
                fill=False,
                edgecolor="k",
                linestyle=":",
                linewidth=0.8,
            )
            ax.add_patch(rect)
    ax.set_xlabel(slice2d.axis_names[0])
    ax.set_ylabel(slice2d.axis_names[1])
    ax.set_aspect("equal", adjustable="box")
    return mesh


def plot_2d_uniform(
    slice2d,
    xlim,
    ylim,
    shape,
    ax=None,
    draw_boxes=False,
    **pcolormesh_kwargs,
):
    """Plot a nearest-neighbor uniform grid assembled from selected AMR blocks."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    a, b, grid = slice2d.to_uniform_grid(xlim, ylim, shape)
    mesh = ax.pcolormesh(a, b, np.ma.masked_invalid(grid), shading="auto", **pcolormesh_kwargs)
    if draw_boxes:
        for _, _, component_a, component_b, _ in slice2d.iter_level_component_grids():
            da = _typical_spacing(component_a)
            db = _typical_spacing(component_b)
            amin = float(component_a[0] - 0.5 * da)
            amax = float(component_a[-1] + 0.5 * da)
            bmin = float(component_b[0] - 0.5 * db)
            bmax = float(component_b[-1] + 0.5 * db)
            ax.add_patch(
                plt.Rectangle(
                    (amin, bmin),
                    amax - amin,
                    bmax - bmin,
                    fill=False,
                    edgecolor="k",
                    linestyle=":",
                    linewidth=0.8,
                )
            )
    ax.set_xlabel(slice2d.axis_names[0])
    ax.set_ylabel(slice2d.axis_names[1])
    ax.set_aspect("equal", adjustable="box")
    return mesh

# Example, kept commented so running the notebook does not immediately read a huge file:
# rho0_file = Path(dirs[0]) / "beta100" / "rho_b.xy.asc"
# rho0 = read_2d_file(rho0_file, variable="rho_b", plane="xy", iteration=0)
# a, b, rho_grid = rho0.to_grid()
# plot_2d_slice(rho0)
# rho_finest_two = read_2d_file(rho0_file, variable="rho_b", plane="xy", iteration=0, ref_level=[-1, -2])
# rho_all = read_2d_file(rho0_file, variable="rho_b", plane="xy", iteration=0, ref_level="all")
# plot_2d_reflevels(rho_all)
