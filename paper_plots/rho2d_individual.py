"""General 2D slice plotting helpers.

The reader/writer plumbing here is generic for any 2D variable in an
`.asc` file. The current default remains ``rho_b`` for existing workflows.
"""
from pathlib import Path
import sys
import re
import warnings
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
import matplotlib.pyplot as plt
import numpy as np
from config import PLOTS_DIR
from helpers.plot_common import save_individual_fig, setup
from helpers.reader import load_sims
from helpers.reader_2d import (
    first_composite_2d_iteration_time_info,
    iter_valid_composite_2d_iteration_time_infos,
    last_composite_2d_iteration_time_info,
    nearest_composite_2d_iteration_time_info,
    plot_2d_reflevels,
    plot_2d_uniform,
    valid_composite_2d_iteration_time_infos,
)
from helpers.style import JET_WHITE_LOW_CMAP
from helpers.time_units import code_time_to_ms


DEFAULT_SIMS = ["A1"]

# Critical knobs.
RHO2D_VARIABLE = "rho_b"
RHO2D_VARIABLE_LATEX = r"\rho_0"
RHO2D_VARIABLE_NAME = "rho_b"
RHO2D_PLANE = "xy"
RHO2D_REQUIRED_REF_LEVELS = list(range(13))
RHO2D_NATIVE_REF_LEVELS = list(range(8, 13))
# Snapshot selectors:
#   integer = exact valid-frame index (0=first, -1=last)
#   float in [0, 1] = fraction of the valid time interval
#   matching BY_TBYPC entry 1 = interpret the value as a target t/P_c
SNAPSHOT_VALUES = [0, -1]
SNAPSHOT_BY_TBYPC = [0, 0]
RHO2D_USE_UNIFORM_RESAMPLE = True
RHO2D_RESAMPLE_GRID = (700, 700)  # (nx, ny)
# Display coordinates in raw code units, disk-rest-mass units, or ADM-mass units.
RHO2D_COORDINATE_NORMALIZATION = "adm_mass"  # "none", "disk_rest_mass", "adm_mass"
# Limits are expressed in the displayed coordinate convention above.
RHO2D_X_LIMITS = (-15.0, 15.0)
RHO2D_Y_LIMITS = (-15.0, 15.0)
SHOW_DATA_LIMITS_IN_TITLE = False
# Use one fixed per-simulation reference for snapshots, panels, and movies.
NORMALIZE_BY_FIXED_INITIAL_MAX = True
USE_RHOMAX_DIAGNOSTIC = RHO2D_VARIABLE == "rho_b"
LOGSCALE = False
INCLUDE_VARIABLE_IN_TITLE = False

# Presentation knobs.
DRAW_REFINEMENT_BOXES = False
DRAW_APPARENT_HORIZON = True
DRAW_AH_SOLID_MASK = True
DRAW_AH_WHITE_OUTLINE = False
AH_MASK_COLOR = "k"
AH_MASK_ALPHA = 1.0
AH_LINE_COLOR = "k"
AH_LINE_WIDTH = 1.0
AH_LINESTYLE = "-"
AH_OUTLINE_COLOR = "white"
AH_OUTLINE_WIDTH = 2.2
AH_OUTLINE_LINESTYLE = ":"
AH_SLICE_ABSOLUTE_TOLERANCE = 5.0e-4
AH_MAX_SLICE_RELATIVE_TOLERANCE = 0.2
AH_SURFACE_FILE_MAX_DELTA = 64
AH_MIN_POINTS = 8
AH_MAX_ANGULAR_GAP_DEG = 20.0
AH_FALLBACK_CIRCLE_POINTS = 256
AH_MIN_RESOLVED_RADIUS_PIXELS = 6.0
SHOW_HORIZON_DEBUG = True

FIGSIZE = (7.0, 5.6)
RHO_MIN = 0.0
RHO_MAX = 1.2
COLORBAR_SHRINK = 0.86
COLORBAR_PAD = 0.03
TITLE_PAD = 8

# Output naming: keep this at "rho2d" for current defaults.
# If you switch RHO2D_VARIABLE for another field, change OUTPUT_PREFIX too.
OUTPUT_PREFIX = "rho2d"

# Optional explicit override for labels (auto-generated from RHO2D_VARIABLE_LATEX by default).
RHO2D_LABEL = None
RHO2D_NORMALIZED_LABEL = None
RHO2D_LOG_NORMALIZED_LABEL = None
AH_SURFACE_RE = re.compile(r"^h\.t(?P<iter>[0-9]+)(?:\.[0-9]+)?\.ah1\.gp$")


def _inline_math(expr: str) -> str:
    expr = _sanitize_latex_text(expr).strip()
    if expr.startswith("$") and expr.endswith("$"):
        return expr
    return f"${expr}$"


def _sanitize_latex_text(expr: str) -> str:
    text = str(expr)
    # Handle accidental non-raw escape like "\\rho_0" written as "\rho_0"
    # in source, which becomes an embedded carriage return.
    text = text.replace("\rho", "\\rho")
    if text.startswith("\r") and text[1:3] == "ho":
        text = "\\rho" + text[3:]
    return text.replace("\x00", "0")


def _field_label_latex() -> str:
    return _sanitize_latex_text(RHO2D_VARIABLE_LATEX or RHO2D_VARIABLE_NAME)


def _field_initial_max(field: str) -> str:
    label = _sanitize_latex_text(field).strip()
    m = re.match(r"^(.*?)(?:_\{(.+?)\}|_([A-Za-z0-9]+))$", label)
    if m:
        base = m.group(1)
        sub = m.group(2) if m.group(2) is not None else m.group(3)
        return f"{base}^{{\\mathrm{{max}}}}_{{{sub},t=0}}"
    return f"{label}^{{\\mathrm{{max}}}}_{{t=0}}"


def rho2d_coordinate_divisor(sim) -> float:
    mode = str(RHO2D_COORDINATE_NORMALIZATION).strip().lower()
    if mode == "none":
        return 1.0
    if mode == "disk_rest_mass":
        divisor = float(getattr(sim.config, "disk_rest_mass", np.nan))
    elif mode == "adm_mass":
        divisor = float(getattr(sim.config, "gw_madm", np.nan))
    else:
        raise ValueError(
            "RHO2D_COORDINATE_NORMALIZATION must be 'none', "
            "'disk_rest_mass', or 'adm_mass'"
        )
    if not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(
            f"{sim.config.name}: invalid {mode} coordinate divisor {divisor!r}"
        )
    return divisor


def _source_coordinate_limits(limits, divisor):
    if limits is None:
        return None
    return tuple(float(value) * divisor for value in limits)


def apply_rho2d_coordinate_scaling(sim, divisor):
    if divisor == 1.0:
        return
    for coordinate in ("x", "y", "z"):
        values = np.asarray(getattr(sim.rho2d, coordinate), dtype=float)
        setattr(sim.rho2d, coordinate, values / divisor)


def rho2d_coordinate_label(axis_name):
    mode = str(RHO2D_COORDINATE_NORMALIZATION).strip().lower()
    if mode == "none":
        return rf"${axis_name}\ [M_\odot]$"
    if mode == "disk_rest_mass":
        return rf"${axis_name}\ [M_{{0,\mathrm{{disk}}}}]$"
    if mode == "adm_mass":
        return rf"${axis_name}\ [M]$"
    raise ValueError(f"Unknown 2D coordinate normalization {mode!r}")


def _cbar_label_fallback(label: str) -> str:
    plain = re.sub(r"\$|\{|\}", "", label)
    plain = plain.replace(r"\mathrm", "")
    plain = plain.replace("\\", "")
    return plain.strip()


def _set_colorbar_label(cbar, label: str):
    try:
        cbar.set_label(label)
    except ValueError:
        cbar.set_label(_cbar_label_fallback(label))


def _log_horizon_status(sim, message: str, enabled: bool) -> None:
    if not enabled:
        return
    sim_name = getattr(getattr(sim, "config", None), "name", "unknown")
    print(f"{sim_name}: horizon: {message}", flush=True)


def rho2d_title(sim, snapshot_label=None):
    plane = str(sim.rho2d.plane).upper()
    plane_label = rf"\mathrm{{{plane}}}"
    field_label = _inline_math(_field_label_latex())
    times = np.asarray(sim.rho2d.time, dtype=float)
    finite_times = times[np.isfinite(times)]
    if finite_times.size == 0:
        if INCLUDE_VARIABLE_IN_TITLE:
            return rf"$\mathrm{{{sim.config.name}}}:~{field_label}~{plane_label}$"
        return rf"$\mathrm{{{sim.config.name}}}:~{plane_label}$"
    t_code = float(np.nanmedian(finite_times))
    t_ms = float(code_time_to_ms(t_code))
    variable_text = f"~{field_label}" if INCLUDE_VARIABLE_IN_TITLE else ""
    if snapshot_label:
        return rf"$\mathrm{{{sim.config.name}}}~{plane_label}{variable_text}\;|\;t={t_code:.1f}\,M~({t_ms:.1f}\,\mathrm{{ms}})\;|\;{snapshot_label}$"
    return rf"$\mathrm{{{sim.config.name}}}~{plane_label}{variable_text}\;|\;t={t_code:.1f}\,M~({t_ms:.1f}\,\mathrm{{ms}})$"


def rho2d_colorbar_label():
    if RHO2D_LABEL is not None:
        return _inline_math(RHO2D_LABEL)
    if NORMALIZE_BY_FIXED_INITIAL_MAX:
        if RHO2D_LOG_NORMALIZED_LABEL is not None:
            return _inline_math(RHO2D_LOG_NORMALIZED_LABEL)
        if RHO2D_NORMALIZED_LABEL is not None:
            return _inline_math(RHO2D_NORMALIZED_LABEL)
        field = _field_label_latex()
        field_with_max = _field_initial_max(field)
        return (
            rf"$\log_{{10}}\left({field}/{field_with_max}\right)$"
            if LOGSCALE
            else rf"${field}/{field_with_max}$"
        )
    return _inline_math(_field_label_latex())


def rho2d_path_for_sim(sim):
    return sim.data_path_2d / f"{RHO2D_VARIABLE}.{RHO2D_PLANE}.asc"


def rho2d_paths_for_sim(sim):
    return sim.rho2d_source_paths(variable=RHO2D_VARIABLE, plane=RHO2D_PLANE)


def ah_surface_path_for_sim(sim, iteration):
    filename = f"h.t{int(iteration)}.ah1.gp"
    for search_dir in ah_surface_search_dirs(sim):
        candidate = search_dir / filename
        if candidate.exists():
            return candidate
    return sim.data_path_2d / filename


def ah_surface_search_dirs(sim):
    # Prefer the source of the loaded slice, then all configured continuation roots.
    roots = []
    loaded_path = getattr(sim, "rho2d_source_path", None)
    if loaded_path is not None:
        roots.append(Path(loaded_path).parent)
    roots.extend(Path(path) for path in sim.data_paths_2d)
    roots.extend(Path(path) for path in sim.data_paths_1d)
    dirs = []
    for root in roots:
        for candidate in (root, root / "Horizon"):
            if candidate not in dirs:
                dirs.append(candidate)
    return tuple(dirs)


def ah_diagnostics_path_for_sim(sim):
    candidates = tuple(path / "BH_diagnostics.ah1.gp" for path in ah_surface_search_dirs(sim))
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _ah_iter_from_path(path):
    name = path.name.lower()
    # Support the naming convention used by the run folders and tolerate a
    # trailing decimal variant, e.g. h.t00001.ah1.gp.
    if not name.endswith(".ah1.gp"):
        return None
    match = AH_SURFACE_RE.match(name)
    if not match:
        return None
    return int(match.group("iter"))


def ah_surface_path_for_iteration(sim, target_iteration, horizon_debug: bool = False):
    exact = ah_surface_path_for_sim(sim, target_iteration)
    if exact.exists():
        return exact, int(target_iteration)

    files_by_iteration = {}
    for search_dir in ah_surface_search_dirs(sim):
        for path in sorted(search_dir.glob("h.t*.ah1.gp")):
            iteration = _ah_iter_from_path(path)
            if iteration is not None:
                files_by_iteration.setdefault(iteration, path)
    files = [files_by_iteration[iteration] for iteration in sorted(files_by_iteration)]
    if not files:
        if horizon_debug:
            dirs = ", ".join(str(d) for d in ah_surface_search_dirs(sim))
            _log_horizon_status(sim, f"no AH surface files matching h.t*.ah1.gp in {dirs}", horizon_debug)
        return None, None
    iters = np.array([_ah_iter_from_path(p) for p in files], dtype=float)
    target = float(target_iteration)
    deltas = np.abs(iters - target)
    idx = int(np.argmin(deltas))
    if not np.isfinite(deltas[idx]):
        return None, None
    if AH_SURFACE_FILE_MAX_DELTA is not None and deltas[idx] > AH_SURFACE_FILE_MAX_DELTA:
        if horizon_debug:
            _log_horizon_status(
                sim,
                f"nearest AH surface too far: |iter-target|={deltas[idx]:.0f} > "
                f"{AH_SURFACE_FILE_MAX_DELTA} (target={target:.0f})",
                horizon_debug,
            )
        return None, None
    return files[idx], int(iters[idx])


def ah_surface_path_for_time(sim, time_code):
    if not np.isfinite(time_code):
        return None, None
    data = _load_ah_diagnostics(sim)
    if data is None:
        return None, None
    data = _prepare_ah_data(data)
    if data is None:
        return None, None
    times = data[:, 1].astype(float)
    iterations = data[:, 0].astype(float)
    if times.size == 0:
        return None, None
    if times.size == 1:
        return ah_surface_path_for_iteration(sim, iterations[0])
    time_query = float(np.interp(float(time_code), times, iterations, left=iterations[0], right=iterations[-1]))
    return ah_surface_path_for_iteration(sim, time_query, horizon_debug=False)


def _load_ah_diagnostics(sim):
    cached = getattr(sim, "_rho2d_ah_diagnostics", None)
    if cached is not None:
        return cached
    try:
        _, data, _, _ = sim.loaddata("beta100/BH_diagnostics.ah1.gp", tcol=1)
    except (OSError, ValueError):
        return None
    if data.size == 0 or data.shape[1] < 8:
        return None
    # iteration, time, centroid x/y/z, min/max/mean coordinate radius
    sim._rho2d_ah_diagnostics = np.asarray(data[:, :8], dtype=float)
    return sim._rho2d_ah_diagnostics


def _prepare_ah_data(data):
    if data is None:
        return None
    finite = np.isfinite(data).all(axis=1)
    if not np.any(finite):
        return None
    data = data[finite]
    order = np.argsort(data[:, 0], kind="mergesort")
    data = data[order]

    # Keep the first row for each repeated iteration.
    _, unique_idx = np.unique(data[:, 0], return_index=True)
    return data[unique_idx]


def ah_center_for_iteration(sim, iteration):
    data = _load_ah_diagnostics(sim)
    if data is None:
        return None
    data = _prepare_ah_data(data)
    if data is None or data.ndim != 2 or data.shape[0] == 0:
        return None
    iterations = data[:, 0].astype(float)
    times = data[:, 1].astype(float)
    cx = data[:, 2].astype(float)
    cy = data[:, 3].astype(float)
    cz = data[:, 4].astype(float)
    if iterations.size == 0:
        return None
    if iterations.size == 1:
        return np.array([cx[0], cy[0], cz[0]], dtype=float)

    iteration_query = float(iteration)
    time_query = float(np.interp(iteration_query, iterations, times, left=times[0], right=times[-1]))
    return ah_center_for_time(sim, time_query)


def ah_center_for_time(sim, time_code):
    data = _load_ah_diagnostics(sim)
    if data is None:
        return None
    data = _prepare_ah_data(data)
    if data is None or data.ndim != 2 or data.shape[0] == 0:
        return None
    if not np.isfinite(time_code):
        return None
    t = data[:, 1].astype(float)
    if t.size == 0:
        return None
    cx = data[:, 2].astype(float)
    cy = data[:, 3].astype(float)
    cz = data[:, 4].astype(float)
    if t.size == 1:
        return np.array([cx[0], cy[0], cz[0]], dtype=float)
    time = float(time_code)
    cx_query = np.interp(time, t, cx, left=cx[0], right=cx[-1])
    cy_query = np.interp(time, t, cy, left=cy[0], right=cy[-1])
    cz_query = np.interp(time, t, cz, left=cz[0], right=cz[-1])
    return np.array(
        [float(cx_query), float(cy_query), float(cz_query)],
        dtype=float,
    )


def _load_ah_points_from_file(path):
    points = []
    with open(path, "r") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            if len(parts) < 6:
                continue
            try:
                points.append([float(parts[3]), float(parts[4]), float(parts[5])])
            except ValueError:
                continue
    if not points:
        return None
    arr = np.asarray(points, dtype=float)
    finite = np.isfinite(arr).all(axis=1)
    if not np.any(finite):
        return None
    return arr[finite]


def spherical_ah_fallback_geometry(sim, frame_time, plane, horizon_debug: bool = False):
    """Return an interpolated spherical AH center and radius in the slice plane."""
    data = _load_ah_diagnostics(sim)
    if data is None or not np.isfinite(frame_time):
        return None

    time = np.asarray(data[:, 1], dtype=float)
    center = np.asarray(data[:, 2:5], dtype=float)
    radius = np.asarray(data[:, 7], dtype=float)  # column 8: mean coordinate radius
    valid = np.isfinite(time) & np.isfinite(radius) & (radius > 0.0) & np.isfinite(center).all(axis=1)
    if not np.any(valid):
        return None
    time = time[valid]
    center = center[valid]
    radius = radius[valid]
    order = np.argsort(time, kind="mergesort")
    time = time[order]
    center = center[order]
    radius = radius[order]
    rev_unique = np.unique(time[::-1], return_index=True)[1]
    keep = np.sort(time.size - 1 - rev_unique)
    time = time[keep]
    center = center[keep]
    radius = radius[keep]

    query = float(frame_time)
    tolerance = 1.0e-10 * max(1.0, abs(query), abs(time[0]), abs(time[-1]))
    if query < time[0] - tolerance or query > time[-1] + tolerance:
        _log_horizon_status(
            sim,
            f"no scalar AH fallback at t={query:g}; diagnostics span [{time[0]:g}, {time[-1]:g}]",
            horizon_debug,
        )
        return None
    query = float(np.clip(query, time[0], time[-1]))
    center_query = np.array(
        [np.interp(query, time, center[:, component]) for component in range(3)],
        dtype=float,
    )
    radius_query = float(np.interp(query, time, radius))

    axis_map = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if plane not in axis_map or not np.isfinite(radius_query) or radius_query <= 0.0:
        return None
    x_index, y_index = axis_map[plane]
    return (center_query[x_index], center_query[y_index]), radius_query


def spherical_ah_fallback_curve(sim, frame_time, plane, horizon_debug: bool = False):
    """Approximate a missing AH surface as a sphere from scalar diagnostics."""
    geometry = spherical_ah_fallback_geometry(
        sim,
        frame_time,
        plane,
        horizon_debug=horizon_debug,
    )
    if geometry is None:
        return None
    center, radius_query = geometry
    theta = np.linspace(0.0, 2.0 * np.pi, int(AH_FALLBACK_CIRCLE_POINTS), endpoint=True)
    x = center[0] + radius_query * np.cos(theta)
    y = center[1] + radius_query * np.sin(theta)
    _log_horizon_status(
        sim,
        f"using spherical scalar fallback at t={float(frame_time):g}, "
        f"mean coordinate radius={radius_query:g}",
        horizon_debug,
    )
    return x, y


def _rho2d_frame_time(sim):
    if not hasattr(sim, "rho2d") or not hasattr(sim.rho2d, "time"):
        return np.nan
    times = np.asarray(sim.rho2d.time, dtype=float)
    finite = times[np.isfinite(times)]
    return float(np.nanmedian(finite)) if finite.size else np.nan


def _polar_ah_boundary(points, x_index, y_index):
    """Sort a thin horizon ring by angle, following the collaborator workflow."""
    xy = np.asarray(points[:, [x_index, y_index]], dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    xy = xy[finite]
    if xy.shape[0] < AH_MIN_POINTS:
        return None

    # Center on the sliced ring itself. This is more robust to a small offset
    # between the scalar AH centroid time and the selected surface file.
    xy_center = 0.5 * (np.nanmin(xy, axis=0) + np.nanmax(xy, axis=0))
    delta = xy - xy_center
    radius = np.hypot(delta[:, 0], delta[:, 1])
    angle = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi)
    valid = np.isfinite(radius) & np.isfinite(angle) & (radius > 0.0)
    if np.count_nonzero(valid) < AH_MIN_POINTS:
        return None
    xy = xy[valid]
    angle = angle[valid]
    order = np.argsort(angle, kind="mergesort")
    xy = xy[order]
    angle = angle[order]
    angular_gaps = np.diff(np.concatenate((angle, [angle[0] + 2.0 * np.pi])))
    max_gap = np.deg2rad(float(AH_MAX_ANGULAR_GAP_DEG))
    if np.max(angular_gaps) > max_gap:
        return None
    xy = np.vstack((xy, xy[0]))
    return xy[:, 0], xy[:, 1]


def load_apparent_horizon_curve(sim, horizon_debug: bool = False):
    if not DRAW_APPARENT_HORIZON or not hasattr(sim, "rho2d"):
        if not DRAW_APPARENT_HORIZON:
            _log_horizon_status(sim, "overlay disabled by DRAW_APPARENT_HORIZON=False", horizon_debug)
        return None
    plane = str(sim.rho2d.plane).lower()
    axis_map = {"xy": (0, 1, 2), "xz": (0, 2, 1), "yz": (1, 2, 0)}
    if plane not in axis_map:
        _log_horizon_status(sim, f"unsupported 2D plane {plane!r}", horizon_debug)
        return None

    frame_time = _rho2d_frame_time(sim)

    path, ah_iteration = ah_surface_path_for_time(sim, frame_time)
    if path is None:
        iteration = int(sim.rho2d.iteration)
        path, ah_iteration = ah_surface_path_for_iteration(sim, iteration, horizon_debug=horizon_debug)
    if path is None:
        _log_horizon_status(
            sim,
            f"no AH surface file found for iteration={int(sim.rho2d.iteration)}; trying scalar fallback",
            horizon_debug,
        )
        return spherical_ah_fallback_curve(sim, frame_time, plane, horizon_debug=horizon_debug)
    _log_horizon_status(
        sim,
        f"using AH surface file {Path(path).name} for matched iteration={ah_iteration}",
        horizon_debug,
    )

    points = _load_ah_points_from_file(path)
    if points is None:
        _log_horizon_status(sim, f"AH surface {Path(path).name} has no usable points", horizon_debug)
        return spherical_ah_fallback_curve(sim, frame_time, plane, horizon_debug=horizon_debug)

    # Match the centroid to the selected surface file, as in the collaborator
    # script. Use frame-time interpolation only when that lookup is unavailable.
    center = ah_center_for_iteration(sim, ah_iteration)
    if center is None:
        center = ah_center_for_time(sim, frame_time)
    if center is None or not np.all(np.isfinite(center)):
        center = np.nanmean(points, axis=0)
        if not np.all(np.isfinite(center)):
            _log_horizon_status(sim, f"AH center unavailable for iteration={ah_iteration}", horizon_debug)
            return spherical_ah_fallback_curve(sim, frame_time, plane, horizon_debug=horizon_debug)

    x_index, y_index, normal_index = axis_map[plane]
    normal_distance = np.abs(points[:, normal_index] - center[normal_index])
    surface_radius = np.linalg.norm(points - center, axis=1)
    finite_radius = surface_radius[np.isfinite(surface_radius) & (surface_radius > 0.0)]
    if finite_radius.size == 0:
        return spherical_ah_fallback_curve(sim, frame_time, plane, horizon_debug=horizon_debug)
    radius_scale = float(np.nanmedian(finite_radius))
    tol = max(np.finfo(float).eps, float(AH_SLICE_ABSOLUTE_TOLERANCE))
    max_tol = max(tol, AH_MAX_SLICE_RELATIVE_TOLERANCE * radius_scale)
    selected = points[normal_distance <= tol]
    while selected.shape[0] < AH_MIN_POINTS and tol < max_tol:
        tol = min(max_tol, tol * 2.0)
        selected = points[normal_distance <= tol]
    if selected.shape[0] < AH_MIN_POINTS:
        _log_horizon_status(
            sim,
            f"AH slice rejected: selected points={selected.shape[0]}, need at least {AH_MIN_POINTS}",
            horizon_debug,
        )
        return spherical_ah_fallback_curve(sim, frame_time, plane, horizon_debug=horizon_debug)

    curve = _polar_ah_boundary(selected, x_index, y_index)
    if curve is None:
        _log_horizon_status(sim, "AH slice has incomplete angular coverage; using scalar fallback", horizon_debug)
        return spherical_ah_fallback_curve(sim, frame_time, plane, horizon_debug=horizon_debug)
    _log_horizon_status(
        sim,
        f"AH outline selected with n={selected.shape[0]} source points "
        f"(tol={tol:.3g}={tol/radius_scale:.3g} r, normal={plane} plane)",
        horizon_debug,
    )
    return curve


def plot_apparent_horizon(sim, ax, horizon_debug: bool = False):
    curve = load_apparent_horizon_curve(sim, horizon_debug=horizon_debug)
    if curve is None:
        return None
    coordinate_divisor = rho2d_coordinate_divisor(sim)
    curve = tuple(np.asarray(component, dtype=float) / coordinate_divisor for component in curve)

    curve_xy = np.column_stack(curve)
    curve_center = np.nanmedian(curve_xy, axis=0)
    display_curve = ax.transData.transform(curve_xy)
    display_center = ax.transData.transform(curve_center)
    display_radius = float(np.nanmedian(np.linalg.norm(display_curve - display_center, axis=1)))

    circle_geometry = None
    if np.isfinite(display_radius) and display_radius < AH_MIN_RESOLVED_RADIUS_PIXELS:
        circle_geometry = spherical_ah_fallback_geometry(
            sim,
            _rho2d_frame_time(sim),
            str(sim.rho2d.plane).lower(),
            horizon_debug=False,
        )
        if circle_geometry is not None:
            center, radius = circle_geometry
            circle_geometry = (
                tuple(float(value) / coordinate_divisor for value in center),
                float(radius) / coordinate_divisor,
            )
            _log_horizon_status(
                sim,
                f"surface radius is only {display_radius:.1f} px; plotting smooth scalar-radius mask",
                horizon_debug,
            )

    artists = []
    if circle_geometry is not None:
        center, radius = circle_geometry
        if DRAW_AH_SOLID_MASK:
            patch = plt.Circle(
                center,
                radius,
                facecolor=AH_MASK_COLOR,
                edgecolor="none",
                alpha=AH_MASK_ALPHA,
                antialiased=True,
                zorder=5,
            )
            ax.add_patch(patch)
            artists.append(patch)
        if DRAW_AH_WHITE_OUTLINE:
            patch = plt.Circle(
                center,
                radius,
                facecolor="none",
                edgecolor=AH_OUTLINE_COLOR,
                linestyle=AH_OUTLINE_LINESTYLE,
                linewidth=AH_OUTLINE_WIDTH,
                antialiased=True,
                zorder=5,
            )
            ax.add_patch(patch)
            artists.append(patch)
        patch = plt.Circle(
            center,
            radius,
            facecolor="none",
            edgecolor=AH_LINE_COLOR,
            linestyle=AH_LINESTYLE,
            linewidth=AH_LINE_WIDTH,
            antialiased=True,
            zorder=6,
        )
        ax.add_patch(patch)
        artists.append(patch)
        _log_horizon_status(sim, "plotted AH outline", horizon_debug)
        return artists

    if DRAW_AH_SOLID_MASK:
        artists.extend(
            ax.fill(
                curve[0],
                curve[1],
                facecolor=AH_MASK_COLOR,
                edgecolor="none",
                alpha=AH_MASK_ALPHA,
                zorder=5,
            )
        )
    if DRAW_AH_WHITE_OUTLINE:
        ax.plot(
            curve[0],
            curve[1],
            color=AH_OUTLINE_COLOR,
            linestyle=AH_OUTLINE_LINESTYLE,
            linewidth=AH_OUTLINE_WIDTH,
            zorder=5,
        )
    _log_horizon_status(sim, "plotted AH outline", horizon_debug)
    artists.extend(
        ax.plot(
            curve[0],
            curve[1],
            color=AH_LINE_COLOR,
            linestyle=AH_LINESTYLE,
            linewidth=AH_LINE_WIDTH,
            zorder=6,
        )
    )
    return artists


def valid_rho2d_iteration_infos(sim):
    authoritative, supplemental = rho2d_paths_for_sim(sim)
    return valid_composite_2d_iteration_time_infos(
        authoritative,
        supplemental_paths=supplemental,
        ref_level=RHO2D_REQUIRED_REF_LEVELS,
    )


def iter_valid_rho2d_iteration_infos(sim):
    authoritative, supplemental = rho2d_paths_for_sim(sim)
    return iter_valid_composite_2d_iteration_time_infos(
        authoritative,
        supplemental_paths=supplemental,
        ref_level=RHO2D_REQUIRED_REF_LEVELS,
    )


def first_rho2d_iteration_info(sim):
    authoritative, supplemental = rho2d_paths_for_sim(sim)
    return first_composite_2d_iteration_time_info(
        authoritative,
        supplemental_paths=supplemental,
        ref_level=RHO2D_REQUIRED_REF_LEVELS,
    )


def last_rho2d_iteration_info(sim):
    authoritative, supplemental = rho2d_paths_for_sim(sim)
    return last_composite_2d_iteration_time_info(
        authoritative,
        supplemental_paths=supplemental,
        ref_level=RHO2D_REQUIRED_REF_LEVELS,
    )


def nearest_rho2d_iteration_info(sim, target_time):
    authoritative, supplemental = rho2d_paths_for_sim(sim)
    return nearest_composite_2d_iteration_time_info(
        authoritative,
        target_time,
        supplemental_paths=supplemental,
        ref_level=RHO2D_REQUIRED_REF_LEVELS,
    )


def _require_snapshot_info(sim, info):
    if info is None:
        raise ValueError(
            f"{sim.config.name}: no valid {RHO2D_VARIABLE} 2D outputs in {rho2d_path_for_sim(sim)}"
        )
    return info


def snapshot_requests(values, by_tbypc):
    values = list(values)
    by_tbypc = list(by_tbypc)
    if len(values) != len(by_tbypc):
        raise ValueError(
            "Snapshot value and BY_TBYPC arrays must have the same length"
        )
    if not values:
        raise ValueError("At least one snapshot value is required")
    requests = []
    for value, flag in zip(values, by_tbypc):
        if flag not in (0, 1, False, True):
            raise ValueError("Snapshot BY_TBYPC entries must be 0 or 1")
        requests.append((value, bool(flag)))
    return requests


def _normalize_snapshot_selection(selection, by_tbypc=False):
    if by_tbypc:
        value = float(selection)
        if not np.isfinite(value):
            raise ValueError("t/P_c snapshot targets must be finite")
        return value, "tbypc"

    if isinstance(selection, (float, np.floating)):
        value = float(selection)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("Fractional snapshot selectors must be in [0, 1]")
        return value, "fraction"

    if isinstance(selection, (int, np.integer)) and not isinstance(selection, bool):
        return int(selection), "index"

    raise ValueError(
        "Snapshot values must be integer frame indices or float fractions in [0, 1]"
    )


def _selection_label(kind, value):
    if kind == "tbypc":
        return f"tbypc{value:g}"
    if kind == "fraction":
        return f"frac{value:g}"
    if value == 0:
        return "idx0"
    if value == -1:
        return "idx-1"
    return f"idx{value}"


def select_rho2d_snapshot_info(
    sim,
    selection,
    by_tbypc=False,
    fraction_time_bounds=None,
):
    value, kind = _normalize_snapshot_selection(selection, by_tbypc=by_tbypc)
    if kind == "tbypc":
        pc = float(getattr(sim.config, "Pc", np.nan))
        if not np.isfinite(pc) or pc <= 0.0:
            raise ValueError(f"{sim.config.name}: invalid P_c={pc!r}")
        return _require_snapshot_info(sim, nearest_rho2d_iteration_info(sim, value * pc))

    if kind == "fraction":
        if fraction_time_bounds is None:
            first = _require_snapshot_info(sim, first_rho2d_iteration_info(sim))
            last = _require_snapshot_info(sim, last_rho2d_iteration_info(sim))
            first_time, last_time = first[1], last[1]
        else:
            first_time, last_time = map(float, fraction_time_bounds)
        target_time = first_time + value * (last_time - first_time)
        return _require_snapshot_info(sim, nearest_rho2d_iteration_info(sim, target_time))

    index = value
    if index == -1:
        return _require_snapshot_info(sim, last_rho2d_iteration_info(sim))
    if index == 0:
        return _require_snapshot_info(sim, first_rho2d_iteration_info(sim))

    valid_infos = valid_rho2d_iteration_infos(sim)
    if not valid_infos:
        raise ValueError(f"{sim.config.name}: no valid rho2D outputs in configured sources")
    return valid_infos[index]


def select_rho2d_snapshot_infos(sim, requests):
    infos = []
    seen = set()
    for raw, by_tbypc in requests:
        value, kind = _normalize_snapshot_selection(raw, by_tbypc=by_tbypc)
        info = select_rho2d_snapshot_info(
            sim,
            selection=raw,
            by_tbypc=by_tbypc,
        )
        key = (int(info[0]),)
        if key in seen:
            continue
        seen.add(key)
        infos.append((raw, _selection_label(kind, value), info))
    if not infos:
        raise ValueError(f"{sim.config.name}: no valid rho2D outputs selected from {rho2d_path_for_sim(sim)}")
    return infos


def select_rho2d_snapshot_iteration(sim, selection, by_tbypc=False):
    return select_rho2d_snapshot_info(
        sim,
        selection=selection,
        by_tbypc=by_tbypc,
    )[0]


def _axis_limits(sim):
    axis0 = getattr(sim.rho2d, sim.rho2d.axis_names[0])
    axis1 = getattr(sim.rho2d, sim.rho2d.axis_names[1])
    x = np.asarray(axis0, dtype=float)
    y = np.asarray(axis1, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return None, None
    return (float(np.min(x)), float(np.max(x))), (float(np.min(y)), float(np.max(y)))


def fixed_initial_max_reference(sim):
    cached = getattr(sim, "rho2d_normalization_reference", None)
    if cached is not None:
        return cached

    reference = None
    method = None
    if USE_RHOMAX_DIAGNOSTIC:
        try:
            sim.load_rhomax()
            times = np.asarray(sim.rhomax_t, dtype=float)
            values = np.asarray(sim.rhomax, dtype=float)
            valid = np.isfinite(times) & np.isfinite(values) & (values > 0.0)
            if np.any(valid):
                times = times[valid]
                values = values[valid]
                first = int(np.argmin(times))
                first_time = float(times[first])
                reference = float(values[first])
                method = f"earliest rho_max diagnostic at t={first_time:g}"
                tolerance = 1.0e-12 * max(1.0, abs(first_time))
                if abs(first_time) > tolerance:
                    warnings.warn(
                        f"{sim.config.name}: earliest rho_max diagnostic is at t={first_time:g}, not t=0; "
                        "using that fixed value for 2D normalization",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    f"{sim.config.name}: rho_max diagnostic has no positive finite values; "
                    "using the first valid 2D slice maximum",
                    RuntimeWarning,
                    stacklevel=2,
                )
        except (OSError, ValueError, IndexError) as exc:
            warnings.warn(
                f"{sim.config.name}: could not load rho_max diagnostic ({exc}); "
                "using the first valid 2D slice maximum",
                RuntimeWarning,
                stacklevel=2,
            )

    if reference is None:
        info = first_rho2d_iteration_info(sim)
        if info is None:
            raise ValueError(f"{sim.config.name}: no complete valid 2D iteration in configured sources")
        iteration, time_code, start_byte = info
        coordinate_divisor = rho2d_coordinate_divisor(sim)
        sim.load_rho2d(
            variable=RHO2D_VARIABLE,
            plane=RHO2D_PLANE,
            iteration=iteration,
            ref_level="all" if RHO2D_USE_UNIFORM_RESAMPLE else RHO2D_NATIVE_REF_LEVELS,
            start_byte=start_byte,
            required_ref_levels=RHO2D_REQUIRED_REF_LEVELS,
            region=(
                _source_coordinate_limits(RHO2D_X_LIMITS, coordinate_divisor),
                _source_coordinate_limits(RHO2D_Y_LIMITS, coordinate_divisor),
            ) if RHO2D_USE_UNIFORM_RESAMPLE else None,
            selection_grid_shape=RHO2D_RESAMPLE_GRID,
        )
        reference = float(np.nanmax(sim.rho2d.data))
        method = f"first valid 2D slice at iteration={iteration}, t={time_code:g}"

    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError(f"{sim.config.name}: invalid initial 2D normalization {reference!r}")

    sim.rho2d_normalization_reference = reference
    print(
        f"{sim.config.name}: rho2D normalization reference={reference:.8e} ({method})",
        flush=True,
    )
    return reference


def apply_rho2d_scaling(sim, reference):
    if not NORMALIZE_BY_FIXED_INITIAL_MAX:
        return
    sim.rho2d.data = sim.rho2d.data / reference
    if LOGSCALE:
        with np.errstate(divide="ignore", invalid="ignore"):
            sim.rho2d.data = np.log10(sim.rho2d.data)


def load_rho2d_slice(sim, iteration=-1, start_byte=None, xlim=None, ylim=None):
    reference = fixed_initial_max_reference(sim) if NORMALIZE_BY_FIXED_INITIAL_MAX else None
    xlim = RHO2D_X_LIMITS if xlim is None else xlim
    ylim = RHO2D_Y_LIMITS if ylim is None else ylim
    if RHO2D_USE_UNIFORM_RESAMPLE and (xlim is None or ylim is None):
        raise ValueError("Uniform 2D resampling requires explicit x and y plot limits")
    coordinate_divisor = rho2d_coordinate_divisor(sim)
    sim.load_rho2d(
        variable=RHO2D_VARIABLE,
        plane=RHO2D_PLANE,
        iteration=iteration,
        ref_level="all" if RHO2D_USE_UNIFORM_RESAMPLE else RHO2D_NATIVE_REF_LEVELS,
        start_byte=start_byte,
        required_ref_levels=RHO2D_REQUIRED_REF_LEVELS,
        region=(
            _source_coordinate_limits(xlim, coordinate_divisor),
            _source_coordinate_limits(ylim, coordinate_divisor),
        ) if RHO2D_USE_UNIFORM_RESAMPLE else None,
        selection_grid_shape=RHO2D_RESAMPLE_GRID,
    )
    apply_rho2d_coordinate_scaling(sim, coordinate_divisor)
    apply_rho2d_scaling(sim, reference)
    return sim.rho2d


def plot_rho2d_on_axis(
    sim,
    ax,
    title=None,
    title_pad=TITLE_PAD,
    xlim=None,
    ylim=None,
    show_data_limits_in_title=None,
    horizon_debug: bool = False,
):
    if xlim is None:
        xlim = RHO2D_X_LIMITS
    if ylim is None:
        ylim = RHO2D_Y_LIMITS
    if xlim is None or ylim is None:
        physical_xlim, physical_ylim = _axis_limits(sim)
        if xlim is None and physical_xlim is not None:
            xlim = physical_xlim
        if ylim is None and physical_ylim is not None:
            ylim = physical_ylim
    plot_kwargs = {
        "ax": ax,
        "draw_boxes": DRAW_REFINEMENT_BOXES,
        "cmap": JET_WHITE_LOW_CMAP,
        "vmin": RHO_MIN,
        "vmax": RHO_MAX,
    }
    if RHO2D_USE_UNIFORM_RESAMPLE:
        mesh = plot_2d_uniform(
            sim.rho2d,
            xlim,
            ylim,
            RHO2D_RESAMPLE_GRID,
            **plot_kwargs,
        )
    else:
        mesh = plot_2d_reflevels(sim.rho2d, **plot_kwargs)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if show_data_limits_in_title is None:
        show_data_limits_in_title = SHOW_DATA_LIMITS_IN_TITLE

    if show_data_limits_in_title and xlim is not None and ylim is not None:
        axis0, axis1 = sim.rho2d.axis_names
        limits = rf"${axis0}\in[{xlim[0]:.1f},{xlim[1]:.1f}],\ {axis1}\in[{ylim[0]:.1f},{ylim[1]:.1f}]$"
        title = f"{title}\n{limits}" if title is not None else limits
    ax.set_title(rho2d_title(sim) if title is None else title, pad=title_pad)
    ax.set_xlabel(rho2d_coordinate_label(sim.rho2d.axis_names[0]))
    ax.set_ylabel(rho2d_coordinate_label(sim.rho2d.axis_names[1]))
    plot_apparent_horizon(sim, ax, horizon_debug=horizon_debug)
    return mesh


def _snapshot_filename(sim_name, plane, iteration, tag):
    return f"{OUTPUT_PREFIX}_{sim_name}_{plane}_iter{iteration}_{tag}.png"


def _remove_superseded_snapshot(args, sim_name, plane, tag, keep_filename):
    if args.no_save:
        return
    case_dir = args.outdir / sim_name
    pattern = f"{OUTPUT_PREFIX}_{sim_name}_{plane}_iter*_{tag}.png"
    for path in case_dir.glob(pattern):
        if path.name != keep_filename:
            path.unlink()
            print(f"removed superseded {path}")


def plot_one_snapshot(sim, tag, snapshot_info, args):
    iteration, _, start_byte = snapshot_info
    load_rho2d_slice(sim, iteration=iteration, start_byte=start_byte)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    try:
        mesh = plot_rho2d_on_axis(
            sim,
            ax,
            title=rho2d_title(sim),
            horizon_debug=SHOW_HORIZON_DEBUG,
        )
        cbar = fig.colorbar(mesh, ax=ax, shrink=COLORBAR_SHRINK, pad=COLORBAR_PAD)
        _set_colorbar_label(cbar, rho2d_colorbar_label())
        fig.tight_layout()
        filename = _snapshot_filename(sim.config.name, sim.rho2d.plane, iteration, tag)
        save_individual_fig(fig, args, sim, filename)
        _remove_superseded_snapshot(
            args,
            sim.config.name,
            sim.rho2d.plane,
            tag,
            filename,
        )
    finally:
        plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Plot configured 2D rho_b snapshots.")
    parser.add_argument("--sims", nargs="+", default=DEFAULT_SIMS)
    parser.add_argument("--outdir", type=Path, default=PLOTS_DIR)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args(argv)
    setup(args)

    sims = load_sims([], names=args.sims)
    for sim in sims:
        requests = snapshot_requests(
            SNAPSHOT_VALUES,
            SNAPSHOT_BY_TBYPC,
        )
        selected_infos = select_rho2d_snapshot_infos(sim, requests)
        for _, tag, snapshot_info in selected_infos:
            plot_one_snapshot(sim, tag, snapshot_info, args)


if __name__ == "__main__":
    main()
