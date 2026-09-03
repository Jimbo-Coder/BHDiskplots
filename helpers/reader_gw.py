"""GW Psi4 and strain readers/converters."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gw_psi4 import (  # noqa: E402
    DEFAULT_GW_WORK_ROOT,
    N_PSI4_COLUMNS,
    Psi4File,
    convert_to_strain,
    merge_psi4_restart_files,
    psi4_file_label_for_index,
    read_psi4_extraction_radii,
    read_sim_psi4,
    selected_psi4_mode,
)


PSI4_MODE_COLUMN_STOP = N_PSI4_COLUMNS - 4
PSI4_RADIUS_RTOL = 0.10
PSI4_RADIUS_ATOL = 3.0


def filter_psi4_by_expected_radius(files, radii):
    """Reject mislabeled extraction files using their stored areal radius.

    Numbered ``Psi4_rad.mon.N`` files correspond to parfile radius index
    ``N-1``. Some runs also contain a literal ``Psi4_rad.mon.*`` artifact;
    it is usable only if its stored areal radius agrees with that mapping.
    """
    valid = {}
    for label, psi4_file in files.items():
        try:
            radius_index = int(label) - 1
        except (TypeError, ValueError):
            valid[label] = psi4_file
            continue
        expected = radii.get(radius_index)
        observed = float(np.nanmedian(np.asarray(psi4_file.r_areal, dtype=float)))
        if expected is None or not np.isfinite(observed):
            valid[label] = psi4_file
            continue
        if np.isclose(observed, expected, rtol=PSI4_RADIUS_RTOL, atol=PSI4_RADIUS_ATOL):
            valid[label] = psi4_file
            continue
        print(
            f"skipping {psi4_file.path}: label {label} implies r={expected:g}, "
            f"but stored median areal radius is {observed:g}"
        )
    return valid


def load_psi4(sim_paths):
    """Load and merge ordered simulation roots, with later restarts winning."""
    if isinstance(sim_paths, (str, Path)):
        sim_paths = (Path(sim_paths),)
    else:
        sim_paths = tuple(Path(path) for path in sim_paths)

    files_by_label = {}
    radii = {}
    for sim_path in sim_paths:
        source_radii = read_psi4_extraction_radii(sim_path)
        if not source_radii and sim_path.name == "data":
            source_radii = read_psi4_extraction_radii(sim_path.parent)
        if source_radii:
            radii.update(source_radii)
        for label, psi4_file in read_sim_psi4(sim_path).items():
            files_by_label.setdefault(label, []).append(psi4_file)

    merged = {
        label: merge_psi4_restart_files(files)
        for label, files in files_by_label.items()
    }
    return filter_psi4_by_expected_radius(merged, radii), radii


def select_psi4_file(psi4_files, parfile_index: int, file_label: str | None = None):
    requested_label = file_label is not None
    label = str(file_label) if requested_label else psi4_file_label_for_index(parfile_index)
    if label in psi4_files:
        return label, psi4_files[label]
    if requested_label:
        return label, None
    if psi4_files:
        fallback = sorted(psi4_files.keys(), key=lambda value: int(value) if str(value).isdigit() else 999)[-1]
        return fallback, psi4_files[fallback]
    return label, None


def convert_psi4_to_strain(
    psi4_file,
    workdir: Path,
    omega_orbital: float,
    madm: float,
    regenerate: bool = False,
    generate_if_missing: bool = True,
    backend: str = "python",
):
    return convert_to_strain(
        psi4_file,
        workdir=workdir,
        omega_orbital=omega_orbital,
        madm=madm,
        reuse_existing=not regenerate,
        generate_if_missing=generate_if_missing,
        backend=backend,
    )


def subtract_psi4_on_retarded_time(
    case_psi4: Psi4File,
    case_t_ret,
    reference_psi4: Psi4File,
    reference_t_ret,
    *,
    label: str,
) -> Psi4File:
    """Subtract all Psi4 modes on the case waveform's common retarded-time grid."""

    def finite_monotone_rows(psi4_file, t_ret):
        t_ret = np.asarray(t_ret, dtype=float)
        rows = np.asarray(psi4_file.data, dtype=float)
        n = min(t_ret.size, rows.shape[0])
        t_ret = t_ret[:n]
        rows = rows[:n]
        mode_values = rows[:, 1:PSI4_MODE_COLUMN_STOP]
        keep = np.isfinite(t_ret) & np.all(np.isfinite(mode_values), axis=1)
        t_ret = t_ret[keep]
        rows = rows[keep]
        order = np.argsort(t_ret, kind="mergesort")
        t_ret = t_ret[order]
        rows = rows[order]
        _, unique = np.unique(t_ret, return_index=True)
        unique = np.sort(unique)
        return t_ret[unique], rows[unique]

    case_t_ret, case_rows = finite_monotone_rows(case_psi4, case_t_ret)
    reference_t_ret, reference_rows = finite_monotone_rows(reference_psi4, reference_t_ret)
    if case_t_ret.size < 2 or reference_t_ret.size < 2:
        raise ValueError("Psi4 subtraction requires at least two finite samples per waveform")

    t_min = max(float(case_t_ret[0]), float(reference_t_ret[0]))
    t_max = min(float(case_t_ret[-1]), float(reference_t_ret[-1]))
    keep = (case_t_ret >= t_min) & (case_t_ret <= t_max)
    if np.count_nonzero(keep) < 2:
        raise ValueError("Psi4 waveforms have no usable common retarded-time interval")

    target_t_ret = case_t_ret[keep]
    difference = case_rows[keep].copy()
    for column in range(1, PSI4_MODE_COLUMN_STOP):
        reference_values = np.interp(target_t_ret, reference_t_ret, reference_rows[:, column])
        difference[:, column] -= reference_values

    return Psi4File(
        path=case_psi4.path,
        label=label,
        data=difference,
        repeated_times=0,
        source_kind="retarded-time-difference",
    )
