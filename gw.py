"""GW data pipeline: extraction files, legacy FFI, caches, and ML subtraction.

The FFI equations follow the Fortran reference. Plotting and detector analysis
are separate; source simulation files are never modified.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from config import (
    DiskSimConfig, FORTRAN_GW_ROOT, GW_WORK_ROOT, GW_STRAIN_BACKEND,
    all_sim_configs, MASSLESS_SIM_NAME,
)


N_PSI4_COLUMNS = 47
N_PSI4_MODES = (N_PSI4_COLUMNS - 5) // 2


def mode_order(num_modes: int = N_PSI4_MODES) -> List[Tuple[int, int]]:
    modes = []
    ell = 2
    while len(modes) < num_modes:
        for emm in range(ell, -ell - 1, -1):
            modes.append((ell, emm))
            if len(modes) == num_modes:
                return modes
        ell += 1
    return modes


MODES = mode_order()
MODE_TO_INDEX = {mode: i for i, mode in enumerate(MODES)}


def mode_columns(ell: int, emm: int) -> Tuple[int, int]:
    """Return zero-based Re/Im columns in Psi4/rhphc-style 47-column files."""
    mode_index = MODE_TO_INDEX[(ell, emm)]
    return 1 + 2 * mode_index, 2 + 2 * mode_index


@dataclass
class Psi4File:
    path: Path
    label: str
    data: np.ndarray
    repeated_times: int
    source_kind: str = "numbered"

    @property
    def time(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def r_areal(self) -> np.ndarray:
        return self.data[:, -4]

    def psi4(self, ell: int = 2, emm: int = 2, multiply_by_r: bool = False) -> np.ndarray:
        re_col, im_col = mode_columns(ell, emm)
        z = self.data[:, re_col] + 1j * self.data[:, im_col]
        if multiply_by_r:
            z = self.r_areal * z
        return z


@dataclass
class StrainResult:
    workdir: Path
    psi4_input: Path
    rhphc: Optional[np.ndarray]
    rhphcdot: Optional[np.ndarray]
    omega22: Optional[np.ndarray]
    ejv_gw: Optional[np.ndarray]
    stdout: str
    stderr: str
    backend: str = "unknown"
    rpsi4_uniform: Optional[np.ndarray] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def time(self) -> np.ndarray:
        if self.rhphc is None:
            return np.array([])
        return self.rhphc[:, 0]

    def hplus_hcross(self, ell: int = 2, emm: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        if self.rhphc is None:
            raise ValueError("rhphc.dat was not produced")
        re_col, im_col = mode_columns(ell, emm)
        return self.rhphc[:, re_col], self.rhphc[:, im_col]

    def rpsi4(self, ell: int = 2, emm: int = 2) -> np.ndarray:
        """Return one uniformly sampled natural-sign ``r Psi4`` mode."""
        if self.rpsi4_uniform is None:
            raise ValueError(
                "rpsi4_uniform.dat is unavailable; regenerate this cache with "
                "generate_gw.py using GW_STRAIN_BACKEND='python'"
            )
        re_col, im_col = mode_columns(ell, emm)
        return self.rpsi4_uniform[:, re_col] + 1j * self.rpsi4_uniform[:, im_col]

    def rpsi4_modes(
        self,
        modes: Iterable[Tuple[int, int]],
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
        """Return the shared uniform time grid and requested natural-sign modes."""
        requested = tuple(modes)
        return self.time, {mode: self.rpsi4(*mode) for mode in requested}


def discover_psi4_paths(sim_path: Path, include_star_file: bool = False) -> Dict[str, Path]:
    sim_path = Path(sim_path)
    paths: Dict[str, Path] = {}
    for path in sorted(sim_path.glob("Psi4_rad.mon.[0-9]*"), key=_psi4_sort_key):
        paths[path.name.rsplit(".", 1)[-1]] = path
    asterisk_path = sim_path / "Psi4_rad.mon.*"
    if include_star_file and asterisk_path.exists():
        paths["10"] = asterisk_path
    return paths


def read_psi4_file(
    path: Path,
    label: Optional[str] = None,
    source_kind: str = "numbered",
    sort_by_time: bool = True,
    unique_by_time: bool = True,
) -> Psi4File:
    path = Path(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] != N_PSI4_COLUMNS:
        raise ValueError(f"{path} has {data.shape[1]} columns, expected {N_PSI4_COLUMNS}")

    repeated_times = 0
    if sort_by_time:
        data = data[np.argsort(data[:, 0], kind="mergesort")]
    if unique_by_time and len(data) > 1:
        # Appended/restarted files can contain an older and a corrected row at
        # the same coordinate time. Match the scalar-reader policy: the last
        # row in source order is authoritative.
        keep_from_end = np.unique(data[::-1, 0], return_index=True)[1]
        keep = np.sort(data.shape[0] - 1 - keep_from_end)
        repeated_times = len(data) - len(keep)
        data = data[keep]

    if label is None:
        label = path.name
    return Psi4File(path=path, label=label, data=data, repeated_times=repeated_times, source_kind=source_kind)


def read_sim_psi4(
    sim_path: Path,
    include_star_file: bool = False,
) -> Dict[str, Psi4File]:
    """Read all Psi4 extraction files for a sim.

    File mapping follows the ET output convention used here:
    ``Psi4_rad.mon.1`` maps to ``radius_GW_Psi4[0]``,
    ``Psi4_rad.mon.9`` maps to ``radius_GW_Psi4[8]``. A literal
    ``Psi4_rad.mon.*`` artifact is excluded unless explicitly requested;
    ordinary numbered files, including ``Psi4_rad.mon.10``, are unaffected.
    """
    out: Dict[str, Psi4File] = {}
    for label, path in discover_psi4_paths(sim_path, include_star_file=include_star_file).items():
        kind = "literal-asterisk" if label == "10" else "numbered"
        out[label] = read_psi4_file(path, label=label, source_kind=kind)
    return out


def merge_psi4_restart_files(files: Sequence[Psi4File]) -> Psi4File:
    """Merge ordered Psi4 restart files without requiring identical overlap times."""
    if not files:
        raise ValueError("At least one Psi4 file is required")
    merged = np.empty((0, N_PSI4_COLUMNS), dtype=float)
    repeated_times = 0
    for psi4_file in files:
        current = np.asarray(psi4_file.data)
        if current.size == 0:
            continue
        restart_time = float(current[0, 0])
        merged = merged[merged[:, 0] < restart_time]
        merged = np.concatenate((merged, current), axis=0)
        repeated_times += int(psi4_file.repeated_times)
    if merged.size == 0:
        raise ValueError("Configured Psi4 files contain no samples")
    order = np.argsort(merged[:, 0], kind="mergesort")
    merged = merged[order]
    rev_unique = np.unique(merged[::-1, 0], return_index=True)[1]
    keep = np.sort(merged.shape[0] - 1 - rev_unique)
    repeated_times += merged.shape[0] - keep.size
    merged = merged[keep]
    latest = files[-1]
    return Psi4File(
        path=latest.path,
        label=latest.label,
        data=merged,
        repeated_times=repeated_times,
        source_kind="restart-merged" if len(files) > 1 else latest.source_kind,
    )


def read_psi4_extraction_radii(sim_path: Path) -> Dict[int, float]:
    """Return ``radius_GW_Psi4[index]`` values from a simulation par file."""
    sim_path = Path(sim_path)
    for par_path in (sim_path / "bh_disk.par", sim_path / "bhdisk.par", sim_path / "beta100" / "bh_disk.par", sim_path / "beta100" / "bhdisk.par"):
        if not par_path.exists():
            continue
        radii: Dict[int, float] = {}
        for line in par_path.read_text(errors="ignore").splitlines():
            line = line.split("#", 1)[0].strip()
            match = re.match(r"gw_extraction::radius_GW_Psi4\s*\[\s*(\d+)\s*\]\s*=\s*([-+0-9.eEdD]+)", line)
            if match:
                index = int(match.group(1))
                value = float(match.group(2).replace("D", "E").replace("d", "e"))
                radii[index] = value
        if radii:
            return radii
    return {}


def write_sorted_psi4_input(psi4_file: Psi4File, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, psi4_file.data, fmt="%25.15E")
    return out_path


def read_strain_cache(workdir: Path, psi4_input: Path) -> StrainResult:
    if not (workdir / "rhphc.dat").is_file():
        raise FileNotFoundError(f"Missing strain cache {workdir}. Run generate_gw.py before plotting.")
    manifest_path = workdir / "strain_cache.json"
    backend = "fortran"
    metadata = {}
    if manifest_path.is_file():
        try:
            metadata = json.loads(manifest_path.read_text())
            backend = str(metadata.get("backend", backend))
        except (AttributeError, OSError, ValueError, TypeError):
            backend = "unknown"
            metadata = {}
    return StrainResult(
        workdir=workdir,
        psi4_input=psi4_input,
        rhphc=_load_optional(workdir / "rhphc.dat"),
        rhphcdot=_load_optional(workdir / "rhphcdot.dat"),
        omega22=_load_optional(workdir / "omega22.dat"),
        ejv_gw=_load_optional(workdir / "ejv_GW.dat"),
        stdout=f"reused existing {backend} strain outputs",
        stderr="",
        backend=backend,
        rpsi4_uniform=_load_optional(workdir / "rpsi4_uniform.dat"),
        metadata=metadata,
    )


def _source_metadata(
    psi4_file: Psi4File,
    omega_orbital: float,
    madm: float,
    t_start: Optional[float],
    t_end: Optional[float],
) -> Dict[str, object]:
    return {
        "source": str(psi4_file.path),
        "source_kind": psi4_file.source_kind,
        "source_label": psi4_file.label,
        "source_rows": int(psi4_file.data.shape[0]),
        "source_t_min": float(psi4_file.time[0]),
        "source_t_max": float(psi4_file.time[-1]),
        "repeated_times_removed": int(psi4_file.repeated_times),
        "omega_orbital": float(omega_orbital),
        "madm": float(madm),
        "requested_t_start": None if t_start is None else float(t_start),
        "requested_t_end": None if t_end is None else float(t_end),
    }


def generate_python_strain(
    psi4_file: Psi4File,
    workdir: Path,
    omega_orbital: float,
    madm: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
) -> StrainResult:
    """Generate legacy-compatible strain products with NumPy."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    products = reconstruct_strain(
        psi4_file.data,
        tuple(mode_order((psi4_file.data.shape[1] - 5) // 2)),
        omega_orbital=omega_orbital,
        madm=madm,
        t_start=t_start,
        t_end=t_end,
    )
    metadata = _source_metadata(psi4_file, omega_orbital, madm, t_start, t_end)
    metadata.update(
        {
            "interpolation": "local-four-point-polynomial",
            "ffi_cutoff": "max(abs(m) * omega_orbital, omega_orbital)",
        }
    )
    manifest = write_products(
        products,
        workdir,
        metadata,
    )
    return StrainResult(
        workdir=workdir,
        psi4_input=psi4_file.path,
        rhphc=products.rhphc,
        rhphcdot=products.rhphcdot,
        omega22=products.omega22,
        ejv_gw=products.ejv_gw,
        stdout="generated Python FFI strain outputs",
        stderr="",
        backend="python",
        rpsi4_uniform=products.rpsi4_uniform,
        metadata=manifest,
    )


def generate_strain(
    psi4_file: Psi4File,
    workdir: Path,
    omega_orbital: float,
    madm: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    backend: str = GW_STRAIN_BACKEND,
) -> StrainResult:
    """Dispatch to the maintained Python backend or the Fortran reference."""
    backend = backend.lower()
    common = dict(
        psi4_file=psi4_file,
        workdir=workdir,
        omega_orbital=omega_orbital,
        madm=madm,
        t_start=t_start,
        t_end=t_end,
    )
    if backend == "python":
        return generate_python_strain(**common)
    if backend == "fortran":
        return generate_fortran_strain(**common)
    raise ValueError(f"Unknown GW strain backend {backend!r}; expected 'python' or 'fortran'")


def generate_fortran_strain(
    psi4_file: Psi4File,
    workdir: Path,
    omega_orbital: float,
    madm: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    psi4_hlm_dir: Path = FORTRAN_GW_ROOT,
    executable: str = "rhphc",
) -> StrainResult:
    """Run the preserved psi4_hlm_ref rhphc executable.

    This preserves the old Fortran FFI algorithm. The only Python-side changes are:
    sort/remove repeated coordinate times, write ccc_ffi.input, and run in a
    separate work directory so Milton's scratch tree is never modified.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    psi4_input = workdir / f"Psi4_rad.mon_sorted.{psi4_file.label}"

    if t_start is None:
        t_start = float(psi4_file.time[0])
    if t_end is None:
        t_end = float(psi4_file.time[-1])

    psi4_input = write_sorted_psi4_input(psi4_file, psi4_input)
    input_text = (
        "# psi4 filename           number of columns    w_lower_cut                Madm                 t_start    t_end\n"
        f"'{psi4_input.name}'         {N_PSI4_COLUMNS}             {omega_orbital:.16E}      "
        f"{madm:.16E}    {t_start:.16E}    {t_end:.16E}\n\n"
        "#   ! Note: The parameter w_lower_cut refers to the *orbital* angular velocity in code unit,\n"
        "#   !       not the (2,2) mode of GW frequency.\n"
        "#   ! GW strain with t<t_start and t>t_end will be set to 0\n"
        "#   !\n"
    )
    (workdir / "ccc_ffi.input").write_text(input_text)

    exe_path = Path(psi4_hlm_dir) / executable
    proc = subprocess.run(
        [str(exe_path)],
        cwd=str(workdir),
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{exe_path} failed with exit code {proc.returncode}\n{proc.stderr}\n{proc.stdout}")

    (workdir / "rpsi4_uniform.dat").unlink(missing_ok=True)
    metadata = _source_metadata(psi4_file, omega_orbital, madm, t_start, t_end)
    metadata.update({"format_version": 1, "backend": "fortran"})
    (workdir / "strain_cache.json").write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    return StrainResult(
        workdir=workdir,
        psi4_input=psi4_input,
        rhphc=_load_optional(workdir / "rhphc.dat"),
        rhphcdot=_load_optional(workdir / "rhphcdot.dat"),
        omega22=_load_optional(workdir / "omega22.dat"),
        ejv_gw=_load_optional(workdir / "ejv_GW.dat"),
        stdout=proc.stdout,
        stderr=proc.stderr,
        backend="fortran",
        metadata=metadata,
    )


def _load_optional(path: Path) -> Optional[np.ndarray]:
    if not path.exists() or path.stat().st_size == 0:
        return None
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data


def _psi4_sort_key(path: Path) -> Tuple[int, str]:
    suffix = path.name.rsplit(".", 1)[-1]
    try:
        return int(suffix), path.name
    except ValueError:
        return 10**9, path.name


# Fixed-frequency integration.

@dataclass(frozen=True)
class FFIProducts:
    """Arrays written to the reusable GW cache."""

    rhphc: np.ndarray
    rhphcdot: np.ndarray
    omega22: np.ndarray
    ejv_gw: np.ndarray
    times: np.ndarray
    rpsi4_uniform: np.ndarray
    fft_size: int
    dt: float


def _padded_fft_size(sample_count: int) -> int:
    """Match the deliberately generous power-of-two padding in the Fortran."""
    if sample_count < 2:
        raise ValueError("Psi4 conversion requires at least two samples")
    exponent = int(np.log(float(sample_count)) / np.log(2.0) - 1.0e-10) + 3
    return 2**exponent


def _retarded_time(data: np.ndarray, madm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the legacy uniform coordinate-time grid, retarded time, and dt."""
    sample_count = data.shape[0]
    if sample_count < 4:
        raise ValueError("Psi4 conversion requires at least four samples")

    t0 = float(data[0, 0])
    tend = float(data[-1, 0])
    dt = (tend - t0) / (sample_count - 1)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Psi4 coordinate times must span a positive interval")
    coordinate_time = t0 + dt * np.arange(sample_count, dtype=float)

    radius = np.asarray(data[:, -4], dtype=float)
    gtt = np.asarray(data[:, -3], dtype=float)
    gtr = np.asarray(data[:, -2], dtype=float)
    grr = np.asarray(data[:, -1], dtype=float)
    discriminant = gtr * gtr - gtt * grr
    lapse_factor = 1.0 - 2.0 * madm / radius
    if (
        np.any(~np.isfinite(radius))
        or np.any(radius <= 2.0 * madm)
        or np.any(discriminant < 0.0)
        or np.any(gtt == 0.0)
        or np.any(lapse_factor == 0.0)
    ):
        raise ValueError("Invalid radius or inverse-metric data in Psi4 input")

    dtsch_dt = (gtr - np.sqrt(discriminant)) / gtt / lapse_factor
    schwarzschild_time = np.empty(sample_count, dtype=float)
    schwarzschild_time[0] = t0
    schwarzschild_time[1:] = t0 + np.cumsum(
        0.5 * (dtsch_dt[:-1] + dtsch_dt[1:]) * dt
    )
    rstar = radius + 2.0 * madm * np.log(radius / (2.0 * madm) - 1.0)
    tret = schwarzschild_time - rstar
    if np.any(~np.isfinite(tret)) or np.any(np.diff(tret) <= 0.0):
        raise ValueError("Gauge-corrected retarded time is not finite and increasing")
    times = np.column_stack((coordinate_time, schwarzschild_time, tret, rstar))
    return times, tret, dt


def _four_point_interpolate(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Local cubic interpolation matching the Fortran ``polint`` stencil."""
    if source_time.size < 4:
        raise ValueError("Four-point interpolation requires at least four samples")
    interval = np.searchsorted(source_time, target_time, side="right") - 1
    starts = np.clip(interval - 1, 0, source_time.size - 4)
    indices = starts[:, None] + np.arange(4)[None, :]
    x = source_time[indices]
    y = source_values[indices]

    weights = np.ones_like(x)
    for column in range(4):
        for other in range(4):
            if column != other:
                weights[:, column] *= (
                    (target_time - x[:, other])
                    / (x[:, column] - x[:, other])
                )
    return np.sum(weights[:, :, None] * y, axis=1)


def _fixed_frequency_integrate(
    rpsi4: np.ndarray,
    dt: float,
    modes: tuple[tuple[int, int], ...],
    omega_orbital: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate complex ``r Psi4`` once and twice using the legacy FFI filter."""
    sample_count = rpsi4.shape[0]
    spectrum = np.fft.ifft(rpsi4, axis=0) * sample_count
    index = np.arange(sample_count)
    signed_index = np.where(index < sample_count // 2, index, index - sample_count)
    omega = 2.0 * np.pi * signed_index / (sample_count * dt)

    hdot_spectrum = np.empty_like(spectrum)
    h_spectrum = np.empty_like(spectrum)
    for mode_index, (_, emm) in enumerate(modes):
        cutoff = max(abs(emm) * omega_orbital, omega_orbital)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("FFI orbital-frequency cutoff must be positive")
        effective_abs = np.maximum(np.abs(omega), cutoff)
        effective_signed = np.copysign(effective_abs, omega)
        effective_signed[0] = cutoff
        hdot_spectrum[:, mode_index] = 1j * spectrum[:, mode_index] / effective_signed
        h_spectrum[:, mode_index] = -spectrum[:, mode_index] / effective_abs**2

    hdot = np.fft.fft(hdot_spectrum, axis=0) / sample_count
    strain = np.fft.fft(h_spectrum, axis=0) / sample_count
    return strain, hdot


def _mode_table(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    table = np.empty((time.size, 1 + 2 * values.shape[1]), dtype=float)
    table[:, 0] = time
    table[:, 1::2] = values.real
    table[:, 2::2] = values.imag
    return table


def _omega22(
    time: np.ndarray,
    rpsi4: np.ndarray,
    strain: np.ndarray,
    strain_dot: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reproduce the three legacy estimates of the (2,2) angular frequency."""
    rows = []
    phase = np.angle(strain[:, 0])
    for i in range(time.size):
        if abs(rpsi4[i, 0]) == 0.0:
            continue
        next_phase = np.angle(strain[i + 1, 0])
        previous_phase = phase[0] if i == 0 else phase[i - 1]
        delta = next_phase - previous_phase
        if delta < -np.pi:
            delta += 2.0 * np.pi
        if delta > np.pi:
            delta -= 2.0 * np.pi
        if i > 0:
            delta *= 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate_a = -1j * rpsi4[i, 0] / strain_dot[i, 0]
            estimate_b = -1j * strain_dot[i, 0] / strain[i, 0]
        rows.append(
            (
                time[i],
                delta / dt,
                estimate_a.real,
                estimate_a.imag,
                estimate_b.real,
                estimate_b.imag,
            )
        )
    return np.asarray(rows, dtype=float).reshape((-1, 6))


def _radiated_diagnostics(
    time: np.ndarray,
    strain: np.ndarray,
    strain_dot: np.ndarray,
    modes: tuple[tuple[int, int], ...],
    madm: float,
    dt: float,
) -> np.ndarray:
    """Port the legacy energy, angular-momentum, and recoil accumulation."""
    mode_index = {mode: index for index, mode in enumerate(modes)}
    factor = dt / (32.0 * np.pi)
    lmax = max(ell for ell, _ in modes) + 1
    sample_count = time.size
    zero = np.zeros(sample_count, dtype=complex)

    def series(ell, emm):
        index = mode_index.get((ell, emm))
        return zero if index is None else strain_dot[:, index]

    mode_power = np.sum(np.abs(strain_dot) ** 2, axis=1)
    angular_density = np.zeros(sample_count, dtype=float)
    xy_density = np.zeros(sample_count, dtype=complex)
    z_density = np.zeros(sample_count, dtype=float)
    for ell in range(2, lmax):
        for emm in range(-ell, ell + 1):
            index = mode_index.get((ell, emm))
            if index is None:
                continue
            hd = strain_dot[:, index]
            angular_density += emm * np.imag(np.conj(hd) * strain[:, index])

            c_plus = -np.sqrt(
                (ell + emm + 1.0)
                * (ell + emm + 2.0)
                * (ell - 1.0)
                * (ell + 3.0)
                / ((2.0 * ell + 1.0) * (2.0 * ell + 3.0))
            ) / (ell + 1.0)
            c_zero = 2.0 * np.sqrt((ell - emm) * (ell + emm + 1.0)) / (ell * (ell + 1.0))
            c_minus = np.sqrt(
                (ell - emm - 1.0)
                * (ell - emm)
                * (ell - 2.0)
                * (ell + 2.0)
                / ((2.0 * ell - 1.0) * (2.0 * ell + 1.0))
            ) / ell
            d_plus = np.sqrt(
                (ell - emm + 1.0)
                * (ell + emm + 1.0)
                * (ell - 1.0)
                * (ell + 3.0)
                / ((2.0 * ell + 1.0) * (2.0 * ell + 3.0))
            ) / (ell + 1.0)
            d_zero = 2.0 * emm / (ell * (ell + 1.0))

            xy_density += np.conj(hd) * (
                c_plus * series(ell + 1, emm + 1)
                + c_zero * series(ell, emm + 1)
                + c_minus * series(ell - 1, emm + 1)
            )
            z_density += (
                2.0 * d_plus * np.real(np.conj(hd) * series(ell + 1, emm))
                + d_zero * np.abs(hd) ** 2
            )

    def trapezoidal_cumulative(density):
        return factor * np.cumsum(density[:-1] + density[1:])

    energy = trapezoidal_cumulative(mode_power)
    angular_momentum = -trapezoidal_cumulative(angular_density)
    px = trapezoidal_cumulative(xy_density.real)
    py = trapezoidal_cumulative(xy_density.imag)
    pz = trapezoidal_cumulative(z_density)
    velocity_scale = 299792.458 / (madm - energy)
    vx, vy, vz = px * velocity_scale, py * velocity_scale, pz * velocity_scale
    flux = mode_power[:-1] / (16.0 * np.pi)
    return np.column_stack(
        (
            time[1:],
            energy,
            angular_momentum,
            np.sqrt(vx * vx + vy * vy + vz * vz),
            vx,
            vy,
            vz,
            flux,
        )
    )


def reconstruct_strain(
    data: np.ndarray,
    modes: tuple[tuple[int, int], ...],
    omega_orbital: float,
    madm: float,
    t_start: float | None = None,
    t_end: float | None = None,
) -> FFIProducts:
    """Create the complete set of legacy-compatible cached waveform products."""
    data = np.asarray(data, dtype=float)
    expected_columns = 1 + 2 * len(modes) + 4
    if data.ndim != 2 or data.shape[1] != expected_columns:
        raise ValueError(f"Psi4 data must have shape (N, {expected_columns})")
    if np.any(~np.isfinite(data)):
        raise ValueError("Psi4 input contains non-finite values")

    times, tret, dt = _retarded_time(data, madm)
    coordinate_time = times[:, 0]
    if t_start is None:
        t_start = float(coordinate_time[0])
    if t_end is None or t_end >= coordinate_time[-1]:
        t_end = float(coordinate_time[-1] - 0.5 * dt)
    start_index = int(np.clip(np.floor((t_start - coordinate_time[0]) / dt), 0, data.shape[0] - 1))
    end_index = int(np.clip(np.ceil((t_end - coordinate_time[0]) / dt), 0, data.shape[0] - 1))
    if end_index < start_index:
        raise ValueError("Requested Psi4 time interval is empty")

    fft_size = _padded_fft_size(data.shape[0])
    uniform_tret = tret[0] + dt * np.arange(fft_size, dtype=float)
    active = (uniform_tret >= tret[start_index]) & (uniform_tret <= tret[end_index])
    raw_modes = data[:, 1 : 1 + 2 * len(modes) : 2] + 1j * data[:, 2 : 1 + 2 * len(modes) : 2]
    # The legacy routine integrates Re(r Psi4) - i Im(r Psi4), then stores its
    # real and imaginary parts as r h_plus and r h_cross.
    legacy_rpsi4 = data[:, -4, None] * np.conj(raw_modes)
    uniform_legacy = np.zeros((fft_size, len(modes)), dtype=complex)
    uniform_legacy[active] = _four_point_interpolate(tret, legacy_rpsi4, uniform_tret[active])

    strain, strain_dot = _fixed_frequency_integrate(
        uniform_legacy,
        dt,
        modes,
        omega_orbital,
    )
    output_time = uniform_tret[: data.shape[0]]
    output_strain = strain[: data.shape[0]]
    output_strain_dot = strain_dot[: data.shape[0]]
    output_rpsi4 = uniform_legacy[: data.shape[0]]

    times_output = np.column_stack(
        (
            times,
            legacy_rpsi4[:, 0].real,
            -legacy_rpsi4[:, 0].imag,
        )
    )
    return FFIProducts(
        rhphc=_mode_table(output_time, output_strain),
        rhphcdot=_mode_table(output_time, output_strain_dot),
        omega22=_omega22(output_time, uniform_legacy, strain, strain_dot, dt),
        ejv_gw=_radiated_diagnostics(output_time, output_strain, output_strain_dot, modes, madm, dt),
        times=times_output,
        rpsi4_uniform=_mode_table(output_time, np.conj(output_rpsi4)),
        fft_size=fft_size,
        dt=dt,
    )


def write_products(
    products: FFIProducts,
    workdir: Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Write legacy-compatible products plus reusable preprocessing metadata."""
    workdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "rhphc.dat": products.rhphc,
        "rhphcdot.dat": products.rhphcdot,
        "omega22.dat": products.omega22,
        "ejv_GW.dat": products.ejv_gw,
        "times.dat": products.times,
        "rpsi4_uniform.dat": products.rpsi4_uniform,
    }
    for name, values in outputs.items():
        np.savetxt(workdir / name, values, fmt="%25.15E")
    manifest = dict(metadata)
    manifest.update(
        {
            "format_version": 1,
            "backend": "python",
            "fft_size": products.fft_size,
            "dt": products.dt,
        }
    )
    (workdir / "strain_cache.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


# Restart-aware extraction loading and subtraction.

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


# Simulation-level access.

@dataclass
class Waveform:
    """A single extraction radius, with every mode available in its arrays."""

    config: DiskSimConfig
    parfile_index: int
    psi4: Psi4File
    strain_result: StrainResult
    psi4_radius: float | None

    @property
    def rh_t(self):
        return self.strain_result.time


def strain_cache_dir(config, file_label, reference_name=None):
    name = f"{config.name}_psi4{file_label}"
    if reference_name is not None:
        name += f"_minus_{reference_name}"
    return GW_WORK_ROOT / name


class GWRun:
    """Read extraction files once and reuse each radius's cached waveform."""

    def __init__(self, config):
        self.config = config
        self.psi4_files, self.radii = load_psi4(config.data_roots)
        self.waveforms = {}

    def at_radius(self, parfile_index):
        index = int(parfile_index)
        if index not in self.waveforms:
            label = str(index + self.config.gw_psi4_file_index_offset)
            if label not in self.psi4_files:
                raise FileNotFoundError(f"{self.config.name}: missing Psi4 extraction {label} for radius index {index}")
            psi4 = self.psi4_files[label]
            strain = read_strain_cache(strain_cache_dir(self.config, label), psi4.path)
            self.waveforms[index] = Waveform(self.config, index, psi4, strain, self.radii.get(int(label)-1))
        return self.waveforms[index]


def read_difference(case, reference):
    """Read h(Psi4_case - Psi4_reference); never reconstruct while plotting."""
    path = strain_cache_dir(case.config, case.psi4.label, reference.config.name)
    return read_strain_cache(path, case.psi4.path)


def load_gw_sims(names=None, psi4_parfile_index=None, skip_missing=True):
    from helpers.style import apply_styles

    waveforms = []
    for config in all_sim_configs(names):
        index = config.psi4_parfile_index if psi4_parfile_index is None else psi4_parfile_index
        try:
            waveforms.append(GWRun(config).at_radius(index))
        except (OSError, ValueError, IndexError) as exc:
            if not skip_missing:
                raise
            print(f"{config.name}: skipping; {exc}")
    return apply_styles(waveforms)


def names_with_massless(names):
    if names is None:
        return None
    out = []
    seen = set()
    for name in list(names) + [MASSLESS_SIM_NAME]:
        key = str(name).upper()
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def subtract_waveforms(case_t, case_values, reference_t, reference_values):
    """Complex mode difference on the shared case grid, never extrapolating."""
    def clean(t, values):
        t, values = np.asarray(t, dtype=float), np.asarray(values, dtype=complex)
        n = min(t.size, values.size)
        t, values = t[:n], values[:n]
        keep = np.isfinite(t) & np.isfinite(values)
        order = np.argsort(t[keep], kind="stable")
        t, values = t[keep][order], values[keep][order]
        _, unique = np.unique(t, return_index=True)
        return t[unique], values[unique]

    case_t, case_values = clean(case_t, case_values)
    reference_t, reference_values = clean(reference_t, reference_values)
    if min(case_t.size, reference_t.size) < 2:
        return None
    keep = ((case_t >= max(case_t[0], reference_t[0])) &
            (case_t <= min(case_t[-1], reference_t[-1])))
    if np.count_nonzero(keep) < 2:
        return None
    t = case_t[keep]
    interpolated = (np.interp(t, reference_t, reference_values.real) +
                    1j * np.interp(t, reference_t, reference_values.imag))
    return t, case_values[keep] - interpolated


def loaded_areal_radius(sim):
    psi4 = getattr(sim, "psi4", None)
    if psi4 is None:
        return None
    radius = np.nanmedian(np.asarray(psi4.r_areal, dtype=float))
    return float(radius) if np.isfinite(radius) else None


def same_loaded_radius(sim, reference, rtol=5.0e-3, atol=1.0):
    radius = loaded_areal_radius(sim)
    reference_radius = loaded_areal_radius(reference)
    if radius is None or reference_radius is None:
        return True
    return bool(np.isclose(radius, reference_radius, rtol=rtol, atol=atol))
