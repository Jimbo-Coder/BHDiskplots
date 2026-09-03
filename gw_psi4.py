import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from config import FORTRAN_GW_ROOT, GW_WORK_ROOT
from helpers.gw_ffi import reconstruct_strain, write_products


N_PSI4_COLUMNS = 47
N_PSI4_MODES = (N_PSI4_COLUMNS - 5) // 2
PSI4_HLM_DIR = FORTRAN_GW_ROOT
DEFAULT_GW_WORK_ROOT = GW_WORK_ROOT
DEFAULT_GW_STRAIN_BACKEND = "python"


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

    @property
    def gtt(self) -> np.ndarray:
        return self.data[:, -3]

    @property
    def gtr(self) -> np.ndarray:
        return self.data[:, -2]

    @property
    def grr(self) -> np.ndarray:
        return self.data[:, -1]

    def psi4(self, ell: int = 2, emm: int = 2, multiply_by_r: bool = False) -> np.ndarray:
        re_col, im_col = mode_columns(ell, emm)
        z = self.data[:, re_col] + 1j * self.data[:, im_col]
        if multiply_by_r:
            z = self.r_areal * z
        return z

    def summary(self) -> Dict[str, float]:
        return {
            "rows": len(self.data),
            "t_min": float(self.time[0]),
            "t_max": float(self.time[-1]),
            "dt_median": float(np.median(np.diff(self.time))) if len(self.time) > 1 else np.nan,
            "r_start": float(self.r_areal[0]),
            "r_end": float(self.r_areal[-1]),
            "r_median": float(np.median(self.r_areal)),
            "repeated_times_removed": int(self.repeated_times),
        }


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


def psi4_file_label_for_index(index: int) -> str:
    """Map zero-based parfile ``radius_GW_Psi4[index]`` to a loaded Psi4 label.

    The run writes parfile index 0 as ``Psi4_rad.mon.1`` and parfile index 9
    as the literal ``Psi4_rad.mon.*`` file, loaded here under label ``"10"``.
    """
    index = int(index)
    if index < 0:
        raise ValueError(f"Psi4 parfile index must be >= 0, got {index}")
    return "10" if index == 9 else str(index + 1)
def write_sorted_psi4_input(psi4_file: Psi4File, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, psi4_file.data, fmt="%25.15E")
    return out_path


def _load_strain_cache(workdir: Path, psi4_input: Path) -> StrainResult:
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


def convert_to_strain_with_python(
    psi4_file: Psi4File,
    workdir: Path,
    omega_orbital: float,
    madm: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    reuse_existing: bool = True,
    generate_if_missing: bool = True,
) -> StrainResult:
    """Generate or reuse legacy-compatible strain products with NumPy."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    if reuse_existing and (workdir / "rhphc.dat").is_file():
        return _load_strain_cache(workdir, psi4_file.path)
    if not generate_if_missing:
        raise FileNotFoundError(
            f"Missing cached strain product: {workdir / 'rhphc.dat'}. "
            "Run generate_gw.py before plotting."
        )

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


def convert_to_strain(
    psi4_file: Psi4File,
    workdir: Path,
    omega_orbital: float,
    madm: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    reuse_existing: bool = True,
    generate_if_missing: bool = True,
    backend: str = DEFAULT_GW_STRAIN_BACKEND,
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
        reuse_existing=reuse_existing,
        generate_if_missing=generate_if_missing,
    )
    if backend == "python":
        return convert_to_strain_with_python(**common)
    if backend == "fortran":
        return convert_to_strain_with_rhphc(**common)
    raise ValueError(f"Unknown GW strain backend {backend!r}; expected 'python' or 'fortran'")


def convert_to_strain_with_rhphc(
    psi4_file: Psi4File,
    workdir: Path,
    omega_orbital: float,
    madm: float,
    t_start: Optional[float] = None,
    t_end: Optional[float] = None,
    psi4_hlm_dir: Path = PSI4_HLM_DIR,
    executable: str = "rhphc",
    reuse_existing: bool = True,
    generate_if_missing: bool = True,
) -> StrainResult:
    """Run the preserved psi4_hlm_ref rhphc executable.

    This preserves the old Fortran FFI algorithm. The only Python-side changes are:
    sort/remove repeated coordinate times, write ccc_ffi.input, and run in a
    separate work directory so Milton's scratch tree is never modified.
    """
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    psi4_input = workdir / f"Psi4_rad.mon_sorted.{psi4_file.label}"

    if reuse_existing and (workdir / "rhphc.dat").exists():
        return _load_strain_cache(workdir, psi4_input)

    if not generate_if_missing:
        raise FileNotFoundError(
            f"Missing cached Fortran strain product: {workdir / 'rhphc.dat'}. "
            "Run generate_gw.py before plotting."
        )

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


def summarize_psi4_files(files: Mapping[str, Psi4File]) -> List[Dict[str, object]]:
    rows = []
    for label, psi4_file in files.items():
        row = {"label": label, "kind": psi4_file.source_kind, "path": str(psi4_file.path)}
        row.update(psi4_file.summary())
        rows.append(row)
    return rows


def selected_psi4_mode(
    psi4_file: Psi4File,
    ell: int,
    emm: int,
    multiply_by_r: bool = True,
) -> np.ndarray:
    """Return one explicit (ell, m) Psi4 mode from a loaded extraction file."""
    return psi4_file.psi4(ell=ell, emm=emm, multiply_by_r=multiply_by_r)


def plot_raw_rpsi4_mode(
    files: Mapping[str, Psi4File],
    ell: int,
    emm: int,
    ax=None,
    labels: Optional[Iterable[str]] = None,
):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    selected = labels if labels is not None else files.keys()
    for label in selected:
        psi4_file = files[label]
        z = selected_psi4_mode(psi4_file, ell=ell, emm=emm, multiply_by_r=True)
        ax.plot(psi4_file.time, z.real, label=f"{label} Re")
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(rf"$r\,\mathrm{{Re}}(\Psi_4^{{{ell}{emm}}})$")
    ax.legend()
    ax.grid()
    return ax


def plot_raw_rpsi4_22(files: Mapping[str, Psi4File], ax=None, labels: Optional[Iterable[str]] = None):
    return plot_raw_rpsi4_mode(files, ell=2, emm=2, ax=ax, labels=labels)


def plot_rhphc_mode(result: StrainResult, ell: int, emm: int, ax=None):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    hp, hc = result.hplus_hcross(ell=ell, emm=emm)
    ax.plot(result.time, hp, label=rf"$r h_+^{{{ell}{emm}}}$")
    ax.plot(result.time, hc, label=rf"$r h_\times^{{{ell}{emm}}}$")
    ax.set_xlabel(r"$t_\mathrm{ret}$")
    ax.set_ylabel(r"$r h$")
    ax.legend()
    ax.grid()
    return ax


def plot_rhphc_22(result: StrainResult, ax=None):
    return plot_rhphc_mode(result, ell=2, emm=2, ax=ax)


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
