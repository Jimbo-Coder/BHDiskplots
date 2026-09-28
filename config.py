"""Configuration for BHDisk diagnostic plots.

Simulation constants live next to the simulation path so adding/updating a run
never requires counting through parallel arrays.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

# Repository-owned paths. Generated products never escape the checkout.
REPOSITORY_ROOT = Path(__file__).resolve().parent
PLOTS_DIR = REPOSITORY_ROOT / "figures"
CACHE_ROOT = REPOSITORY_ROOT / "cache"
GW_WORK_ROOT = REPOSITORY_ROOT / "gw_work"
FORTRAN_GW_ROOT = REPOSITORY_ROOT / "psi4_hlm_ref"
INITIAL_DATA_ROOT = REPOSITORY_ROOT / "data" / "initial_profiles"

# External, read-only simulation data. These are the only machine-specific
# roots; every case path below is relative to one of them.
MILTON_DATA_ROOT = Path("/anvil/scratch/x-ruizm/BHdisk_2025")
SUPPLEMENTAL_DATA_ROOT = Path(
    "/anvil/projects/x-mca99s008/BHDisk_illinois_maxjamie"
)
EXTRA2D_ROOT = SUPPLEMENTAL_DATA_ROOT / "extra2d"

# Shared GW radius policy. Index 8 is the outermost valid common extraction
# sphere and is the production default. Index 4 is the first sphere retained
# as a defensible wave-zone comparison with longer time coverage.
GW_FIRST_WAVEZONE_PARFILE_INDEX = 4
GW_OUTERMOST_PARFILE_INDEX = 8
GW_COMPARISON_PARFILE_INDICES = (
    GW_FIRST_WAVEZONE_PARFILE_INDEX,
    GW_OUTERMOST_PARFILE_INDEX,
)
PSI4_PARFILE_INDEX = GW_OUTERMOST_PARFILE_INDEX
PSI4_MODE = (2, 2)
GW_PLOT_MODES = ((2, 2), (2, 1), (2, 0), (4, 0))
GW_MULTIMODE_PLOT_MODES = ((2, 2), (2, 0), (2, 1), (4, 0))
GW_MULTIMODE_LEFT_MODES = ((2, 0), (4, 0))
GW_MULTIMODE_RIGHT_MODES = ((2, 2), (2, 1))
GW_DIFFERENCE_PARFILE_INDICES = GW_COMPARISON_PARFILE_INDICES
GW_DIFFERENCE_OUTPUT_SUBDIR = "gw/difference"
MASSLESS_SIM_NAME = "ML"


# Shared units and workflow. Plot-specific limits and layout stay in each plot.
TIME_NORMALIZE_BY_PC = True
TIME_CODE_UNIT_MASS_MSUN = 1.0
GW_NORMALIZE_BY_M = True
GW_TIME_SCALE = "M_BH"  # "M_BH", "M_ADM", "P_c", or "code"

RUN_GW_CACHE = True
RUN_PAPER_PLOTS = True
RUN_WIP_PLOTS = True
RUN_INDIVIDUAL_PLOTS = True
RUN_INDIVIDUAL_EXTRAS = True
INDIVIDUAL_CASES = ("all",)

# Explicit cache generation includes new source data; plotting only reads caches.
GW_SIM_NAMES = ("A1", "A2", "A3", "B1", "B2", "B3", "ML")
GW_STRAIN_BACKEND = "python"  # "fortran" is the regression reference.
GW_REGENERATE_EXISTING = True
GW_GENERATE_PSI4_DIFFERENCES = True
GW_STOP_ON_ERROR = False
GW_IFORT_MODULE = "intel/19.0.5.281"


# Detectability-specific scientific choices live at the top of gw_detectability.py.


@dataclass(frozen=True)
class DiskSimConfig:
    name: str
    # Ordered oldest to newest. Later roots replace overlapping restart data.
    data_roots: tuple[Path, ...]
    q: float
    gamma: float
    kappa: float
    # Initial M0_disk / M_BH,0 from notes/reference/restmass.txt; dimensionless.
    disk_to_bh_mass_ratio: float
    Pc: float
    # Initial central-BH mass scale in the simulation's geometrized code units.
    # Keep this distinct from gw_madm, the total ADM mass also used when
    # normalizing paper coordinates by M.
    mlittle: float = 0.05
    gw_omega_orbital: float | None = None
    gw_madm: float | None = None
    initial_data_path: Path | None = None
    psi4_parfile_index: int = PSI4_PARFILE_INDEX
    gw_psi4_file_index_offset: int = 1
    legend: bool = True
    # Directories containing variable.plane.asc files. These only add missing
    # 2D snapshots; they never contribute scalar or GW diagnostics.
    supplemental_2d_paths: tuple[Path, ...] = ()

    @property
    def disk_rest_mass_code(self) -> float:
        """Initial disk rest mass in G = c = M_sun = 1 code units."""
        return self.disk_to_bh_mass_ratio * self.mlittle

    @property
    def data_path(self) -> Path:
        """Primary source retained for existing readers and path inference."""
        return self.data_roots[0]

    @property
    def continuation_data_paths(self) -> tuple[Path, ...]:
        """Later restart roots retained for existing readers."""
        return self.data_roots[1:]

    def with_initial_data_path(self) -> "DiskSimConfig":
        if self.initial_data_path is not None:
            return self
        return replace(self, initial_data_path=initial_data_path_from_run(self.data_path))


def initial_data_path_from_run(data_path: Path) -> Path:
    data_path = Path(str(data_path).rstrip("/"))
    run_name = data_path.name
    if run_name == "data":
        run_name = data_path.parent.name
    for suffix in ["_higherres", "_first", "_v2", "_PunctureTracker"]:
        if run_name.endswith(suffix):
            run_name = run_name[:-len(suffix)]
    return INITIAL_DATA_ROOT / run_name


DISK_SIMS = [
    DiskSimConfig(
        name="A1",
        data_roots=(MILTON_DATA_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_01",),
        q=1.99,
        gamma=1.6,
        kappa=1.0,
        disk_to_bh_mass_ratio=1.87e-2,
        Pc=16.29,
        gw_omega_orbital=0.385761187900437,
        gw_madm=0.0512207666580524,
    ),
    DiskSimConfig(
        name="A2",
        data_roots=(MILTON_DATA_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_05",),
        supplemental_2d_paths=(
            EXTRA2D_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_05/beta100",
        ),
        q=1.99,
        gamma=1.6,
        kappa=1.0,
        disk_to_bh_mass_ratio=1.98e-1,
        Pc=18.36,
        gw_omega_orbital=0.342298350854386,
        gw_madm=0.0603349020955639,
    ),
    DiskSimConfig(
        name="A3",
        data_roots=(MILTON_DATA_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_07",),
        supplemental_2d_paths=(
            EXTRA2D_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.99_l4.10_r0.40_sol_07/beta100",
        ),
        q=1.99,
        gamma=1.6,
        kappa=1.0,
        disk_to_bh_mass_ratio=6.02e-1,
        Pc=20.95,
        gw_omega_orbital=0.299915740314243,
        gw_madm=0.0807943547824248,
    ),
    DiskSimConfig(
        name="B1",
        data_roots=(MILTON_DATA_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_28",),
        supplemental_2d_paths=(
            EXTRA2D_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_28/beta100",
        ),
        q=1.85,
        gamma=1.6,
        kappa=1.0,
        disk_to_bh_mass_ratio=1.07e-2,
        Pc=17.27,
        gw_omega_orbital=0.363729746185521,
        gw_madm=0.0508101490724165,
    ),
    DiskSimConfig(
        name="B2",
        data_roots=(MILTON_DATA_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_32",),
        supplemental_2d_paths=(
            EXTRA2D_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_32/beta100",
        ),
        q=1.85,
        gamma=1.6,
        kappa=1.0,
        disk_to_bh_mass_ratio=1.21e-1,
        Pc=20.56,
        gw_omega_orbital=0.305581353575123,
        gw_madm=0.0564177477296656,
    ),
    DiskSimConfig(
        name="B3",
        data_roots=(MILTON_DATA_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_35",),
        supplemental_2d_paths=(
            EXTRA2D_ROOT / "bhtD2.0_K1_g1.60_fAJS0.80_000_000_q1.85_l3.50_r0.40_sol_35/beta100",
        ),
        q=1.85,
        gamma=1.6,
        kappa=1.0,
        disk_to_bh_mass_ratio=6.65e-1,
        Pc=24.39,
        gw_omega_orbital=0.257573177238334,
        gw_madm=0.0840478939414561,
    ),
    DiskSimConfig(
        name="ML",
        data_roots=(
            SUPPLEMENTAL_DATA_ROOT / "jamiescalars/bhtD2.0_fAJS0.80_000_000_q2.00_l4.00_r0.40_gamma4o3_sol_01_v2/data",
            MILTON_DATA_ROOT / "massless",
        ),
        q=2.00,
        gamma=4/3,
        kappa=1.45708,
        disk_to_bh_mass_ratio=1.8e-7,
        Pc=14.24,
        gw_omega_orbital=0.441137056341708,
        gw_madm=0.05,
        gw_psi4_file_index_offset=0,
    ),
]


def all_sim_configs(names: Iterable[str] | None = None) -> list[DiskSimConfig]:
    selected = None if names is None else {name.upper() for name in names}
    configs = [cfg.with_initial_data_path() for cfg in DISK_SIMS]
    if selected is None:
        return configs
    return [cfg for cfg in configs if cfg.name.upper() in selected]
