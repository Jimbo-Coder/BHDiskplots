#!/usr/bin/env python3
"""Generate the reusable GW time-series cache.

The maintained NumPy backend follows the established ``rhphc`` algorithm and
writes compatible products. The original Fortran executable remains available
as a regression reference. Plot scripts never regenerate data implicitly.

Two products are generated:

1. Each simulation's Psi4 at every available extraction radius.
2. Disk-minus-massless Psi4 at the configured shared radii.

Simulation source directories are read only. All derived files are written
below the repository-local ``GW_WORK_ROOT``.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess

from config import FORTRAN_GW_ROOT, GW_WORK_ROOT, all_sim_configs
from helpers.gw_difference import GW_DIFFERENCE_PARFILE_INDICES, MASSLESS_SIM_NAME
from helpers.reader import DiskSim
from helpers.reader_gw import convert_psi4_to_strain


# Scientific/data-selection knobs.
SIM_NAMES = ("A1", "A2", "A3", "B1", "B2", "B3", "ML")
VALIDATE_MODES = ((2, 2), (2, 1), (2, 0), (4, 0))
GENERATE_PSI4_DIFFERENCES = True
# Psi4-to-strain producer: "python" is the maintained NumPy implementation;
# "fortran" retains the original executable as a regression reference.
GW_STRAIN_BACKEND = "python"

# Operational knobs.
# A requested cache pass rebuilds all products so newly appended source data
# cannot be hidden behind an older complete cache.
REGENERATE_EXISTING = True
STOP_ON_ERROR = False
IFORT_MODULE = "intel/19.0.5.281"


def ensure_fortran_executable():
    executable = FORTRAN_GW_ROOT / "rhphc"
    if executable.is_file():
        return executable

    source = FORTRAN_GW_ROOT / "ccc_ffi_hplus_hcross_ejkick.f90"
    if not source.is_file():
        raise FileNotFoundError(f"Fortran GW source not found: {source}")

    print(f"building {executable.name} from {source.name}")
    compiler = shutil.which("ifort") or shutil.which("gfortran")
    if compiler is not None:
        command = [compiler, "-o", str(executable), str(source)]
    else:
        build = (
            f"module load {shlex.quote(IFORT_MODULE)} >/dev/null && "
            f"ifort -o {shlex.quote(str(executable))} {shlex.quote(str(source))}"
        )
        command = ["bash", "-lc", build]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Could not build {executable} with ifort or gfortran. "
            f"Load {IFORT_MODULE} or another supported compiler and retry."
        ) from error
    return executable


def _label_sort_key(item):
    label, _ = item
    try:
        return int(label)
    except ValueError:
        return 10**9


def _mode_warnings(result):
    warnings = []
    for ell, emm in VALIDATE_MODES:
        try:
            hplus, hcross = result.hplus_hcross(ell=ell, emm=emm)
        except (IndexError, KeyError, ValueError) as exc:
            warnings.append(f"({ell},{emm}): {exc}")
            continue
        if hplus.size == 0 or hcross.size == 0:
            warnings.append(f"({ell},{emm}): empty strain columns")
    return warnings


def _convert(psi4_file, sim, workdir):
    result = convert_psi4_to_strain(
        psi4_file,
        workdir=workdir,
        omega_orbital=sim.config.gw_omega_orbital,
        madm=sim.config.gw_madm,
        regenerate=REGENERATE_EXISTING,
        generate_if_missing=True,
        backend=GW_STRAIN_BACKEND,
    )
    warnings = _mode_warnings(result)
    if warnings:
        print(f"{sim.config.name}: mode warnings: {'; '.join(warnings)}")
    print(
        f"{sim.config.name}: cached {result.time.size} samples with "
        f"the {result.backend} backend"
    )
    return result


def generate_simulation(sim):
    if sim.config.gw_omega_orbital is None or sim.config.gw_madm is None:
        raise ValueError("missing gw_omega_orbital or gw_madm")
    if not sim.load_psi4():
        raise ValueError("no Psi4 extraction files")

    processed = 0
    for label, psi4_file in sorted(sim.psi4_files.items(), key=_label_sort_key):
        workdir = sim.gw_workdir(label)
        print(f"{sim.config.name}: Psi4 {label} -> {workdir}")
        _convert(psi4_file, sim, workdir)
        processed += 1
    return processed


def generate_difference_radius(parfile_index):
    sims = {config.name: DiskSim(config) for config in all_sim_configs(SIM_NAMES)}
    massless = sims[MASSLESS_SIM_NAME]
    if not massless.load_strain(
        regenerate_gw=False,
        psi4_parfile_index=parfile_index,
        psi4_mode=VALIDATE_MODES[0],
    ):
        raise ValueError(f"could not load {MASSLESS_SIM_NAME} at index {parfile_index}")

    processed = 0
    for name in SIM_NAMES:
        if name == MASSLESS_SIM_NAME:
            continue
        sim = sims[name]
        if not sim.load_strain(
            regenerate_gw=False,
            psi4_parfile_index=parfile_index,
            psi4_mode=VALIDATE_MODES[0],
        ):
            raise ValueError(f"{name}: could not load strain at index {parfile_index}")
        result = sim.strain_from_psi4_difference(
            massless,
            regenerate_gw=REGENERATE_EXISTING,
            generate_if_missing=True,
            backend=GW_STRAIN_BACKEND,
        )
        warnings = _mode_warnings(result)
        print(f"{name}: Psi4-{MASSLESS_SIM_NAME} -> {result.workdir}")
        if warnings:
            print(f"{name}: mode warnings: {'; '.join(warnings)}")
        processed += 1
    return processed


def main():
    if GW_STRAIN_BACKEND == "fortran":
        executable = ensure_fortran_executable()
        print(f"Fortran GW executable: {executable}")
    print(f"GW strain backend: {GW_STRAIN_BACKEND}")
    print(f"GW cache root: {GW_WORK_ROOT}")
    print(f"regenerate existing cache: {REGENERATE_EXISTING}")

    processed = 0
    failures = 0
    for config in all_sim_configs(SIM_NAMES):
        try:
            processed += generate_simulation(DiskSim(config))
        except Exception as exc:
            failures += 1
            print(f"{config.name}: GW generation failed: {exc}")
            if STOP_ON_ERROR:
                raise

    if GENERATE_PSI4_DIFFERENCES:
        for parfile_index in GW_DIFFERENCE_PARFILE_INDICES:
            try:
                processed += generate_difference_radius(parfile_index)
            except Exception as exc:
                failures += 1
                print(f"Psi4 difference index {parfile_index} failed: {exc}")
                if STOP_ON_ERROR:
                    raise

    print(f"GW generation complete: processed={processed}, failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
