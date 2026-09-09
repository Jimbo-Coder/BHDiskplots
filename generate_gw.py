#!/usr/bin/env python3
"""Generate all-radius strain, then shared-radius disk-minus-ML strain.

Settings are in config.py. Source files are read-only; products go in gw_work/.
"""
from __future__ import annotations

import shlex
import shutil
import subprocess

from config import FORTRAN_GW_ROOT, GW_WORK_ROOT, all_sim_configs
from config import GW_DIFFERENCE_PARFILE_INDICES, MASSLESS_SIM_NAME
from gw import (
    GWRun, generate_strain, read_strain_cache, strain_cache_dir,
    subtract_psi4_on_retarded_time,
)
from config import (
    GW_SIM_NAMES, GW_PLOT_MODES, GW_GENERATE_PSI4_DIFFERENCES,
    GW_STRAIN_BACKEND, GW_REGENERATE_EXISTING, GW_STOP_ON_ERROR, GW_IFORT_MODULE,
)


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
            f"module load {shlex.quote(GW_IFORT_MODULE)} >/dev/null && "
            f"ifort -o {shlex.quote(str(executable))} {shlex.quote(str(source))}"
        )
        command = ["bash", "-lc", build]
    try:
        subprocess.run(command, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Could not build {executable} with ifort or gfortran. "
            f"Load {GW_IFORT_MODULE} or another supported compiler and retry."
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
    for ell, emm in GW_PLOT_MODES:
        try:
            hplus, hcross = result.hplus_hcross(ell=ell, emm=emm)
        except (IndexError, KeyError, ValueError) as exc:
            warnings.append(f"({ell},{emm}): {exc}")
            continue
        if hplus.size == 0 or hcross.size == 0:
            warnings.append(f"({ell},{emm}): empty strain columns")
    return warnings


def generate_product(psi4_file, config, workdir):
    if not GW_REGENERATE_EXISTING and (workdir / "rhphc.dat").is_file():
        result = read_strain_cache(workdir, psi4_file.path)
    else:
        result = generate_strain(
            psi4_file, workdir, config.gw_omega_orbital, config.gw_madm,
            backend=GW_STRAIN_BACKEND,
        )
    warnings = _mode_warnings(result)
    if warnings:
        print(f"{config.name}: mode warnings: {'; '.join(warnings)}")
    print(
        f"{config.name}: cached {result.time.size} samples with "
        f"the {result.backend} backend"
    )
    return result


def generate_simulation(sim):
    if sim.config.gw_omega_orbital is None or sim.config.gw_madm is None:
        raise ValueError("missing gw_omega_orbital or gw_madm")
    if not sim.psi4_files:
        raise ValueError("no Psi4 extraction files")

    processed = 0
    for label, psi4_file in sorted(sim.psi4_files.items(), key=_label_sort_key):
        workdir = strain_cache_dir(sim.config, label)
        print(f"{sim.config.name}: Psi4 {label} -> {workdir}")
        generate_product(psi4_file, sim.config, workdir)
        processed += 1
    return processed


def generate_difference_radius(runs, parfile_index):
    massless = runs[MASSLESS_SIM_NAME].at_radius(parfile_index)
    processed = 0
    for name, run in runs.items():
        if name == MASSLESS_SIM_NAME:
            continue
        sim = run.at_radius(parfile_index)
        difference = subtract_psi4_on_retarded_time(
            sim.psi4, sim.rh_t, massless.psi4, massless.rh_t,
            label=f"{sim.psi4.label}_minus_{MASSLESS_SIM_NAME}",
        )
        result = generate_product(difference, sim.config,
                                  strain_cache_dir(sim.config, sim.psi4.label, MASSLESS_SIM_NAME))
        print(f"{name}: Psi4-{MASSLESS_SIM_NAME} -> {result.workdir}")
        processed += 1
    return processed


def main():
    if GW_STRAIN_BACKEND == "fortran":
        executable = ensure_fortran_executable()
        print(f"Fortran GW executable: {executable}")
    print(f"GW strain backend: {GW_STRAIN_BACKEND}")
    print(f"GW cache root: {GW_WORK_ROOT}")
    print(f"regenerate existing cache: {GW_REGENERATE_EXISTING}")

    processed = 0
    failures = 0
    runs = {}
    for config in all_sim_configs(GW_SIM_NAMES):
        try:
            run = GWRun(config)
            processed += generate_simulation(run)
            runs[config.name] = run
        except Exception as exc:
            failures += 1
            print(f"{config.name}: GW generation failed: {exc}")
            if GW_STOP_ON_ERROR:
                raise

    if GW_GENERATE_PSI4_DIFFERENCES:
        for parfile_index in GW_DIFFERENCE_PARFILE_INDICES:
            try:
                processed += generate_difference_radius(runs, parfile_index)
            except Exception as exc:
                failures += 1
                print(f"Psi4 difference index {parfile_index} failed: {exc}")
                if GW_STOP_ON_ERROR:
                    raise

    print(f"GW generation complete: processed={processed}, failures={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
