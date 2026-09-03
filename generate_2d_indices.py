#!/usr/bin/env python3
"""Populate the repository 2D iteration cache for every disk simulation."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import CACHE_ROOT
from helpers.reader import load_sims
from paper_plots.rho2d_individual import valid_rho2d_iteration_infos

# Operational knobs.
CASES = ("A1", "A2", "A3", "B1", "B2", "B3")
# Each worker can stream a 100+ GiB ASCII source. Keep the default conservative
# for login-node memory and filesystem load; completed indices are reused.
MAX_WORKERS = 1
REGENERATE_2D_PLOTS = True
PANEL_LAYOUT = "all"


def build_one(sim):
    start = time.perf_counter()
    infos = valid_rho2d_iteration_infos(sim)
    return sim.config.name, len(infos), time.perf_counter() - start


def main():
    sims = load_sims([], names=CASES)
    print(f"2D cache: {CACHE_ROOT / '2d_indices'}", flush=True)
    print(f"cases: {', '.join(CASES)}; workers: {MAX_WORKERS}", flush=True)

    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        pending = {executor.submit(build_one, sim): sim.config.name for sim in sims}
        for future in as_completed(pending):
            name = pending[future]
            try:
                _, frame_count, elapsed = future.result()
            except Exception as error:
                failures.append((name, error))
                print(f"{name}: FAILED: {error}", flush=True)
            else:
                print(
                    f"{name}: cached {frame_count} valid frames in {elapsed:.1f} s",
                    flush=True,
                )

    if failures:
        names = ", ".join(name for name, _ in failures)
        raise RuntimeError(f"2D cache generation failed for: {names}")
    print("2D cache complete", flush=True)

    if REGENERATE_2D_PLOTS:
        print("regenerating individual 2D snapshots", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "paper_plots/rho2d_individual.py",
                "--sims",
                *CASES,
            ],
            check=True,
        )
        print("regenerating 2D panels", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-u",
                "paper_plots/rho2d_panels.py",
                PANEL_LAYOUT,
            ],
            check=True,
        )
        print("2D plot regeneration complete", flush=True)


if __name__ == "__main__":
    main()
