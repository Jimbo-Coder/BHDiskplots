#!/usr/bin/env python3
"""Run the complete non-movie BHDisk post-processing workflow."""
from __future__ import annotations

from pathlib import Path
import subprocess
import sys


from config import (
    RUN_GW_CACHE, RUN_PAPER_PLOTS, RUN_WIP_PLOTS,
    RUN_INDIVIDUAL_PLOTS, RUN_INDIVIDUAL_EXTRAS, INDIVIDUAL_CASES,
)


ROOT = Path(__file__).resolve().parent


def run_stage(label, script, *args):
    command = [sys.executable, str(ROOT / script), *args]
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main(*, run_gw_cache=None):
    run_gw_cache = RUN_GW_CACHE if run_gw_cache is None else run_gw_cache
    if run_gw_cache:
        import generate_gw

        print("\n=== GW cache ===", flush=True)
        generate_gw.main()

    if RUN_PAPER_PLOTS:
        run_stage("paper plots", "run_paper.py")

    if RUN_WIP_PLOTS:
        run_stage("combined WIP plots", "run_wip.py")

    if RUN_INDIVIDUAL_PLOTS:
        args = list(INDIVIDUAL_CASES)
        if RUN_INDIVIDUAL_EXTRAS:
            args.append("--extra")
        run_stage("individual plots", "wip_plots/run_individual.py", *args)

    print("\nComplete non-movie workflow finished.")


if __name__ == "__main__":
    main()
