#!/usr/bin/env python3

"""Run every individual BHDisk plot script for one or more target simulations."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PLOTS_DIR
from paper_plots import rho2d_individual
from wip_plots.gw_waveforms import individual_main
from gw_detectability import validation_main

def main(argv=None):
    parser = argparse.ArgumentParser(description="Run every individual plot for one or more simulations.")
    parser.add_argument("sims", nargs="+", help="Target simulation names, e.g. A1 B2 ML, or all for A1-A3/B1-B3")
    parser.add_argument("--outdir", type=Path, default=None, help="Output root; each simulation is saved in its own subdirectory")
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Also run extra/diagnostic GW detectability plots per simulation.",
    )
    parser.add_argument("--no-save", action="store_true", help="Run plots without saving figures")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    args = parser.parse_args(argv)
    if "all" in args.sims and args.sims != ["all"]:
        parser.error("Use 'all' on its own, or list simulation names")
    sims = ["A1", "A2", "A3", "B1", "B2", "B3"] if args.sims == ["all"] else args.sims
    outdir = args.outdir if args.outdir is not None else PLOTS_DIR
    for sim in sims:
        module_argv = ["--sims", sim, "--outdir", str(outdir)]
        if args.no_save:
            module_argv.append("--no-save")
        if args.show:
            module_argv.append("--show")
        print(f"running rho2d_individual for {sim}")
        rho2d_individual.main(module_argv)

        individual_main(module_argv)

        if args.extra:
            print(f"running detectability validation for {sim}")
            validation_main(module_argv)


if __name__ == "__main__":
    main()
