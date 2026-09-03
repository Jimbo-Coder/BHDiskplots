#!/usr/bin/env python3

"""Run every individual BHDisk plot script for one or more target simulations."""
from __future__ import annotations

import argparse
from importlib import import_module
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PLOTS_DIR

# Critical knobs.
ALL_SIM_ALIASES = {"all": ["A1", "A2", "A3", "B1", "B2", "B3"]}

INDIVIDUAL_PLOT_MODULES = [
    ("paper_plots", "rho2d_individual"),
    ("wip_plots", "gw_psi4_radii_individual"),
    ("wip_plots", "gw_strain_radii_individual"),
    ("wip_plots", "gw_psi4_modes_individual"),
    ("wip_plots", "gw_strain_modes_individual"),
]

INDIVIDUAL_EXTRA_MODULES = [
    ("wip_plots", "gw_detectability_diagnostics_individual"),
]

# Operational knobs.
UNICODE_OPTION_DASHES = "\u2012\u2013\u2014\u2015\u2212"


def normalize_option_dashes(argv):
    normalized = []
    for arg in argv:
        if arg and arg[0] in UNICODE_OPTION_DASHES:
            normalized.append("--" + arg.lstrip(UNICODE_OPTION_DASHES))
        else:
            normalized.append(arg)
    return normalized


def expand_sim_args(sims):
    expanded = []
    seen = set()
    for sim in sims:
        for name in ALL_SIM_ALIASES.get(sim.lower(), [sim]):
            if name not in seen:
                expanded.append(name)
                seen.add(name)
    return expanded


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
    raw_argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(normalize_option_dashes(raw_argv))

    sims = expand_sim_args(args.sims)
    outdir = args.outdir if args.outdir is not None else PLOTS_DIR
    for sim in sims:
        for package_name, module_name in INDIVIDUAL_PLOT_MODULES:
            print(f"running {module_name} for {sim}")
            module = import_module(f"{package_name}.{module_name}")
            module_argv = ["--sims", sim, "--outdir", str(outdir)]
            if args.no_save:
                module_argv.append("--no-save")
            if args.show:
                module_argv.append("--show")
            module.main(module_argv)

        if args.extra:
            for package_name, module_name in INDIVIDUAL_EXTRA_MODULES:
                print(f"running {module_name} for {sim}")
                module = import_module(f"{package_name}.{module_name}")
                module_argv = ["--sims", sim, "--outdir", str(outdir)]
                if args.no_save:
                    module_argv.append("--no-save")
                if args.show:
                    module_argv.append("--show")
                module.main(module_argv)


if __name__ == "__main__":
    main()
