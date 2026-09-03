#!/usr/bin/env python3
"""Run all approved paper plots, or selected plots by short name."""
from __future__ import annotations

import argparse
from importlib import import_module


# This registry is intentionally explicit. A plot enters the paper workflow
# only when it is moved to paper_plots and added here after review.
PAPER_PLOTS = {
    "hamiltonian": "constraints_all",
    "modes": "modes_all",
    "phase": "phase_all",
    "rhomax": "rhomax_all",
    "spin": "spin_all",
    "masses": "triple_m_all",
    "accretion": "m0dot_all",
    "initial_data": "initial_data_all",
    "centers": "centers_all",
    "panels": "rho2d_panels",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plots",
        nargs="*",
        help="Optional plot names. With no names, run the complete paper set.",
    )
    args = parser.parse_args(argv)
    invalid = [name for name in args.plots if name not in PAPER_PLOTS]
    if invalid:
        parser.error(
            f"unknown plot(s): {', '.join(invalid)}; "
            f"choose from {', '.join(PAPER_PLOTS)}"
        )
    names = args.plots or list(PAPER_PLOTS)

    for name in names:
        module_name = PAPER_PLOTS[name]
        print(f"running paper plot: {name}")
        module = import_module(f"paper_plots.{module_name}")
        module.main([])


if __name__ == "__main__":
    main()
