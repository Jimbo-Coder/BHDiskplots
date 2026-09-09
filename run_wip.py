#!/usr/bin/env python3
"""Run one or more combined WIP plots by short name."""
from __future__ import annotations

import argparse
from importlib import import_module


WIP_PLOTS = {
    "displacement": "disp_all",
    "horizon": "j_rs_all",
    "irreducible_mass": "mirr_all",
    "radii": "rs_minmax_all",
    "psi4": "gw_waveforms",
    "psi4_minus_ml": "gw_waveforms",
    "strain": "gw_waveforms",
    "strain_panel": "gw_strain_polarization_panel",
    "strain_minus_ml": "gw_waveforms",
    "strain_from_psi4_minus_ml": "gw_waveforms",
    "detectability": "gw_detectability_all",
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plots",
        nargs="*",
        help="Optional plot names. With no names, run every combined WIP plot.",
    )
    args = parser.parse_args(argv)
    invalid = [name for name in args.plots if name not in WIP_PLOTS]
    if invalid:
        parser.error(
            f"unknown plot(s): {', '.join(invalid)}; "
            f"choose from {', '.join(WIP_PLOTS)}"
        )

    requested = args.plots or list(WIP_PLOTS)
    waveforms_done = False
    for name in requested:
        module_name = WIP_PLOTS[name]
        print(f"running WIP plot: {name}")
        module = import_module(f"wip_plots.{module_name}")
        if module_name == "gw_waveforms":
            if not waveforms_done:
                module.main([], kinds=[key for key in requested if WIP_PLOTS[key] == module_name])
                waveforms_done = True
        else:
            module.main([])


if __name__ == "__main__":
    main()
