#!/usr/bin/env python3
"""Run one or more combined WIP plots by short name."""
from __future__ import annotations

import argparse
from importlib import import_module


WIP_PLOTS = {
    "accretion": "wip_plots.accretion",
    "restmass": "wip_plots.restmass",
    "modes_345": "wip_plots.modes_high_all",
    "mode_appendix": "wip_plots.mode_appendix",
    "shibata": "wip_plots.disk_compactness_comparison",
    "toomre": "wip_plots.toomre",
    "displacement": "wip_plots.disp_all",
    "horizon": "wip_plots.j_rs_all",
    "irreducible_mass": "wip_plots.mirr_all",
    "radii": "wip_plots.rs_minmax_all",
    "psi4": "wip_plots.gw_waveforms",
    "psi4_minus_ml": "wip_plots.gw_waveforms",
    "strain": "wip_plots.gw_waveforms",
    "strain_panel": "wip_plots.gw_waveforms",
    "strain_observer": "wip_plots.gw_waveforms",
    "gw_radiated": "wip_plots.gw_radiated",
    "strain_minus_ml": "wip_plots.gw_waveforms",
    "strain_from_psi4_minus_ml": "wip_plots.gw_waveforms",
    "detectability": "gw_detectability",
    "temporal_trial": "gw_detectability",  # Explicit opt-in until the model is supported.
    "gw_mismatch": "gw_mismatch",  # Requires kuibit; explicit opt-in diagnostic.
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "plots",
        nargs="*",
        help="Optional plot names. No names excludes temporal_trial and gw_mismatch.",
    )
    args = parser.parse_args(argv)
    invalid = [name for name in args.plots if name not in WIP_PLOTS]
    if invalid:
        parser.error(
            f"unknown plot(s): {', '.join(invalid)}; "
            f"choose from {', '.join(WIP_PLOTS)}"
        )

    requested = args.plots or [name for name in WIP_PLOTS if name not in {"temporal_trial", "gw_mismatch"}]
    waveforms_done = False
    for name in requested:
        module_name = WIP_PLOTS[name]
        print(f"running WIP plot: {name}")
        module = import_module(module_name)
        if name == "temporal_trial":
            module.temporal_main([])
        elif module_name == "wip_plots.gw_waveforms":
            if not waveforms_done:
                module.main([], kinds=[key for key in requested if WIP_PLOTS[key] == module_name])
                waveforms_done = True
        elif name == "detectability":
            module.main(["--plots", *module.BATCH_PLOTS])
        else:
            module.main([])


if __name__ == "__main__":
    main()
