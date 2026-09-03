from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from config import GW_COMPARISON_PARFILE_INDICES
from helpers.gw_units import (
    add_gw_time_secondary_axis,
    gw_time_xlabel,
    gw_time_values,
    normalize_strain,
    strain_ylabel,
)
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    GW_LEGEND_KWARGS,
    PAPER_TWO_PANEL_HEIGHT,
    figure_size,
    format_shared_gw_yaxes,
    ordered_sim_interpanel_legend,
    set_symmetric_gw_ticks,
)

# Critical knobs.
SHARED_PSI4_PARFILE_INDICES = GW_COMPARISON_PARFILE_INDICES
PSI4_MODE = ((2, 2), (2, 1), (2, 0), (4, 0))
OUTPUT_TEMPLATE = "gw/rhphc_{mode}_{radius}.png"

# Presentation knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.96, "bottom": 0.12, "top": 0.86}
HSPACE = 0.28


def mode_tag(mode):
    ell, emm = mode
    return f"l{ell}m{emm}" if emm >= 0 else f"l{ell}mneg{abs(emm)}"


def mode_label(mode):
    ell, emm = mode
    return rf"$(\ell,m)=({ell},{emm})$"


def radius_tag(sims, parfile_index):
    for sim in sims:
        radius = getattr(sim, "psi4_radius", None)
        if radius is not None and np.isfinite(radius):
            value = float(radius)
            text = str(int(round(value))) if np.isclose(value, round(value)) else f"{value:g}"
            return f"r{text.replace('.', 'p')}"
    return f"extraction_index{int(parfile_index)}"


def selected_modes(psi4_mode):
    if len(psi4_mode) == 2 and all(np.isscalar(part) for part in psi4_mode):
        return (tuple(int(part) for part in psi4_mode),)
    return tuple(tuple(int(part) for part in mode) for mode in psi4_mode)


def plot(sims, psi4_mode, parfile_index):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size("double", PAPER_TWO_PANEL_HEIGHT),
        sharex=True,
        gridspec_kw={"hspace": HSPACE},
    )
    final_times = []
    hplus_values = []
    hcross_values = []
    ell, emm = psi4_mode
    mode = f"{ell}{emm}"

    for sim in sims:
        hplus_raw, hcross_raw = sim.strain_result.hplus_hcross(ell=ell, emm=emm)
        time = gw_time_values(sim.strain_result.time, sim)
        hplus = normalize_strain(hplus_raw, sim)
        hcross = normalize_strain(hcross_raw, sim)
        final_times.append(time[-1])
        hplus_values.append(hplus)
        hcross_values.append(hcross)
        axes[0].plot(
            time,
            hplus,
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )
        axes[1].plot(
            time,
            hcross,
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )

    radius_label = (
        sims[0].gw_extraction_plot_label(parfile_index)
        if sims
        else rf"$i_{{par}}={parfile_index}$"
    )
    axes[0].set_title(f"{mode_label(psi4_mode)}; {radius_label}")
    axes[0].set_ylabel(strain_ylabel("plus", mode))
    axes[1].set_ylabel(strain_ylabel("cross", mode))
    axes[1].set_xlabel(gw_time_xlabel())
    for ax in axes:
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    add_gw_time_secondary_axis(axes[0])
    if final_times:
        axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(final_times))
    set_symmetric_gw_ticks([axes[0]], hplus_values)
    set_symmetric_gw_ticks([axes[1]], hcross_values)
    format_shared_gw_yaxes(axes)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(
        fig,
        axes[1],
        axes,
        ncols=3,
        x=SUBPLOT_MARGINS["left"],
        **GW_LEGEND_KWARGS,
    )
    return fig


def main(argv=None):
    args = parser("Plot GW strain.").parse_args(argv)
    setup(args)
    modes = selected_modes(PSI4_MODE)
    for parfile_index in SHARED_PSI4_PARFILE_INDICES:
        sims = load_sims(
            ["strain"],
            names=args.sims,
            psi4_parfile_index=parfile_index,
            psi4_mode=modes[0],
        )
        for psi4_mode in modes:
            fig = plot(sims, psi4_mode, parfile_index)
            savefig(
                fig,
                args,
                OUTPUT_TEMPLATE.format(
                    mode=mode_tag(psi4_mode),
                    radius=radius_tag(sims, parfile_index),
                ),
            )


if __name__ == "__main__":
    main()
