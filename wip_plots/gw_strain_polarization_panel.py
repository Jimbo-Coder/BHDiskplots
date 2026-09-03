#!/usr/bin/env python3
"""Plot both strain polarizations in one cell for each disk simulation."""
from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from config import GW_COMPARISON_PARFILE_INDICES
from helpers.gw_units import (
    add_gw_time_secondary_axis,
    gw_time_xlabel,
    gw_time_values,
    normalize_strain,
)
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    figure_size,
    format_paper_axes,
    format_shared_gw_yaxes,
    set_symmetric_gw_ticks,
)


# Scientific knobs.
SIMULATION_GRID = (("A1", "A2", "A3"), ("B1", "B2", "B3"))
PSI4_PARFILE_INDICES = GW_COMPARISON_PARFILE_INDICES
PSI4_MODES = ((2, 2), (2, 1), (2, 0), (4, 0))

# Output knob. Radius is resolved from the loaded data, not inferred from index.
OUTPUT_TEMPLATE = "gw/rhphc_panel_{mode}_{radius}.png"

# Presentation knobs.
FIGURE_HEIGHT = 4.9
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.03
SUBPLOT_MARGINS = {
    "left": 0.10,
    "right": 0.985,
    "bottom": 0.11,
    "top": 0.80,
    "wspace": 0.10,
    "hspace": 0.27,
}
PLUS_LINESTYLE = "-"
CROSS_LINESTYLE = "--"


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
    return f"index{int(parfile_index)}"


def radius_label(sims, parfile_index):
    if sims:
        return sims[0].gw_extraction_plot_label(parfile_index)
    return rf"$i_{{\mathrm{{par}}}}={int(parfile_index)}$"


def plot(sims, mode, parfile_index):
    """Build the complete six-case panel; all layout choices live here."""
    sim_by_name = {sim.config.name: sim for sim in sims}
    fig, axes = plt.subplots(
        2,
        3,
        figsize=figure_size("double", FIGURE_HEIGHT),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    final_times = []
    strain_values = []
    mode_text = f"{mode[0]}{mode[1]}"
    for row, names in enumerate(SIMULATION_GRID):
        for col, name in enumerate(names):
            ax = axes[row, col]
            sim = sim_by_name.get(name)
            if sim is None:
                ax.set_visible(False)
                continue

            hplus_raw, hcross_raw = sim.strain_result.hplus_hcross(
                ell=mode[0],
                emm=mode[1],
            )
            time = np.asarray(
                gw_time_values(sim.strain_result.time, sim),
                dtype=float,
            )
            hplus = np.asarray(normalize_strain(hplus_raw, sim), dtype=float)
            hcross = np.asarray(normalize_strain(hcross_raw, sim), dtype=float)
            sample_count = min(time.size, hplus.size, hcross.size)
            time = time[:sample_count]
            hplus = hplus[:sample_count]
            hcross = hcross[:sample_count]
            finite = np.isfinite(time) & np.isfinite(hplus) & np.isfinite(hcross)
            time = time[finite]
            hplus = hplus[finite]
            hcross = hcross[finite]
            if not time.size:
                ax.set_visible(False)
                continue

            final_times.append(time[-1])
            strain_values.extend((hplus, hcross))
            ax.plot(
                time,
                hplus,
                color=sim.color,
                linestyle=PLUS_LINESTYLE,
                label=r"$h_+$",
            )
            ax.plot(
                time,
                hcross,
                color=sim.color,
                linestyle=CROSS_LINESTYLE,
                label=r"$h_\times$",
            )
            ax.text(
                0.04,
                0.94,
                name,
                transform=ax.transAxes,
                ha="left",
                va="top",
            )
            ax.grid()
            format_paper_axes(ax)

    if final_times:
        axes[0, 0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * max(final_times))

    ylabel = rf"$r_A h_{{+,\times}}^{{{mode_text}}}/M_{{\mathrm{{BH}}}}$"
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(ylabel)
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(gw_time_xlabel())

    visible_axes = [ax for ax in axes.flat if ax.get_visible()]
    set_symmetric_gw_ticks(visible_axes, strain_values)
    format_shared_gw_yaxes(visible_axes)
    for ax in axes[0, :]:
        if ax.get_visible():
            add_gw_time_secondary_axis(ax)

    fig.text(
        0.10,
        0.975,
        f"{mode_label(mode)}, {radius_label(sims, parfile_index)}",
        ha="left",
        va="center",
    )
    fig.legend(
        handles=(
            Line2D([], [], color="k", linestyle=PLUS_LINESTYLE, label=r"$h_+$"),
            Line2D([], [], color="k", linestyle=CROSS_LINESTYLE, label=r"$h_\times$"),
        ),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.985),
        ncols=2,
        frameon=True,
    )
    fig.subplots_adjust(**SUBPLOT_MARGINS)
    return fig


def main(argv=None):
    args = parser("Plot a 2x3 per-simulation GW polarization panel.").parse_args(argv)
    setup(args)
    requested_names = [name for row in SIMULATION_GRID for name in row]
    for parfile_index in PSI4_PARFILE_INDICES:
        sims = load_sims(
            ["strain"],
            names=requested_names if args.sims is None else args.sims,
            psi4_parfile_index=parfile_index,
            psi4_mode=PSI4_MODES[0],
        )
        for mode in PSI4_MODES:
            filename = OUTPUT_TEMPLATE.format(
                mode=mode_tag(mode),
                radius=radius_tag(sims, parfile_index),
            )
            savefig(plot(sims, mode, parfile_index), args, filename)


if __name__ == "__main__":
    main()
