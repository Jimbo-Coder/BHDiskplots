from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    COMPACT_LEGEND_KWARGS,
    PAPER_SINGLE_PANEL_HEIGHT,
    figure_size,
    ordered_sim_legend,
)
from helpers.time_units import add_time_secondary_axis, time_xlabel, time_values

# Critical knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
OUTPUT_FILENAME = "rhomax.png"

# Presentation knobs.
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.96, "bottom": 0.17, "top": 0.90}


def plot(sims):
    fig, ax = plt.subplots(figsize=figure_size("double", PAPER_SINGLE_PANEL_HEIGHT))
    final_times = []
    for sim in sims:
        time = time_values(sim.rhomax_t, sim)
        final_times.append(time[-1])
        ax.plot(
            time,
            sim.rhomax / sim.rhomax[0],
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )
    ax.set_ylabel(r"$\rho^{\mathrm{max}}_0/\rho^{\mathrm{max}}_{0,t=0}$")
    ax.set_xlabel(time_xlabel())
    add_time_secondary_axis(ax)
    # ax.set_yscale("log")
    ax.set_ylim(8e-1, 3.0)
    ax.grid()
    if final_times:
        ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(final_times))
    ordered_sim_legend(
        ax,
        ncols=4,
        loc="upper left",
        **COMPACT_LEGEND_KWARGS,
    )
    fig.subplots_adjust(**SUBPLOT_MARGINS)
    return fig


def main(argv=None):
    args = parser("Plot maximum rest-mass density.").parse_args(argv)
    setup(args)
    fig = plot(load_sims(["rhomax"], names=args.sims))
    savefig(fig, args, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
