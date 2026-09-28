"""Compare the higher density modes C3, C4, and C5."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    COMPACT_LEGEND_KWARGS,
    figure_size,
    ordered_sim_interpanel_legend,
)
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel


MODE_INDICES = (3, 4, 5)
MODE_YLIM = (1e-9, 1e-1)
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.03
OUTPUT_FILENAME = "wip/modes_345.png"

FIG_HEIGHT = 7.8
HSPACE = 0.25
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.98, "bottom": 0.09, "top": 0.93}


def plot(sims):
    fig, axes = plt.subplots(
        3, 1, figsize=figure_size("double", FIG_HEIGHT), sharex=True,
        sharey=True, gridspec_kw={"hspace": HSPACE},
    )
    final_times = []
    for sim in sims:
        if sim.modes.shape[1] <= max(MODE_INDICES):
            raise ValueError(f"{sim.config.name}: density-mode file lacks C5")
        time = time_values(sim.modes_t, sim)
        final_times.append(time[-1])
        c0 = np.abs(sim.modes[:, 0])
        for mode, ax in zip(MODE_INDICES, axes):
            amplitude = np.divide(
                np.abs(sim.modes[:, mode]), c0,
                out=np.zeros_like(c0), where=c0 != 0,
            )
            ax.plot(time, amplitude, label=sim.legend_name,
                    linestyle=sim.linestyle, color=sim.color)

    for mode, ax in zip(MODE_INDICES, axes):
        ax.set_ylabel(rf"$|C_{mode}|/C_0$")
        ax.set_yscale("log")
        ax.set_ylim(*MODE_YLIM)
        ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
        ax.yaxis.set_minor_locator(
            mticker.LogLocator(base=10, subs=np.arange(2, 10), numticks=80)
        )
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
        ax.tick_params(axis="y", pad=4)
        ax.tick_params(axis="y", which="minor", width=0.55, length=3)
        if final_times:
            ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * max(final_times))

    axes[-1].set_xlabel(time_xlabel())
    add_time_secondary_axis(axes[0])
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(
        fig, axes[0], axes, ncols=4, loc="center right",
        x=SUBPLOT_MARGINS["right"], **COMPACT_LEGEND_KWARGS,
    )
    return fig


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    setup(args)
    savefig(plot(load_sims(["modes"], names=args.sims)), args, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
