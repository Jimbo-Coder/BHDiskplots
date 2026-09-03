from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, PAPER_SINGLE_PANEL_HEIGHT, figure_size, ordered_sim_legend
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Critical knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
OUTPUT_FILENAME = "phase.png"

# Presentation knobs.
FIG_HEIGHT = PAPER_SINGLE_PANEL_HEIGHT
SUBPLOT_MARGINS = {"left": 0.17, "right": 0.96, "bottom": 0.16, "top": 0.90}


def plot(sims):
    fig, ax = plt.subplots(1, 1, figsize=figure_size("double", FIG_HEIGHT))
    tfs = []
    for sim in sims:
        t = time_values(sim.modes_t, sim)
        tfs.append(t[-1])
        c1 = sim.modes[:, 1]
        phases = np.unwrap(np.arctan2(np.imag(c1), np.real(c1)) + np.pi)
        ax.plot(t, phases / (2 * np.pi), label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
    ax.set_ylabel(r"$\phi/(2\pi)$")
    ax.set_xlabel(time_xlabel())
    add_time_secondary_axis(ax)
    if tfs:
        ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(tfs))
    ax.margins(y=0.08)
    ax.grid()
    ordered_sim_legend(ax, ncols=4, loc="lower right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(**SUBPLOT_MARGINS)
    return fig


def main(argv=None):
    args = parser("Plot C1 phase.").parse_args(argv)
    setup(args)
    sims = load_sims(["modes"], names=args.sims)
    fig = plot(sims)
    savefig(fig, args, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
