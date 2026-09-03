from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, PAPER_TWO_PANEL_HEIGHT, figure_size, ordered_sim_interpanel_legend
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Critical knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.03
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.98, "bottom": 0.11, "top": 0.90}
MODE_YLIM = (1e-6, 1.25)
MODE_Y_MAJOR_TICKS = [10.0**power for power in range(-6, 1)]
OUTPUT_FILENAME = "modes.png"

# Presentation knobs.
FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
HSPACE = 0.20
SAVEFIG_PAD_INCHES = 0.05


def plot(sims):
    fig, axes = plt.subplots(2, 1, figsize=figure_size("double", FIG_HEIGHT), sharex=True, gridspec_kw={"hspace": HSPACE})
    tfs = []
    for sim in sims:
        t = time_values(sim.modes_t, sim)
        tfs.append(t[-1])
        c0 = sim.modes[:, 0]
        c1 = sim.modes[:, 1]
        c2 = sim.modes[:, 2]
        axes[0].plot(t, np.divide(np.abs(c1), np.abs(c0), out=np.zeros(len(c1)), where=np.abs(c0) != 0), label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        axes[1].plot(t, np.divide(np.abs(c2), np.abs(c0), out=np.zeros(len(c2)), where=np.abs(c0) != 0), label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
    axes[0].set_ylabel(r"$|C_1|/C_0$")
    axes[1].set_ylabel(r"$|C_2|/C_0$")
    axes[1].set_xlabel(time_xlabel())
    add_time_secondary_axis(axes[0])
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(*MODE_YLIM)
        ax.yaxis.set_major_locator(mticker.FixedLocator(MODE_Y_MAJOR_TICKS))
        ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10), numticks=80))
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
        ax.tick_params(axis="y", pad=4)
        ax.tick_params(axis="y", which="minor", width=0.55, length=3)
    if tfs:
        for ax in axes:
            ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(tfs))
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(fig, axes[0], axes, ncols=4, loc="center right", x=SUBPLOT_MARGINS["right"], **COMPACT_LEGEND_KWARGS)
    return fig


def main(argv=None):
    args = parser("Plot density mode amplitudes.").parse_args(argv)
    setup(args)
    sims = load_sims(["modes"], names=args.sims)
    fig = plot(sims)
    with plt.rc_context({"savefig.pad_inches": SAVEFIG_PAD_INCHES}):
        savefig(fig, args, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
