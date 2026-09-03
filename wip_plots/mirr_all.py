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
TIME_XMAX_PAD = 1.03
OUTPUT_FILENAME = "wip/bh_irreducible_mass.png"

# Presentation knobs.
FIG_HEIGHT = PAPER_SINGLE_PANEL_HEIGHT

def plot(sims):
    fig, ax = plt.subplots(figsize=figure_size("double", FIG_HEIGHT))
    final_times = []
    for sim in sims:
        time = time_values(sim.Rs_t, sim)
        final_times.append(time[-1])
        ax.plot(
            time,
            sim.Rsdata[:, 26],
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )
    ax.set_xlabel(time_xlabel())
    add_time_secondary_axis(ax)
    ax.set_ylabel(r"$M_{\mathrm{irr}}\ [M_\odot]$")
    ax.grid()
    if final_times:
        ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(final_times))
    ordered_sim_legend(ax, ncols=4, **COMPACT_LEGEND_KWARGS)
    fig.tight_layout()
    return fig

def main(argv=None):
    args = parser("Plot irreducible mass.").parse_args(argv)
    setup(args)
    sims = load_sims(["Rs"], names=args.sims)
    savefig(plot(sims), args, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
