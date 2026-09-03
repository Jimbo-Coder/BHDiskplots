from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, PAPER_TWO_PANEL_HEIGHT, figure_size, ordered_sim_legend
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Critical knobs.
FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
HSPACE = 0.12
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.98, "bottom": 0.10, "top": 0.90}
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.03
OUTPUT_FILENAME = "wip/bh_horizon_radius_minimum_maximum.png"

def plot(sims):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size("double", FIG_HEIGHT),
        sharex=True,
        sharey=True,
        gridspec_kw={"hspace": HSPACE},
    )
    final_times = []
    for sim in sims:
        time = time_values(sim.Rs_t, sim)
        final_times.append(time[-1])
        axes[0].plot(time, sim.Rsdata[:, 5], label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        axes[1].plot(time, sim.Rsdata[:, 6], label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
    axes[0].set_ylabel(r"$R_{\mathrm{BH},\min}\ [M_\odot]$")
    axes[1].set_ylabel(r"$R_{\mathrm{BH},\max}\ [M_\odot]$")
    axes[1].set_xlabel(time_xlabel())
    add_time_secondary_axis(axes[0])
    for ax in axes:
        ax.set_yscale("log")
        ax.grid()
        if final_times:
            ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(final_times))
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_legend(
        axes[0],
        ncols=4,
        loc="upper right",
        **COMPACT_LEGEND_KWARGS,
    )
    return fig

def main(argv=None):
    args = parser("Plot AH radius min/max.").parse_args(argv)
    setup(args)
    sims = load_sims(["Rs"], names=args.sims)
    savefig(plot(sims), args, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
