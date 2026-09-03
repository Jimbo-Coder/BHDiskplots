from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, PAPER_THREE_PANEL_HEIGHT, PAPER_TWO_PANEL_HEIGHT, figure_size, ordered_sim_interpanel_legend
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Critical knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
SHOW_RESTMASS_PANEL = False
SPIN_YLIM = (0.6, 1.0)
OUTPUT_FILENAME_DOUBLE = "tripleM.png"
OUTPUT_FILENAME_TRIPLE = "tripleM.png"

# Presentation knobs.
DOUBLE_FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
TRIPLE_FIG_HEIGHT = PAPER_THREE_PANEL_HEIGHT
HSPACE = 0.20
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.97, "bottom": 0.10, "top": 0.90}

def plot(sims):
    nrows = 3 if SHOW_RESTMASS_PANEL else 2
    fig_height = TRIPLE_FIG_HEIGHT if SHOW_RESTMASS_PANEL else DOUBLE_FIG_HEIGHT
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=figure_size("double", fig_height),
        sharex=True,
        gridspec_kw={"hspace": HSPACE},
    )
    tfs = []
    for sim in sims:
        rhomax_t = time_values(sim.rhomax_t, sim)
        j_t = time_values(sim.J_t, sim)
        axes[0].plot(rhomax_t, sim.rhomax/sim.rhomax[0], label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        axes[1].plot(j_t, sim.J/sim.Mbh**2, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        final_times = [rhomax_t[-1], j_t[-1]]
        if SHOW_RESTMASS_PANEL:
            m0_t = time_values(sim.M0MADM_t, sim)
            axes[2].plot(m0_t, sim.restmass/sim.restmass[0], label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
            final_times.append(m0_t[-1])
        tfs.append(max(final_times))

    axes[0].set_ylabel(r"$\rho^{\mathrm{max}}_0/\rho^{\mathrm{max}}_{0,t=0}$")
    axes[0].set_ylim(0.6,3)
    axes[1].set_ylabel(r"$J_{\mathrm{BH}}/M_{\mathrm{BH}}^2$")
    axes[1].set_ylim(*SPIN_YLIM)
    if SHOW_RESTMASS_PANEL:
        axes[2].set_ylabel(r"$M_0/M_0(0)$")
        axes[2].set_ylim(0.9,1.1)
    axes[-1].set_xlabel(time_xlabel())
    add_time_secondary_axis(axes[0])
    for ax in axes:
        ax.grid()
    if tfs:
        for ax in axes:
            ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD*np.max(tfs))
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(
        fig,
        axes[0],
        axes,
        ncols=4,
        loc="center right",
        x=SUBPLOT_MARGINS["right"],
        **COMPACT_LEGEND_KWARGS,
    )
    return fig

def main(argv=None):
    args = parser("Plot rho max, spin, and rest mass.").parse_args(argv)
    setup(args)
    diagnostics = ["tripleM"] if SHOW_RESTMASS_PANEL else ["rhomax", "spin_parameter"]
    fig = plot(load_sims(diagnostics, names=args.sims))
    filename = OUTPUT_FILENAME_TRIPLE if SHOW_RESTMASS_PANEL else OUTPUT_FILENAME_DOUBLE
    savefig(fig, args, filename)


if __name__ == "__main__":
    main()
