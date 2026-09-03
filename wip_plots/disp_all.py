from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import adjustText as adjText
import matplotlib.pyplot as plt
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, PAPER_TWO_PANEL_HEIGHT, figure_size, ordered_sim_spanning_legend
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Critical knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.03
THETA_XMIN = 0.0
THETA_XMAX_PAD = 1.03
OUTPUT_FILENAME = "wip/bh_displacement.png"

# Presentation knobs.
FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
HSPACE = 0.24
SUBPLOT_MARGINS = {"left": 0.17, "right": 0.98, "bottom": 0.10, "top": 0.90}
def plot(sims):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size("double", FIG_HEIGHT),
        gridspec_kw={"hspace": HSPACE},
    )
    final_times = []
    final_angles = []
    texts = []
    for sim in sims:
        time = time_values(sim.Rs_t, sim)
        x = sim.Rsdata[:, 2]
        y = sim.Rsdata[:, 3]
        radius = np.hypot(x, y)
        angle = np.unwrap(np.arctan2(y, x)) / (2 * np.pi)
        final_times.append(time[-1])
        final_angles.append(angle[-1])
        axes[0].plot(
            time,
            radius,
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )
        axes[1].scatter(
            angle,
            radius,
            s=2,
            marker=sim.markerstyle,
            label=sim.legend_name,
            color=sim.color,
        )
    axes[0].set_xlabel(time_xlabel())
    add_time_secondary_axis(axes[0])
    axes[0].set_ylabel(r"Disp. $r\ [M_\odot]$")
    axes[1].set_xlabel(r"$\theta/(2\pi)$")
    axes[1].set_ylabel(r"Disp. $r\ [M_\odot]$")
    for ax in axes:
        ax.grid()
    if final_times:
        axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(final_times))
    if final_angles:
        axes[1].set_xlim(THETA_XMIN, THETA_XMAX_PAD * np.max(final_angles))
    if texts:
        adjText.adjust_text(texts, arrowprops={"arrowstyle": "-", "lw": 0.5})
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_spanning_legend(
        fig,
        axes[0],
        axes=axes,
        ncols=4,
        **COMPACT_LEGEND_KWARGS,
    )
    return fig

def main(argv=None):
    args = parser("Plot BH displacement.").parse_args(argv)
    setup(args)
    sims = load_sims(["Rs"], names=args.sims)
    savefig(plot(sims), args, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
