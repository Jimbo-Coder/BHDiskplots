from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, DOUBLE_COLUMN_WIDTH, PAPER_SINGLE_PANEL_HEIGHT, PAPER_TWO_PANEL_HEIGHT, ordered_sim_fig_legend, ordered_sim_spanning_legend
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Critical knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.03
OUTPUT_FILENAME_RADIUS_J = "wip/bh_horizon_radius_and_angular_momentum.png"
OUTPUT_FILENAME_J_COMPONENTS = "wip/bh_angular_momentum_components.png"

# Presentation knobs.
RADIUS_J_FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
RADIUS_J_HSPACE = 0.22
RADIUS_J_MARGINS = {"left": 0.18, "right": 0.98, "bottom": 0.10, "top": 0.90}
J_COMPONENTS_FIG_HEIGHT = PAPER_SINGLE_PANEL_HEIGHT
def plot_radius_j(sims):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(DOUBLE_COLUMN_WIDTH, RADIUS_J_FIG_HEIGHT),
        gridspec_kw={"hspace": RADIUS_J_HSPACE},
    )
    radius_final_times = []
    angular_momentum_final_times = []
    for sim in sims:
        radius_time = time_values(sim.Rs_t, sim)
        angular_momentum_time = time_values(sim.J_t, sim)
        radius_final_times.append(radius_time[-1])
        angular_momentum_final_times.append(angular_momentum_time[-1])
        axes[0].plot(
            radius_time,
            sim.Rs,
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )
        axes[1].plot(
            angular_momentum_time,
            sim.J,
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )
    axes[0].set_xlabel(time_xlabel())
    axes[0].set_ylabel(r"$R_{\mathrm{BH}}\ [M_\odot]$")
    axes[1].set_xlabel(time_xlabel())
    axes[1].set_ylabel(r"$J_{\mathrm{BH}}\ [M_\odot^2]$")
    add_time_secondary_axis(axes[0])
    for ax in axes:
        ax.grid()
    if radius_final_times:
        axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(radius_final_times))
        axes[1].set_xlim(
            TIME_XMIN,
            TIME_XMAX_PAD * np.max(angular_momentum_final_times),
        )
    fig.subplots_adjust(hspace=RADIUS_J_HSPACE, **RADIUS_J_MARGINS)
    ordered_sim_spanning_legend(
        fig,
        axes[0],
        axes=axes,
        ncols=4,
        **COMPACT_LEGEND_KWARGS,
    )
    return fig

def plot_j_components(sims):
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(DOUBLE_COLUMN_WIDTH, J_COMPONENTS_FIG_HEIGHT),
        sharey=False,
    )
    final_times = []
    titles = [r"$J_{\mathrm{BH},x}$", r"$J_{\mathrm{BH},y}$", r"$J_{\mathrm{BH},z}$"]
    for i, (ax, title) in enumerate(zip(axes, titles)):
        ax.set_title(title)
        ax.set_xlabel(time_xlabel())
        secax = add_time_secondary_axis(ax)
        if secax is not None and i != 1:
            secax.set_xlabel("")
        ax.grid()
    axes[0].set_ylabel(r"$J_{\mathrm{BH},i}\ [M_\odot^2]$")
    for sim in sims:
        time = time_values(sim.J_t, sim)
        final_times.append(time[-1])
        for index, ax in enumerate(axes):
            ax.plot(
                time,
                sim.Jdata[:, index + 1],
                label=sim.legend_name,
                linestyle=sim.linestyle,
                color=sim.color,
            )
    if final_times:
        for ax in axes:
            ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(final_times))
    for ax in axes:
        ymin, ymax = ax.get_ylim()
        pad = 0.08 * (ymax - ymin) if ymax > ymin else 1.0e-6
        ax.set_ylim(ymin - pad, ymax + pad)
    ordered_sim_fig_legend(
        fig,
        axes[0],
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        **COMPACT_LEGEND_KWARGS,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    return fig

def main(argv=None):
    args = parser("Plot BH radius/J and J components.").parse_args(argv)
    setup(args)
    sims = load_sims(["J_Rs"], names=args.sims)
    savefig(plot_radius_j(sims), args, OUTPUT_FILENAME_RADIUS_J)
    savefig(plot_j_components(sims), args, OUTPUT_FILENAME_J_COMPONENTS)


if __name__ == "__main__":
    main()
