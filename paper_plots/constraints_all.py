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
XLIM_LEFT_FACTOR = 0.0
XLIM_RIGHT_FACTOR = 1.03
OUTPUT_FILENAME = "constraints.png"
# B3 contains a six-sample burst in the excised Hamiltonian numerator while
# its denominator and outside-horizon diagnostics remain smooth. Break the
# plotted line across that known bad interval; do not alter or interpolate it.
HAMILTONIAN_EXCLUDED_CODE_TIME_INTERVALS = {
    "B3": ((267.50, 267.84),),
}

# Presentation knobs.
HSPACE = 0.05
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.98, "bottom": 0.11, "top": 0.90}


def _hamiltonian_for_plot(sim):
    values = np.asarray(sim.ham_r, dtype=float).copy()
    time = np.asarray(sim.ham_t, dtype=float)
    for start, stop in HAMILTONIAN_EXCLUDED_CODE_TIME_INTERVALS.get(
        sim.config.name,
        (),
    ):
        values[(time >= start) & (time <= stop)] = np.nan
    return values

def plot(sims):
    fig, axes = plt.subplots(2, 1, figsize=figure_size("double", PAPER_TWO_PANEL_HEIGHT), sharex=True, gridspec_kw={"hspace": HSPACE})
    tfs = []
    for sim in sims:
        ham_t = time_values(sim.ham_t, sim)
        mom_t = time_values(sim.mom_t, sim)
        tfs.append(ham_t[-1])
        axes[0].plot(ham_t, _hamiltonian_for_plot(sim), label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        axes[1].plot(mom_t, sim.mom_r, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
    axes[0].set_ylabel(r"$\|\mathcal{H}\|$")
    axes[1].set_ylabel(r"$\|\mathcal{M}^{i}\|$")
    axes[1].set_xlabel(time_xlabel())
    add_time_secondary_axis(axes[0])
    for ax in axes:
        ax.set_yscale("log")
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    if tfs:
        for ax in axes:
            ax.set_xlim(XLIM_LEFT_FACTOR * np.min(tfs), XLIM_RIGHT_FACTOR * np.max(tfs))
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_legend(axes[0], ncols=4, loc="upper right", **COMPACT_LEGEND_KWARGS)
    return fig


def main(argv=None):
    args = parser("Plot Hamiltonian and momentum constraints.").parse_args(argv)
    setup(args)
    sims = load_sims(["constraints"], names=args.sims)
    fig = plot(sims)
    savefig(fig, args, OUTPUT_FILENAME)

if __name__ == "__main__":
    main()
