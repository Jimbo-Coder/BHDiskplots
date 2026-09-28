"""Standalone version of the former tripleM rest-mass panel.

Total conserved rest mass M0 (bhns.don column 2), including matter inside
the AH. This is not the outside-AH mass used to fit the GW temporal tail.
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, figure_size, ordered_sim_legend
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Physical/display range; None lets the measured data determine the range.
YLIM = None
OUTPUT_FILENAME = "wip/restmass.png"


def plot(sims):
    fig, ax = plt.subplots(figsize=figure_size("double"))
    starts_at_zero = True
    for sim in sims:
        initial = sim.restmass[0]
        if not np.isfinite(initial) or initial <= 0:
            raise ValueError(f"{sim.config.name}: initial rest mass must be positive")
        if sim.M0MADM_t[0] != 0:
            starts_at_zero = False
            print(f"{sim.config.name}: normalizing to first scalar sample at t={sim.M0MADM_t[0]:g}, not t=0")
        ax.plot(time_values(sim.M0MADM_t, sim), sim.restmass/initial,
                color=sim.color, linestyle=sim.linestyle, label=sim.legend_name)
    ax.set(xlabel=time_xlabel(), ylabel=r"$M_0/M_0(0)$" if starts_at_zero else r"$M_0/M_0(t_{\mathrm{first}})$")
    ax.set_xlim(left=0)
    if YLIM is not None:
        ax.set_ylim(*YLIM)
    ax.margins(x=.02, y=.12)
    ax.grid(which="major", alpha=.25)
    ncols = 4 if any(sim.config.name in {"ML", "WL"} for sim in sims) else 3
    ordered_sim_legend(ax, ncols=ncols, loc="upper left", **COMPACT_LEGEND_KWARGS)
    add_time_secondary_axis(ax)
    fig.subplots_adjust(left=.15, right=.98, bottom=.18, top=.90)
    return fig


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    setup(args)
    savefig(plot(load_sims(["M0MADM"], names=args.sims)), args, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
