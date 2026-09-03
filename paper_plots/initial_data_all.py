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
    PAPER_FONT_SIZE,
    PAPER_TWO_PANEL_HEIGHT,
    figure_size,
    ordered_sim_legend,
)

# Critical knobs.
XMIN_OVER_M = 1.0
XMAX_PHYSICAL = 100.0
OUTPUT_FILENAME = "initial_rho_ell_xp.png"

POWER_LAW_SPECS = {
    1.99: {"x0": 0.18, "power": 0.01, "offset_sign": -1.0, "label_fraction": 0.87, "label_yshift": -0.04},
    1.85: {"x0": 0.16, "power": 0.15, "offset_sign": 1.0, "label_fraction": 0.84, "label_yshift": 0.0},
}

# Presentation knobs.
FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
HSPACE = 0.13
SUBPLOT_MARGINS = {"left": 0.23, "right": 0.98, "bottom": 0.11, "top": 0.98}


def plot(sims):
    fig, (ax_rho, ax_ell) = plt.subplots(
        2,
        1,
        figsize=figure_size("double", FIG_HEIGHT),
        sharex=True,
        gridspec_kw={"hspace": HSPACE},
    )

    rho_vals = []
    ell_vals = []
    xvals = []
    mlittles = []
    refs = {}
    for sim in sims:
        mlittles.append(sim.config.mlittle)
        x_rho = sim.emdg_x / sim.config.mlittle
        x_ell = sim.ell_x / sim.config.mlittle

        rm = (x_rho > 0) & np.isfinite(sim.rho_initial) & (sim.rho_initial > 0)
        if np.any(rm):
            ax_rho.plot(x_rho[rm], sim.rho_initial[rm], label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
            rho_vals.append(sim.rho_initial[rm])
            xvals.append(x_rho[rm])

        em = (x_ell > 0) & np.isfinite(sim.ell)
        erm = em & (sim.ell != 0)
        if np.any(em):
            ax_ell.plot(x_ell[em], sim.ell[em], label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
            ell_vals.append(sim.ell[em])
            xvals.append(x_ell[em])
        if np.any(erm):
            refs.setdefault(round(float(sim.config.q), 2), {"x": x_ell[erm], "y": sim.ell[erm], "mlittle": sim.config.mlittle})

    ax_rho.set_ylabel(r"$\rho_0$")
    ax_ell.set_ylabel(r"$\ell=-u_\phi/u_t$")
    ax_ell.set_xlabel(r"$x/m$")

    if rho_vals:
        rv = np.concatenate(rho_vals)
        rv = rv[rv > 0]
        if rv.size:
            ax_rho.set_ylim(np.min(rv) / 1.5, np.max(rv) * 1.35)
    if ell_vals:
        ev = np.concatenate(ell_vals)
        ymin = np.min(ev)
        ymax = np.max(ev)
        pad = 0.05 * (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.05, 1e-6)
        ax_ell.set_ylim(ymin - pad - 0.05, ymax + pad)
        ell_offset = 0.03 * (ymax - ymin) if ymax > ymin else max(abs(ymax) * 0.03, 1e-6)
    else:
        ell_offset = 0.0

    if xvals:
        xv = np.concatenate(xvals)
        xv = xv[np.isfinite(xv) & (xv > 0)]
        if xv.size:
            xmax_over_m = XMAX_PHYSICAL / min(mlittles)
            ax_ell.set_xlim(XMIN_OVER_M, min(np.max(xv), xmax_over_m))

    x_right = ax_ell.get_xlim()[1]
    for q, spec in POWER_LAW_SPECS.items():
        if q not in refs:
            continue
        ref = refs[q]
        x = ref["x"]
        y = ref["y"]
        x0_over_m = spec["x0"] / ref["mlittle"]
        candidates = np.where(x >= x0_over_m)[0]
        if not candidates.size:
            continue
        i0 = candidates[0]
        x0 = x[i0]
        y0 = y[i0] + spec["offset_sign"] * ell_offset
        xr = np.logspace(np.log10(x0), np.log10(x_right), 200)
        yr = y0 * np.power(xr / x0, spec["power"])
        ax_ell.plot(xr, yr, color="0.35", linestyle=":", linewidth=1.6, zorder=1)
        li = int(spec.get("label_fraction", 0.82) * (len(xr) - 1))
        ax_ell.text(
            xr[li],
            yr[li] + spec.get("label_yshift", 0.0),
            rf"$x^{{{spec['power']:.2f}}}$",
            color="0.35",
            fontsize=PAPER_FONT_SIZE,
            ha="center",
            va="bottom" if spec["offset_sign"] > 0 else "top",
        )

    ax_ell.set_xscale("log")
    ax_rho.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax_ell.xaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=(1, 2, 3, 4, 5, 6, 7, 8, 9), numticks=9))
    for ax in (ax_rho, ax_ell):
        ax.grid(which="major")

    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    fig.align_ylabels((ax_rho, ax_ell))
    ordered_sim_legend(ax_rho, ncols=3, loc="upper right", **COMPACT_LEGEND_KWARGS)
    return fig


def main(argv=None):
    args = parser("Plot initial rho and angular momentum profiles.").parse_args(argv)
    setup(args)
    fig = plot(load_sims(["initial_data"], names=args.sims))
    savefig(fig, args, OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
