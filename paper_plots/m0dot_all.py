from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    COMPACT_LEGEND_KWARGS,
    PAPER_SINGLE_PANEL_HEIGHT,
    PAPER_TWO_PANEL_HEIGHT,
    figure_size,
    ordered_sim_interpanel_legend,
    ordered_sim_legend,
)
from helpers.time_units import add_time_secondary_axis, time_values, time_xlabel

# Critical knobs.
M0DOT_NORMALIZATION = "disk_rest_mass"  # "disk_rest_mass" or "initial_m0dot"
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
OUTPUT_FILENAME_RATE = "M0dot_BH.png"
OUTPUT_FILENAME_RATE_AND_TIMESCALE = "M0dot_M0_ratios.png"

# Presentation knobs.
FIG_HEIGHT = PAPER_SINGLE_PANEL_HEIGHT
SUBPLOT_MARGINS = {"left": 0.20, "right": 0.96, "bottom": 0.17, "top": 0.92}
RATIO_FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
RATIO_SUBPLOT_MARGINS = {"left": 0.20, "right": 0.96, "bottom": 0.12, "top": 0.92}
RATIO_HSPACE = 0.20


def _normalization_labels():
    if M0DOT_NORMALIZATION == "disk_rest_mass":
        return (
            r"$|\dot{M}_0|/M_{0,\mathrm{disk}}$",
            r"$M_{0,\mathrm{disk}}/|\dot{M}_0|$",
        )
    if M0DOT_NORMALIZATION == "initial_m0dot":
        return (
            r"$|\dot{M}_0|/|\dot{M}_0(t=0)|$",
            r"$|\dot{M}_0(t=0)|/|\dot{M}_0|$",
        )
    raise ValueError(
        "M0DOT_NORMALIZATION must be 'disk_rest_mass' or 'initial_m0dot', "
        f"not {M0DOT_NORMALIZATION!r}"
    )


def _normalization(sim, mdot):
    labels = _normalization_labels()
    if M0DOT_NORMALIZATION == "disk_rest_mass":
        denominator = float(sim.config.disk_rest_mass)
    else:
        denominator = float(mdot[0])
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError(f"{sim.config.name}: invalid M0dot normalization {denominator}")
    return denominator, labels


def plot(sims):
    fig, ax = plt.subplots(figsize=figure_size("double", FIG_HEIGHT))
    tfs = []
    values = []
    labels = _normalization_labels()
    for sim in sims:
        t = np.asarray(time_values(sim.M0dot_BH_t, sim), dtype=float)
        mdot = np.abs(np.asarray(sim.M0dot_BH, dtype=float))
        n = min(t.size, mdot.size)
        t = t[:n]
        mdot = mdot[:n]
        valid = np.isfinite(t) & np.isfinite(mdot) & (mdot > 0.0)
        if not np.any(valid):
            continue
        t = t[valid]
        mdot = mdot[valid]
        denominator, labels = _normalization(sim, mdot)
        mdot = mdot / denominator
        tfs.append(t[-1])
        values.append(mdot)
        ax.plot(t, mdot, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
    ax.set_ylabel(labels[0])
    ax.set_xlabel(time_xlabel())
    add_time_secondary_axis(ax)
    ax.set_yscale("log")
    ax.grid()
    if tfs:
        ax.set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(tfs))
    if values:
        all_values = np.concatenate(values)
        ax.set_ylim(np.min(all_values) / 1.5, np.max(all_values) * 3.0)
    ordered_sim_legend(ax, ncols=4, loc="lower right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(**SUBPLOT_MARGINS)
    return fig


def plot_m0dot_ratios(sims):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size("double", RATIO_FIG_HEIGHT),
        sharex=True,
        gridspec_kw={"hspace": RATIO_HSPACE},
    )
    tfs = []
    labels = _normalization_labels()
    for sim in sims:
        t = time_values(sim.M0dot_BH_t, sim)
        mdot = np.abs(np.asarray(sim.M0dot_BH, dtype=float))
        n = min(t.size, mdot.size)
        t = t[:n]
        mdot = mdot[:n]
        valid = np.isfinite(t) & np.isfinite(mdot) & (mdot > 0.0)
        if not np.any(valid):
            continue
        t = t[valid]
        mdot = mdot[valid]
        denominator, labels = _normalization(sim, mdot)
        tfs.append(t[-1])
        axes[0].plot(
            t,
            mdot / denominator,
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )
        axes[1].plot(
            t,
            denominator / mdot,
            label=sim.legend_name,
            linestyle=sim.linestyle,
            color=sim.color,
        )

    axes[0].set_ylabel(labels[0])
    axes[1].set_ylabel(labels[1])
    axes[1].set_xlabel(time_xlabel())
    add_time_secondary_axis(axes[0])
    for ax in axes:
        ax.set_yscale("log")
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    if tfs:
        axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(tfs))
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=RATIO_HSPACE, **RATIO_SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(
        fig,
        axes[0],
        axes,
        ncols=4,
        loc="center right",
        x=RATIO_SUBPLOT_MARGINS["right"],
        **COMPACT_LEGEND_KWARGS,
    )
    return fig


def main(argv=None):
    args = parser("Plot BH rest-mass accretion rate.").parse_args(argv)
    setup(args)
    sims = load_sims(["M0MADM"], names=args.sims)
    savefig(plot(sims), args, OUTPUT_FILENAME_RATE)
    savefig(plot_m0dot_ratios(sims), args, OUTPUT_FILENAME_RATE_AND_TIMESCALE)


if __name__ == "__main__":
    main()
