from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

from config import GW_OUTERMOST_PARFILE_INDEX
from helpers.plot_common import parser, save_individual_fig, setup
from helpers.reader import load_sims, selected_psi4_mode
from helpers.gw_units import add_gw_time_secondary_axis, gw_time_values, gw_time_xlabel, normalize_rpsi4, rpsi4_multimode_ylabel
from helpers.style import (
    FIGURE_LEGEND_BORDERAXESPAD,
    GW_LEGEND_KWARGS,
    INTERPANEL_LEGEND_INSET,
    PAPER_TWO_PANEL_HEIGHT,
    figure_size,
    format_shared_gw_yaxes,
    set_symmetric_gw_ticks,
)

# Critical knobs.
MULTIMODE_PSI4_PARFILE_INDEX = GW_OUTERMOST_PARFILE_INDEX
MULTIMODE_GW_MODES = ((2, 2), (2, 0), (2, 1), (4, 0))
LEFT_AXIS_MODES = ((2, 0), (4, 0))
RIGHT_AXIS_MODES = ((2, 2), (2, 1))
OUTPUT_TEMPLATE = "gw_psi4_modes_{sim}_{modes}_{radius}.png"

# Presentation knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.88, "bottom": 0.12, "top": 0.84}
HSPACE = 0.22
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def mode_tag(mode):
    ell, emm = mode
    return f"l{ell}m{emm}" if emm >= 0 else f"l{ell}mneg{abs(emm)}"


def mode_label(mode):
    ell, emm = mode
    return rf"$({ell},{emm})$"


def modes_tag(modes):
    return "_".join(mode_tag(mode) for mode in modes)


def modes_title(modes):
    return ", ".join(f"({ell},{emm})" for ell, emm in modes)


def radius_tag(sim, parfile_index):
    radius = getattr(sim, "psi4_radius", None)
    if radius is not None and np.isfinite(radius):
        value = float(radius)
        text = str(int(round(value))) if np.isclose(value, round(value)) else f"{value:g}"
        return f"r{text.replace('.', 'p')}"
    return f"extraction_index{int(parfile_index)}"


def nice_step_at_least(value):
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = int(np.floor(np.log10(value)))
    fraction = value / (10.0**exponent)
    for nice in (1.0, 2.0, 4.0, 5.0, 8.0, 10.0):
        if fraction <= nice:
            return nice * (10.0**exponent)
    return 10.0 ** (exponent + 1)


def finite_abs_values(values):
    arrays = [np.asarray(value) for value in values if np.asarray(value).size]
    if not arrays:
        return np.array([])
    combined = np.concatenate([np.ravel(array) for array in arrays])
    combined = np.abs(combined[np.isfinite(combined)])
    return combined[combined > 0]


def set_positive_gw_ticks(ax, values, ticks=3):
    abs_values = finite_abs_values(values)
    if not abs_values.size:
        return
    ymax = np.nanmax(abs_values)
    if not np.isfinite(ymax) or ymax <= 0:
        return
    step = nice_step_at_least(ymax / ticks)
    tick_max = ticks * step
    ax.set_ylim(0.0, tick_max)
    ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(ticks + 1) * step))


def mode_axis_side(mode):
    if tuple(mode) in {tuple(value) for value in RIGHT_AXIS_MODES}:
        return "right"
    return "left"


def mode_group_label(modes):
    return "$" + ",".join(f"({ell},{emm})" for ell, emm in modes) + "$"


def interpanel_mode_legend(fig, axes, handles, labels):
    positions = sorted((axis.get_position() for axis in axes), key=lambda pos: pos.y0, reverse=True)
    gap_midpoint = 0.5 * (positions[0].y0 + positions[1].y1)
    return fig.legend(
        handles,
        labels,
        ncols=2,
        loc="center left",
        bbox_to_anchor=(SUBPLOT_MARGINS["left"] + INTERPANEL_LEGEND_INSET, gap_midpoint),
        borderaxespad=FIGURE_LEGEND_BORDERAXESPAD,
        **GW_LEGEND_KWARGS,
    )


def plot_one(sim):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size("double", PAPER_TWO_PANEL_HEIGHT),
        sharex=True,
        gridspec_kw={"hspace": HSPACE},
    )
    right_axes = [ax.twinx() for ax in axes]
    t = gw_time_values(sim.rh_t, sim)
    tfs = [t[-1]]
    real_yvals = {"left": [], "right": []}
    abs_yvals = {"left": [], "right": []}
    legend_handles = []
    legend_labels = []

    plotted_modes = []
    for i, mode in enumerate(MULTIMODE_GW_MODES):
        ell, emm = mode
        try:
            z = normalize_rpsi4(selected_psi4_mode(sim.psi4, ell=ell, emm=emm, multiply_by_r=True), sim)
        except (AttributeError, IndexError, KeyError, ValueError) as exc:
            print(f"{sim.config.name}: skipping Psi4 mode {mode}; {exc}")
            continue
        n = min(t.size, z.size)
        z_real = np.real(z)
        z_abs = np.abs(z)
        side = mode_axis_side(mode)
        real_yvals[side].append(z_real[:n])
        abs_yvals[side].append(z_abs[:n])
        plot_axes = axes if side == "left" else right_axes
        color = COLORS[i % len(COLORS)]
        line, = plot_axes[0].plot(t[:n], z_real[:n], label=mode_label(mode), color=color)
        plot_axes[1].plot(t[:n], z_abs[:n], label=mode_label(mode), color=color)
        legend_handles.append(line)
        legend_labels.append(mode_label(mode))
        plotted_modes.append(mode)

    radius_label = sim.gw_extraction_plot_label(MULTIMODE_PSI4_PARFILE_INDEX)
    title_modes = modes_title(plotted_modes) if plotted_modes else modes_title(MULTIMODE_GW_MODES)
    axes[0].set_title(f"{sim.config.name}: {radius_label}; modes {title_modes}")
    axes[0].set_ylabel(rpsi4_multimode_ylabel("real") + "\n" + mode_group_label(LEFT_AXIS_MODES))
    axes[1].set_ylabel(rpsi4_multimode_ylabel("abs") + "\n" + mode_group_label(LEFT_AXIS_MODES))
    right_axes[0].set_ylabel(rpsi4_multimode_ylabel("real") + "\n" + mode_group_label(RIGHT_AXIS_MODES))
    right_axes[1].set_ylabel(rpsi4_multimode_ylabel("abs") + "\n" + mode_group_label(RIGHT_AXIS_MODES))
    axes[1].set_xlabel(gw_time_xlabel())

    for ax in axes:
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    for ax in right_axes:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    add_gw_time_secondary_axis(axes[0])
    if tfs:
        axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(tfs))
    set_symmetric_gw_ticks([axes[0]], real_yvals["left"])
    set_symmetric_gw_ticks([right_axes[0]], real_yvals["right"])
    set_positive_gw_ticks(axes[1], abs_yvals["left"])
    set_positive_gw_ticks(right_axes[1], abs_yvals["right"])

    all_y_axes = list(axes) + right_axes
    format_shared_gw_yaxes(all_y_axes)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    interpanel_mode_legend(fig, axes, legend_handles, legend_labels)
    return fig


def main(argv=None):
    args = parser("Plot Psi4 modes at one extraction radius for each target simulation.").parse_args(argv)
    setup(args)
    for sim in load_sims(
        ["strain"],
        names=args.sims,
        psi4_parfile_index=MULTIMODE_PSI4_PARFILE_INDEX,
        psi4_mode=MULTIMODE_GW_MODES[0],
    ):
        save_individual_fig(
            plot_one(sim),
            args,
            sim,
            OUTPUT_TEMPLATE.format(
                sim=sim.config.name,
                modes=modes_tag(MULTIMODE_GW_MODES),
                radius=radius_tag(sim, MULTIMODE_PSI4_PARFILE_INDEX),
            ),
        )


if __name__ == "__main__":
    main()
