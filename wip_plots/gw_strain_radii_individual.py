from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from config import GW_COMPARISON_PARFILE_INDICES
from helpers.gw_units import (
    add_gw_time_secondary_axis,
    gw_time_xlabel,
    gw_time_values,
    normalize_strain,
    strain_ylabel,
)
from helpers.plot_common import parser, save_individual_fig, setup
from helpers.reader import load_sims
from helpers.style import (
    GW_LEGEND_KWARGS,
    PAPER_TWO_PANEL_HEIGHT,
    figure_size,
    format_shared_gw_yaxes,
    interpanel_legend,
    set_symmetric_gw_ticks,
)

# Critical knobs.
INDIVIDUAL_GW_PARFILE_INDICES = GW_COMPARISON_PARFILE_INDICES
PSI4_MODE = (2, 2)
OUTPUT_TEMPLATE = "rhphc_{sim}_{mode}_{radii}.png"

# Presentation knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.96, "bottom": 0.12, "top": 0.84}
HSPACE = 0.28
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


def mode_tag(mode):
    ell, emm = mode
    return f"l{ell}m{emm}" if emm >= 0 else f"l{ell}mneg{abs(emm)}"


def mode_label(mode):
    ell, emm = mode
    return rf"$(\ell,m)=({ell},{emm})$"


def radii_tag(sim, indices):
    radii = []
    for index in indices:
        radius_index = sim.gw_extraction_radius_index(index)
        radius = sim.psi4_extraction_radii.get(radius_index)
        if radius is None or not np.isfinite(radius):
            continue
        value = float(radius)
        text = str(int(round(value))) if np.isclose(value, round(value)) else f"{value:g}"
        radii.append(f"r{text.replace('.', 'p')}")
    if radii:
        return "-".join(radii)
    return "extraction_indices" + "-".join(str(int(index)) for index in indices)


def plot_one(sim):
    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size("double", PAPER_TWO_PANEL_HEIGHT),
        sharex=True,
        gridspec_kw={"hspace": HSPACE},
    )
    final_times = []
    hplus_values = []
    hcross_values = []
    mode = f"{sim.psi4_mode[0]}{sim.psi4_mode[1]}"
    radius_labels = []

    for color_index, parfile_index in enumerate(INDIVIDUAL_GW_PARFILE_INDICES):
        result_key = str(parfile_index + 1)
        if result_key not in sim.strain_radii:
            continue
        color = COLORS[color_index % len(COLORS)]
        radius_label = sim.gw_extraction_plot_label(parfile_index)
        time = gw_time_values(sim.rh_t_radii[result_key], sim)
        hplus = normalize_strain(sim.rh_plus_lm_radii[result_key], sim)
        hcross = normalize_strain(sim.rh_cross_lm_radii[result_key], sim)

        radius_labels.append(radius_label)
        final_times.append(time[-1])
        hplus_values.append(hplus)
        hcross_values.append(hcross)
        axes[0].plot(time, hplus, label=radius_label, color=color)
        axes[1].plot(time, hcross, label=radius_label, color=color)

    radius_title = ", ".join(radius_labels) if radius_labels else "requested radii"
    axes[0].set_ylabel(strain_ylabel("plus", mode))
    axes[1].set_ylabel(strain_ylabel("cross", mode))
    axes[1].set_xlabel(gw_time_xlabel())
    axes[0].set_title(
        f"{sim.config.name}: {mode_label(sim.psi4_mode)}; {radius_title}"
    )
    for ax in axes:
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    add_gw_time_secondary_axis(axes[0])
    if final_times:
        axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(final_times))
    set_symmetric_gw_ticks([axes[0]], hplus_values)
    set_symmetric_gw_ticks([axes[1]], hcross_values)
    format_shared_gw_yaxes(axes)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    interpanel_legend(
        fig,
        axes[0],
        axes,
        ncols=3,
        x=SUBPLOT_MARGINS["left"],
        **GW_LEGEND_KWARGS,
    )
    return fig


def main(argv=None):
    args = parser("Plot GW strain extraction radii.").parse_args(argv)
    setup(args)
    sims = load_sims(
        ["strain_radii"],
        names=args.sims,
        gw_parfile_indices=INDIVIDUAL_GW_PARFILE_INDICES,
        psi4_mode=PSI4_MODE,
    )
    for sim in sims:
        save_individual_fig(
            plot_one(sim),
            args,
            sim,
            OUTPUT_TEMPLATE.format(
                sim=sim.config.name,
                mode=mode_tag(PSI4_MODE),
                radii=radii_tag(sim, INDIVIDUAL_GW_PARFILE_INDICES),
            ),
        )


if __name__ == "__main__":
    main()
