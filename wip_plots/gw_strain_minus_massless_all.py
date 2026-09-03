from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from helpers.gw_difference import (
    GW_DIFFERENCE_OUTPUT_SUBDIR,
    GW_DIFFERENCE_PARFILE_INDICES,
    MASSLESS_SIM_NAME,
    loaded_areal_radius,
    loaded_radius_label,
    loaded_radius_tag,
    names_with_massless,
    same_loaded_radius,
)
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.gw_units import (
    add_gw_time_secondary_axis,
    difference_strain_ylabel,
    gw_time_values,
    gw_time_xlabel,
    normalize_strain_by_disk_mass,
)
from helpers.style import (
    GW_LEGEND_KWARGS,
    PAPER_TWO_PANEL_HEIGHT,
    figure_size,
    format_shared_gw_yaxes,
    ordered_sim_interpanel_legend,
    set_symmetric_gw_ticks,
)

# Critical knobs.
PSI4_MODE = ((2, 2), (2, 1), (2, 0), (4, 0))
OUTPUT_TEMPLATE = "gw_strain_minus_{reference}_all_cases_{mode}_{radius}.png"

# Presentation knobs.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.96, "bottom": 0.12, "top": 0.86}
HSPACE = 0.28


def mode_tag(mode):
    ell, emm = mode
    return f"l{ell}m{emm}" if emm >= 0 else f"l{ell}mneg{abs(emm)}"


def mode_label(mode):
    ell, emm = mode
    return rf"$({ell},{emm})$"


def selected_modes(psi4_mode):
    if len(psi4_mode) == 2 and all(np.isscalar(part) for part in psi4_mode):
        return (tuple(int(part) for part in psi4_mode),)
    return tuple(tuple(int(part) for part in mode) for mode in psi4_mode)


def finite_monotone_time_and_values(t, *values):
    t = np.asarray(t, dtype=float)
    arrays = [np.asarray(value, dtype=float) for value in values]
    n = min([t.size] + [array.size for array in arrays])
    t = t[:n]
    arrays = [array[:n] for array in arrays]
    keep = np.isfinite(t)
    for array in arrays:
        keep &= np.isfinite(array)
    if np.count_nonzero(keep) < 2:
        return (np.array([], dtype=float),) + tuple(np.array([], dtype=float) for _ in arrays)
    t = t[keep]
    arrays = [array[keep] for array in arrays]
    order = np.argsort(t, kind="mergesort")
    t = t[order]
    arrays = [array[order] for array in arrays]
    _, keep_unique = np.unique(t, return_index=True)
    keep_unique = np.sort(keep_unique)
    return (t[keep_unique],) + tuple(array[keep_unique] for array in arrays)


def residual_on_case_grid(sim, massless):
    case_t, case_hp, case_hc = finite_monotone_time_and_values(
        sim.rh_t,
        sim.rh_plus_lm,
        sim.rh_cross_lm,
    )
    ml_t, ml_hp, ml_hc = finite_monotone_time_and_values(
        massless.rh_t,
        massless.rh_plus_lm,
        massless.rh_cross_lm,
    )
    if case_t.size < 2 or ml_t.size < 2:
        return None

    t_min = max(float(case_t[0]), float(ml_t[0]))
    t_max = min(float(case_t[-1]), float(ml_t[-1]))
    if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
        return None

    keep = (case_t >= t_min) & (case_t <= t_max)
    if np.count_nonzero(keep) < 2:
        return None

    t = case_t[keep]
    hp = normalize_strain_by_disk_mass(case_hp[keep] - np.interp(t, ml_t, ml_hp), sim)
    hc = normalize_strain_by_disk_mass(case_hc[keep] - np.interp(t, ml_t, ml_hc), sim)
    return t, hp, hc


def plot(sims, psi4_mode, parfile_index):
    massless = next((sim for sim in sims if sim.config.name.upper() == MASSLESS_SIM_NAME), None)
    if massless is None:
        print(f"{MASSLESS_SIM_NAME}: missing; cannot make massless-subtracted GW plot")
        return None

    fig, axes = plt.subplots(
        2,
        1,
        figsize=figure_size("double", PAPER_TWO_PANEL_HEIGHT),
        sharex=True,
        gridspec_kw={"hspace": HSPACE},
    )
    tfs = []
    hp_yvals = []
    hc_yvals = []
    mode = f"{psi4_mode[0]}{psi4_mode[1]}"
    plotted = False

    for sim in sims:
        if sim.config.name.upper() == MASSLESS_SIM_NAME:
            continue
        if not same_loaded_radius(sim, massless):
            sim_radius = loaded_areal_radius(sim)
            ml_radius = loaded_areal_radius(massless)
            print(
                f"{sim.config.name}: loaded r_A={sim_radius} is not compatible with "
                f"{MASSLESS_SIM_NAME} r_A={ml_radius}; skipping"
            )
            continue
        residual = residual_on_case_grid(sim, massless)
        if residual is None:
            print(f"{sim.config.name}: no common retarded-time overlap with {MASSLESS_SIM_NAME}; skipping")
            continue
        t, hp, hc = residual
        x = gw_time_values(t, sim)
        tfs.append(x[-1])
        hp_yvals.append(hp)
        hc_yvals.append(hc)
        axes[0].plot(x, hp, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        axes[1].plot(x, hc, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        plotted = True

    if not plotted:
        plt.close(fig)
        print("No massless-subtracted GW residuals were plotted.")
        return None

    radius_label = loaded_radius_label(massless, parfile_index)
    axes[0].set_title(f"{mode_label(psi4_mode)}; {radius_label}; disk-{MASSLESS_SIM_NAME}")
    axes[0].set_ylabel(difference_strain_ylabel("plus", mode))
    axes[1].set_ylabel(difference_strain_ylabel("cross", mode))
    axes[1].set_xlabel(gw_time_xlabel())

    for ax in axes:
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    add_gw_time_secondary_axis(axes[0])
    if tfs:
        axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(tfs))
    set_symmetric_gw_ticks([axes[0]], hp_yvals)
    set_symmetric_gw_ticks([axes[1]], hc_yvals)
    format_shared_gw_yaxes(axes)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(fig, axes[1], axes, ncols=3, x=SUBPLOT_MARGINS["left"], **GW_LEGEND_KWARGS)
    return fig


def main(argv=None):
    args = parser("Plot GW strain after subtracting the massless run at the same extraction radius.").parse_args(argv)
    args.outdir = args.outdir / GW_DIFFERENCE_OUTPUT_SUBDIR
    setup(args)
    for parfile_index in GW_DIFFERENCE_PARFILE_INDICES:
        for psi4_mode in selected_modes(PSI4_MODE):
            sims = load_sims(
                ["strain"],
                names=names_with_massless(args.sims),
                psi4_parfile_index=parfile_index,
                psi4_mode=psi4_mode,
            )
            fig = plot(sims, psi4_mode, parfile_index)
            if fig is None:
                continue
            massless = next(
                sim for sim in sims if sim.config.name.upper() == MASSLESS_SIM_NAME
            )
            savefig(
                fig,
                args,
                OUTPUT_TEMPLATE.format(
                    reference=MASSLESS_SIM_NAME,
                    mode=mode_tag(psi4_mode),
                    radius=loaded_radius_tag(massless, parfile_index),
                ),
            )


if __name__ == "__main__":
    main()
