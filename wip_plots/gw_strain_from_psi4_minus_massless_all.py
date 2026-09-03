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
from helpers.gw_units import (
    add_gw_time_secondary_axis,
    difference_strain_ylabel,
    gw_time_values,
    gw_time_xlabel,
    normalize_strain_by_disk_mass,
)
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
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
OUTPUT_TEMPLATE = (
    "gw_strain_from_psi4_minus_{reference}_all_cases_{mode}_{radius}.png"
)

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


def difference_strain_results(sims):
    massless = next((sim for sim in sims if sim.config.name.upper() == MASSLESS_SIM_NAME), None)
    if massless is None:
        print(f"{MASSLESS_SIM_NAME}: missing; cannot reconstruct strain from Psi4 differences")
        return None, []

    results = []
    for sim in sims:
        if sim.config.name.upper() == MASSLESS_SIM_NAME:
            continue
        if not same_loaded_radius(sim, massless):
            print(
                f"{sim.config.name}: loaded r_A={loaded_areal_radius(sim)} is not compatible with "
                f"{MASSLESS_SIM_NAME} r_A={loaded_areal_radius(massless)}; skipping"
            )
            continue
        try:
            result = sim.strain_from_psi4_difference(massless)
        except (OSError, RuntimeError, ValueError) as exc:
            print(f"{sim.config.name}: could not reconstruct strain from Psi4-{MASSLESS_SIM_NAME}: {exc}")
            continue
        results.append((sim, result))
    return massless, results


def plot(results, massless, psi4_mode, parfile_index):
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
    ell, emm = psi4_mode
    mode = f"{ell}{emm}"

    for sim, result in results:
        hp, hc = result.hplus_hcross(ell=ell, emm=emm)
        hp = normalize_strain_by_disk_mass(hp, sim)
        hc = normalize_strain_by_disk_mass(hc, sim)
        t = gw_time_values(result.time, sim)
        n = min(t.size, hp.size, hc.size)
        if n < 2:
            print(f"{sim.config.name}: empty reconstructed ({ell},{emm}) strain; skipping")
            continue
        t = t[:n]
        hp = hp[:n]
        hc = hc[:n]
        tfs.append(t[-1])
        hp_yvals.append(hp)
        hc_yvals.append(hc)
        axes[0].plot(t, hp, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
        axes[1].plot(t, hc, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)

    if not tfs:
        plt.close(fig)
        return None

    axes[0].set_title(
        f"{mode_label(psi4_mode)}; {loaded_radius_label(massless, parfile_index)}; "
        "strain from disk-ML "
        r"$\Psi_4$"
    )
    axes[0].set_ylabel(difference_strain_ylabel("plus", mode, source=r"\Delta\Psi_4"))
    axes[1].set_ylabel(difference_strain_ylabel("cross", mode, source=r"\Delta\Psi_4"))
    axes[1].set_xlabel(gw_time_xlabel())
    for ax in axes:
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    add_gw_time_secondary_axis(axes[0])
    axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * np.max(tfs))
    set_symmetric_gw_ticks([axes[0]], hp_yvals)
    set_symmetric_gw_ticks([axes[1]], hc_yvals)
    format_shared_gw_yaxes(axes)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=HSPACE, **SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(
        fig,
        axes[1],
        axes,
        ncols=3,
        x=SUBPLOT_MARGINS["left"],
        **GW_LEGEND_KWARGS,
    )
    return fig


def main(argv=None):
    args = parser("Plot strain reconstructed from disk-minus-massless Psi4.").parse_args(argv)
    args.outdir = args.outdir / GW_DIFFERENCE_OUTPUT_SUBDIR
    setup(args)
    modes = selected_modes(PSI4_MODE)
    for parfile_index in GW_DIFFERENCE_PARFILE_INDICES:
        sims = load_sims(
            ["strain"],
            names=names_with_massless(args.sims),
            psi4_parfile_index=parfile_index,
            psi4_mode=modes[0],
        )
        massless, results = difference_strain_results(sims)
        if massless is None or not results:
            print(
                f"No strain-from-Psi4-difference results were available at "
                f"parfile index {parfile_index}."
            )
            continue

        for psi4_mode in modes:
            fig = plot(results, massless, psi4_mode, parfile_index)
            if fig is None:
                continue
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
