"""Combined Psi4, strain, and ML-difference figures. Run with run_wip.py."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config import (
    GW_COMPARISON_PARFILE_INDICES, GW_PLOT_MODES, MASSLESS_SIM_NAME,
    GW_OUTERMOST_PARFILE_INDEX, PSI4_MODE,
    GW_MULTIMODE_PLOT_MODES, GW_MULTIMODE_LEFT_MODES, GW_MULTIMODE_RIGHT_MODES,
    all_sim_configs,
)
from gw import GWRun, load_gw_sims, names_with_massless, same_loaded_radius, subtract_waveforms, read_difference
from helpers.gw_units import (
    add_gw_time_secondary_axis, gw_time_values, gw_time_xlabel,
    normalize_rpsi4, normalize_strain, normalize_rpsi4_by_disk_mass,
    normalize_strain_by_disk_mass, rpsi4_ylabel, strain_ylabel,
    difference_rpsi4_ylabel, difference_strain_ylabel,
    loaded_radius_label, loaded_radius_tag,
    rpsi4_multimode_ylabel, strain_multimode_ylabel,
    extraction_radius_label,
)
from helpers.plot_common import parser, savefig, save_individual_fig, setup
from helpers.style import (
    GW_LEGEND_KWARGS, PAPER_TWO_PANEL_HEIGHT, figure_size,
    format_shared_gw_yaxes, ordered_sim_interpanel_legend, set_symmetric_gw_ticks,
    interpanel_legend, INTERPANEL_LEGEND_INSET, FIGURE_LEGEND_BORDERAXESPAD,
)

# Data selection lives in config.py. These are plot-only presentation settings.
TIME_XMIN = 0.0
TIME_XMAX_PAD = 1.08
SUBPLOT_MARGINS = {"left": 0.18, "right": 0.96, "bottom": 0.12, "top": 0.86}
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
OUTPUTS = {
    "psi4": "gw/rpsi4_{mode}_{radius}.png",
    "strain": "gw/rhphc_{mode}_{radius}.png",
    "psi4_minus_ml": "gw/difference/gw_psi4_minus_ML_all_cases_{mode}_{radius}.png",
    "strain_minus_ml": "gw/difference/gw_strain_minus_ML_all_cases_{mode}_{radius}.png",
    "strain_from_psi4_minus_ml": "gw/difference/gw_strain_from_psi4_minus_ML_all_cases_{mode}_{radius}.png",
}


def mode_tag(mode):
    ell, emm = mode
    return f"l{ell}m{emm}" if emm >= 0 else f"l{ell}mneg{abs(emm)}"


def selected_modes(value):
    if len(value) == 2 and all(np.isscalar(part) for part in value):
        return (tuple(int(part) for part in value),)
    return tuple(tuple(int(part) for part in mode) for mode in value)


def psi4_series(sim, mode):
    values = sim.psi4.psi4(*mode, multiply_by_r=True)
    n = min(sim.rh_t.size, values.size)
    values = normalize_rpsi4(values[:n], sim)
    return sim.rh_t[:n], values.real, np.abs(values)


def strain_series(sim, mode):
    hp, hc = sim.strain_result.hplus_hcross(*mode)
    return sim.rh_t, normalize_strain(hp, sim), normalize_strain(hc, sim)


def psi4_difference_series(sim, reference, mode):
    residual = subtract_waveforms(
        sim.rh_t, sim.psi4.psi4(*mode, multiply_by_r=True),
        reference.rh_t, reference.psi4.psi4(*mode, multiply_by_r=True),
    )
    if residual is None:
        return None
    time, values = residual
    values = normalize_rpsi4_by_disk_mass(values, sim)
    return time, values.real, np.abs(values)


def strain_difference_series(sim, reference, mode):
    hp, hc = sim.strain_result.hplus_hcross(*mode)
    ref_hp, ref_hc = reference.strain_result.hplus_hcross(*mode)
    residual = subtract_waveforms(sim.rh_t, hp + 1j * hc,
                                  reference.rh_t, ref_hp + 1j * ref_hc)
    if residual is None:
        return None
    time, values = residual
    values = normalize_strain_by_disk_mass(values, sim)
    return time, values.real, values.imag


def integrated_difference_series(sim, result, mode):
    hp, hc = result.hplus_hcross(*mode)
    return result.time, normalize_strain_by_disk_mass(hp, sim), normalize_strain_by_disk_mass(hc, sim)


def waveform(sim, mode, kind, reference=None, difference_result=None):
    """Select an explicit data path; the drawing code is shared."""
    if kind == "psi4":
        return psi4_series(sim, mode)
    if kind == "strain":
        return strain_series(sim, mode)
    if kind == "psi4_minus_ml":
        return psi4_difference_series(sim, reference, mode)
    if kind == "strain_minus_ml":
        return strain_difference_series(sim, reference, mode)
    if kind == "strain_from_psi4_minus_ml":
        return integrated_difference_series(sim, difference_result, mode)
    raise ValueError(f"Unknown waveform plot {kind!r}")


def plot(sims, mode, parfile_index, kind, difference_results=None):
    is_difference = kind not in ("psi4", "strain")
    is_psi4 = kind in ("psi4", "psi4_minus_ml")
    reference = next((s for s in sims if s.config.name == MASSLESS_SIM_NAME), None)
    if is_difference and reference is None:
        print("ML missing; cannot make difference plot")
        return None

    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=figure_size("double", PAPER_TWO_PANEL_HEIGHT))
    final_times, top_values, bottom_values = [], [], []
    for sim in sims:
        if is_difference and sim is reference:
            continue
        if is_difference and not same_loaded_radius(sim, reference):
            print(f"{sim.config.name}: extraction radius incompatible with ML; skipping")
            continue
        result = (difference_results or {}).get(sim.config.name)
        if kind == "strain_from_psi4_minus_ml" and result is None:
            continue
        data = waveform(sim, mode, kind, reference if is_difference else None, result)
        if data is None or len(data[0]) < 2:
            print(f"{sim.config.name}: no usable common waveform interval; skipping")
            continue
        time, top, bottom = data
        x = gw_time_values(time, sim)
        final_times.append(x[-1])
        top_values.append(top)
        bottom_values.append(bottom)
        for ax, y in zip(axes, (top, bottom)):
            ax.plot(x, y, label=sim.legend_name, linestyle=sim.linestyle, color=sim.color)
    if not final_times:
        plt.close(fig)
        return None

    ell, emm = mode
    mode_text = f"{ell}{emm}"
    if is_difference:
        title = rf"$({ell},{emm})$; {loaded_radius_label(reference, parfile_index)}; "
        title += r"strain from disk-ML $\Psi_4$" if kind == "strain_from_psi4_minus_ml" else "disk-ML"
        label = difference_rpsi4_ylabel if is_psi4 else difference_strain_ylabel
    else:
        title = rf"$(\ell,m)=({ell},{emm})$; {extraction_radius_label(sims[0])}"
        label = rpsi4_ylabel if is_psi4 else strain_ylabel
    axes[0].set_title(title)
    label_kwargs = {"source": r"\Delta\Psi_4"} if kind == "strain_from_psi4_minus_ml" else {}
    for ax, component in zip(axes, ("real", "abs") if is_psi4 else ("plus", "cross")):
        ax.set_ylabel(label(component, mode_text, **label_kwargs))
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    axes[1].set_xlabel(gw_time_xlabel())
    add_gw_time_secondary_axis(axes[0])
    axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * max(final_times))
    if not is_psi4:
        set_symmetric_gw_ticks([axes[0]], top_values)
        set_symmetric_gw_ticks([axes[1]], bottom_values)
    format_shared_gw_yaxes(axes)
    fig.align_ylabels(axes)
    fig.subplots_adjust(hspace=0.22 if is_psi4 else 0.28, **SUBPLOT_MARGINS)
    ordered_sim_interpanel_legend(fig, axes[1], axes, ncols=3,
                                 x=SUBPLOT_MARGINS["left"], **GW_LEGEND_KWARGS)
    return fig


def main(argv=None, kinds=None):
    args = parser(__doc__).parse_args(argv)
    setup(args)
    kinds = tuple(OUTPUTS) if kinds is None else tuple(kinds)
    if any(kind not in OUTPUTS for kind in kinds):
        raise ValueError(f"Choose waveform plots from {tuple(OUTPUTS)}")
    modes = selected_modes(GW_PLOT_MODES)
    names = names_with_massless(args.sims) if any(k not in ("psi4", "strain") for k in kinds) else args.sims
    for radius in GW_COMPARISON_PARFILE_INDICES:
        sims = load_gw_sims(names=names, psi4_parfile_index=radius)
        if not sims:
            continue
        reference = next((s for s in sims if s.config.name == MASSLESS_SIM_NAME), None)
        differences = {}
        if "strain_from_psi4_minus_ml" in kinds and reference is not None:
            for sim in sims:
                if sim is reference or not same_loaded_radius(sim, reference):
                    continue
                try:
                    differences[sim.config.name] = read_difference(sim, reference)
                except (OSError, RuntimeError, ValueError) as exc:
                    print(f"{sim.config.name}: missing difference cache: {exc}")
        for kind in kinds:
            for mode in modes:
                fig = plot(sims, mode, radius, kind, differences)
                if fig is not None:
                    label_sim = sims[0] if kind in ("psi4", "strain") else reference
                    savefig(fig, args, OUTPUTS[kind].format(
                        mode=mode_tag(mode), radius=loaded_radius_tag(label_sim, radius)))


def _positive_ticks(ax, values, ticks=3):
    values = np.abs(np.concatenate(values)) if values else np.array([])
    values = values[np.isfinite(values) & (values > 0)]
    if not values.size:
        return
    target = values.max() / ticks
    scale = 10.0 ** np.floor(np.log10(target))
    step = next(n for n in (1, 2, 4, 5, 8, 10) if target / scale <= n) * scale
    ax.set_ylim(0, ticks * step)
    ax.yaxis.set_major_locator(mticker.FixedLocator(np.arange(ticks + 1) * step))


def plot_individual(radii, kind, multimode=False):
    """One case: compare radii, or compare modes with separate m=0 axes."""
    is_psi4 = kind == "psi4"
    fig, axes = plt.subplots(2, 1, sharex=True,
                             figsize=figure_size("double", PAPER_TWO_PANEL_HEIGHT))
    right_axes = [ax.twinx() for ax in axes] if multimode else []
    plotted = {"left": [[], []], "right": [[], []]}
    handles, labels, final_times = [], [], []
    selections = GW_MULTIMODE_PLOT_MODES if multimode else GW_COMPARISON_PARFILE_INDICES
    for i, selection in enumerate(selections):
        mode = selection if multimode else PSI4_MODE
        radius = GW_OUTERMOST_PARFILE_INDEX if multimode else selection
        if radius not in radii:
            continue
        sim = radii[radius]
        try:
            time, top, bottom = waveform(sim, mode, kind)
        except (OSError, AttributeError, IndexError, KeyError, ValueError) as exc:
            print(f"{sim.config.name}: skipping mode {mode}, radius index {radius}: {exc}")
            continue
        side = "right" if multimode and mode in GW_MULTIMODE_RIGHT_MODES else "left"
        target_axes = right_axes if side == "right" else axes
        label = rf"$({mode[0]},{mode[1]})$" if multimode else extraction_radius_label(sim)
        x = gw_time_values(time, sim)
        final_times.append(x[-1])
        for j, (ax, y) in enumerate(zip(target_axes, (top, bottom))):
            line, = ax.plot(x, y, label=label, color=COLORS[i % len(COLORS)])
            plotted[side][j].append(y)
            if j == 0:
                handles.append(line)
        labels.append(label)
    if not final_times:
        plt.close(fig)
        return None

    components = ("real", "abs") if is_psi4 else ("plus", "cross")
    if multimode:
        label_fn = rpsi4_multimode_ylabel if is_psi4 else strain_multimode_ylabel
        for axis_group, group in ((axes, GW_MULTIMODE_LEFT_MODES), (right_axes, GW_MULTIMODE_RIGHT_MODES)):
            group_label = "$" + ",".join(f"({ell},{emm})" for ell, emm in group) + "$"
            for ax, component in zip(axis_group, components):
                ax.set_ylabel(label_fn(component) + "\n" + group_label)
        title = extraction_radius_label(sim)
        title += "; modes " + ", ".join(f"({ell},{emm})" for ell, emm in GW_MULTIMODE_PLOT_MODES)
    else:
        label_fn = rpsi4_ylabel if is_psi4 else strain_ylabel
        for ax, component in zip(axes, components):
            ax.set_ylabel(label_fn(component, f"{PSI4_MODE[0]}{PSI4_MODE[1]}"))
        title = rf"$(\ell,m)=({PSI4_MODE[0]},{PSI4_MODE[1]})$; " + ", ".join(labels)
    axes[0].set_title(f"{sim.config.name}: {title}")
    axes[1].set_xlabel(gw_time_xlabel())
    for ax in axes:
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    for ax in right_axes:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    add_gw_time_secondary_axis(axes[0])
    axes[0].set_xlim(TIME_XMIN, TIME_XMAX_PAD * max(final_times))
    for side, axis_group in (("left", axes), ("right", right_axes)):
        for j, ax in enumerate(axis_group):
            if not is_psi4 or (multimode and j == 0):
                set_symmetric_gw_ticks([ax], plotted[side][j])
            elif multimode:
                _positive_ticks(ax, plotted[side][j])
    format_shared_gw_yaxes(list(axes) + right_axes)
    fig.align_ylabels(axes)
    margins = dict(SUBPLOT_MARGINS, top=0.84, right=0.88 if multimode else 0.96)
    fig.subplots_adjust(hspace=0.22 if is_psi4 else 0.28, **margins)
    if multimode:
        midpoint = (axes[0].get_position().y0 + axes[1].get_position().y1) / 2
        fig.legend(handles, labels, ncols=2, loc="center left",
                   bbox_to_anchor=(margins["left"] + INTERPANEL_LEGEND_INSET, midpoint),
                   borderaxespad=FIGURE_LEGEND_BORDERAXESPAD, **GW_LEGEND_KWARGS)
    else:
        interpanel_legend(fig, axes[0], axes, ncols=3, x=margins["left"], **GW_LEGEND_KWARGS)
    return fig


def individual_main(argv=None):
    args = parser("Individual GW radius and mode comparisons.").parse_args(argv)
    setup(args)
    for config in all_sim_configs(args.sims):
        try:
            run = GWRun(config)
        except (OSError, ValueError) as exc:
            print(f"{config.name}: skipping GW plots; {exc}")
            continue
        radii = {}
        for radius in dict.fromkeys((*GW_COMPARISON_PARFILE_INDICES, GW_OUTERMOST_PARFILE_INDEX)):
            try:
                radii[radius] = run.at_radius(radius)
            except (OSError, ValueError) as exc:
                print(f"{config.name}: skipping radius index {radius}; {exc}")
        radius_tags = [loaded_radius_tag(radii[r], r) for r in GW_COMPARISON_PARFILE_INDICES if r in radii]
        for multimode in (False, True):
            for kind in ("psi4", "strain"):
                fig = plot_individual(radii, kind, multimode)
                if fig is None:
                    continue
                if multimode:
                    modes = "_".join(mode_tag(mode) for mode in GW_MULTIMODE_PLOT_MODES)
                    filename = f"gw_{kind}_modes_{config.name}_{modes}_{loaded_radius_tag(radii[GW_OUTERMOST_PARFILE_INDEX], GW_OUTERMOST_PARFILE_INDEX)}.png"
                else:
                    prefix = "rpsi4" if kind == "psi4" else "rhphc"
                    filename = f"{prefix}_{config.name}_{mode_tag(PSI4_MODE)}_{'-'.join(radius_tags)}.png"
                save_individual_fig(fig, args, config.name, filename)


if __name__ == "__main__":
    main()
