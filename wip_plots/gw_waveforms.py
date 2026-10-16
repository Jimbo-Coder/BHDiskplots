"""GW time histories: Psi4, FFI strain, disk-minus-ML, and observer views.

Run: python -m wip_plots.gw_waveforms --kinds strain --radius-index 6 --sims A1 A3

Read settings first, then waveform() for the actual data preparation, then
plotting. The batch/individual commands and CLI overrides are at the bottom.
Use --kinds strain_panel or --kinds strain_observer for other strain views.
The default draws all views at the configured comparison radii; observer
strain and radiated diagnostics use the outer radius. Select --kinds
gw_radiated for energy, angular momentum and phase-frequency histories.
"""
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

from config import (
    GW_COMPARISON_PARFILE_INDICES, GW_PLOT_MODES, MASSLESS_SIM_NAME,
    GW_OUTERMOST_PARFILE_INDEX, PSI4_MODE,
    GW_MULTIMODE_PLOT_MODES, GW_MULTIMODE_LEFT_MODES, GW_MULTIMODE_RIGHT_MODES,
    all_sim_configs,
)
from gw import (
    GWRun, names_with_massless, same_loaded_radius, subtract_waveforms,
    read_difference, MODES as CACHED_MODES, read_strain_cache,
    strain_cache_dir, mode_columns, _radiated_diagnostics,
)
from helpers import gw_units as units
from gw_detectability import (
    FlatLambdaCDM, METERS_PER_M_SUN, METERS_PER_MPC, SECONDS_PER_M_SUN,
    MODES as RADIATED_MODES, TRANSIENT_CUTOFF_MBH,
)

from helpers.plot_common import parser, savefig, save_individual_fig, setup
from helpers.style import (
    GW_LEGEND_KWARGS, PAPER_TWO_PANEL_HEIGHT, PAPER_THREE_PANEL_HEIGHT, figure_size,
    COLOR_OPTS, COMPACT_LEGEND_KWARGS, format_paper_axes, apply_styles, ordered_sim_legend,
    format_shared_gw_yaxes, ordered_sim_interpanel_legend, set_symmetric_gw_ticks,
    interpanel_legend, INTERPANEL_LEGEND_INSET, FIGURE_LEGEND_BORDERAXESPAD,
)

# Shared case/mode/radius defaults and units live in config.py. View-specific
# physical choices and appearance are below; command overrides are at the end.
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


# Two presentations of the same FFI strain; neither changes its integration.
PANEL_OUTPUT = "gw/rhphc_panel_{mode}_{radius}.png"
PANEL_HEIGHT = 4.9
PANEL_TIME_XMAX_PAD = 1.03
PANEL_MARGINS = dict(left=.10, right=.985, bottom=.11, top=.80, wspace=.10, hspace=.27)
PLUS_LINESTYLE, CROSS_LINESTYLE = "-", "--"

# Source and data selection. M_BH is the initial BH mass, never M_ADM.
CASE_GROUPS = (("A1", "A2", "A3"), ("B1", "B2", "B3"))
OBSERVER_SOURCE_BH_MASS_MSUN = 50.0
OBSERVER_DISTANCE_MPC = 100.0
OBSERVER_INCLUDE_REDSHIFT = True  # Same cosmology as detectability; z ~ 0.022 at 100 Mpc.
OBSERVER_MODES = ((2, 2), (2, 1), (2, -2), (2, -1))  # Or "all" (cached ell=2..4).
OBSERVER_MIN_TRET_MBH = 1000.0  # Same initial-relaxation exclusion as detectability.
OBSERVER_ALIGN_PEAK = True  # Shift each retained hypot(h_plus, h_cross) maximum to zero.
OBSERVER_TIME_LIMITS_S = None  # None shows the full retained duration; no extrapolation.

# Output and presentation. Dashed means cross here, as in Shibata Fig. 4;
# the cases have separate rows, so this does not replace the A/B line convention.
OBSERVER_OUTPUT = "gw/strain_observer_{family}.png"
OBSERVER_HEIGHT = PAPER_THREE_PANEL_HEIGHT
OBSERVER_SHARE_Y_LIMITS = False  # Individual symmetric ranges, as in the paper.
OBSERVER_LINE_WIDTH = 1.0
OBSERVER_MARGINS = dict(left=0.20, right=0.975, bottom=0.09, top=0.92, hspace=0.18)


# Radiated energy, J_z and phase-frequency diagnostic, using the same caches.
RADIATED_SMOOTHING_ORBITS = 1.0
RADIATED_OUTPUT = "wip/gw_radiated.png"
RADIATED_SUMMARY = "wip/gw_radiated.json"

# Batch defaults include all presentations; select only what you need.
PLOTS = (*OUTPUTS, "strain_panel", "strain_observer", "gw_radiated")


def mode_tag(mode):
    ell, emm = mode
    return f"l{ell}m{emm}" if emm >= 0 else f"l{ell}mneg{abs(emm)}"


# 1. DATA PREPARATION: select columns, align/subtract if needed, then normalize.
def waveform(sim, mode, kind, reference=None, difference_result=None):
    """Return (time, upper-panel values, lower-panel values) from cached arrays.

    With GW_NORMALIZE_BY_M enabled, Psi4 uses M_BH*r*Psi4 and strain uses
    r*h/M_BH. Difference curves instead use the initial disk mass, keeping
    subtraction after FFI distinct from FFI of the Psi4 difference. Shared
    unit functions also supply the axis labels for these same conventions.
    """
    if kind == "psi4":
        values = sim.psi4.psi4(*mode, multiply_by_r=True)
        count = min(sim.rh_t.size, values.size)
        values = units.normalize_rpsi4(values[:count], sim)
        return sim.rh_t[:count], values.real, np.abs(values)

    if kind == "strain":
        hp, hc = sim.strain_result.hplus_hcross(*mode)
        return sim.rh_t, units.normalize_strain(hp, sim), units.normalize_strain(hc, sim)

    if kind == "psi4_minus_ml":
        residual = subtract_waveforms(
            sim.rh_t, sim.psi4.psi4(*mode, multiply_by_r=True),
            reference.rh_t, reference.psi4.psi4(*mode, multiply_by_r=True),
        )
        if residual is None:
            return None
        time, values = residual
        values = units.normalize_rpsi4_by_disk_mass(values, sim)
        return time, values.real, np.abs(values)

    if kind == "strain_minus_ml":
        hp, hc = sim.strain_result.hplus_hcross(*mode)
        ref_hp, ref_hc = reference.strain_result.hplus_hcross(*mode)
        # Pack the two real channels together for the same time interpolation.
        residual = subtract_waveforms(
            sim.rh_t, hp + 1j * hc, reference.rh_t, ref_hp + 1j * ref_hc)
        if residual is None:
            return None
        time, values = residual
        values = units.normalize_strain_by_disk_mass(values, sim)
        return time, values.real, values.imag

    if kind == "strain_from_psi4_minus_ml":
        # This cache was made by integrating the Psi4 difference, not by
        # subtracting the two separately integrated strains above.
        hp, hc = difference_result.hplus_hcross(*mode)
        return (difference_result.time, units.normalize_strain_by_disk_mass(hp, sim),
                units.normalize_strain_by_disk_mass(hc, sim))

    raise ValueError(f"Unknown waveform plot {kind!r}")


# Flux/frequency preparation: the parity-tested FFI accumulation lives in gw.py.
def measured_rows(result):
    """Rows up to the last measured Psi4 sample; later rows are zero fill."""
    values = result.rpsi4_uniform[:, 1:]
    measured = np.flatnonzero(np.any(values != 0.0, axis=1))
    return slice(measured[0], measured[-1] + 1)


def radiated(sim, modes):
    """Cumulative E and J_z in code units after the transient cut."""
    result = sim.strain_result
    if result.rhphcdot is None or result.rpsi4_uniform is None:
        raise ValueError(f"{sim.config.name}: regenerate the GW cache with generate_gw.py")
    rows = measured_rows(result)
    time = result.time[rows]
    keep = time >= TRANSIENT_CUTOFF_MBH * sim.config.mlittle
    if np.count_nonzero(keep) < 8:
        raise ValueError(f"{sim.config.name}: too few samples after the transient cut")
    columns = [mode_columns(*mode) for mode in modes]
    # Legacy (conjugated) h and hdot, exactly as reconstruct_strain passes them.
    strain = np.stack([result.rhphc[rows, re] + 1j*result.rhphc[rows, im] for re, im in columns], axis=1)
    strain_dot = np.stack([result.rhphcdot[rows, re] + 1j*result.rhphcdot[rows, im] for re, im in columns], axis=1)
    dt = float(np.median(np.diff(time)))
    table = _radiated_diagnostics(time[keep], strain[keep], strain_dot[keep], tuple(modes),
                                  sim.config.gw_madm, dt)
    return table[:, 0], table[:, 1], table[:, 2]


def frequency_22(sim):
    """Instantaneous (2,2) Psi4 frequency divided by the orbital frequency."""
    result = sim.strain_result
    rows = measured_rows(result)
    time, psi4 = result.time[rows], result.rpsi4(2, 2)[rows]
    keep = time >= TRANSIENT_CUTOFF_MBH * sim.config.mlittle
    time, psi4 = time[keep], psi4[keep]
    rate = np.abs(np.gradient(np.unwrap(np.angle(psi4)), time))
    width = max(1, int(round(RADIATED_SMOOTHING_ORBITS * sim.config.Pc / np.median(np.diff(time)))))
    if width >= rate.size:
        raise ValueError(f"{sim.config.name}: series shorter than the smoothing window")
    smooth = np.convolve(rate, np.ones(width) / width, mode="valid")
    centre = time[width//2: width//2 + smooth.size]
    return centre, smooth / sim.config.gw_omega_orbital


# 2. OBSERVER PROJECTION: cached modal strain -> directional strain at D_L.
def observer_waveform(case, strain, *, source_mass=OBSERVER_SOURCE_BH_MASS_MSUN,
                      distance=OBSERVER_DISTANCE_MPC, redshift=0.0, modes=OBSERVER_MODES,
                      min_tret_mbh=OBSERVER_MIN_TRET_MBH, align_peak=OBSERVER_ALIGN_PEAK):
    """Return observer seconds, dimensionless h+/cross, and source peak t/M_BH.

    Cache columns are r*h_plus and r*h_cross in G=c=Msun=1. Divide by
    the code BH mass, project, then scale by G*M_BH*(1+z)/(c^2*D_L).
    The legacy conjugation is already in h_cross; at phi=0 the harmonic
    coefficient is real, so no further sign flip or factor of two is needed.
    """
    if not all(np.isfinite(v) and v > 0 for v in (case.mlittle, source_mass, distance)):
        raise ValueError("Code BH mass, source BH mass and distance must be positive")
    if not np.isfinite(redshift) or redshift < 0 or not np.isfinite(min_tret_mbh) or min_tret_mbh < 0:
        raise ValueError("Redshift and initial time cut must be finite and nonnegative")
    if isinstance(modes, str):
        if modes != "all":
            raise ValueError('MODES must be (ell,m) pairs or "all"')
        modes = CACHED_MODES
    modes = tuple(tuple(mode) for mode in modes)
    if not modes or len(set(modes)) != len(modes) or any(mode not in CACHED_MODES for mode in modes):
        raise ValueError("Select distinct modes present in the GW cache")

    time = np.asarray(strain.time, dtype=float) / case.mlittle
    hp, hc = np.zeros_like(time), np.zeros_like(time)
    for ell, emm in modes:
        if emm != 2:
            continue
        plus, cross = strain.hplus_hcross(ell, emm)
        weight = np.sqrt((2 * ell + 1) / (4 * np.pi)) / case.mlittle
        hp += weight * plus
        hc += weight * cross
    keep = (time >= min_tret_mbh) & np.isfinite(time) & np.isfinite(hp) & np.isfinite(hc)
    time, hp, hc = time[keep], hp[keep], hc[keep]
    if time.size < 2 or np.any(np.diff(time) <= 0):
        raise ValueError(f"{case.name}: need at least two increasing retained retarded times")
    peak_time = float(time[np.argmax(np.hypot(hp, hc))])
    if align_peak:
        time = time - peak_time
    mass_z = source_mass * (1 + redshift)
    amplitude = mass_z * METERS_PER_M_SUN / (distance * METERS_PER_MPC)
    return time * mass_z * SECONDS_PER_M_SUN, hp * amplitude, hc * amplitude, peak_time


# 3. PLOT AND FORMAT: combined cases, then individual radius/mode comparisons.

def plot_observer_group(cases, waveforms):
    """One row per case, both polarizations together, with shared time limits."""
    fig, axes = plt.subplots(len(cases), 1, sharex=True, squeeze=False,
                             figsize=figure_size("double", OBSERVER_HEIGHT))
    axes = axes[:, 0]
    all_values, all_times = [], []
    for ax, case in zip(axes, cases):
        time, hp, hc, _ = waveforms[case.name]
        color = COLOR_OPTS[int(case.name[1:]) - 1]
        ax.plot(time, hp, color=color, linestyle="-", linewidth=OBSERVER_LINE_WIDTH)
        ax.plot(time, hc, color=color, linestyle="--", linewidth=OBSERVER_LINE_WIDTH)
        ax.text(0.98, 0.93, case.name, transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_ylabel(r"$h_{+,\times}$")
        ax.grid(visible=True, which="major", alpha=0.25)
        format_paper_axes(ax)
        set_symmetric_gw_ticks([ax], [hp, hc])
        all_values.extend((hp, hc))
        all_times.append(time)
    if OBSERVER_SHARE_Y_LIMITS:
        set_symmetric_gw_ticks(axes, all_values)
    format_shared_gw_yaxes(axes)
    limits = OBSERVER_TIME_LIMITS_S or (min(t[0] for t in all_times), max(t[-1] for t in all_times))
    axes[-1].set_xlim(*limits)
    xlabel = r"$(t_{\mathrm{ret}}-t_{\mathrm{peak}})\,[\mathrm{s}]$" if OBSERVER_ALIGN_PEAK else r"$t_{\mathrm{ret}}\,[\mathrm{s}]$"
    axes[-1].set_xlabel(xlabel)
    number = ScalarFormatter(useMathText=True).format_data
    fig.text(OBSERVER_MARGINS["left"], 0.963,
             rf"$M_{{\mathrm{{BH}}}}={number(OBSERVER_SOURCE_BH_MASS_MSUN)}\,M_\odot$, "
             rf"$D_L={number(OBSERVER_DISTANCE_MPC)}\,\mathrm{{Mpc}}$", va="center")
    fig.legend(handles=[Line2D([], [], color="k", ls=ls, label=label)
                        for ls, label in (("-", r"$h_+$"), ("--", r"$h_\times$"))],
               loc="center right", bbox_to_anchor=(0.975, 0.963), ncols=2,
               **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(**OBSERVER_MARGINS)
    fig.align_ylabels(axes)
    return fig


def plot_radiated(sims):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("double", 7.0))
    summary = {}
    for sim in sims:
        name, disk, bh = sim.config.name, sim.config.disk_rest_mass_code, sim.config.mlittle
        time, energy, angular = radiated(sim, RADIATED_MODES)
        _, energy_all, angular_all = radiated(sim, CACHED_MODES)
        f_time, f_ratio = frequency_22(sim)
        style = dict(color=sim.color, linestyle=sim.linestyle, label=sim.legend_name)
        axes[0].plot(units.gw_time_values(time, sim), energy / disk, **style)
        axes[1].plot(units.gw_time_values(time, sim), angular / (disk * bh), **style)
        axes[2].plot(units.gw_time_values(f_time, sim), f_ratio, **style)
        summary[name] = dict(
            modes=[list(mode) for mode in RADIATED_MODES],
            energy_over_disk_mass=float(energy[-1] / disk),
            angular_momentum_over_disk_mass_bh_mass=float(angular[-1] / (disk * bh)),
            all_modes_energy_over_disk_mass=float(energy_all[-1] / disk),
            all_modes_angular_momentum_over_disk_mass_bh_mass=float(angular_all[-1] / (disk * bh)),
            late_f22_over_forb_median=float(np.median(f_ratio[f_ratio.size // 2:])),
            interval_code=[float(time[0]), float(time[-1])],
        )
        print(f"{name}: E_GW/M_disk={energy[-1]/disk:.3g} (all modes {energy_all[-1]/disk:.3g}), "
              f"late f22/f_orb={summary[name]['late_f22_over_forb_median']:.3f}", flush=True)
    axes[0].set(ylabel=r"$E_{\rm GW}/M_{\rm disk,0}$")
    axes[1].set(ylabel=r"$J_{\rm GW}/(M_{\rm disk,0}M_{\rm BH})$")
    axes[2].set(ylabel=r"$f_{22}/f_{\rm orb}$", xlabel=units.gw_time_xlabel())
    axes[2].axhline(2.0, color="0.5", linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.grid(which="major", alpha=.25)
    ordered_sim_legend(axes[0], ncols=3, loc="upper left", **COMPACT_LEGEND_KWARGS)
    units.add_gw_time_secondary_axis(axes[0])
    fig.subplots_adjust(left=.15, right=.97, bottom=.08, top=.94, hspace=.16)
    return fig, summary


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
        x = units.gw_time_values(time, sim)
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
        title = rf"$({ell},{emm})$; {units.loaded_radius_label(reference)}; "
        title += r"strain from disk-ML $\Psi_4$" if kind == "strain_from_psi4_minus_ml" else "disk-ML"
        label = units.difference_rpsi4_ylabel if is_psi4 else units.difference_strain_ylabel
    else:
        title = rf"$(\ell,m)=({ell},{emm})$; {units.extraction_radius_label(sims[0])}"
        label = units.rpsi4_ylabel if is_psi4 else units.strain_ylabel
    axes[0].set_title(title)
    label_kwargs = {"source": r"\Delta\Psi_4"} if kind == "strain_from_psi4_minus_ml" else {}
    for ax, component in zip(axes, ("real", "abs") if is_psi4 else ("plus", "cross")):
        ax.set_ylabel(label(component, mode_text, **label_kwargs))
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    axes[1].set_xlabel(units.gw_time_xlabel())
    units.add_gw_time_secondary_axis(axes[0])
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
        time, top, bottom = waveform(sim, mode, kind)
        side = "right" if multimode and mode in GW_MULTIMODE_RIGHT_MODES else "left"
        target_axes = right_axes if side == "right" else axes
        label = rf"$({mode[0]},{mode[1]})$" if multimode else units.extraction_radius_label(sim)
        x = units.gw_time_values(time, sim)
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
        label_fn = units.rpsi4_multimode_ylabel if is_psi4 else units.strain_multimode_ylabel
        for axis_group, group in ((axes, GW_MULTIMODE_LEFT_MODES), (right_axes, GW_MULTIMODE_RIGHT_MODES)):
            group_label = "$" + ",".join(f"({ell},{emm})" for ell, emm in group) + "$"
            for ax, component in zip(axis_group, components):
                ax.set_ylabel(label_fn(component) + "\n" + group_label)
        title = units.extraction_radius_label(sim)
        title += "; modes " + ", ".join(f"({ell},{emm})" for ell, emm in GW_MULTIMODE_PLOT_MODES)
    else:
        label_fn = units.rpsi4_ylabel if is_psi4 else units.strain_ylabel
        for ax, component in zip(axes, components):
            ax.set_ylabel(label_fn(component, f"{PSI4_MODE[0]}{PSI4_MODE[1]}"))
        title = rf"$(\ell,m)=({PSI4_MODE[0]},{PSI4_MODE[1]})$; " + ", ".join(labels)
    axes[0].set_title(f"{sim.config.name}: {title}")
    axes[1].set_xlabel(units.gw_time_xlabel())
    for ax in axes:
        ax.grid()
        ax.tick_params(axis="x", top=True, which="both")
    for ax in right_axes:
        ax.grid(False)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    units.add_gw_time_secondary_axis(axes[0])
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


# Polarization panel and observer presentation of the same cached strain.
def plot_polarization_panel(sims, mode):
    """Six-case view of the same normalized plus/cross strain as plot()."""
    sim_by_name = {sim.config.name: sim for sim in sims}
    fig, axes = plt.subplots(2, 3, figsize=figure_size("double", PANEL_HEIGHT),
                             sharex=True, sharey=True, squeeze=False)
    final_times, strain_values = [], []
    for row, names in enumerate(CASE_GROUPS):
        for col, name in enumerate(names):
            ax = axes[row, col]
            sim = sim_by_name.get(name)
            if sim is None:
                ax.set_visible(False)
                continue
            # Reader -> shared normalization -> finite common samples.
            hp, hc = sim.strain_result.hplus_hcross(ell=mode[0], emm=mode[1])
            time = np.asarray(units.gw_time_values(sim.strain_result.time, sim), dtype=float)
            hp = np.asarray(units.normalize_strain(hp, sim), dtype=float)
            hc = np.asarray(units.normalize_strain(hc, sim), dtype=float)
            count = min(time.size, hp.size, hc.size)
            time, hp, hc = time[:count], hp[:count], hc[:count]
            finite = np.isfinite(time) & np.isfinite(hp) & np.isfinite(hc)
            time, hp, hc = time[finite], hp[finite], hc[finite]
            if not time.size:
                ax.set_visible(False)
                continue
            final_times.append(time[-1])
            strain_values.extend((hp, hc))
            ax.plot(time, hp, color=sim.color, linestyle=PLUS_LINESTYLE, label=r"$h_+$")
            ax.plot(time, hc, color=sim.color, linestyle=CROSS_LINESTYLE, label=r"$h_\times$")
            ax.text(.04, .94, name, transform=ax.transAxes, ha="left", va="top")
            ax.grid()
            format_paper_axes(ax)
    if not final_times:
        plt.close(fig)
        return None
    axes[0, 0].set_xlim(TIME_XMIN, PANEL_TIME_XMAX_PAD * max(final_times))
    mode_text = f"{mode[0]}{mode[1]}"
    for ax in axes[:, 0]:
        if ax.get_visible():
            ax.set_ylabel(rf"$r_A h_{{+,\times}}^{{{mode_text}}}/M_{{\mathrm{{BH}}}}$")
    for ax in axes[-1, :]:
        if ax.get_visible():
            ax.set_xlabel(units.gw_time_xlabel())
    visible_axes = [ax for ax in axes.flat if ax.get_visible()]
    set_symmetric_gw_ticks(visible_axes, strain_values)
    format_shared_gw_yaxes(visible_axes)
    for ax in axes[0, :]:
        if ax.get_visible():
            units.add_gw_time_secondary_axis(ax)
    fig.text(.10, .975,
             rf"$(\ell,m)=({mode[0]},{mode[1]})$, {units.extraction_radius_label(sims[0])}",
             ha="left", va="center")
    fig.legend(handles=(Line2D([], [], color="k", linestyle=PLUS_LINESTYLE, label=r"$h_+$"),
                        Line2D([], [], color="k", linestyle=CROSS_LINESTYLE, label=r"$h_\times$")),
               loc="upper right", bbox_to_anchor=(.985, .985), ncols=2, frameon=True)
    fig.subplots_adjust(**PANEL_MARGINS)
    return fig


def save_observer_groups(args, radius):
    """Read modal FFI caches, project face-on, scale, then draw the A/B groups.

    This view needs rhphc.dat and strain_cache.json, not raw Psi4 or scalars.
    """
    redshift = (FlatLambdaCDM().redshift_at_luminosity_distance(OBSERVER_DISTANCE_MPC)
                if OBSERVER_INCLUDE_REDSHIFT else 0.0)
    for group in CASE_GROUPS:
        names = [name for name in group if args.sims is None or name in args.sims]
        if not names:
            continue
        cases, waveforms = all_sim_configs(names), {}
        for case in cases:
            label = str(radius + case.gw_psi4_file_index_offset)
            directory = strain_cache_dir(case, label)
            strain = read_strain_cache(directory, case.data_path / f"Psi4_rad.mon.{label}")
            if str(strain.metadata.get("source_label")) != label or not np.isclose(
                float(strain.metadata.get("madm", np.nan)), case.gw_madm, rtol=1e-10, atol=0,
            ):
                raise ValueError(f"{directory}: cache metadata does not match this simulation")
            waveforms[case.name] = observer_waveform(case, strain, redshift=redshift)
            time, hp, hc, peak = waveforms[case.name]
            print(f"{case.name}: face-on peak |h|={np.hypot(hp, hc).max():.6e}, "
                  f"source peak t_ret/M_BH={peak:.3f}, "
                  f"observer span={time[0]:.4f}..{time[-1]:.4f} s")
        savefig(plot_observer_group(cases, waveforms), args,
                OBSERVER_OUTPUT.format(family=group[0][0]))


# 4. COMMANDS: read once per radius, prepare, draw, and save.
def main(argv=None):
    command = parser(__doc__)
    command.add_argument("--kinds", nargs="+", choices=PLOTS, default=PLOTS,
                         metavar="NAME", help="Choose from %(choices)s; default: all waveform views")
    command.add_argument("--radius-index", type=int, choices=range(9), default=None,
                         help="One parfile extraction index; default: configured comparison radii")
    args = command.parse_args(argv)
    setup(args)
    kinds = args.kinds
    modal_kinds = [kind for kind in kinds if kind in OUTPUTS]
    draw_panel = "strain_panel" in kinds
    draw_radiated = "gw_radiated" in kinds
    draw_observer = "strain_observer" in kinds

    # Select cases. Only the difference views need the massless reference.
    names = args.sims
    if any(kind not in ("psi4", "strain") for kind in modal_kinds):
        names = names_with_massless(names)
    if names is None and not modal_kinds:
        names = sum(CASE_GROUPS, ())

    # Select radii. Modal/panel views compare the configured extractions;
    # radiated diagnostics use the outer extraction. An explicit choice wins.
    radii = []
    if modal_kinds or draw_panel:
        if args.radius_index is None:
            radii.extend(GW_COMPARISON_PARFILE_INDICES)
        else:
            radii.append(args.radius_index)
    single_radius = args.radius_index  # Observer and radiated views use one radius.
    if single_radius is None:
        single_radius = GW_OUTERMOST_PARFILE_INDEX
    if draw_radiated and single_radius not in radii:
        radii.append(single_radius)

    # Read once per case, then reuse matching radius and difference caches.
    # Observer-only runs read strain caches below and never open raw Psi4.
    cases = []
    if radii:
        cases = all_sim_configs(names)
    runs = [GWRun(case) for case in cases]
    for radius in radii:
        sims = apply_styles([run.at_radius(radius) for run in runs])
        if not sims:
            command.error("No selected simulations have GW data")
        reference = next((s for s in sims if s.config.name == MASSLESS_SIM_NAME), None)
        differences = {}
        if "strain_from_psi4_minus_ml" in kinds and reference is not None:
            for sim in sims:
                if sim is reference or not same_loaded_radius(sim, reference):
                    continue
                differences[sim.config.name] = read_difference(sim, reference)
        # Prepare, plot and save. waveform() above contains every data branch.
        for kind in modal_kinds:
            for mode in GW_PLOT_MODES:
                fig = plot(sims, mode, radius, kind, differences)
                if fig is not None:
                    label_sim = sims[0] if kind in ("psi4", "strain") else reference
                    savefig(fig, args, OUTPUTS[kind].format(
                        mode=mode_tag(mode), radius=units.loaded_radius_tag(label_sim)))

        if draw_panel:
            for mode in GW_PLOT_MODES:
                fig = plot_polarization_panel(sims, mode)
                if fig is not None:
                    savefig(fig, args, PANEL_OUTPUT.format(
                        mode=mode_tag(mode), radius=units.loaded_radius_tag(sims[0])))
        if draw_radiated and radius == single_radius:
            disk_sims = [sim for sim in sims if sim.config.name != MASSLESS_SIM_NAME]
            if not disk_sims:
                command.error("The radiated diagnostic requires a disk simulation")
            fig, summary = plot_radiated(disk_sims)
            savefig(fig, args, RADIATED_OUTPUT)
            if not args.no_save:
                path = args.outdir / RADIATED_SUMMARY
                path.write_text(json.dumps(summary, indent=2) + "\n")
                print(f"saved {path}")
    if draw_observer:
        save_observer_groups(args, single_radius)


def individual_main(argv=None):
    args = parser("Individual GW radius and mode comparisons.").parse_args(argv)
    setup(args)
    for config in all_sim_configs(args.sims):
        run = GWRun(config)
        radii = {}
        for radius in dict.fromkeys((*GW_COMPARISON_PARFILE_INDICES, GW_OUTERMOST_PARFILE_INDEX)):
            radii[radius] = run.at_radius(radius)
        radius_tags = [units.loaded_radius_tag(radii[r]) for r in GW_COMPARISON_PARFILE_INDICES if r in radii]
        for multimode in (False, True):
            for kind in ("psi4", "strain"):
                fig = plot_individual(radii, kind, multimode)
                if fig is None:
                    continue
                if multimode:
                    modes = "_".join(mode_tag(mode) for mode in GW_MULTIMODE_PLOT_MODES)
                    filename = f"gw_{kind}_modes_{config.name}_{modes}_{units.loaded_radius_tag(radii[GW_OUTERMOST_PARFILE_INDEX])}.png"
                else:
                    prefix = "rpsi4" if kind == "psi4" else "rhphc"
                    filename = f"{prefix}_{config.name}_{mode_tag(PSI4_MODE)}_{'-'.join(radius_tags)}.png"
                save_individual_fig(fig, args, config.name, filename)


if __name__ == "__main__":
    main()
