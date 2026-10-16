#!/usr/bin/env python3
"""Additional detectability figures; calculations are in gw_detectability.py.

Run all existing diagnostics: python -m wip_plots.gw_detectability_all
Run selected diagnostics:    python -m wip_plots.gw_detectability_all --plots radius averaging

Edit DEFAULT_PLOTS below to select outputs. This file draws the diagnostics;
gw_detectability.py owns the measured calculation and the combined figure.
The CLI and the explicit sequence of analysis/drawing calls are at the bottom.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
import numpy as np

import gw_detectability as analysis

import gw_detectability_extras as extras
from helpers.plot_common import parser, savefig, save_individual_fig, setup
from helpers.style import (
    AXES_LEGEND_BORDERAXESPAD, COMPACT_LEGEND_KWARGS,
    PAPER_SINGLE_PANEL_HEIGHT, PAPER_TWO_PANEL_HEIGHT, PAPER_THREE_PANEL_HEIGHT,
    apply_styles, figure_size, format_paper_axes,
    ordered_sim_legend, COLOR_OPTS,
)

# This file contains diagnostic plots. Production math and drawing are in analysis.

# One output list replaces the interacting per-panel switches. This default
# preserves the existing batch exports, including the labelled extrapolation.
# Use --plots combined for measured data alone, or the root gw_detectability.py.
DEFAULT_PLOTS = ("radius", "averaging", "combined", "reference", "extrapolation")
AVAILABLE_PLOTS = ("combined", "strain", "distance", "radius", "averaging", "reference", "extrapolation")
OUTPUT_FILENAME_CHARACTERISTIC_STRAIN = "gw/gw_detectability_characteristic_strain.png"
# Existing per-panel names remain usable when calling the drawing functions.
OUTPUT_FILENAME_TIME_ON = "gw/gw_detectability_characteristic_strain_time_on.png"
OUTPUT_FILENAME_HORIZON_TIME_ON = "gw/gw_detectability_horizon_time_on.png"
OUTPUT_FILENAME_SHIBATA = "gw/gw_detectability_shibata.png"
OUTPUT_FILENAME_HORIZON = "gw/gw_detectability_horizon.png"
OUTPUT_FILENAME_RADIUS_COMPARISON = "gw/gw_detectability_radius_comparison.png"
OUTPUT_FILENAME_METHOD_COMPARISON = "gw/gw_detectability_method_comparison.png"

# Presentation only: case colours and one detector-style key across both panels.
# Label below each curve, away from the signal bands; shared by every spectrum.
DETECTOR_LABEL_HZ = {"ligo": 120.0, "et": 900.0, "ce": 300.0, "decigo": 0.15, "lisa": 0.04}
DETECTOR_LABEL_PAD_POINTS = {"ligo": 2, "et": 3, "ce": 3, "decigo": 6, "lisa": 3}
DETECTOR_LABEL_SIZE = 9
COMPARISON_HEIGHT = PAPER_TWO_PANEL_HEIGHT
TARGETS_HEIGHT = 3.4
SHIBATA_HEIGHT = PAPER_SINGLE_PANEL_HEIGHT
HORIZON_HEIGHT = 3.4
VALIDATION_HEIGHT = PAPER_THREE_PANEL_HEIGHT
TEMPORAL_VALIDATION_HEIGHT = PAPER_THREE_PANEL_HEIGHT
TEMPORAL_NOTE_SIZE = 11
TEMPORAL_PREDICTION_COLOR = "#b85b00"
TEMPORAL_BLEND_COLOR = "0.35"
TEMPORAL_SPECTRUM_DYNAMIC_RANGE = 1e6  # Display only, relative to the larger peak.
# Display limits only; spectra and SNR still use every retained analysis bin.


def _label_noise_curve(ax, frequency, noise, detector):
    """Bend serif glyph outlines along the displayed log-log curve, below it.

    Call after axis limits and layout are final. Work in screen coordinates
    so font size and normal offset stay consistent despite different scales.
    """
    label = analysis.DETECTOR_CURVE_LABELS.get(detector, analysis.DETECTOR_LABELS[detector])
    glyphs = TextPath((0, 0), label, size=DETECTOR_LABEL_SIZE,
                      prop=FontProperties(family="serif", weight="normal"),
                      usetex=plt.rcParams["text.usetex"])
    curve = ax.transData.transform(np.column_stack((frequency, noise)))
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(curve, axis=0), axis=1))]
    tangent = np.column_stack((np.gradient(curve[:, 0], arc), np.gradient(curve[:, 1], arc)))
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    scale = ax.figure.dpi / 72.
    bounds = glyphs.get_extents()
    center = np.interp(np.log(DETECTOR_LABEL_HZ[detector]), np.log(frequency), arc)
    center = np.clip(center, bounds.width * scale / 2., arc[-1] - bounds.width * scale / 2.)
    position = center + (glyphs.vertices[:, 0] - (bounds.x0 + bounds.x1) / 2.) * scale
    xy = np.column_stack([np.interp(position, arc, curve[:, i]) for i in (0, 1)])
    direction = np.column_stack([np.interp(position, arc, tangent[:, i]) for i in (0, 1)])
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    normal = np.column_stack((-direction[:, 1], direction[:, 0]))
    offset = (glyphs.vertices[:, 1] - bounds.y1 - DETECTOR_LABEL_PAD_POINTS[detector]) * scale
    vertices = ax.transData.inverted().transform(xy + normal * offset[:, None])
    ax.add_patch(PathPatch(MplPath(vertices, glyphs.codes), facecolor=analysis.DETECTOR_COLOR,
                           edgecolor="none", zorder=4, clip_on=True))


def _plot_radius_comparison(result, args):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figure_size("double", COMPARISON_HEIGHT))
    for name, source in result.sources.items():
        if name not in result.inner:
            continue
        x, hc = extras.orbital_spectrum(source.config, source.spectrum)
        inner_x, inner_hc = extras.orbital_spectrum(source.config, result.inner[name].spectrum)
        keep = analysis._spectrum_plot_mask(x, hc)
        axes[0].plot(x[keep], hc[keep], color=source.color, linestyle=source.linestyle, label=source.legend_name)
        ratio_x, ratio = extras.interpolated_spectral_ratio(x, hc, inner_x, inner_hc)
        significant = np.isfinite(ratio) & (np.interp(ratio_x, x, hc) >= 1e-3 * np.max(hc))
        axes[1].plot(ratio_x[significant], ratio[significant], color=source.color, linestyle=source.linestyle)
    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$ (outer)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel(r"$h_c(\mathrm{inner})/h_c(\mathrm{outer})$")
    axes[1].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[1].axhline(1, color="0.45", linewidth=0.8, linestyle=":")
    analysis._format_axes(axes)
    ordered_sim_legend(axes[0], ncols=3, loc="upper right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_RADIUS_COMPARISON)


def _plot_method_comparison(result, args):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figure_size("double", COMPARISON_HEIGHT))
    for name, source in result.sources.items():
        x, hc = extras.orbital_spectrum(source.config, source.spectrum)
        keep = analysis._spectrum_plot_mask(x, hc)
        axes[0].plot(x[keep], hc[keep], color=source.color, linestyle=source.linestyle, label=source.legend_name)
        representative = result.representative.get(name)
        if representative is not None:
            _, rep_hc = extras.orbital_spectrum(source.config, representative)
            ratio = np.divide(hc, rep_hc, out=np.full_like(hc, np.nan), where=rep_hc > 0)
            mask = keep & np.isfinite(ratio)
            axes[1].plot(x[mask], ratio[mask], color=source.color, linestyle=source.linestyle)
    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$")
    axes[0].set_yscale("log")
    average_label = r"\langle h_c\rangle_\Omega" if analysis.SOURCE_AVERAGING == "mean" else r"h_{c,\mathrm{RMS}}"
    axes[1].set_ylabel(rf"${average_label}/h_c(\pi/2.34)$")
    axes[1].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[1].axhline(1, color="0.45", linewidth=0.8, linestyle=":")
    analysis._format_axes(axes)
    ordered_sim_legend(axes[0], ncols=3, loc="upper right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_METHOD_COMPARISON)


def _plot_characteristic_strain(result, args, *, fixed_source=False, filename=None, note=None):
    height = SHIBATA_HEIGHT if fixed_source else TARGETS_HEIGHT
    fig, ax = plt.subplots(figsize=figure_size("double", height))
    analysis._draw_characteristic_strain(ax, result, fixed_source=fixed_source, note=note)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.98)
    for detector, (frequency, hn) in result.noise.items():
        _label_noise_curve(ax, frequency, hn, detector)
    savefig(fig, args, filename or (OUTPUT_FILENAME_SHIBATA if fixed_source else OUTPUT_FILENAME_CHARACTERISTIC_STRAIN))


def _plot_horizon(result, args, *, filename=OUTPUT_FILENAME_HORIZON, note=None):
    fig, ax = plt.subplots(figsize=figure_size("double", HORIZON_HEIGHT))
    analysis._draw_horizon(ax, result, note=note)
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.18, top=0.98)
    savefig(fig, args, filename)


def _plot_extrapolation(result, args):
    """Keep the direct-Psi4 baseline; modify only the (2,2) termination."""
    spectra, reports, extended, conditional = {}, {}, [], []
    for name, source in result.sources.items():
        trial = extras.temporal_trial_for_case(source.config)
        fit = trial.fit
        spectra[name] = source.spectrum
        reports[name] = dict(mass_source=trial.mass_source, gamma_mbh=fit.gamma,
                             fit_errors=fit.errors, holdout_error=trial.holdout_error,
                             issues=trial.issues, conditional=trial.conditional,
                             extended=False)
        save_individual_fig(plot_temporal_validation(source.config, trial), args,
                            name, "gw_temporal_fit.png")
        if trial.extended_strain is None:
            print(f"{name}: measured only; " + "; ".join(trial.issues), flush=True)
            if not args.no_save:
                for filename in ("gw_temporal_transition.png", "gw_temporal_psi4_transition.png"):
                    (Path(args.outdir)/name/filename).unlink(missing_ok=True)
            continue
        time, modes = analysis.read_rpsi4_modes(source.config, analysis.OUTER_RADIUS_INDEX)
        spectra[name], _ = analysis.spectrum_for_case(source.config, time, modes, continuation=trial)
        extended.append(name)
        if trial.conditional:
            conditional.append(name)
        reports[name]["extended"] = True
        reports[name]["total_duration_over_measured"] = float(np.ptp(trial.extended_time)/np.ptp(trial.time))
        save_individual_fig(plot_temporal_transition(source.config, trial), args,
                            name, "gw_temporal_transition.png")
        time, modes = analysis.retained_psi4_modes(time, modes, source.config.mlittle)
        tau = time/source.config.mlittle
        p = source.config.mlittle*modes[2, 2]
        te, pe = extras.continued_psi4_mode(tau, p, trial)
        psi4_trial = replace(trial, time=tau, strain=p, extended_time=te, extended_strain=pe)
        save_individual_fig(plot_temporal_transition(source.config, psi4_trial, psi4=True),
                            args, name, "gw_temporal_psi4_transition.png")
        model = (-fit.gamma+1j*fit.omega)**2*fit.strain(tau)
        late = tau >= fit.start
        reports[name]["psi4_model_relative_residual"] = float(np.linalg.norm(p[late]-model[late])/np.linalg.norm(p[late]))
        print(f"{name}: direct-Psi4 tail; conditional={trial.conditional}; gamma={fit.gamma:.4g}", flush=True)
    unchanged = [name for name in spectra if name not in extended]
    lines = []
    for names, label in ((conditional, r": conditional $(2,2)$ tail"),
                         ([n for n in extended if n not in conditional], r": $(2,2)$ tail"),
                         (unchanged, ": measured")):
        if names:
            lines.append(", ".join(names)+label)
    note = "\n".join(lines)
    curves, cosmology = analysis.load_detector_curves(), analysis.FlatLambdaCDM()
    targets = analysis.example_targets(spectra, curves, cosmology)
    updated = replace(result, targets=targets)
    # This diagnostic always draws the two panels together. Its spectra and
    # threshold examples use the extrapolated model; the measured result stays
    # available separately, with its own unchanged output name.
    updated.horizons = analysis.horizon_curves(spectra, curves, cosmology)
    analysis.plot_combined(
        updated, args, filename=analysis.OUTPUT_FILENAME_COMBINED_TIME_ON, note=note)
    report = dict(input="direct Psi4 FFT for both off and on", modes=analysis.MODES,
                  extended_mode=[2, 2], cases=reports,
                  off=[dict(mass=t.mass, distance=t.distance, snr=t.snr) for t in result.targets],
                  on=[dict(mass=t.mass, distance=t.distance, snr=t.snr) for t in targets])
    if not args.no_save:
        path = Path(args.outdir)/"gw/gw_detectability_time_comparison.json"
        path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def plot_validation(sim_name):
    radial, tapers, cutoffs = extras.validation_data(sim_name)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("double", VALIDATION_HEIGHT))
    for index, x, y in radial:
        endpoint = index in {analysis.FIRST_RADIUS_INDEX, analysis.OUTER_RADIUS_INDEX}
        label = "Outer" if index == analysis.OUTER_RADIUS_INDEX else "Inner" if index == analysis.FIRST_RADIUS_INDEX else "Intermediate"
        axes[0].plot(x, y, label=rf"{label} ($i={index}$)",
                     linewidth=1.5 if endpoint else 1, alpha=1 if endpoint else 0.65)
    for alpha, x, ratio in tapers:
        axes[1].plot(x, ratio, label=rf"$\alpha={alpha:g}$")
    for cutoff, x, ratio in cutoffs:
        axes[2].plot(x, ratio, label=rf"$t_{{\mathrm{{cut}}}}={cutoff:g}M_{{\mathrm{{BH}}}}$")
    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$")
    axes[1].set_ylabel(r"$h_c/h_{c,\mathrm{base}}$")
    axes[2].set_ylabel(r"$h_c/h_{c,\mathrm{base}}$")
    axes[2].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    for ax in axes:
        ax.legend(loc="upper right", borderaxespad=AXES_LEGEND_BORDERAXESPAD,
                  **COMPACT_LEGEND_KWARGS)
        ax.set_yscale("log")
    for ax in axes[1:]:
        ax.axhline(1, color="0.45", linewidth=0.8, linestyle=":")
    analysis._format_axes(axes)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.09, top=0.98, hspace=0.08)
    return fig


def plot_temporal_validation(case, trial):
    """Compare measured mass, amplitude and phase with their late-time fits."""
    fig, axes = plt.subplots(3, 1, sharex=True,
                             figsize=figure_size("double", TEMPORAL_VALIDATION_HEIGHT))
    t, q, fit = trial.time, trial.strain, trial.fit
    color = COLOR_OPTS[int(case.name[1:])-1]
    window = (trial.mass_time >= fit.start) & (trial.mass_time <= fit.end)
    axes[0].plot(trial.mass_time[window], trial.mass[window]/trial.initial_mass,
                 color=color, label="Simulation")
    fitted = t >= fit.start
    axes[1].plot(t[fitted], np.abs(q[fitted]), color=color)
    axes[0].plot(t[fitted], fit.mass(t[fitted])/trial.initial_mass, "k--", label="Fit")
    axes[1].plot(t[fitted], np.abs(fit.strain(t[fitted])), "k--")
    phase = np.unwrap(np.angle(q[fitted]))
    trend = fit.omega*(t[fitted]-fit.end)+fit.phase
    phase += 2*np.pi*np.round(np.mean(trend-phase)/(2*np.pi))
    axes[2].plot(t[fitted], phase-phase[0], color=color)
    axes[2].plot(t[fitted], trend-phase[0], "k--")
    for ax in axes:
        ax.set_xlim(fit.start, fit.end)
        format_paper_axes(ax)
    axes[0].set_ylabel(r"$M_{0,\mathrm{flux}}/M_{0,t=0}$" if trial.mass_source == "horizon_flux"
                       else r"$M_{0,\mathrm{disk}}/M_{0,t=0}$")
    axes[1].set_ylabel(r"$|r h_{22}|/M_{\mathrm{BH}}$")
    axes[2].set_ylabel(r"$\phi_{22}\,[\mathrm{rad}]$")
    axes[0].set_xlabel(r"$t/M_{\mathrm{BH}}$")
    axes[0].tick_params(labelbottom=True)
    axes[2].set_xlabel(r"$t_{\mathrm{ret}}/M_{\mathrm{BH}}$")
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((-2, 2))
    axes[1].yaxis.set_major_formatter(formatter)
    axes[0].yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, [f"{case.name} data", labels[1]], loc="upper center",
               bbox_to_anchor=(.58, .995), ncols=2, fontsize=TEMPORAL_NOTE_SIZE)
    fig.subplots_adjust(left=.18, right=.98, bottom=.09, top=.93, hspace=.40)
    fig.align_ylabels(axes)
    return fig


def plot_temporal_transition(case, trial, *, psi4=False):
    """Separate original data, the modified join, and the actual future tail."""
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figure_size("double", 5.2))
    period = case.Pc/case.mlittle
    end = trial.time[-1]
    join_start = max(trial.time[0], end-extras.TAIL_JOIN_ORBITS*period)
    left = max(trial.time[0], join_start-period)
    right = min(trial.extended_time[-1], end+2*period)
    color = COLOR_OPTS[int(case.name[1:])-1]
    regions = ((join_start, end, TEMPORAL_BLEND_COLOR, "Blend"),
               (end, right, TEMPORAL_PREDICTION_COLOR, "Extrapolation"))
    labels = ((r"$\mathrm{Re}(M_{\mathrm{BH}}r\Psi_4^{22})$", r"$M_{\mathrm{BH}}|r\Psi_4^{22}|$")
              if psi4 else (r"$\mathrm{Re}(r h_{22})/M_{\mathrm{BH}}$", r"$|r h_{22}|/M_{\mathrm{BH}}$"))
    for ax, transform, ylabel in zip(axes, (np.real, np.abs), labels):
        measured = trial.time >= left
        ax.plot(trial.time[measured], transform(trial.strain[measured]),
                color=color, lw=1.5, label="Simulation")
        values = transform(trial.extended_strain)
        for start, stop, line_color, label in regions:
            inside = (trial.extended_time > start) & (trial.extended_time < stop)
            # Add boundary points only for drawing; the numerical trial is untouched.
            times = np.r_[start, trial.extended_time[inside], stop]
            ax.plot(times, np.interp(times, trial.extended_time, values),
                    color=line_color, lw=1.5, label=label)
        ax.axvline(end, color="0.15", ls="--", lw=1, label="Last simulation sample")
        ax.set(xlim=(left, right), ylabel=ylabel)
        ax.margins(y=.12)
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))
        ax.yaxis.set_major_formatter(formatter)
        format_paper_axes(ax)
    handles = axes[0].get_lines()[:3]
    fig.legend(handles, [f"{case.name} data", "Blend", "Extrapolation"], loc="upper center",
               bbox_to_anchor=(.575, .995), ncols=3, fontsize=TEMPORAL_NOTE_SIZE)
    axes[0].text(end, 1.025, "Data end", ha="center",
                 transform=axes[0].get_xaxis_transform(), fontsize=TEMPORAL_NOTE_SIZE)
    axes[-1].set_xlabel(r"$t_{\mathrm{ret}}/M_{\mathrm{BH}}$")
    fig.subplots_adjust(left=.17, right=.98, bottom=.13, top=.88, hspace=.18)
    fig.align_ylabels(axes)
    return fig


def temporal_main(argv=None, *, cases=None):
    """Opt-in trial only; never overwrite any direct-Psi4 production figure."""
    import json

    args = parser("Test Wessel's disk-mass-driven (2,2) strain continuation.").parse_args(argv)
    setup(args)
    cases = analysis.config.all_sim_configs(args.sims or analysis.SIM_NAMES) if cases is None else cases
    reports = {}
    curves, cosmology = analysis.load_detector_curves(), analysis.FlatLambdaCDM()
    for case in cases:
        trial = extras.temporal_trial_for_case(case)
        fit = trial.fit
        reports[case.name] = dict(fit_interval_mbh=[fit.start, fit.end], gamma_mbh=fit.gamma,
                                 mass_source=trial.mass_source,
                                 omega_mbh=fit.omega, errors=fit.errors,
                                 holdout_error=trial.holdout_error, issues=trial.issues,
                                 conditional=trial.conditional,
                                 window_fits=[dict(start=f.start, end=f.end, gamma=f.gamma,
                                                   omega=f.omega, errors=f.errors)
                                              for f in trial.window_fits])
        if trial.density_time is not None:
            late = (trial.density_time >= fit.start) & (trial.density_time <= fit.end)
            bins = np.array_split(trial.density_modes[late], 4)
            reports[case.name]["density_mode_quarter_means"] = [b.mean(axis=0).tolist() for b in bins]
            later = trial.density_time > fit.end
            if later.sum() >= 4:
                bins = np.array_split(trial.density_modes[later], 4)
                reports[case.name]["later_density_mode_quarter_means"] = [b.mean(axis=0).tolist() for b in bins]
            reports[case.name]["mass_end_actual_over_prediction"] = float(
                trial.mass[-1]/fit.mass(trial.mass_time[-1]))
            reports[case.name]["scalar_end_mbh"] = float(trial.mass_time[-1])
        print(f"{case.name}: gamma={fit.gamma:.4g}, phase RMS={fit.errors['phase']:.3g}, "
              f"holdout={trial.holdout_error:.3g}; " + ("; ".join(trial.issues) or "trial accepted"), flush=True)
        fig = plot_temporal_validation(case, trial)
        save_individual_fig(fig, args, case.name, "gw_temporal_fit.png")
        if trial.extended_spectrum is None or (trial.issues and not trial.conditional):
            if not args.no_save:
                for filename in ("gw_temporal_on_off.png", "gw_temporal_transition.png"):
                    (Path(args.outdir)/case.name/filename).unlink(missing_ok=True)
            continue
        fig = plot_temporal_transition(case, trial)
        save_individual_fig(fig, args, case.name, "gw_temporal_transition.png")
        fig, ax = plt.subplots(figsize=figure_size("double", PAPER_SINGLE_PANEL_HEIGHT))
        case_color = COLOR_OPTS[int(case.name[1:])-1]
        reports[case.name]["total_duration_over_measured"] = float(
            np.ptp(trial.extended_time)/np.ptp(trial.time))
        mass, distance = analysis.SHIBATA_BH_MASS_MSUN, analysis.SHIBATA_DISTANCE_MPC
        z = cosmology.redshift_at_luminosity_distance(distance)
        result = analysis.observed_target({"Off": trial.finite_spectrum, "On": trial.extended_spectrum},
                                          mass, distance, z, curves)
        reports[case.name]["snr"] = result.snr
        reports[case.name]["source"] = dict(mass_msun=mass, distance_mpc=distance, redshift=z)
        for label, (frequency, hc) in result.strain.items():
            ax.loglog(frequency, hc, label={"Off": "Measured", "On": "With extrapolation"}[label],
                       color=case_color if label == "Off" else TEMPORAL_PREDICTION_COLOR)
        peak = max(float(hc.max()) for _, hc in result.strain.values())
        floor = peak/TEMPORAL_SPECTRUM_DYNAMIC_RANGE
        visible = np.concatenate([f[hc >= floor] for f, hc in result.strain.values()])
        ax.set(xlim=(visible.min()/1.1, visible.max()*1.1), ylim=(floor, 3*peak))
        ax.set(xlabel=r"$f_{\mathrm{obs}}\,[\mathrm{Hz}]$", ylabel=r"$h_c^{22}$")
        format_paper_axes(ax)
        ax.legend(title=case.name, title_fontsize=TEMPORAL_NOTE_SIZE,
                  loc="upper right", borderaxespad=AXES_LEGEND_BORDERAXESPAD,
                  **COMPACT_LEGEND_KWARGS)
        fig.text(.17, .02, rf"$M_{{\mathrm{{BH}}}}={analysis._scientific_label(mass)}\,M_\odot$, "
                 rf"$D_L={analysis._scientific_label(distance)}\,\mathrm{{Mpc}}$",
                 fontsize=TEMPORAL_NOTE_SIZE)
        fig.subplots_adjust(left=.17, right=.98, bottom=.24, top=.98)
        save_individual_fig(fig, args, case.name, "gw_temporal_on_off.png")
    if not args.no_save:
        destination = Path(args.outdir)/"gw"/"temporal_trial.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(reports, indent=2)+"\n")
    return reports


def main(argv=None):
    command = parser(__doc__)
    command.add_argument(
        "--plots", nargs="+", choices=AVAILABLE_PLOTS, default=DEFAULT_PLOTS, metavar="NAME",
        help="Choose from %(choices)s; default: %(default)s. Extrapolation uses scalar data.")
    args = command.parse_args(argv)
    requested = set(args.plots)
    setup(args)

    # Read and calculate only what the requested figures need. A measured-only
    # run never opens scalar files or runs the extrapolation fit.
    result = analysis.analyze(
        args.sims,
        compare_radius="radius" in requested,
        compare_direction="averaging" in requested,
        compute_horizons=bool(requested & {"combined", "distance"}),
        include_reference="reference" in requested,
    )
    apply_styles(list(result.sources.values()))

    # Draw: these names match DEFAULT_PLOTS and --plots above.
    if "radius" in requested and result.inner:
        _plot_radius_comparison(result, args)
    if "averaging" in requested:
        _plot_method_comparison(result, args)
    if "combined" in requested:
        analysis.plot_combined(result, args)
    if "strain" in requested:
        _plot_characteristic_strain(result, args)
    if "distance" in requested:
        _plot_horizon(result, args)
    if "reference" in requested:
        _plot_characteristic_strain(result, args, fixed_source=True)
    if "extrapolation" in requested:
        _plot_extrapolation(result, args)
    return result


def individual_main(argv=None):
    args = parser("Validate direct-Psi4 detectability for individual simulations.").parse_args(argv)
    setup(args)
    for name in args.sims or analysis.SIM_NAMES:
        fig = plot_validation(name)
        save_individual_fig(fig, args, name, f"gw_detectability_method_validation_{name}.png")


if __name__ == "__main__":
    main()
