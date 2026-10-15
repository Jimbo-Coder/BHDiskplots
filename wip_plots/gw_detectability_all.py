#!/usr/bin/env python3
"""All detectability figures; the scientific pipeline is in gw_detectability.py.

Combined: run_wip.py detectability. Individual: run_individual.py A1 --extra.
"""
from pathlib import Path
from dataclasses import replace
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, ScalarFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
import numpy as np
import shutil
from matplotlib.lines import Line2D

import gw_detectability as analysis
from helpers.plot_common import parser, savefig, save_individual_fig, setup
from helpers.style import (
    AXES_LEGEND_BORDERAXESPAD, COMPACT_LEGEND_KWARGS,
    PAPER_SINGLE_PANEL_HEIGHT, PAPER_TWO_PANEL_HEIGHT, PAPER_THREE_PANEL_HEIGHT,
    apply_styles, figure_size, format_paper_axes,
    ordered_sim_legend, COLOR_OPTS,
)

# Figure selection and output names. Scientific choices are in gw_detectability.py.
PLOT_CHARACTERISTIC_STRAIN = True
PLOT_TIME_COMPARISON = True  # Additional conditional tail chart; baseline unchanged.
PLOT_SHIBATA_COMPARISON = True
PLOT_HORIZON = True
PLOT_RADIUS_COMPARISON = True
PLOT_METHOD_COMPARISON = True
OUTPUT_FILENAME_CHARACTERISTIC_STRAIN = "gw/gw_detectability_characteristic_strain.png"
OUTPUT_FILENAME_TIME_ON = "gw/gw_detectability_characteristic_strain_time_on.png"
OUTPUT_FILENAME_HORIZON_TIME_ON = "gw/gw_detectability_horizon_time_on.png"
OUTPUT_FILENAME_SHIBATA = "gw/gw_detectability_shibata.png"
OUTPUT_FILENAME_HORIZON = "gw/gw_detectability_horizon.png"
OUTPUT_FILENAME_COMBINED = "gw/gw_detectability_combined.png"
OUTPUT_FILENAME_COMBINED_TIME_ON = "gw/gw_detectability_combined_time_on.png"
OUTPUT_FILENAME_RADIUS_COMPARISON = "gw/gw_detectability_radius_comparison.png"
OUTPUT_FILENAME_METHOD_COMPARISON = "gw/gw_detectability_method_comparison.png"

# Presentation only: case colours and one detector-style key across both panels.
DETECTOR_LABELS = {
    "ligo": r"$\mathrm{LIGO\ A+}$", "ce": r"$\mathrm{Cosmic\ Explorer}$",
    "decigo": r"$\mathrm{DECIGO}$", "lisa": r"$\mathrm{LISA\ (SciRDv1)}$",
    "et": r"$\mathrm{ET\ (10\,km)}$",
}
DETECTOR_LINESTYLES = {"ligo": ":", "et": (0, (3, 1, 1, 1, 1, 1)),
                      "ce": "-.", "decigo": "-", "lisa": "--"}
DETECTOR_COLOR = "0.15"
# Label below each curve, away from the signal bands; shared by every spectrum.
DETECTOR_CURVE_LABELS = {"ce": r"$\mathrm{CE}$", "et": r"$\mathrm{ET}$", "lisa": r"$\mathrm{LISA}$"}
DETECTOR_LABEL_HZ = {"ligo": 120.0, "et": 900.0, "ce": 300.0, "decigo": 0.15, "lisa": 0.04}
DETECTOR_LABEL_PAD_POINTS = {"ligo": 2, "et": 3, "ce": 3, "decigo": 6, "lisa": 3}
DETECTOR_LABEL_SIZE = 9
COMPARISON_HEIGHT = PAPER_TWO_PANEL_HEIGHT
TARGETS_HEIGHT = 3.4
SHIBATA_HEIGHT = PAPER_SINGLE_PANEL_HEIGHT
HORIZON_HEIGHT = 3.4
DETECTABILITY_HEIGHT = 5.4
COMBINED_FREQUENCY_LIMITS = (5e-4, 3e2)  # Display only; full bins enter SNR.
COMBINED_STRAIN_LIMITS = (1e-25, 1e-18)
EXAMPLE_LABEL_STYLE = "caption"  # Wessel-style default; "inline" or "key" optional.
EXAMPLE_KEY_HEIGHT = .7
EXAMPLE_KEY_FONT_SIZE = 10
VALIDATION_HEIGHT = PAPER_THREE_PANEL_HEIGHT
TEMPORAL_VALIDATION_HEIGHT = PAPER_THREE_PANEL_HEIGHT
TEMPORAL_NOTE_SIZE = 11
TEMPORAL_PREDICTION_COLOR = "#b85b00"
TEMPORAL_BLEND_COLOR = "0.35"
TEMPORAL_SPECTRUM_DYNAMIC_RANGE = 1e6  # Display only, relative to the larger peak.
# Display limits only; spectra and SNR still use every retained analysis bin.
CHARACTERISTIC_YLIM = (1e-25, 1e-17)
SOURCE_LABEL_SIZE = 11
SOURCE_LABEL_POSITIONS = {"ce": (0.76, 0.97), "decigo": (0.40, 0.97), "lisa": (0.17, 0.97)}
HORIZON_YLIM = (1e-2, 1e4)
HORIZON_REDSHIFT_TICKS = (0.001, 0.01, 0.05, 0.1, 0.5, 1.0)
HORIZON_REDSHIFT_TICK_SIZE = 8
HORIZON_LABEL_POSITIONS = {"ligo": (20, 0.018), "ce": (25, 150),
                           "et": (150, 1.1),
                           "decigo": (5000, 300), "lisa": (3e5, 1100)}


def _scientific_label(value):
    """Compact LaTeX number for source scales, without e-notation."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.6e}".split("e")
    coefficient, exponent = float(mantissa), int(exponent)
    if exponent == 0:
        return f"{coefficient:g}"
    prefix = "" if coefficient == 1 else rf"{coefficient:g}\times "
    return prefix + rf"10^{{{exponent}}}"


def _spectrum_plot_mask(frequency, values):
    keep = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0) & (values > 0)
    if np.any(keep):
        keep &= values >= 1e-10 * np.max(values[keep])
    return keep


def example_source_info(result):
    """Actual source scales and largest SNR across cases and plotted detectors."""
    examples = []
    for index, target in enumerate(result.targets):
        candidates = [(float(snr), name, detector)
                      for name, detectors in target.snr.items()
                      for detector, snr in detectors.items() if np.isfinite(snr)]
        snr, name, detector = max(candidates)
        band_snr, band_case = max((float(values[target.detector]), name)
                                  for name, values in target.snr.items())
        examples.append(dict(label=chr(ord("a") + index), mass_msun=target.mass,
                             distance_mpc=target.distance, redshift=target.redshift,
                             max_snr=snr, case=name, detector=detector,
                             band_detector=target.detector,
                             band_snr=band_snr, band_case=band_case,
                             distance_policy=getattr(target, "distance_policy", "fixed"),
                             horizon_case=getattr(target, "horizon_case", None)))
    return examples


def _example_caption(examples, note=None):
    details = []
    for row in examples:
        mass = _scientific_label(row["mass_msun"])
        distance = _scientific_label(row["distance_mpc"])
        detector = DETECTOR_CURVE_LABELS.get(row["detector"], DETECTOR_LABELS[row["detector"]])
        details.append(
            rf'({row["label"]}) $M_{{\rm BH}}={mass}\,M_\odot$, '
            rf'$D_L={distance}\,{{\rm Mpc}}$, $z={row["redshift"]:.4g}$, '
            rf'$\rho_{{\max}}={row["max_snr"]:.3g}$ '
            rf'({row["case"]}, {detector})')
    caption = (
        r"\caption{Top: SNR-threshold luminosity-distance reach versus source-frame "
        r"initial BH mass, with redshift on the right axis. Bottom: source-direction "
        r"mean characteristic strain $\langle h_c(f)\rangle$ for four illustrative "
        r"source scales and the corresponding effective detector characteristic "
        r"noise $h_n^{\mathrm{eff}}(f)=\sqrt{f S_n^{\mathrm{eff}}(f)}$. Colours identify the "
        r"simulation cases in both panels; line styles identify the detectors "
        r"in the distance curves and characteristic-noise curves using the same key. "
        r"The crosses in the distance panel and letter labels in the strain panel "
        r"refer to the same example sources: " + "; ".join(details) + ". "
        r"Each reported SNR is the largest among the plotted simulation cases and "
        r"detectors, using the selected-mode source-direction mean-amplitude "
        r"spectrum. Each example lies on the uppermost SNR=8 reach curve "
        r"at its fixed source mass, ordered from low to high mass: CE for "
        r"(a) and (b), DECIGO for (c), and LISA for (d). Distances are solved "
        r"using the same cosmology and "
        r"source-direction mean-amplitude statistic as the plotted reaches. "
        r"Example (a) uses a $10\,M_\odot$ BH as an illustrative BH--NS "
        r"remnant scale, not a simulated merger waveform. ")
    for row in examples:
        if row["band_detector"] != row["detector"]:
            detector = DETECTOR_CURVE_LABELS.get(
                row["band_detector"], DETECTOR_LABELS[row["band_detector"]])
            caption += (rf'For example ({row["label"]}), the largest SNR in '
                        + detector + r' is $' + f'{row["band_snr"]:.3g}' + '$. ')
    if note:
        caption += "The continuation shown here is conditional; " + note.replace("\n", "; ") + ". "
    else:
        caption += "Only the measured finite-duration signal is used. "
    return caption + "}\n"


def _example_key(ax, examples):
    columns = ["Example", r"$M_{\rm BH}/M_\odot$", r"$D_L\,[{\rm Mpc}]$",
               r"$z$", r"$\rho_{\rm band}$", "Case", "Detector"]
    rows = [[f'({row["label"]})', '$' + _scientific_label(row["mass_msun"]) + '$',
             '$' + _scientific_label(row["distance_mpc"]) + '$',
             f'{row["redshift"]:.4g}', f'{row["band_snr"]:.3g}', row["band_case"],
             DETECTOR_CURVE_LABELS.get(row["band_detector"], DETECTOR_LABELS[row["band_detector"]])]
            for row in examples]
    ax.set_axis_off()
    table = ax.table(cellText=rows, colLabels=columns, cellLoc="center",
                     colWidths=[.11, .18, .18, .13, .13, .1, .17], bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(EXAMPLE_KEY_FONT_SIZE)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor("0.65")
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor("0.95")


def _format_axes(axes):
    for ax in np.asarray(axes).flat:
        ax.set_xscale("log")
        ax.grid(False, which="minor")
        ax.grid(True, which="major")
        format_paper_axes(ax)


def _label_noise_curve(ax, frequency, noise, detector):
    """Bend serif glyph outlines along the displayed log-log curve, below it.

    Call after axis limits and layout are final. Work in screen coordinates
    so font size and normal offset stay consistent despite different scales.
    """
    label = DETECTOR_CURVE_LABELS.get(detector, DETECTOR_LABELS[detector])
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
    ax.add_patch(PathPatch(MplPath(vertices, glyphs.codes), facecolor=DETECTOR_COLOR,
                           edgecolor="none", zorder=4, clip_on=True))


def _plot_radius_comparison(result, args):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figure_size("double", COMPARISON_HEIGHT))
    for name, source in result.sources.items():
        if name not in result.inner:
            continue
        x, hc = analysis.orbital_spectrum(source.config, source.spectrum)
        inner_x, inner_hc = analysis.orbital_spectrum(source.config, result.inner[name].spectrum)
        keep = _spectrum_plot_mask(x, hc)
        axes[0].plot(x[keep], hc[keep], color=source.color, linestyle=source.linestyle, label=source.legend_name)
        ratio_x, ratio = analysis.interpolated_spectral_ratio(x, hc, inner_x, inner_hc)
        significant = np.isfinite(ratio) & (np.interp(ratio_x, x, hc) >= 1e-3 * np.max(hc))
        axes[1].plot(ratio_x[significant], ratio[significant], color=source.color, linestyle=source.linestyle)
    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$ (outer)")
    axes[0].set_yscale("log")
    axes[1].set_ylabel(r"$h_c(\mathrm{inner})/h_c(\mathrm{outer})$")
    axes[1].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[1].axhline(1, color="0.45", linewidth=0.8, linestyle=":")
    _format_axes(axes)
    ordered_sim_legend(axes[0], ncols=3, loc="upper right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_RADIUS_COMPARISON)


def _plot_method_comparison(result, args):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figure_size("double", COMPARISON_HEIGHT))
    for name, source in result.sources.items():
        x, hc = analysis.orbital_spectrum(source.config, source.spectrum)
        keep = _spectrum_plot_mask(x, hc)
        axes[0].plot(x[keep], hc[keep], color=source.color, linestyle=source.linestyle, label=source.legend_name)
        representative = result.representative.get(name)
        if representative is not None:
            _, rep_hc = analysis.orbital_spectrum(source.config, representative)
            ratio = np.divide(hc, rep_hc, out=np.full_like(hc, np.nan), where=rep_hc > 0)
            mask = keep & np.isfinite(ratio)
            axes[1].plot(x[mask], ratio[mask], color=source.color, linestyle=source.linestyle)
    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$")
    axes[0].set_yscale("log")
    average_label = r"\langle h_c\rangle_\Omega" if analysis.SOURCE_AVERAGING == "mean" else r"h_{c,\mathrm{RMS}}"
    axes[1].set_ylabel(rf"${average_label}/h_c(\pi/2.34)$")
    axes[1].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[1].axhline(1, color="0.45", linewidth=0.8, linestyle=":")
    _format_axes(axes)
    ordered_sim_legend(axes[0], ncols=3, loc="upper right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_METHOD_COMPARISON)


def _draw_characteristic_strain(ax, result, *, fixed_source=False, note=None,
                                legend=True, example_labels="scales"):
    targets = [result.shibata] if fixed_source else result.targets
    examples = example_source_info(result) if example_labels != "scales" else None
    for index, target in enumerate(targets):
        for name, (frequency, hc) in target.strain.items():
            source = result.sources[name]
            keep = _spectrum_plot_mask(frequency, hc)
            ax.plot(frequency[keep], hc[keep], color=source.color,
                    linestyle=source.linestyle, label=source.legend_name if index == 0 else "_nolegend_")
            print(f"{name}, M_BH={target.mass:g} Msun, D_L={target.distance:g} Mpc: "
                  + ", ".join(f"{detector.upper()} SNR={snr:.3g}" for detector, snr in target.snr[name].items()))
        mass_text = _scientific_label(target.mass)
        distance_text = _scientific_label(target.distance)
        position = SOURCE_LABEL_POSITIONS["ce"] if fixed_source else SOURCE_LABEL_POSITIONS[target.detector]
        if not fixed_source and target.detector == "ce" and target.mass == 10:
            position = (.92, .75)
        text = (rf"$M_{{\mathrm{{BH}}}}={mass_text}\,M_\odot$" + "\n"
                + rf"$D_L={distance_text}\,\mathrm{{Mpc}}$")
        size = SOURCE_LABEL_SIZE
        alignment = "top"
        if example_labels == "inline":
            row = examples[index]
            detector = DETECTOR_CURVE_LABELS.get(row["detector"], DETECTOR_LABELS[row["detector"]])
            text = (rf'$({row["label"]})\quad M_{{\rm BH}}={mass_text}\,M_\odot$' + "\n"
                    + rf"$D_L={distance_text}\,\mathrm{{Mpc}}$" + "\n"
                    + rf'$z={target.redshift:.4g}$' + "\n"
                    + rf'$\rho_{{\max}}={row["max_snr"]:.3g}$'
                    + f' ({row["case"]}, {detector})')
            # Keep the full information on the figure, above the data region.
            # The inline layout reserves a larger gap between the two panels.
            ax.text(*position, rf'$({row["label"]})$', transform=ax.transAxes,
                    ha="center", va="top", fontsize=SOURCE_LABEL_SIZE)
            position = {"ce": (.69, 1.04), "decigo": (.40, 1.04), "lisa": (.12, 1.04)}[target.detector]
            if target.detector == "ce" and target.mass == 10:
                position = (.94, 1.04)
            alignment = "bottom"
            size = 9
        elif example_labels in {"caption", "key"}:
            text = rf'$({examples[index]["label"]})$'
            # Place each letter next to its best in-band SNR contribution.
            frequency, hc = target.strain[examples[index]["band_case"]]
            noise_f, noise_h = result.noise[target.detector]
            keep = _spectrum_plot_mask(frequency, hc)
            keep &= (frequency >= noise_f[0]) & (frequency <= noise_f[-1])
            if np.any(keep):
                f, h = frequency[keep], hc[keep]
                hn = np.exp(np.interp(np.log(f), np.log(noise_f), np.log(noise_h)))
                peak = int(np.argmax(h / hn))
                ax.annotate(text, (f[peak], h[peak]), xytext=(3, 12),
                            textcoords="offset points", ha="center", va="bottom",
                            fontsize=10)
                continue
        ax.text(*position, text, transform=ax.transAxes,
                ha="center", va=alignment, fontsize=size)
    for detector, (frequency, hn) in result.noise.items():
        ax.plot(frequency, hn, color=DETECTOR_COLOR, linewidth=1.25,
                linestyle=DETECTOR_LINESTYLES[detector], label="_nolegend_")
    ax.set(xlabel=r"$f_{\mathrm{obs}}\,[\mathrm{Hz}]$", ylabel=r"$\langle h_c(f)\rangle$",
           yscale="log", ylim=CHARACTERISTIC_YLIM,
           xlim=(min(f[0] for f, _ in result.noise.values()), max(f[-1] for f, _ in result.noise.values())))
    _format_axes([ax])
    if legend:
        ordered_sim_legend(ax, ncols=3, loc="lower left", **COMPACT_LEGEND_KWARGS)
    if note:
        ax.text(.98, .04, note, transform=ax.transAxes, ha="right", va="bottom",
                fontsize=SOURCE_LABEL_SIZE)


def _plot_characteristic_strain(result, args, *, fixed_source=False, filename=None, note=None):
    height = SHIBATA_HEIGHT if fixed_source else TARGETS_HEIGHT
    fig, ax = plt.subplots(figsize=figure_size("double", height))
    _draw_characteristic_strain(ax, result, fixed_source=fixed_source, note=note)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.98)
    for detector, (frequency, hn) in result.noise.items():
        _label_noise_curve(ax, frequency, hn, detector)
    savefig(fig, args, filename or (OUTPUT_FILENAME_SHIBATA if fixed_source else OUTPUT_FILENAME_CHARACTERISTIC_STRAIN))


def _draw_horizon(ax, result, *, note=None, legend=True):
    masses, horizons = result.horizons
    for index, (detector, cases) in enumerate(horizons.items()):
        for name, horizon in cases.items():
            source = result.sources[name]
            ax.plot(masses, horizon, color=source.color,
                    linestyle=DETECTOR_LINESTYLES[detector],
                    label=source.legend_name if index == 0 else "_nolegend_")
        label = DETECTOR_CURVE_LABELS.get(detector, DETECTOR_LABELS[detector])
        ax.text(*HORIZON_LABEL_POSITIONS[detector], label, ha="center", va="bottom",
                fontsize=SOURCE_LABEL_SIZE, color=DETECTOR_COLOR)
    ax.set(xlabel=r"$M_{\mathrm{BH}}\,[M_\odot]$", ylabel=r"$D_L\,[\mathrm{Mpc}]$",
           yscale="log", xlim=(masses[0], masses[-1]), ylim=HORIZON_YLIM)
    ax.text(0.97, 0.96, rf"$\mathrm{{SNR}}={analysis.SNR_THRESHOLD:g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=SOURCE_LABEL_SIZE)
    _format_axes([ax])
    cosmology = analysis.FlatLambdaCDM()
    redshift_axis = ax.secondary_yaxis(
        "right", functions=(cosmology.redshift_at_luminosity_distance,
                            cosmology.luminosity_distance_mpc))
    redshift_axis.set_ylabel(r"$z$")
    redshift_axis.yaxis.set_major_locator(FixedLocator(HORIZON_REDSHIFT_TICKS))
    redshift_axis.yaxis.set_major_formatter(
        FuncFormatter(lambda z, _: rf"${z:g}$"))
    redshift_axis.yaxis.set_minor_locator(NullLocator())
    redshift_axis.tick_params(axis="y", which="both", direction="in",
                              left=False, right=True, labelleft=False, labelright=True,
                              labelsize=HORIZON_REDSHIFT_TICK_SIZE)
    redshift_axis.grid(False, which="both")
    # The right edge represents redshift, not mirrored distance ticks.
    ax.tick_params(axis="y", which="both", right=False)
    if legend:
        ordered_sim_legend(ax, ncols=3, loc="upper left", **COMPACT_LEGEND_KWARGS)
    if note:
        ax.text(.03, .83, note, transform=ax.transAxes, ha="left", va="top",
                fontsize=SOURCE_LABEL_SIZE)


def _plot_horizon(result, args, *, filename=OUTPUT_FILENAME_HORIZON, note=None):
    fig, ax = plt.subplots(figsize=figure_size("double", HORIZON_HEIGHT))
    _draw_horizon(ax, result, note=note)
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.18, top=0.98)
    savefig(fig, args, filename)


def _plot_detectability(result, args, *, filename=OUTPUT_FILENAME_COMBINED, note=None,
                        example_labels=EXAMPLE_LABEL_STYLE, write_metadata=True):
    """Distance above characteristic strain, following Wessel Fig. 10's layout."""
    if example_labels not in {"inline", "caption", "key"}:
        raise ValueError("Choose inline, caption, or key example labels")
    examples = example_source_info(result)
    extra_rows = max(0, len(examples) - 3)
    height = DETECTABILITY_HEIGHT + (EXAMPLE_KEY_HEIGHT + .18 * extra_rows
                                    if example_labels == "key" else 0)
    fig, axes = plt.subplots(2, 1, figsize=figure_size("double", height),
                             gridspec_kw={"height_ratios": (1.5, 1)})
    _draw_horizon(axes[0], result, legend=False)
    for row in examples:
        axes[0].plot(row["mass_msun"], row["distance_mpc"], marker="+",
                     color="black", markersize=7, markeredgewidth=1.1,
                     linestyle="none", zorder=5)
        offset = (5, -14)
        if row["band_detector"] == "ce":
            offset = (-16, 6) if row["mass_msun"] == 10 else (5, 5)
        axes[0].annotate(rf'$({row["label"]})$', (row["mass_msun"], row["distance_mpc"]),
                         xytext=offset, textcoords="offset points", fontsize=10)
    _draw_characteristic_strain(axes[1], result, legend=False, example_labels=example_labels)
    axes[1].set(xlim=COMBINED_FREQUENCY_LIMITS, ylim=COMBINED_STRAIN_LIMITS)
    # Scale typography and strokes to these deliberately short panels.
    for ax in axes:
        ax.xaxis.label.set_fontsize(11)
        ax.yaxis.label.set_fontsize(11)
        ax.tick_params(which="major", labelsize=9, length=4, width=.8, pad=3)
        ax.tick_params(which="minor", length=2, width=.6)
        ax.grid(True, which="major", linewidth=.5, color="0.75")
        ax.grid(True, which="minor", linewidth=.3, color="0.88")
        for spine in ax.spines.values():
            spine.set_linewidth(.8)
        for line in ax.lines:
            line.set_linewidth(1.05)
        for text in ax.texts:
            text.set_fontsize(9)
    for secondary in axes[0].child_axes:
        secondary.yaxis.label.set_fontsize(11)
        secondary.tick_params(which="major", labelsize=HORIZON_REDSHIFT_TICK_SIZE,
                              length=4, width=.8, pad=3)
        for spine in secondary.spines.values():
            spine.set_linewidth(.8)
    # Both lower-panel quantities share the same dimensionless amplitude scale.
    noise_axis = axes[1].secondary_yaxis("right")
    noise_axis.set_ylabel(r"$h_n^{\mathrm{eff}}(f)$", fontsize=11, labelpad=6)
    noise_axis.yaxis.set_major_locator(NullLocator())
    noise_axis.yaxis.set_minor_locator(NullLocator())
    noise_axis.spines["right"].set_linewidth(.8)
    axes[1].tick_params(axis="y", which="both", right=False)
    # Keep both keys outside the data, in one compact horizontal strip.
    bottom = .23 + .02 * extra_rows if example_labels == "key" else .13
    gap = .66 if example_labels == "inline" else .42 if note else .32
    top = .90 if example_labels == "inline" else .97
    fig.subplots_adjust(left=.11, right=.89, bottom=bottom, top=top, hspace=gap)
    lower_top = axes[1].get_position().y1
    legend_y = (.96 if example_labels == "inline" else
                lower_top + (.26 if note and example_labels != "key" else .10)/height)
    if note and example_labels in {"key", "inline"}:
        fig.text(.13, .008, note.replace("\n", "; "), ha="left", va="bottom", fontsize=9)
    elif note:
        fig.text(.50, lower_top + .10/height, note.replace("\n", "; "),
                 ha="center", va="center", fontsize=8)
    case_handles = [Line2D([], [], color=source.color, linestyle=source.linestyle,
                           label=source.legend_name) for source in result.sources.values()]
    fig.legend(handles=case_handles, ncols=len(case_handles),
               loc="center left", bbox_to_anchor=(.11, legend_y),
               fontsize=9, handlelength=1.3, columnspacing=.7, handletextpad=.4,
               borderaxespad=0, borderpad=0, frameon=False)
    detector_handles = [Line2D([], [], color=DETECTOR_COLOR, linestyle=DETECTOR_LINESTYLES[d],
                               linewidth=1.05,
                               label=DETECTOR_CURVE_LABELS.get(d, DETECTOR_LABELS[d]))
                        for d in result.noise]
    fig.legend(handles=detector_handles, ncols=len(detector_handles),
               loc="center right", bbox_to_anchor=(.89, legend_y),
               fontsize=9, handlelength=1.8, columnspacing=.7, handletextpad=.4,
               borderaxespad=0, borderpad=0, frameon=False)
    if example_labels == "key":
        key_axis = fig.add_axes([.12, .035, .86, .11 + .02 * extra_rows])
        _example_key(key_axis, examples)
    if not args.no_save:
        path = Path(args.outdir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve the requested full-page 7-inch width in both exports.
        with plt.rc_context({"savefig.bbox": None}):
            fig.savefig(path.with_suffix(".pdf"))
        if write_metadata:
            path.with_name(path.stem + "_examples.json").write_text(json.dumps(examples, indent=2) + "\n")
            path.with_name(path.stem + "_caption.tex").write_text(_example_caption(examples, note))
    with plt.rc_context({"savefig.bbox": None}):
        savefig(fig, args, filename)
    if not args.no_save and filename in {OUTPUT_FILENAME_COMBINED, OUTPUT_FILENAME_COMBINED_TIME_ON}:
        # Keep related variants together and preserve existing paper-facing paths.
        variant = "conditional" if filename == OUTPUT_FILENAME_COMBINED_TIME_ON else "measured"
        family = Path(args.outdir) / "gw" / "detectability"
        family.mkdir(parents=True, exist_ok=True)
        suffixes = [".png", ".pdf"] + (["_caption.tex", "_examples.json"] if write_metadata else [])
        for suffix in suffixes:
            shutil.copyfile(path.with_name(path.stem + suffix), family / (variant + suffix))


def _plot_time_on(result, args):
    """Keep the direct-Psi4 baseline; modify only the (2,2) termination."""
    spectra, reports, extended, conditional = {}, {}, [], []
    for name, source in result.sources.items():
        trial = analysis.temporal_trial_for_case(source.config)
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
        te, pe = analysis.continued_psi4_mode(tau, p, trial)
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
    if PLOT_HORIZON:
        updated.horizons = analysis.horizon_curves(spectra, curves, cosmology)
    if PLOT_CHARACTERISTIC_STRAIN and PLOT_HORIZON:
        _plot_detectability(updated, args, filename=OUTPUT_FILENAME_COMBINED_TIME_ON, note=note)
    else:
        _plot_characteristic_strain(updated, args, filename=OUTPUT_FILENAME_TIME_ON, note=note)
        if PLOT_HORIZON:
            _plot_horizon(updated, args, filename=OUTPUT_FILENAME_HORIZON_TIME_ON, note=note)
    report = dict(input="direct Psi4 FFT for both off and on", modes=analysis.MODES,
                  extended_mode=[2, 2], cases=reports,
                  off=[dict(mass=t.mass, distance=t.distance, snr=t.snr) for t in result.targets],
                  on=[dict(mass=t.mass, distance=t.distance, snr=t.snr) for t in targets])
    if not args.no_save:
        path = Path(args.outdir)/"gw/gw_detectability_time_comparison.json"
        path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def plot_validation(sim_name):
    radial, tapers, cutoffs = analysis.validation_data(sim_name)
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
    _format_axes(axes)
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
    join_start = max(trial.time[0], end-analysis.TAIL_JOIN_ORBITS*period)
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
        trial = analysis.temporal_trial_for_case(case)
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
        fig.text(.17, .02, rf"$M_{{\mathrm{{BH}}}}={_scientific_label(mass)}\,M_\odot$, "
                 rf"$D_L={_scientific_label(distance)}\,\mathrm{{Mpc}}$",
                 fontsize=TEMPORAL_NOTE_SIZE)
        fig.subplots_adjust(left=.17, right=.98, bottom=.24, top=.98)
        save_individual_fig(fig, args, case.name, "gw_temporal_on_off.png")
    if not args.no_save:
        destination = Path(args.outdir)/"gw"/"temporal_trial.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(reports, indent=2)+"\n")
    return reports


def main(argv=None):
    args = parser("Plot GW characteristic strain, noise and detectability horizons.").parse_args(argv)
    setup(args)
    result = analysis.analyze(args.sims, compare_radius=PLOT_RADIUS_COMPARISON,
                              compare_direction=PLOT_METHOD_COMPARISON, compute_horizons=PLOT_HORIZON)
    apply_styles(list(result.sources.values()))
    if PLOT_RADIUS_COMPARISON and result.inner:
        _plot_radius_comparison(result, args)
    if PLOT_METHOD_COMPARISON:
        _plot_method_comparison(result, args)
    if PLOT_CHARACTERISTIC_STRAIN and PLOT_HORIZON:
        _plot_detectability(result, args)
    elif PLOT_CHARACTERISTIC_STRAIN:
        _plot_characteristic_strain(result, args)
    elif PLOT_HORIZON:
        _plot_horizon(result, args)
    if PLOT_SHIBATA_COMPARISON:
        _plot_characteristic_strain(result, args, fixed_source=True)
    if PLOT_TIME_COMPARISON:
        _plot_time_on(result, args)


def individual_main(argv=None):
    args = parser("Validate direct-Psi4 detectability for individual simulations.").parse_args(argv)
    setup(args)
    for name in args.sims or analysis.SIM_NAMES:
        fig = plot_validation(name)
        save_individual_fig(fig, args, name, f"gw_detectability_method_validation_{name}.png")


if __name__ == "__main__":
    main()
