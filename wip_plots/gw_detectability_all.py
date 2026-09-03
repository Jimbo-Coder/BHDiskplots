#!/usr/bin/env python3
"""Finite-duration GW detectability directly from extracted r*Psi4.

The production method follows the collaborator BH-cluster implementation and
the finite-signal analysis in Wessel et al. (2021): discard the initial
transient, combine all modes through ell=3, apply a Tukey window, transform
Psi4 directly, divide by omega^2 in frequency space, and average the resulting
polarization amplitude over source directions. This script deliberately does
not perform radial or temporal waveform extrapolation.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
package_root = Path(__file__).resolve().parents[1]
if str(package_root) not in sys.path:
    sys.path.insert(0, str(package_root))

from config import (
    GW_FIRST_WAVEZONE_PARFILE_INDEX,
    GW_OUTERMOST_PARFILE_INDEX,
    PLOTS_DIR,
)
from helpers.gw_detectability import (
    DimensionlessSpectrum,
    FlatLambdaCDM,
    METERS_PER_MPC,
    METERS_PER_M_SUN,
    SECONDS_PER_M_SUN,
    characteristic_strain,
    direct_psi4_spectrum,
    observer_spectrum,
    prepared_rpsi4_modes,
)
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    COMPACT_LEGEND_KWARGS,
    FIGURE_LEGEND_BORDERAXESPAD,
    figure_size,
    format_paper_axes,
    ordered_sim_fig_legend,
    ordered_sim_legend,
)
from gw_psi4 import MODES as AVAILABLE_GW_MODES


# Scientific knobs.
DETECTABILITY_SIM_NAMES = ("A1", "A2", "A3", "B1", "B2", "B3")
# r=120 is the first extraction sphere retained as a wave-zone systematic;
# r=170 is the outermost valid common sphere and is the central value.
DETECTABILITY_FIRST_WAVEZONE_PARFILE_INDEX = GW_FIRST_WAVEZONE_PARFILE_INDEX
DETECTABILITY_OUTER_PARFILE_INDEX = GW_OUTERMOST_PARFILE_INDEX
DETECTABILITY_PSI4_PARFILE_INDEX = DETECTABILITY_OUTER_PARFILE_INDEX
DETECTABILITY_MODES = tuple(mode for mode in AVAILABLE_GW_MODES if mode[0] <= 3)
# Wessel et al. remove the first 1000 M_BH to suppress initial-data relaxation.
DETECTABILITY_TRANSIENT_CUTOFF_MBH = 1000.0
# The collaborator's direct-Psi4 implementation uses a Tukey alpha of 0.05.
DETECTABILITY_TAPER_ALPHA = 0.05
DETECTABILITY_ZERO_PAD_FACTOR = 2.0
# Do not interpret frequencies represented by fewer than three retained cycles.
DETECTABILITY_LOW_FREQUENCY_CYCLES = 3.0
DETECTABILITY_THETA_NODES = 24
DETECTABILITY_PHI_NODES = 48
DETECTABILITY_SOURCE_AVERAGING = "mean"

# Wessel et al. finite-signal comparison points: (source BH mass, distance).
DETECTABILITY_TARGETS = ((10.0, 150.0), (1.0e3, 4.0e4), (2.0e5, 7.0e3))
DETECTABILITY_SNR_THRESHOLD = 8.0
DETECTABILITY_HORIZON_MASS_RANGE_MSUN = (1.0, 1.0e7)
DETECTABILITY_HORIZON_MASS_SAMPLES = 44
DETECTABILITY_HORIZON_REDSHIFT_MAX = 10.0
DETECTABILITY_HORIZON_REDSHIFT_SAMPLES = 64
DETECTABILITY_HORIZON_SPECTRAL_BINS = 2048

# Plot-selection knobs.
DETECTABILITY_PLOT_CHARACTERISTIC_STRAIN = True
DETECTABILITY_PLOT_HORIZON = True
DETECTABILITY_PLOT_METHOD_COMPARISON = True
DETECTABILITY_PLOT_RADIUS_COMPARISON = True
DETECTABILITY_ACTIVE_DETECTORS = ("ligo", "ce", "decigo", "lisa")

# Output knobs.
OUTPUT_FILENAME_RADIUS_COMPARISON = "gw/gw_detectability_radius_comparison.png"
OUTPUT_FILENAME_METHOD_COMPARISON = "gw/gw_detectability_method_comparison.png"
OUTPUT_FILENAME_CHARACTERISTIC_STRAIN = "gw/gw_detectability_characteristic_strain.png"
OUTPUT_FILENAME_HORIZON = "gw/gw_detectability_horizon.png"


SCRIPT_DIR = Path(__file__).resolve().parent
DETECTOR_CURVE_DIR = SCRIPT_DIR.parent / "detector_curves"
DETECTOR_BOUNDS = {
    "ligo": (5.0, 2.5e3),
    "ce": (5.0, 5.0e3),
    "decigo": (1.0e-2, 10.0),
    "lisa": (1.0e-4, 1.0),
}
DETECTOR_LABELS = {
    "ligo": r"$\mathrm{LIGO\ A+}$",
    "ce": r"$\mathrm{Cosmic\ Explorer}$",
    "decigo": r"$\mathrm{DECIGO}$",
    "lisa": r"$\mathrm{LISA\ (SciRDv1)}$",
}
DETECTOR_LINESTYLES = {
    "ligo": ":",
    "ce": "-.",
    "decigo": "-",
    "lisa": "--",
}


@dataclass(frozen=True)
class DetectorCurve:
    frequency: np.ndarray
    asd: np.ndarray


@dataclass(frozen=True)
class BinnedSpectralPower:
    frequency: np.ndarray
    power_dfrequency: np.ndarray


def active_detectors():
    return tuple(name for name in DETECTABILITY_ACTIVE_DETECTORS if name in DETECTOR_BOUNDS)


def _read_two_column_curve(filename: str, quantity: str) -> DetectorCurve:
    path = DETECTOR_CURVE_DIR / filename
    raw = np.loadtxt(path, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    frequency = np.asarray(raw[:, 0], dtype=float)
    values = np.asarray(raw[:, 1], dtype=float)
    keep = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0.0) & (values > 0.0)
    frequency = frequency[keep]
    values = values[keep]
    if frequency.size < 2:
        raise ValueError(f"Detector curve {path} has fewer than two positive samples")
    order = np.argsort(frequency)
    asd = np.sqrt(values) if quantity == "psd" else values
    return DetectorCurve(frequency[order], asd[order])


def _decigo_instrument_asd(frequency) -> np.ndarray:
    """Single effective L-shaped DECIGO interferometer, Yagi-Seto Eq. (5)."""
    frequency = np.maximum(np.asarray(frequency, dtype=float), 1.0e-12)
    pivot = 7.36
    psd = (
        6.53e-49 * (1.0 + (frequency / pivot) ** 2)
        + 4.45e-51 * frequency ** -4 / (1.0 + (frequency / pivot) ** 2)
        + 4.94e-52 * frequency ** -4
    )
    return np.sqrt(psd)


def load_detector_curves():
    return {
        "ligo": _read_two_column_curve("AplusDesign.txt", "asd"),
        "ce": _read_two_column_curve("CE2_40km_strain.txt", "asd"),
        # This file is the LISA SciRDv1 equivalent sky-and-polarization-averaged PSD.
        "lisa": _read_two_column_curve("LISA_Alloc_Sh.txt", "psd"),
    }


def _log_interpolate_curve(curve: DetectorCurve, frequency) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=float)
    clipped = np.clip(frequency, curve.frequency[0], curve.frequency[-1])
    return np.power(
        10.0,
        np.interp(np.log10(clipped), np.log10(curve.frequency), np.log10(curve.asd)),
    )


def effective_detector_asd(detector: str, frequency, curves) -> np.ndarray:
    """Noise ASD in the same Wessel convention as polarization-averaged h_res.

    A+, CE, and DECIGO start as optimal single-interferometer curves. The
    factor sqrt(5) performs the standard right-angle sky/polarization response
    average. Wessel et al. then multiply all sky-and-polarization-averaged
    curves by sqrt(2), because h_res already contains the 1/sqrt(2)
    polarization average. The LISA file already includes the first average.
    """
    if detector == "decigo":
        return np.sqrt(10.0) * _decigo_instrument_asd(frequency)
    asd = _log_interpolate_curve(curves[detector], frequency)
    if detector in {"ligo", "ce"}:
        return np.sqrt(10.0) * asd
    if detector == "lisa":
        return np.sqrt(2.0) * asd
    raise ValueError(f"Unknown detector {detector!r}")


def _spectrum_plot_mask(frequency, values, relative_floor=1.0e-10):
    frequency = np.asarray(frequency, dtype=float)
    values = np.asarray(values, dtype=float)
    keep = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0.0) & (values > 0.0)
    if np.any(keep) and relative_floor > 0.0:
        keep &= values >= relative_floor * np.max(values[keep])
    return keep


def _noise_characteristic_strain(frequency, asd):
    return np.sqrt(np.asarray(frequency, dtype=float)) * np.asarray(asd, dtype=float)


def _load_source_spectra_at_radius(args, parfile_index, *, include_representative=False):
    names = args.sims if args.sims is not None else DETECTABILITY_SIM_NAMES
    sims = load_sims(
        ["strain"],
        names=names,
        psi4_parfile_index=parfile_index,
        psi4_mode=DETECTABILITY_MODES[0],
    )
    spectra = {}
    representative = {}
    for sim in sims:
        mass = float(sim.config.mlittle)
        try:
            retarded_time, mode_data, time_method = prepared_rpsi4_modes(
                sim.strain_result,
                sim.psi4,
                DETECTABILITY_MODES,
            )
        except ValueError as exc:
            print(f"{sim.config.name}: skipping detectability; {exc}")
            continue
        print(f"{sim.config.name}: rPsi4 input from {time_method}")
        try:
            spectrum, info = direct_psi4_spectrum(
                retarded_time,
                mode_data,
                mass,
                transient_cutoff_mbh=DETECTABILITY_TRANSIENT_CUTOFF_MBH,
                taper_alpha=DETECTABILITY_TAPER_ALPHA,
                zero_pad_factor=DETECTABILITY_ZERO_PAD_FACTOR,
                low_frequency_cycles=DETECTABILITY_LOW_FREQUENCY_CYCLES,
                theta_nodes=DETECTABILITY_THETA_NODES,
                phi_nodes=DETECTABILITY_PHI_NODES,
                averaging=DETECTABILITY_SOURCE_AVERAGING,
            )
            spectra[sim.config.name] = spectrum
            print(
                f"{sim.config.name}: direct rPsi4 spectrum, {len(info.modes)} modes, "
                f"{info.samples} samples over {info.duration_mbh:.1f} M_BH, "
                f"nu=[{info.frequency_min:.3g},{info.frequency_max:.3g}]"
            )
            if include_representative:
                representative[sim.config.name], _ = direct_psi4_spectrum(
                    retarded_time,
                    mode_data,
                    mass,
                    transient_cutoff_mbh=DETECTABILITY_TRANSIENT_CUTOFF_MBH,
                    taper_alpha=DETECTABILITY_TAPER_ALPHA,
                    zero_pad_factor=DETECTABILITY_ZERO_PAD_FACTOR,
                    low_frequency_cycles=DETECTABILITY_LOW_FREQUENCY_CYCLES,
                    representative_direction=(np.pi / 2.34, 0.0),
                )
        except ValueError as exc:
            print(f"{sim.config.name}: skipping detectability; {exc}")
    return sims, spectra, representative


def _load_source_spectra(args):
    sims, spectra, representative = _load_source_spectra_at_radius(
        args,
        DETECTABILITY_OUTER_PARFILE_INDEX,
        include_representative=DETECTABILITY_PLOT_METHOD_COMPARISON,
    )
    _, first_wavezone_spectra, _ = _load_source_spectra_at_radius(
        args,
        DETECTABILITY_FIRST_WAVEZONE_PARFILE_INDEX,
    )
    missing = sorted(set(spectra) - set(first_wavezone_spectra))
    if missing:
        print(
            "Radius-systematic spectrum unavailable at the first wave-zone radius for: "
            + ", ".join(missing)
        )
    return sims, spectra, representative, first_wavezone_spectra


def _interpolated_spectral_ratio(reference_x, reference_y, test_x, test_y):
    lower = max(float(np.min(reference_x)), float(np.min(test_x)))
    upper = min(float(np.max(reference_x)), float(np.max(test_x)))
    keep = (
        (reference_x >= lower)
        & (reference_x <= upper)
        & np.isfinite(reference_y)
        & (reference_y > 0.0)
    )
    x = reference_x[keep]
    interpolated = np.interp(x, test_x, test_y)
    ratio = np.divide(
        interpolated,
        reference_y[keep],
        out=np.full(x.shape, np.nan),
        where=reference_y[keep] > 0.0,
    )
    return x, ratio


def _plot_radius_comparison(sims, outer_spectra, first_wavezone_spectra, args):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figure_size("double", 5.7))
    registry = {sim.config.name: sim for sim in sims}
    for name, outer in outer_spectra.items():
        first = first_wavezone_spectra.get(name)
        if first is None:
            continue
        sim = registry[name]
        orbital_frequency = (
            float(sim.config.gw_omega_orbital) * float(sim.config.mlittle) / (2.0 * np.pi)
        )
        outer_x = outer.frequency / orbital_frequency
        outer_hc = characteristic_strain(outer.frequency, outer.strain_ft)
        first_x = first.frequency / orbital_frequency
        first_hc = characteristic_strain(first.frequency, first.strain_ft)
        keep = _spectrum_plot_mask(outer_x, outer_hc)
        axes[0].plot(
            outer_x[keep],
            outer_hc[keep],
            color=sim.color,
            linestyle=sim.linestyle,
            label=sim.legend_name,
        )
        ratio_x, ratio = _interpolated_spectral_ratio(outer_x, outer_hc, first_x, first_hc)
        significant = (
            np.isfinite(ratio)
            & (np.interp(ratio_x, outer_x, outer_hc) >= 1.0e-3 * np.nanmax(outer_hc))
        )
        axes[1].plot(
            ratio_x[significant],
            ratio[significant],
            color=sim.color,
            linestyle=sim.linestyle,
        )

    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$ at $r=170$")
    axes[0].set_yscale("log")
    axes[1].set_ylabel(r"$h_c(r=120)/h_c(r=170)$")
    axes[1].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[1].axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        format_paper_axes(ax)
    ordered_sim_legend(axes[0], ncols=3, loc="lower right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_RADIUS_COMPARISON)


def _plot_method_comparison(sims, spectra, representative, args):
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figure_size("double", 5.7))
    registry = {sim.config.name: sim for sim in sims}
    for name, spectrum in spectra.items():
        sim = registry[name]
        orbital_frequency = float(sim.config.gw_omega_orbital) * float(sim.config.mlittle) / (2.0 * np.pi)
        x = spectrum.frequency / orbital_frequency
        mean_hc = characteristic_strain(spectrum.frequency, spectrum.strain_ft)
        rep = representative.get(name)
        keep = _spectrum_plot_mask(x, mean_hc)
        axes[0].plot(x[keep], mean_hc[keep], color=sim.color, linestyle=sim.linestyle, label=sim.legend_name)
        if rep is not None and np.array_equal(rep.frequency, spectrum.frequency):
            rep_hc = characteristic_strain(rep.frequency, rep.strain_ft)
            ratio = np.divide(mean_hc, rep_hc, out=np.full_like(mean_hc, np.nan), where=rep_hc > 0.0)
            ratio_keep = keep & np.isfinite(ratio)
            axes[1].plot(x[ratio_keep], ratio[ratio_keep], color=sim.color, linestyle=sim.linestyle)

    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$")
    axes[0].set_yscale("log")
    axes[1].set_ylabel(r"$\langle h_c\rangle_\Omega/h_c(\pi/2.34)$")
    axes[1].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[1].axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        format_paper_axes(ax)
    ordered_sim_legend(axes[0], ncols=3, loc="lower right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_METHOD_COMPARISON)


def _plot_characteristic_strain(sims, spectra, curves, cosmology, args):
    registry = {sim.config.name: sim for sim in sims}
    fig, axes = plt.subplots(
        len(DETECTABILITY_TARGETS),
        1,
        sharex=True,
        figsize=figure_size("double", 7.4),
    )
    axes = np.atleast_1d(axes)
    frequency_min = min(DETECTOR_BOUNDS[name][0] for name in active_detectors())
    frequency_max = max(DETECTOR_BOUNDS[name][1] for name in active_detectors())
    for ax, (mass, distance) in zip(axes, DETECTABILITY_TARGETS):
        redshift = cosmology.redshift_at_luminosity_distance(distance)
        target_snr = []
        for name, spectrum in spectra.items():
            sim = registry[name]
            frequency, strain_ft = observer_spectrum(spectrum, mass, distance, redshift)
            hc = characteristic_strain(frequency, strain_ft)
            keep = _spectrum_plot_mask(frequency, hc)
            ax.plot(
                frequency[keep],
                hc[keep],
                color=sim.color,
                linestyle=sim.linestyle,
                label=sim.legend_name,
            )
            binned = _bin_spectral_power(spectrum, DETECTABILITY_HORIZON_SPECTRAL_BINS)
            detector_values = [
                f"{detector.upper()}={_snr_from_binned_power(binned, detector, mass * (1.0 + redshift), distance, curves):.3g}"
                for detector in active_detectors()
            ]
            target_snr.append(f"{name}: " + ", ".join(detector_values))
        for detector in active_detectors():
            lower, upper = DETECTOR_BOUNDS[detector]
            frequency = np.logspace(np.log10(lower), np.log10(upper), 500)
            asd = effective_detector_asd(detector, frequency, curves)
            ax.plot(
                frequency,
                _noise_characteristic_strain(frequency, asd),
                color="0.15",
                linestyle=DETECTOR_LINESTYLES[detector],
                linewidth=1.25,
                label="_nolegend_",
            )
        ax.text(
            0.98,
            0.95,
            rf"$M_{{\mathrm{{BH}}}}={mass:g}\,M_\odot$, "
            rf"$D_L={distance:g}\,\mathrm{{Mpc}}$, $z={redshift:.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )
        ax.set_ylabel(r"$h_c(f)$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(frequency_min, frequency_max)
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        format_paper_axes(ax)
        print(
            f"target M_BH={mass:g} Msun, D_L={distance:g} Mpc, z={redshift:.3g}: "
            + " | ".join(target_snr)
        )
    axes[-1].set_xlabel(r"$f_{\mathrm{obs}}\,[\mathrm{Hz}]$")
    ordered_sim_fig_legend(
        fig,
        axes[0],
        ncols=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        **COMPACT_LEGEND_KWARGS,
    )
    detector_handles = [
        Line2D(
            [0],
            [0],
            color="0.15",
            linestyle=DETECTOR_LINESTYLES[name],
            linewidth=1.25,
            label=DETECTOR_LABELS[name],
        )
        for name in active_detectors()
    ]
    fig.legend(
        detector_handles,
        [handle.get_label() for handle in detector_handles],
        ncols=len(detector_handles),
        loc="lower center",
        borderaxespad=FIGURE_LEGEND_BORDERAXESPAD,
    )
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.095, top=0.91, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_CHARACTERISTIC_STRAIN)


def _bin_spectral_power(spectrum: DimensionlessSpectrum, bins: int) -> BinnedSpectralPower:
    frequency = np.asarray(spectrum.frequency, dtype=float)
    power = np.square(np.abs(np.asarray(spectrum.strain_ft, dtype=complex)))
    keep = np.isfinite(frequency) & np.isfinite(power) & (frequency > 0.0) & (power >= 0.0)
    frequency = frequency[keep]
    power = power[keep]
    if frequency.size < 2:
        return BinnedSpectralPower(np.array([]), np.array([]))
    segment_frequency = np.sqrt(frequency[:-1] * frequency[1:])
    segment_power = 0.5 * (power[:-1] + power[1:]) * np.diff(frequency)
    keep = np.isfinite(segment_power) & (segment_power > 0.0)
    segment_frequency = segment_frequency[keep]
    segment_power = segment_power[keep]
    edges = np.logspace(
        np.log10(segment_frequency[0]),
        np.log10(segment_frequency[-1]) + 8.0 * np.finfo(float).eps,
        max(64, int(bins)) + 1,
    )
    indices = np.clip(np.searchsorted(edges, segment_frequency, side="right") - 1, 0, edges.size - 2)
    power_binned = np.bincount(indices, weights=segment_power, minlength=edges.size - 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    nonzero = power_binned > 0.0
    return BinnedSpectralPower(centers[nonzero], power_binned[nonzero])


def _snr_from_binned_power(binned, detector, redshifted_mass, distance, curves):
    if binned.frequency.size == 0 or redshifted_mass <= 0.0 or distance <= 0.0:
        return 0.0
    time_scale = redshifted_mass * SECONDS_PER_M_SUN
    frequency = binned.frequency / time_scale
    lower, upper = DETECTOR_BOUNDS[detector]
    keep = (frequency >= lower) & (frequency <= upper)
    if np.count_nonzero(keep) < 2:
        return 0.0
    asd = effective_detector_asd(detector, frequency[keep], curves)
    amplitude = redshifted_mass * METERS_PER_M_SUN / (distance * METERS_PER_MPC)
    snr_squared = (
        4.0
        * amplitude**2
        * time_scale
        * np.sum(binned.power_dfrequency[keep] / np.square(asd))
    )
    return float(np.sqrt(max(snr_squared, 0.0)))


def _horizon_curves(spectra, curves, cosmology):
    masses = np.logspace(
        np.log10(DETECTABILITY_HORIZON_MASS_RANGE_MSUN[0]),
        np.log10(DETECTABILITY_HORIZON_MASS_RANGE_MSUN[1]),
        int(DETECTABILITY_HORIZON_MASS_SAMPLES),
    )
    samples = max(24, int(DETECTABILITY_HORIZON_REDSHIFT_SAMPLES))
    redshift = np.unique(
        np.concatenate(
            (
                np.geomspace(1.0e-6, 0.1, samples),
                np.linspace(0.1, DETECTABILITY_HORIZON_REDSHIFT_MAX, samples),
            )
        )
    )
    distance = cosmology.luminosity_distance_mpc(redshift)
    result = {detector: {} for detector in active_detectors()}
    for sim_name, spectrum in spectra.items():
        binned = _bin_spectral_power(spectrum, DETECTABILITY_HORIZON_SPECTRAL_BINS)
        for detector in active_detectors():
            horizon = np.full_like(masses, np.nan)
            for mass_index, source_mass in enumerate(masses):
                snr = np.array(
                    [
                        _snr_from_binned_power(
                            binned,
                            detector,
                            source_mass * (1.0 + z),
                            d,
                            curves,
                        )
                        for z, d in zip(redshift, distance)
                    ]
                )
                detected = np.flatnonzero(snr >= DETECTABILITY_SNR_THRESHOLD)
                if detected.size == 0:
                    continue
                last = int(detected[-1])
                z_horizon = redshift[last]
                if last + 1 < redshift.size and snr[last + 1] < DETECTABILITY_SNR_THRESHOLD:
                    y0 = np.log(max(snr[last], np.finfo(float).tiny))
                    y1 = np.log(max(snr[last + 1], np.finfo(float).tiny))
                    target = np.log(DETECTABILITY_SNR_THRESHOLD)
                    if not np.isclose(y0, y1):
                        fraction = np.clip((target - y0) / (y1 - y0), 0.0, 1.0)
                        z_horizon += fraction * (redshift[last + 1] - redshift[last])
                horizon[mass_index] = cosmology.luminosity_distance_mpc(z_horizon)
            result[detector][sim_name] = horizon
    return masses, result


def _plot_horizon(sims, spectra, curves, cosmology, args):
    registry = {sim.config.name: sim for sim in sims}
    masses, horizon_curves = _horizon_curves(spectra, curves, cosmology)
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=figure_size("double", 6.2))
    for ax, detector in zip(axes.flat, active_detectors()):
        for sim_name, horizon in horizon_curves[detector].items():
            sim = registry[sim_name]
            ax.plot(
                masses,
                horizon,
                color=sim.color,
                linestyle=sim.linestyle,
                label=sim.legend_name,
            )
            finite = horizon[np.isfinite(horizon)]
            if finite.size:
                print(f"{sim_name}: {detector.upper()} maximum horizon = {np.max(finite):.4g} Mpc")
        ax.text(0.97, 0.95, DETECTOR_LABELS[detector], transform=ax.transAxes, ha="right", va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        format_paper_axes(ax)
    for ax in axes[-1, :]:
        ax.set_xlabel(r"Source-frame $M_{\mathrm{BH}}\,[M_\odot]$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Horizon $D_L\,[\mathrm{Mpc}]$")
    ordered_sim_legend(axes.flat[0], ncols=3, loc="lower left", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.10, top=0.98, wspace=0.08, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_HORIZON)


STALE_OUTPUTS = (
    "gw_skyavg_modesum.png",
    "gw_characteristic_strain_dimensionless.png",
    "gw_detectability_signal.png",
    "gw_detectability_snr_ligo.png",
    "gw_detectability_snr_ce.png",
    "gw_detectability_snr_decigo.png",
    "gw_detectability_snr_lisa.png",
    "gw_detectability_finite_targets.png",
    "gw_detectability_extended_targets.png",
    "gw_detectability_horizon_finite.png",
    "gw_detectability_horizon_extended.png",
)


def _remove_stale_outputs(outdir):
    for filename in STALE_OUTPUTS:
        path = Path(outdir) / filename
        if path.is_file():
            path.unlink()
            print(f"removed stale {path}")


def main(argv: list[str] | None = None):
    arg_parser = parser("Plot finite-duration GW detectability directly from rPsi4.")
    args = arg_parser.parse_args(argv)
    args.outdir = getattr(args, "outdir", PLOTS_DIR)
    setup(args)

    print(
        "GW detectability method: finite outer-radius rPsi4 on the shared cached t_ret, "
        "all ell<=3 modes, "
        f"discard first {DETECTABILITY_TRANSIENT_CUTOFF_MBH:g} M_BH, "
        f"Tukey alpha={DETECTABILITY_TAPER_ALPHA:g}, source-direction mean."
    )
    print("No radial extrapolation and no temporal tail extrapolation are applied.")
    curves = load_detector_curves()
    sims, spectra, representative, first_wavezone_spectra = _load_source_spectra(args)
    if not spectra:
        print("No valid direct-Psi4 spectra were produced.")
        return

    cosmology = FlatLambdaCDM(z_max=max(12.0, DETECTABILITY_HORIZON_REDSHIFT_MAX + 0.5))
    if DETECTABILITY_PLOT_METHOD_COMPARISON:
        _plot_method_comparison(sims, spectra, representative, args)
    if DETECTABILITY_PLOT_RADIUS_COMPARISON and first_wavezone_spectra:
        _plot_radius_comparison(sims, spectra, first_wavezone_spectra, args)
    if DETECTABILITY_PLOT_CHARACTERISTIC_STRAIN:
        _plot_characteristic_strain(sims, spectra, curves, cosmology, args)
    if DETECTABILITY_PLOT_HORIZON:
        _plot_horizon(sims, spectra, curves, cosmology, args)

    # Only remove superseded products after every requested replacement saved.
    _remove_stale_outputs(args.outdir)


if __name__ == "__main__":
    main()
