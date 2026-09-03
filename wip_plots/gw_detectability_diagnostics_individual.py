#!/usr/bin/env python3
"""Numerical validation for the finite direct-Psi4 detectability method."""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers.gw_detectability import (
    characteristic_strain,
    direct_psi4_spectrum,
    prepared_rpsi4_modes,
)
from helpers.plot_common import parser, save_individual_fig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, figure_size, format_paper_axes
from wip_plots.gw_detectability_all import (
    DETECTABILITY_LOW_FREQUENCY_CYCLES,
    DETECTABILITY_MODES,
    DETECTABILITY_FIRST_WAVEZONE_PARFILE_INDEX,
    DETECTABILITY_OUTER_PARFILE_INDEX,
    DETECTABILITY_PHI_NODES,
    DETECTABILITY_SOURCE_AVERAGING,
    DETECTABILITY_TAPER_ALPHA,
    DETECTABILITY_THETA_NODES,
    DETECTABILITY_TRANSIENT_CUTOFF_MBH,
    DETECTABILITY_ZERO_PAD_FACTOR,
)


# Validation knobs.
VALIDATION_INTERMEDIATE_RADIAL_INDICES = (7,)
VALIDATION_RADIAL_INDICES = (
    DETECTABILITY_FIRST_WAVEZONE_PARFILE_INDEX,
    *VALIDATION_INTERMEDIATE_RADIAL_INDICES,
    DETECTABILITY_OUTER_PARFILE_INDEX,
)
VALIDATION_TAPER_ALPHAS = (0.02, DETECTABILITY_TAPER_ALPHA, 0.10)
VALIDATION_TRANSIENT_CUTOFFS_MBH = (500.0, DETECTABILITY_TRANSIENT_CUTOFF_MBH, 1500.0)
VALIDATION_COARSE_ANGULAR_GRID = (12, 24)


def _spectrum_for_sim(
    sim,
    *,
    taper_alpha,
    theta_nodes,
    phi_nodes,
    transient_cutoff_mbh=DETECTABILITY_TRANSIENT_CUTOFF_MBH,
):
    retarded_time, mode_data, time_method = prepared_rpsi4_modes(
        sim.strain_result,
        sim.psi4,
        DETECTABILITY_MODES,
    )
    if time_method != "cached-exact":
        print(f"{sim.config.name}: rPsi4 input from {time_method}")
    return direct_psi4_spectrum(
        retarded_time,
        mode_data,
        float(sim.config.mlittle),
        transient_cutoff_mbh=transient_cutoff_mbh,
        taper_alpha=taper_alpha,
        zero_pad_factor=DETECTABILITY_ZERO_PAD_FACTOR,
        low_frequency_cycles=DETECTABILITY_LOW_FREQUENCY_CYCLES,
        theta_nodes=theta_nodes,
        phi_nodes=phi_nodes,
        averaging=DETECTABILITY_SOURCE_AVERAGING,
    )[0]


def _dimensionless_axes(sim, spectrum):
    orbital_frequency = float(sim.config.gw_omega_orbital) * float(sim.config.mlittle) / (2.0 * np.pi)
    return (
        spectrum.frequency / orbital_frequency,
        characteristic_strain(spectrum.frequency, spectrum.strain_ft),
    )


def _interp_ratio(reference_x, reference_y, test_x, test_y):
    lower = max(np.min(reference_x), np.min(test_x))
    upper = min(np.max(reference_x), np.max(test_x))
    keep = (reference_x >= lower) & (reference_x <= upper) & (reference_y > 0.0)
    interpolated = np.interp(reference_x[keep], test_x, test_y)
    ratio = np.divide(
        interpolated,
        reference_y[keep],
        out=np.full(np.count_nonzero(keep), np.nan),
        where=reference_y[keep] > 0.0,
    )
    return reference_x[keep], ratio


def _report_ratio(label, x, ratio, reference_x, reference_y):
    reference = np.interp(x, reference_x, reference_y)
    significant = (
        np.isfinite(ratio)
        & np.isfinite(reference)
        & (reference >= 1.0e-3 * np.nanmax(reference_y))
    )
    if not np.any(significant):
        print(f"{label}: no significant-band overlap")
        return
    fractional = np.abs(ratio[significant] - 1.0)
    print(
        f"{label}: significant-band median fractional change={np.median(fractional):.3g}, "
        f"95th percentile={np.percentile(fractional, 95):.3g}"
    )


def plot_validation(sim_name):
    radial_sims = {}
    for index in VALIDATION_RADIAL_INDICES:
        loaded = load_sims(
            ["strain"],
            names=[sim_name],
            psi4_parfile_index=index,
            psi4_mode=DETECTABILITY_MODES[0],
        )
        if loaded:
            radial_sims[index] = loaded[0]
    if DETECTABILITY_OUTER_PARFILE_INDEX not in radial_sims:
        raise ValueError(f"{sim_name}: outer extraction radius is unavailable")

    outer_sim = radial_sims[DETECTABILITY_OUTER_PARFILE_INDEX]
    baseline = _spectrum_for_sim(
        outer_sim,
        taper_alpha=DETECTABILITY_TAPER_ALPHA,
        theta_nodes=DETECTABILITY_THETA_NODES,
        phi_nodes=DETECTABILITY_PHI_NODES,
    )
    baseline_x, baseline_y = _dimensionless_axes(outer_sim, baseline)

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("double", 7.0))

    for index, sim in radial_sims.items():
        spectrum = baseline if index == DETECTABILITY_OUTER_PARFILE_INDEX else _spectrum_for_sim(
            sim,
            taper_alpha=DETECTABILITY_TAPER_ALPHA,
            theta_nodes=DETECTABILITY_THETA_NODES,
            phi_nodes=DETECTABILITY_PHI_NODES,
        )
        x, y = _dimensionless_axes(sim, spectrum)
        radius = sim.psi4_radius
        label = rf"$r={radius:g}$" if radius is not None else rf"$i_{{\mathrm{{par}}}}={index}$"
        is_endpoint = index in {
            DETECTABILITY_FIRST_WAVEZONE_PARFILE_INDEX,
            DETECTABILITY_OUTER_PARFILE_INDEX,
        }
        axes[0].plot(
            x,
            y,
            label=label,
            linewidth=1.5 if is_endpoint else 1.0,
            alpha=1.0 if is_endpoint else 0.65,
        )
        if index != DETECTABILITY_OUTER_PARFILE_INDEX:
            ratio_x, ratio = _interp_ratio(baseline_x, baseline_y, x, y)
            radius_text = f"{radius:g}" if radius is not None else f"index {index}"
            _report_ratio(f"{sim_name} radius {radius_text}/outer", ratio_x, ratio, baseline_x, baseline_y)

    for alpha in VALIDATION_TAPER_ALPHAS:
        spectrum = baseline if np.isclose(alpha, DETECTABILITY_TAPER_ALPHA) else _spectrum_for_sim(
            outer_sim,
            taper_alpha=alpha,
            theta_nodes=DETECTABILITY_THETA_NODES,
            phi_nodes=DETECTABILITY_PHI_NODES,
        )
        x, y = _dimensionless_axes(outer_sim, spectrum)
        ratio_x, ratio = _interp_ratio(baseline_x, baseline_y, x, y)
        reference = np.interp(ratio_x, baseline_x, baseline_y)
        significant = np.isfinite(ratio) & (reference >= 1.0e-3 * np.nanmax(baseline_y))
        axes[1].plot(ratio_x[significant], ratio[significant], label=rf"$\alpha={alpha:g}$")
        if not np.isclose(alpha, DETECTABILITY_TAPER_ALPHA):
            _report_ratio(f"{sim_name} Tukey alpha={alpha:g}/baseline", ratio_x, ratio, baseline_x, baseline_y)

    for cutoff in VALIDATION_TRANSIENT_CUTOFFS_MBH:
        spectrum = baseline if np.isclose(cutoff, DETECTABILITY_TRANSIENT_CUTOFF_MBH) else _spectrum_for_sim(
            outer_sim,
            taper_alpha=DETECTABILITY_TAPER_ALPHA,
            theta_nodes=DETECTABILITY_THETA_NODES,
            phi_nodes=DETECTABILITY_PHI_NODES,
            transient_cutoff_mbh=cutoff,
        )
        x, y = _dimensionless_axes(outer_sim, spectrum)
        ratio_x, ratio = _interp_ratio(baseline_x, baseline_y, x, y)
        reference = np.interp(ratio_x, baseline_x, baseline_y)
        significant = np.isfinite(ratio) & (reference >= 1.0e-3 * np.nanmax(baseline_y))
        axes[2].plot(
            ratio_x[significant],
            ratio[significant],
            label=rf"$t_{{\mathrm{{cut}}}}={cutoff:g}M_{{\mathrm{{BH}}}}$",
        )
        if not np.isclose(cutoff, DETECTABILITY_TRANSIENT_CUTOFF_MBH):
            _report_ratio(
                f"{sim_name} transient cutoff={cutoff:g} M_BH/baseline",
                ratio_x,
                ratio,
                baseline_x,
                baseline_y,
            )

    coarse = _spectrum_for_sim(
        outer_sim,
        taper_alpha=DETECTABILITY_TAPER_ALPHA,
        theta_nodes=VALIDATION_COARSE_ANGULAR_GRID[0],
        phi_nodes=VALIDATION_COARSE_ANGULAR_GRID[1],
    )
    coarse_x, coarse_y = _dimensionless_axes(outer_sim, coarse)
    ratio_x, ratio = _interp_ratio(baseline_x, baseline_y, coarse_x, coarse_y)
    _report_ratio(f"{sim_name} angular 12x24/24x48", ratio_x, ratio, baseline_x, baseline_y)
    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$")
    axes[1].set_ylabel(r"$h_c(\alpha)/h_c(0.05)$")
    axes[2].set_ylabel(r"$h_c(t_{\mathrm{cut}})/h_c(1000M_{\mathrm{BH}})$")
    axes[2].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[0].legend(loc="lower right", **COMPACT_LEGEND_KWARGS)
    axes[1].legend(loc="lower right", **COMPACT_LEGEND_KWARGS)
    axes[2].legend(loc="lower right", **COMPACT_LEGEND_KWARGS)
    axes[0].set_yscale("log")
    for ax in axes[1:]:
        ax.set_yscale("log")
        ax.axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.set_xscale("log")
        ax.grid(True, which="both", linestyle=":", alpha=0.55)
        format_paper_axes(ax)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.09, top=0.98, hspace=0.08)
    return fig


def _remove_stale_diagnostics(outdir, sim_name):
    for path in (Path(outdir) / sim_name).glob(f"gw_detectability_diagnostic_{sim_name}_*.png"):
        path.unlink()
        print(f"removed stale {path}")


def main(argv=None):
    arg_parser = parser("Validate finite direct-Psi4 detectability for one simulation.")
    args = arg_parser.parse_args(argv)
    setup(args)
    names = args.sims or []
    for sim_name in names:
        try:
            fig = plot_validation(sim_name)
        except ValueError as exc:
            print(exc)
            continue
        save_individual_fig(
            fig,
            args,
            sim_name,
            f"gw_detectability_method_validation_{sim_name}.png",
        )
        _remove_stale_diagnostics(args.outdir, sim_name)


if __name__ == "__main__":
    main()
