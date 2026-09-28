#!/usr/bin/env python3
"""Initial Newtonian Toomre diagnostic for the relativistic COCAL disks.

Sigma(R) = integral rho_0(R,z) dz; c_s^2 = Gamma P/rho_0 at z=0;
kappa^2 = R^-3 d(R^4 Omega^2)/dR; Q_N = c_s kappa/(pi Sigma), G=1.
These are coordinate/Newtonian quantities, NOT a relativistic stability test
or a criterion for the global PPI. The Q=1 line is a thin-disk reference.
Definition: https://academic.oup.com/mnras/article/421/1/818/990790

Two companion comparisons leave the original diagnostic unchanged:
* Romeo & Falstad (2013), Eq. (18): Q_R = T Q_N, for one gas component.
  T=1.5 assumes isotropic thermal support, not measured turbulent anisotropy.
  https://arxiv.org/abs/1302.4291
* Meidt (2022), Eq. (43): Q_M = kappa^2/(4 pi rho_mid), G=1, for local
  3D midplane modes. This is NOT another multiplicative thickness correction.
  https://arxiv.org/abs/2208.01888
Both use our existing Newtonian/coordinate inputs. Their local disk
approximations are not validated for these thick, relativistic polytropic
tori; a crossing of unity is exploratory, not a GR/PPI stability verdict.

COCAL IO_output_2D_general writes two spherical-grid meridians as (x,z,emdg),
radius varying fastest, phi=0 then pi. emdg=P/rho_0=K*rho_0^(Gamma-1).
IO_output_1D_xyz writes omeg_xp at theta=pi/2, phi=0, with no unit rescaling.
Use matching converged suffixes for the density and rotation profiles.
"""
from pathlib import Path
import re
import sys
import warnings

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapezoid
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import maximum_filter

from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    COMPACT_LEGEND_KWARGS, PAPER_SINGLE_PANEL_HEIGHT, PAPER_TWO_PANEL_HEIGHT,
    figure_size, format_paper_axes, ordered_sim_interpanel_legend,
    ordered_sim_legend,
)

# Scientific/data knobs. Simulation paths, K, Gamma, and M_ADM live in config.py.
SIM_NAMES = ("A1", "A2", "A3", "B1", "B2", "B3")  # Add "ML" to include massless.
VERTICAL_SAMPLES = 2049
MIDPLANE_DENSITY_FRACTION = 1e-3  # Exclude tenuous edges from Q and derivatives.
MATTER_Q_FRACTION = 1e-12  # Atmosphere cutoff in emdg, NOT in rho_0.
ROMEO_SIGMA_Z_OVER_SIGMA_R = 1.0  # Isotropic thermal support; valid range [0, 1].
XLIM = None  # R/M_ADM; None fits the selected disk profiles.
YLIM = None  # Original Toomre figure only; comparisons autoscale.

# Presentation knobs; shared paper fonts, colors, and legend geometry.
OUTPUT_FILENAME = "wip/toomre.png"
ROMEO_OUTPUT_FILENAME = "wip/toomre_romeo.png"
MEIDT_OUTPUT_FILENAME = "wip/toomre_meidt.png"
CASES_OUTPUT_FILENAME = "wip/toomre_cases.png"
FIG_HEIGHT = PAPER_SINGLE_PANEL_HEIGHT
COMPARISON_HEIGHT = PAPER_TWO_PANEL_HEIGHT


def extract_cocal_profiles(source, destination, suffix):
    """One-time export from raw BHT fluid/grid files; never modify the source.

    COCAL IO_input_matter_BHT_export: q, Omega; radius fastest, then theta,
    then phi. IO_output_solution_3D_BHT: radial, theta, phi coordinate vectors.
    Normal plotting reads only the resulting small text profiles.
    """
    source, destination = Path(source), Path(destination)
    with (source / "rnsgrids_3D.las").open() as stream:
        nr, nt, nphi = map(int, stream.readline().split())
        coords = np.loadtxt(stream)
    if len(coords) != nr + nt + nphi + 3:
        raise ValueError("COCAL grid length disagrees with header")
    radius, theta, phi = np.split(coords, [nr + 1, nr + nt + 2])
    with (source / "rnsflu_3D.las").open() as stream:
        if tuple(map(int, stream.readline().split())) != (nr, nt, nphi):
            raise ValueError("COCAL fluid and grid dimensions disagree")
        fields = np.loadtxt(stream, max_rows=(nr + 1) * (nt + 1) * (nphi + 1))
    fields = fields.reshape(nphi + 1, nt + 1, nr + 1, 2)
    equator = np.argmin(abs(theta - np.pi / 2))
    opposite = np.argmin(abs(phi - np.pi))
    if not np.allclose([theta[equator], phi[0], phi[opposite]], [np.pi / 2, 0, np.pi]):
        raise ValueError("COCAL grid has no exact equator/meridians")
    rotation = np.loadtxt(source / f"omeg_xp{suffix}.txt")
    if (not np.allclose(rotation[:, 0], radius, rtol=1e-9, atol=0) or
            not np.allclose(rotation[:, 1], fields[0, equator, :, 1],
                            rtol=1e-9, atol=1e-14)):
        raise ValueError("Raw COCAL grid/rotation does not match the exported profile")
    destination.mkdir(parents=True, exist_ok=True)
    np.savetxt(destination / f"emdg_xp{suffix}.txt",
               np.column_stack((radius, fields[0, equator, :, 0])), fmt="%.15e")
    meridians = []
    for ip in (0, opposite):
        x = np.sin(theta[:, None]) * radius * np.cos(phi[ip])
        z = np.cos(theta[:, None]) * radius
        meridians.append(np.column_stack((x.ravel(), z.ravel(),
                                         fields[ip, :, :, 0].ravel())))
    np.savetxt(destination / f"emdg_xz{suffix}.txt", np.vstack(meridians), fmt="%.15e")


def read_profiles(sim):
    """Read and verify native COCAL fields against the existing equatorial data."""
    root = sim.data_path_initial_data
    # Toomre needs density and rotation, not the unrelated ell_xp diagnostic.
    suffixes = []
    for prefix in ("emdg_xp", "emdg_xz", "omeg_xp"):
        suffixes.append({int(match[1]) for path in root.glob(f"{prefix}*.txt")
                         if (match := re.fullmatch(rf"{prefix}(\d+)\.txt", path.name))})
    common = set.intersection(*suffixes)
    if not common:
        raise FileNotFoundError(f"{sim.config.name}: missing matching density/rotation profiles")
    suffix = max(common)
    midplane = np.loadtxt(root / f"emdg_xp{suffix}.txt")
    eos = np.loadtxt(root / "peos_parameter_output.dat")
    if not (np.allclose(eos[:, 1], sim.config.kappa) and
            np.allclose(eos[:, 2], sim.config.gamma)):
        raise ValueError(f"{sim.config.name}: EOS table is not the configured single polytrope")
    rotation = np.loadtxt(root / f"omeg_xp{suffix}.txt")
    radius, omega = rotation.T
    if not np.all(np.diff(radius) > 0) or not np.allclose(radius, midplane[:, 0]):
        raise ValueError(f"{sim.config.name}: inconsistent initial radial grids")
    data = np.loadtxt(root / f"emdg_xz{suffix}.txt")
    # Infer dimensions from the matching radial grid, not hardcoded grid sizes.
    meridians = data.reshape(2, -1, radius.size, 3)
    xyz = meridians[0, :, :, :2]
    theta = np.arctan2(xyz[:, 0, 0], xyz[:, 0, 1])
    if (not np.all(np.diff(theta) > 0) or
            not np.allclose(theta[[0, -1]], [0, np.pi]) or
            not np.allclose(xyz[:, :, 0], np.sin(theta[:, None]) * radius) or
            not np.allclose(xyz[:, :, 1], np.cos(theta[:, None]) * radius)):
        raise ValueError(f"{sim.config.name}: unexpected COCAL meridional grid")
    q = meridians[:, :, :, 2].copy()
    floor = MATTER_Q_FRACTION * np.nanmax(q)
    invalid = ~np.isfinite(q)
    # Legacy files have isolated NaNs in atmosphere. Reject holes adjacent to
    # resolved matter instead of silently filling missing physical density.
    near_matter = maximum_filter(np.isfinite(q) & (q > floor), size=(1, 3, 3))
    if np.any(invalid & near_matter):
        raise ValueError(f"{sim.config.name}: nonfinite emdg next to resolved matter")
    if invalid.any():
        warnings.warn(f"{sim.config.name}: treating {invalid.sum()} atmosphere-neighbor "
                      "nonfinite emdg samples as vacuum", stacklevel=2)
    q[invalid | (q <= floor)] = 0
    if not np.allclose(q[0], q[1], rtol=1e-6, atol=1e-8 * q.max()):
        raise ValueError(f"{sim.config.name}: initial meridians are not axisymmetric")
    equator = np.argmin(abs(theta - np.pi / 2))
    if not np.allclose(q[0, equator], midplane[:, 1], rtol=1e-6, atol=1e-8 * q.max()):
        raise ValueError(f"{sim.config.name}: 2D matter does not match the initial 1D profile")
    peak = np.nanargmax(midplane[:, 1])
    if not np.isclose(omega[peak], sim.config.gw_omega_orbital, rtol=1e-5):
        raise ValueError(f"{sim.config.name}: Omega at density peak disagrees with config")
    density = (q[0] / sim.config.kappa) ** (1 / (sim.config.gamma - 1))
    return radius, theta, density, q[0, equator], omega


def column_density(radius, theta, density, cylindrical_radius, samples=VERTICAL_SAMPLES):
    """Integrate both disk halves using linear interpolation on the native grid."""
    if samples < 3 or samples % 2 != 1:
        raise ValueError("VERTICAL_SAMPLES must be odd and at least 3 (includes z=0)")
    occupied = density > 0
    if not occupied.any():
        raise ValueError("No matter in the initial meridional profile")
    # Include one extra native radial shell beyond all matter, for a vacuum
    # boundary outside the interpolation support. No integration to grid infinity.
    last = np.flatnonzero(occupied.any(axis=0))[-1]
    if last + 1 >= radius.size:
        raise ValueError("Matter reaches the radial boundary; column may be truncated")
    zmax = radius[last + 1]
    z = np.linspace(-zmax, zmax, samples)
    rr, zz = np.meshgrid(cylindrical_radius, z, indexing="ij")
    points = np.stack((np.arctan2(rr, zz), np.hypot(rr, zz)), axis=-1)
    interp = RegularGridInterpolator((theta, radius), density,
                                     method="linear", bounds_error=True)
    values = interp(points)
    return trapezoid(values, z, axis=1)


def epicyclic_squared(radius, omega):
    """Newtonian identity, differentiating angular momentum to reduce cancellation."""
    return np.gradient((radius**2 * omega)**2, radius, edge_order=2) / radius**3


def romeo_thickness_factor(anisotropy):
    """Single-component thickness factor, Romeo & Falstad (2013), Eq. (18)."""
    if not np.isfinite(anisotropy) or not 0 <= anisotropy <= 1:
        raise ValueError("ROMEO_SIGMA_Z_OVER_SIGMA_R must be finite and within [0, 1]")
    if anisotropy <= 0.5:
        return 1 + 0.6 * anisotropy**2
    return 0.8 + 0.7 * anisotropy


def meidt_parameter(kappa2, rho_mid):
    """Local 3D midplane Q_M (Meidt 2022, Eq. 43); G=1, gas density only."""
    kappa2, rho_mid = np.broadcast_arrays(kappa2, rho_mid)
    Q = np.full(kappa2.shape, np.nan)
    valid = np.isfinite(kappa2) & np.isfinite(rho_mid) & (kappa2 > 0) & (rho_mid > 0)
    Q[valid] = kappa2[valid] / (4 * np.pi * rho_mid[valid])
    return Q


def calculate(sim, *, vertical_samples=VERTICAL_SAMPLES):
    """Read once; all three diagnostics use the same selected radial samples."""
    radius, theta, density, qmid, omega = read_profiles(sim)
    rho_mid = (qmid / sim.config.kappa) ** (1 / (sim.config.gamma - 1))
    selected = rho_mid >= MIDPLANE_DENSITY_FRACTION * rho_mid.max()
    indices = np.flatnonzero(selected)
    if len(indices) < 5 or not np.all(np.diff(indices) == 1):
        raise ValueError(f"{sim.config.name}: expected one resolved radial disk interval")
    r, om = radius[selected], omega[selected]
    if not np.all(np.isfinite(om)) or np.any(om <= 0):
        raise ValueError(f"{sim.config.name}: invalid angular velocity inside disk")
    sigma = column_density(radius, theta, density, r, vertical_samples)
    cs = np.sqrt(sim.config.gamma * qmid[selected])
    kappa2 = epicyclic_squared(r, om)
    Q = np.full_like(r, np.nan)
    positive = (kappa2 > 0) & (sigma > 0)
    Q[positive] = cs[positive] * np.sqrt(kappa2[positive]) / (np.pi * sigma[positive])
    if not positive.any():
        raise ValueError(f"{sim.config.name}: no positive Newtonian kappa^2 in selected disk")
    print(f"{sim.config.name}: Q_N,min={np.nanmin(Q):.4g}, "
          f"R/M_ADM={r[np.nanargmin(Q)] / sim.config.gw_madm:.4g}; "
          f"{np.count_nonzero(~positive)}/{len(r)} samples have undefined Q_N", flush=True)
    thickness = romeo_thickness_factor(ROMEO_SIGMA_Z_OVER_SIGMA_R)
    Q_M = meidt_parameter(kappa2, rho_mid[selected])
    print(f"{sim.config.name}: T={thickness:.3g}, Q_R,min={np.nanmin(thickness * Q):.4g}, "
          f"Q_M,min={np.nanmin(Q_M):.4g}", flush=True)
    return dict(r=r, sigma=sigma, cs=cs, omega=om, kappa2=kappa2,
                rho_mid=rho_mid[selected], Q_N=Q, Q_R=thickness * Q, Q_M=Q_M)


def plot_comparison(sims, profiles, field, ylabel, filename, args):
    """Original Q above one alternative, using the same case styles and radii."""
    fig, axes = plt.subplots(2, 1, sharex=True, sharey=(field == "Q_R"),
                             figsize=figure_size("double", COMPARISON_HEIGHT))
    try:
        for ax, key, label in zip(axes, ("Q_N", field), (r"$Q_{\mathrm{N}}$", ylabel)):
            for sim, profile in zip(sims, profiles):
                ax.plot(profile["r"] / sim.config.gw_madm, profile[key],
                        color=sim.color, linestyle=sim.linestyle, label=sim.legend_name)
            ax.axhline(1, color="0.4", linestyle=":", linewidth=1)
            ax.set(ylabel=label, yscale="log")
            if XLIM is not None:
                ax.set_xlim(XLIM)
            format_paper_axes(ax)
            ax.grid(True, which="major")
        axes[-1].set_xlabel(r"$r/M$")
        if field == "Q_R":
            thickness = romeo_thickness_factor(ROMEO_SIGMA_Z_OVER_SIGMA_R)
            axes[-1].text(0.97, 0.84, rf"$T={thickness:.3g}$",
                          transform=axes[-1].transAxes, ha="right", va="top")
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.11, top=0.98, hspace=0.12)
        fig.align_ylabels(axes)
        ordered_sim_interpanel_legend(fig, axes[0], axes, ncols=3,
                                      loc="center right", x=0.98, **COMPACT_LEGEND_KWARGS)
        savefig(fig, args, filename)
    finally:
        plt.close(fig)


def plot_case_comparison(sims, profiles):
    """One panel per case, with method colors instead of six separate figures."""
    if not sims or len(sims) != len(profiles):
        raise ValueError("expected one diagnostic profile per simulation")
    ncols = min(3, len(sims))
    nrows = (len(sims) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, sharex=True, sharey=True, squeeze=False,
        figsize=figure_size("double", max(FIG_HEIGHT, nrows * COMPARISON_HEIGHT / 2)),
    )
    for ax, sim, profile in zip(axes.flat, sims, profiles):
        for field, label, color, linestyle in (
            ("Q_N", "Toomre", "tab:blue", "-"),
            ("Q_R", "Romeo", "tab:orange", "--"),
            ("Q_M", "Meidt", "tab:green", "-."),
        ):
            ax.plot(profile["r"] / sim.config.gw_madm, profile[field],
                    color=color, linestyle=linestyle, label=label)
        ax.axhline(1, color="0.4", linestyle=":", linewidth=1)
        ax.set_yscale("log")
        if XLIM is not None:
            ax.set_xlim(XLIM)
        ax.text(0.94, 0.92, sim.legend_name, transform=ax.transAxes,
                ha="right", va="top")
        format_paper_axes(ax)
        ax.grid(True, which="major")
    for ax in axes.flat[len(sims):]:
        ax.set_visible(False)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 0.995),
               ncols=3, **COMPACT_LEGEND_KWARGS)
    fig.supxlabel(r"$r/M$", y=0.01)
    fig.supylabel(r"$Q$", x=0.01)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.91,
                        hspace=0.12, wspace=0.10)
    return fig


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    setup(args)
    sims = load_sims(names=args.sims or SIM_NAMES, skip_missing=False)
    profiles = [calculate(sim) for sim in sims]
    fig, ax = plt.subplots(figsize=figure_size("double", FIG_HEIGHT))
    try:
        for sim, profile in zip(sims, profiles):
            ax.plot(profile["r"] / sim.config.gw_madm, profile["Q_N"], color=sim.color,
                    linestyle=sim.linestyle, label=sim.legend_name)
        ax.axhline(1, color="0.4", linestyle=":", linewidth=1)
        # On the positive equatorial ray, cylindrical R equals coordinate r.
        ax.set(xlabel=r"$r/M$", ylabel=r"$Q_{\mathrm{N}}$", yscale="log")
        if XLIM is not None:
            ax.set_xlim(XLIM)
        if YLIM is not None:
            ax.set_ylim(YLIM)
        format_paper_axes(ax)
        ax.grid(True, which="major")
        ordered_sim_legend(ax, ncols=3, loc="upper right", **COMPACT_LEGEND_KWARGS)
        fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.98)
        savefig(fig, args, OUTPUT_FILENAME)
    finally:
        plt.close(fig)
    plot_comparison(sims, profiles, "Q_R", r"$Q_{\mathrm{R}}=T Q_{\mathrm{N}}$",
                    ROMEO_OUTPUT_FILENAME, args)
    plot_comparison(sims, profiles, "Q_M", r"$Q_{\mathrm{M}}$", MEIDT_OUTPUT_FILENAME, args)
    savefig(plot_case_comparison(sims, profiles), args, CASES_OUTPUT_FILENAME)


if __name__ == "__main__":
    main()
