#!/usr/bin/env python3
"""Face-on strain at a physical distance, following Shibata et al. Fig. 4.

https://arxiv.org/pdf/2101.05440
Use the existing Fortran-compatible FFI cache, not the detectability FFT.
This retains our orbital-frequency FFI cutoff, not Shibata's fixed 8 Hz.
The north-pole observer has _{-2}Y_lm(0,0)=sqrt((2l+1)/(4*pi))*delta_m2.
Thus only (2,2) contributes for the default non-axisymmetric quadrupole subset.
This is directional strain, not a mode amplitude or a sky average.
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import numpy as np

from config import GW_OUTERMOST_PARFILE_INDEX, all_sim_configs
from gw import MODES as CACHED_MODES, read_strain_cache, strain_cache_dir
from gw_detectability import (
    FlatLambdaCDM, METERS_PER_M_SUN, METERS_PER_MPC, SECONDS_PER_M_SUN,
)
from helpers.plot_common import parser, savefig, setup
from helpers.style import (
    COLOR_OPTS, COMPACT_LEGEND_KWARGS, PAPER_THREE_PANEL_HEIGHT,
    figure_size, format_paper_axes, format_shared_gw_yaxes, set_symmetric_gw_ticks,
)


# Source and data selection. M_BH is the initial BH mass, never M_ADM.
CASE_GROUPS = (("A1", "A2", "A3"), ("B1", "B2", "B3"))
SOURCE_BH_MASS_MSUN = 50.0
DISTANCE_MPC = 100.0
INCLUDE_REDSHIFT = True  # Same cosmology as detectability; z ~ 0.022 at 100 Mpc.
RADIUS_INDEX = GW_OUTERMOST_PARFILE_INDEX
MODES = ((2, 2), (2, 1), (2, -2), (2, -1))  # Or "all" (cached ell=2..4).
MIN_TRET_MBH = 1000.0  # Same initial-relaxation exclusion as detectability.
ALIGN_PEAK = True  # Shift each retained hypot(h_plus, h_cross) maximum to zero.
TIME_LIMITS_S = None  # None shows the full retained duration; no extrapolation.

# Output and presentation. Dashed means cross here, as in Shibata Fig. 4;
# the cases have separate rows, so this does not replace the A/B line convention.
OUTPUT_TEMPLATE = "gw/strain_observer_{family}.png"
FIGURE_HEIGHT = PAPER_THREE_PANEL_HEIGHT
SHARE_Y_LIMITS = False  # Individual symmetric ranges, as in the paper.
LINE_WIDTH = 1.0
SUBPLOT_MARGINS = dict(left=0.20, right=0.975, bottom=0.09, top=0.92, hspace=0.18)


def observer_waveform(case, strain, *, source_mass=SOURCE_BH_MASS_MSUN,
                      distance=DISTANCE_MPC, redshift=0.0, modes=MODES,
                      min_tret_mbh=MIN_TRET_MBH, align_peak=ALIGN_PEAK):
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


def plot_group(cases, waveforms):
    """One row per case, both polarizations together, with shared time limits."""
    fig, axes = plt.subplots(len(cases), 1, sharex=True, squeeze=False,
                             figsize=figure_size("double", FIGURE_HEIGHT))
    axes = axes[:, 0]
    all_values, all_times = [], []
    for ax, case in zip(axes, cases):
        time, hp, hc, _ = waveforms[case.name]
        color = COLOR_OPTS[int(case.name[1:]) - 1]
        ax.plot(time, hp, color=color, linestyle="-", linewidth=LINE_WIDTH)
        ax.plot(time, hc, color=color, linestyle="--", linewidth=LINE_WIDTH)
        ax.text(0.98, 0.93, case.name, transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="none", pad=1.5))
        ax.set_ylabel(r"$h_{+,\times}$")
        ax.grid(visible=True, which="major", alpha=0.25)
        format_paper_axes(ax)
        set_symmetric_gw_ticks([ax], [hp, hc])
        all_values.extend((hp, hc))
        all_times.append(time)
    if SHARE_Y_LIMITS:
        set_symmetric_gw_ticks(axes, all_values)
    format_shared_gw_yaxes(axes)
    limits = TIME_LIMITS_S or (min(t[0] for t in all_times), max(t[-1] for t in all_times))
    axes[-1].set_xlim(*limits)
    xlabel = r"$(t_{\mathrm{ret}}-t_{\mathrm{peak}})\,[\mathrm{s}]$" if ALIGN_PEAK else r"$t_{\mathrm{ret}}\,[\mathrm{s}]$"
    axes[-1].set_xlabel(xlabel)
    number = ScalarFormatter(useMathText=True).format_data
    fig.text(SUBPLOT_MARGINS["left"], 0.963,
             rf"$M_{{\mathrm{{BH}}}}={number(SOURCE_BH_MASS_MSUN)}\,M_\odot$, "
             rf"$D_L={number(DISTANCE_MPC)}\,\mathrm{{Mpc}}$", va="center")
    fig.legend(handles=[Line2D([], [], color="k", ls=ls, label=label)
                        for ls, label in (("-", r"$h_+$"), ("--", r"$h_\times$"))],
               loc="center right", bbox_to_anchor=(0.975, 0.963), ncols=2,
               **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(**SUBPLOT_MARGINS)
    fig.align_ylabels(axes)
    return fig


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    setup(args)
    redshift = FlatLambdaCDM().redshift_at_luminosity_distance(DISTANCE_MPC) if INCLUDE_REDSHIFT else 0.0
    print(f"Face-on cached FFI strain: M_BH={SOURCE_BH_MASS_MSUN:g} Msun, "
          f"D_L={DISTANCE_MPC:g} Mpc, z={redshift:.6g}, radius index={RADIUS_INDEX}, "
          f"modes={MODES}, t_ret/M_BH >= {MIN_TRET_MBH:g}")
    for group in CASE_GROUPS:
        names = [name for name in group if args.sims is None or name in args.sims]
        if not names:
            continue
        cases = all_sim_configs(names)
        waveforms = {}
        for case in cases:
            label = str(RADIUS_INDEX + case.gw_psi4_file_index_offset)
            directory = strain_cache_dir(case, label)
            strain = read_strain_cache(directory, case.data_path / f"Psi4_rad.mon.{label}")
            if str(strain.metadata.get("source_label")) != label or not np.isclose(
                float(strain.metadata.get("madm", np.nan)), case.gw_madm, rtol=1e-10, atol=0,
            ):
                raise ValueError(f"{directory}: cache metadata does not match this simulation")
            waveforms[case.name] = observer_waveform(case, strain, redshift=redshift)
            time, hp, hc, peak = waveforms[case.name]
            cutoff_hz = (2 * float(strain.metadata["omega_orbital"]) * case.mlittle
                         / (2 * np.pi * SOURCE_BH_MASS_MSUN * (1 + redshift) * SECONDS_PER_M_SUN))
            print(f"{case.name}: peak |h|={np.hypot(hp, hc).max():.6e}, "
                  f"source peak t_ret/M_BH={peak:.3f}, observer span={time[0]:.4f}..{time[-1]:.4f} s, "
                  f"cached m=2 FFI cutoff={cutoff_hz:.3f} Hz")
        savefig(plot_group(cases, waveforms), args, OUTPUT_TEMPLATE.format(family=group[0][0]))


if __name__ == "__main__":
    main()
