"""Radiated GW energy, angular momentum and (2,2) frequency.

Energy and J_z reuse the parity-tested legacy flux accumulation in gw.py on the
cached FFI strain, restricted to the detectability modes. Accumulation starts
at the same transient cut as detectability, so initial-data junk is excluded.
The (2,2) frequency comes from the Psi4 phase, independent of the FFI cutoff.
These are finite-radius values with no extrapolation to infinity.
"""
from pathlib import Path
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from config import GW_OUTERMOST_PARFILE_INDEX, MASSLESS_SIM_NAME
from gw import MODES as CACHED_MODES_ALL, _radiated_diagnostics, load_gw_sims, mode_columns
from gw_detectability import MODES, TRANSIENT_CUTOFF_MBH
from helpers.gw_units import add_gw_time_secondary_axis, gw_time_values, gw_time_xlabel
from helpers.plot_common import parser, savefig, setup
from helpers.style import COMPACT_LEGEND_KWARGS, figure_size, ordered_sim_legend

RADIUS_INDEX = GW_OUTERMOST_PARFILE_INDEX
FREQUENCY_SMOOTHING_ORBITS = 1.0  # Moving average of the Psi4 phase rate.
OUTPUT_FILENAME = "wip/gw_radiated.png"
SUMMARY_FILENAME = "wip/gw_radiated.json"


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
    width = max(1, int(round(FREQUENCY_SMOOTHING_ORBITS * sim.config.Pc / np.median(np.diff(time)))))
    if width >= rate.size:
        raise ValueError(f"{sim.config.name}: series shorter than the smoothing window")
    smooth = np.convolve(rate, np.ones(width) / width, mode="valid")
    centre = time[width//2: width//2 + smooth.size]
    return centre, smooth / sim.config.gw_omega_orbital


def plot(sims):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("double", 7.0))
    summary = {}
    for sim in sims:
        name, disk, bh = sim.config.name, sim.config.disk_rest_mass_code, sim.config.mlittle
        time, energy, angular = radiated(sim, MODES)
        _, energy_all, angular_all = radiated(sim, CACHED_MODES_ALL)
        f_time, f_ratio = frequency_22(sim)
        style = dict(color=sim.color, linestyle=sim.linestyle, label=sim.legend_name)
        axes[0].plot(gw_time_values(time, sim), energy / disk, **style)
        axes[1].plot(gw_time_values(time, sim), angular / (disk * bh), **style)
        axes[2].plot(gw_time_values(f_time, sim), f_ratio, **style)
        summary[name] = dict(
            modes=[list(mode) for mode in MODES],
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
    axes[2].set(ylabel=r"$f_{22}/f_{\rm orb}$", xlabel=gw_time_xlabel())
    axes[2].axhline(2.0, color="0.5", linewidth=0.8, linestyle=":")
    for ax in axes:
        ax.grid(which="major", alpha=.25)
    ordered_sim_legend(axes[0], ncols=3, loc="upper left", **COMPACT_LEGEND_KWARGS)
    add_gw_time_secondary_axis(axes[0])
    fig.subplots_adjust(left=.15, right=.97, bottom=.08, top=.94, hspace=.16)
    return fig, summary


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    setup(args)
    names = args.sims or ("A1", "A2", "A3", "B1", "B2", "B3")
    sims = [sim for sim in load_gw_sims(names, psi4_parfile_index=RADIUS_INDEX)
            if sim.config.name != MASSLESS_SIM_NAME]
    if not sims:
        raise SystemExit("No GW caches available; run generate_gw.py")
    fig, summary = plot(sims)
    savefig(fig, args, OUTPUT_FILENAME)
    if not args.no_save:
        path = args.outdir / SUMMARY_FILENAME
        path.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"saved {path}")


if __name__ == "__main__":
    main()
