"""Shared time-axis normalization for non-GW diagnostic plots."""
from __future__ import annotations

import matplotlib.ticker as mticker
import numpy as np

from plot_settings import TIME_CODE_UNIT_MASS_MSUN, TIME_NORMALIZE_BY_PC

MILLISECONDS_PER_SOLAR_MASS = 4.925490947e-3
SECONDARY_AXIS_YTICK_PAD = 9
ORBITAL_PERIOD_LATEX = r"P_c"


def time_values(values, sim, normalize_by_pc: bool = TIME_NORMALIZE_BY_PC):
    arr = np.asarray(values, dtype=float)
    pc = float(getattr(sim.config, "Pc", np.nan))
    if normalize_by_pc and np.isfinite(pc) and pc > 0.0:
        arr = arr / pc
    if np.isscalar(values):
        return float(arr)
    return arr


def time_xlabel(normalize_by_pc: bool = TIME_NORMALIZE_BY_PC) -> str:
    if normalize_by_pc:
        return rf"$t/{ORBITAL_PERIOD_LATEX}$"
    return r"$t\,[\mathrm{code}]$"


def code_time_to_ms(values):
    return np.asarray(values, dtype=float) * TIME_CODE_UNIT_MASS_MSUN * MILLISECONDS_PER_SOLAR_MASS


def ms_to_code_time(values):
    scale = TIME_CODE_UNIT_MASS_MSUN * MILLISECONDS_PER_SOLAR_MASS
    return np.asarray(values, dtype=float) / scale


def add_time_secondary_axis(ax, normalize_by_pc: bool = TIME_NORMALIZE_BY_PC):
    if normalize_by_pc:
        return None
    ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    ax.tick_params(axis="y", which="major", pad=SECONDARY_AXIS_YTICK_PAD)
    secax = ax.secondary_xaxis("top", functions=(code_time_to_ms, ms_to_code_time))
    secax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    secax.set_xlabel(r"$t\,[\mathrm{ms}]$", labelpad=4)
    secax.tick_params(axis="x", which="both", direction="in", pad=2)
    return secax
