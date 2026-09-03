"""Mass normalization helpers for GW diagnostic plots."""
from __future__ import annotations

import matplotlib.ticker as mticker
import numpy as np

from plot_settings import GW_NORMALIZE_BY_M, GW_TIME_SCALE
from helpers.time_units import ORBITAL_PERIOD_LATEX, code_time_to_ms, ms_to_code_time

SECONDARY_AXIS_YTICK_PAD = 9
GW_TIME_SCALES = ("M_BH", "M_ADM", "P_c", "code")


def gw_bh_mass(sim) -> float | None:
    """Configured initial central-BH mass scale in raw simulation code units."""
    mass = getattr(sim.config, "mlittle", None)
    if mass is None:
        return None
    mass = float(mass)
    if not np.isfinite(mass) or mass <= 0.0:
        return None
    return mass


def gw_adm_mass(sim) -> float | None:
    """Total ADM mass used by the Psi4-to-strain reconstruction."""
    mass = getattr(sim.config, "gw_madm", None)
    if mass is None:
        return None
    mass = float(mass)
    if not np.isfinite(mass) or mass <= 0.0:
        return None
    return mass


def gw_mass(sim) -> float | None:
    """Compatibility name for the mass used to normalize plotted GWs."""
    return gw_bh_mass(sim)


def disk_rest_mass(sim) -> float:
    mass = float(getattr(sim.config, "disk_rest_mass", np.nan))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{sim.config.name}: invalid configured disk rest mass {mass!r}")
    return mass


def normalize_rpsi4(values, sim, normalize_by_m: bool = GW_NORMALIZE_BY_M):
    values = np.asarray(values)
    mass = gw_mass(sim)
    if normalize_by_m and mass is not None:
        return mass * values
    return values


def normalize_rpsi4_by_disk_mass(values, sim, normalize_by_m: bool = GW_NORMALIZE_BY_M):
    values = np.asarray(values)
    if normalize_by_m:
        return disk_rest_mass(sim) * values
    return values


def normalize_strain(values, sim, normalize_by_m: bool = GW_NORMALIZE_BY_M):
    values = np.asarray(values)
    mass = gw_mass(sim)
    if normalize_by_m and mass is not None:
        return values / mass
    return values


def normalize_strain_by_disk_mass(values, sim, normalize_by_m: bool = GW_NORMALIZE_BY_M):
    values = np.asarray(values)
    if normalize_by_m:
        return values / disk_rest_mass(sim)
    return values


def gw_time_divisor(sim, time_scale: str = GW_TIME_SCALE) -> float:
    if time_scale == "code":
        return 1.0
    if time_scale == "M_BH":
        divisor = gw_bh_mass(sim)
    elif time_scale == "M_ADM":
        divisor = gw_adm_mass(sim)
    elif time_scale == "P_c":
        divisor = float(getattr(sim.config, "Pc", np.nan))
    else:
        raise ValueError(
            f"Unknown GW time scale {time_scale!r}; choose from {GW_TIME_SCALES}"
        )
    if divisor is None or not np.isfinite(divisor) or divisor <= 0.0:
        raise ValueError(
            f"{sim.config.name}: invalid divisor {divisor!r} for GW time scale {time_scale}"
        )
    return float(divisor)


def gw_time_values(values, sim, time_scale: str = GW_TIME_SCALE):
    arr = np.asarray(values, dtype=float)
    arr = arr / gw_time_divisor(sim, time_scale)
    if np.isscalar(values):
        return float(arr)
    return arr


def gw_time_xlabel(time_scale: str = GW_TIME_SCALE) -> str:
    if time_scale == "M_BH":
        return r"$t_{\mathrm{ret}}/M_{\mathrm{BH}}$"
    if time_scale == "M_ADM":
        return r"$t_{\mathrm{ret}}/M$"
    if time_scale == "P_c":
        return rf"$t_{{\mathrm{{ret}}}}/{ORBITAL_PERIOD_LATEX}$"
    if time_scale == "code":
        return r"$t_{\mathrm{ret}}\;(\mathrm{code})$"
    raise ValueError(f"Unknown GW time scale {time_scale!r}; choose from {GW_TIME_SCALES}")


def add_gw_time_secondary_axis(ax, time_scale: str = GW_TIME_SCALE):
    if time_scale != "code":
        return None
    ax.tick_params(axis="x", which="both", top=False, labeltop=False)
    ax.tick_params(axis="y", which="major", pad=SECONDARY_AXIS_YTICK_PAD)
    secax = ax.secondary_xaxis("top", functions=(code_time_to_ms, ms_to_code_time))
    secax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    secax.set_xlabel(r"$t_{\mathrm{ret}}\;(\mathrm{ms})$", labelpad=4)
    secax.tick_params(axis="x", which="both", direction="in", pad=2)
    return secax


def rpsi4_ylabel(kind: str, mode: str, normalize_by_m: bool = GW_NORMALIZE_BY_M) -> str:
    quantity = rf"M_{{\mathrm{{BH}}}} r_A\Psi_4^{{{mode}}}" if normalize_by_m else rf"r_A\Psi_4^{{{mode}}}"
    if kind == "real":
        return rf"$\mathrm{{Re}}({quantity})$"
    if kind == "abs":
        return rf"$|{quantity}|$"
    raise ValueError(f"Unknown rPsi4 label kind {kind!r}")


def difference_rpsi4_ylabel(kind: str, mode: str, normalize_by_m: bool = GW_NORMALIZE_BY_M) -> str:
    quantity = (
        rf"M_{{0,\mathrm{{disk}}}}(t=0)\,\Delta(r_A\Psi_4^{{{mode}}})"
        if normalize_by_m
        else rf"\Delta(r_A\Psi_4^{{{mode}}})"
    )
    if kind == "real":
        return rf"$\mathrm{{Re}}\left({quantity}\right)$"
    if kind == "abs":
        return rf"$\left|{quantity}\right|$"
    raise ValueError(f"Unknown difference rPsi4 label kind {kind!r}")


def rpsi4_multimode_ylabel(kind: str, normalize_by_m: bool = GW_NORMALIZE_BY_M) -> str:
    quantity = r"M_{\mathrm{BH}} r_A\Psi_4^{\ell m}" if normalize_by_m else r"r_A\Psi_4^{\ell m}"
    if kind == "real":
        return rf"$\mathrm{{Re}}({quantity})$"
    if kind == "abs":
        return rf"$|{quantity}|$"
    raise ValueError(f"Unknown rPsi4 label kind {kind!r}")


def strain_ylabel(component: str, mode: str, normalize_by_m: bool = GW_NORMALIZE_BY_M) -> str:
    component_tex = "+" if component == "plus" else r"\times"
    suffix = r"/M_{\mathrm{BH}}" if normalize_by_m else ""
    return rf"$r_A h_{component_tex}^{{{mode}}}{suffix}$"


def difference_strain_ylabel(
    component: str,
    mode: str,
    normalize_by_m: bool = GW_NORMALIZE_BY_M,
    source: str | None = None,
) -> str:
    component_tex = "+" if component == "plus" else r"\times"
    suffix = r"/M_{0,\mathrm{disk}}(t=0)" if normalize_by_m else ""
    source_tex = rf"\,[{source}]" if source else ""
    return rf"$r_A\,\Delta h_{component_tex}^{{{mode}}}{suffix}{source_tex}$"


def strain_multimode_ylabel(component: str, normalize_by_m: bool = GW_NORMALIZE_BY_M) -> str:
    component_tex = "+" if component == "plus" else r"\times"
    suffix = r"/M_{\mathrm{BH}}" if normalize_by_m else ""
    return rf"$r_A h_{component_tex}^{{\ell m}}{suffix}$"


def skyavg_strain_ylabel(normalize_by_m: bool = GW_NORMALIZE_BY_M) -> str:
    suffix = r"/M_{\mathrm{BH}}" if normalize_by_m else ""
    return rf"$\langle |r_A h|^2\rangle_\Omega^{{1/2}}{suffix}$"
