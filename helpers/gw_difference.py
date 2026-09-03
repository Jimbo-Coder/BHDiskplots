"""Shared configuration and radius handling for disk-minus-massless GW plots."""
from __future__ import annotations

import numpy as np

from config import GW_COMPARISON_PARFILE_INDICES


# Critical knobs. These map to the shared physical extraction radii r_A=120,
# 170. The disk runs' apparent label 10 is a mismatched artifact, not r_A=180.
GW_DIFFERENCE_PARFILE_INDICES = GW_COMPARISON_PARFILE_INDICES
GW_DIFFERENCE_OUTPUT_SUBDIR = "gw/difference"
MASSLESS_SIM_NAME = "ML"


def names_with_massless(names):
    if names is None:
        return None
    out = []
    seen = set()
    for name in list(names) + [MASSLESS_SIM_NAME]:
        key = str(name).upper()
        if key not in seen:
            out.append(key)
            seen.add(key)
    return out


def loaded_areal_radius(sim):
    psi4 = getattr(sim, "psi4", None)
    if psi4 is None:
        return None
    radius = np.nanmedian(np.asarray(psi4.r_areal, dtype=float))
    return float(radius) if np.isfinite(radius) else None


def same_loaded_radius(sim, reference, rtol=5.0e-3, atol=1.0):
    radius = loaded_areal_radius(sim)
    reference_radius = loaded_areal_radius(reference)
    if radius is None or reference_radius is None:
        return True
    return bool(np.isclose(radius, reference_radius, rtol=rtol, atol=atol))


def loaded_radius_label(sim, parfile_index):
    radius = loaded_areal_radius(sim)
    if radius is None:
        return sim.gw_extraction_plot_label(parfile_index)
    return rf"$r_A\simeq {radius:.3g}$"


def loaded_radius_tag(sim, parfile_index):
    radius = getattr(sim, "psi4_radius", None)
    if radius is not None:
        radius = float(radius)
    if radius is None or not np.isfinite(radius):
        radius = loaded_areal_radius(sim)
    if radius is None:
        return f"i{int(parfile_index) + 1}"
    if np.isclose(radius, round(radius)):
        radius_text = str(int(round(radius)))
    else:
        radius_text = f"{radius:g}".replace(".", "p")
    return f"r{radius_text}"
