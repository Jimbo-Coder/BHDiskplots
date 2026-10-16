#!/usr/bin/env python3
"""GW detectability: measured spectra, distance reach, and their diagnostic views.

Run: python gw_detectability.py --radius-index 6
Checks: python gw_detectability.py --plots radius averaging --sims A1 A3

Read sections 1-4 in order: settings and readers -> measured spectral and
detector calculation -> analyze() -> plot_combined() and its two panels.
Edit physical choices in section 1 and appearance in section 4. Optional
validation and continuation diagnostics follow in section 5; the command-line
options are at the bottom in section 6.

The default uses measured rPsi4, with no FFI or synthetic continuation.
Units: tau=t/M_BH, q=r*h/M_BH, p=M_BH*r*Psi4=d^2q/dtau^2.
Conventions and derivations: notes/gw_paper_methods.md.
"""
from __future__ import annotations

import config
import json
from pathlib import Path

from dataclasses import dataclass, replace
from functools import cached_property
from math import factorial
from typing import Mapping

import numpy as np
from gw import MODES as CACHED_MODES, mode_columns, strain_cache_dir, read_strain_cache


# 1. SETTINGS AND INPUTS
# Index 6 is Psi4_rad.mon.7 for the A cases. No extrapolation is applied.
SIM_NAMES = ("A1", "A2", "A3")
OUTER_RADIUS_INDEX = config.GW_OUTERMOST_PARFILE_INDEX
FIRST_RADIUS_INDEX = config.GW_FIRST_WAVEZONE_PARFILE_INDEX  # Optional radius checks.
MODES = ((2, 2), (2, 1), (2, -2), (2, -1))  # (2,0) is deliberately excluded.
TRANSIENT_CUTOFF_MBH = 1000.0
TAPER_ALPHA = 0.05
LOW_FREQUENCY_CYCLES = 3.0
SOURCE_AVERAGING = "mean"  # Mean amplitude; not an orientation-averaged SNR.
ACTIVE_DETECTORS = ("ligo", "et", "ce", "decigo", "lisa")
SNR_THRESHOLD = 8.0
# (detector, initial BH mass in solar masses, distance in Mpc).
# None means solve the largest SNR=8 distance among the plotted cases.
EXAMPLE_TARGETS = (
    ("ce", 10., None),
    ("ce", 50., None),
    ("decigo", 1.e3, None),
    ("lisa", 1.e5, None),
)

# Numerical resolution. These control accuracy/cost, not signal duration.
ZERO_PAD_FACTOR = 2.0
THETA_NODES, PHI_NODES = 24, 48
HORIZON_MASS_RANGE_MSUN = (1., 1.e7)
HORIZON_REDSHIFT_MAX = 10.
HORIZON_MASS_SAMPLES, HORIZON_REDSHIFT_SAMPLES = 44, 64
HORIZON_SPECTRAL_BINS = 2048

# Physical constants and the flat matter+Lambda cosmology used for D_L(z).
SECONDS_PER_M_SUN = 4.92549095e-6
METERS_PER_M_SUN = 1476.6250385
METERS_PER_MPC = 3.085677581491367e22
LIGHT_SPEED_KM_S = 299792.458
COSMOLOGY_H0_KM_S_MPC = 67.66
COSMOLOGY_OMEGA_M = 0.30966

# Fixed source used only by the optional reference-comparison figure.
SHIBATA_BH_MASS_MSUN, SHIBATA_DISTANCE_MPC = 50., 100.

# Input tables: filename, ASD or PSD, and zero-based column after frequency.
DETECTOR_CURVE_DIR = config.REPOSITORY_ROOT / "detector_curves"
DETECTOR_TABLES = {
    "ligo": ("AplusDesign.txt", "asd", 1),
    "et": ("ET10kmcolumns.txt", "psd", 3),
    "ce": ("CE2_40km_strain.txt", "asd", 1),
    "lisa": ("LISA_Alloc_Sh.txt", "psd", 1),
}
DETECTOR_BOUNDS = {  # Observed Hz.
    "ligo": (5., 2500.),
    "et": (1., 1.e4),
    "ce": (5., 5000.),
    "decigo": (.01, 10.),
    "lisa": (1.e-4, 1.),
}


@dataclass(frozen=True)
class DimensionlessSpectrum:
    """Fourier transform of q=r*h/M_BH with respect to tau=t/M_BH."""

    frequency: np.ndarray
    strain_ft: np.ndarray
    averaging: str


@dataclass(frozen=True)
class Psi4SpectrumInfo:
    """Numerical choices and resolved sampling for a direct-Psi4 spectrum."""

    modes: tuple[tuple[int, int], ...]
    samples: int
    duration_mbh: float
    dt_mbh: float
    frequency_min: float
    frequency_max: float
    theta_nodes: int
    phi_nodes: int
    taper_alpha: float
    zero_pad_factor: float
    low_frequency_cycles: float


@dataclass
class SourceSpectrum:
    config: config.DiskSimConfig
    spectrum: DimensionlessSpectrum
    info: Psi4SpectrumInfo


@dataclass
class ObservedTarget:
    mass: float
    distance: float
    redshift: float
    strain: dict  # case -> (observed frequency in Hz, characteristic strain)
    snr: dict  # case -> detector -> SNR of the chosen source-amplitude statistic
    detector: str | None  # Example's detector band; None for the Shibata reference.
    distance_policy: str = "fixed"
    horizon_case: str | None = None


@dataclass
class DetectabilityResult:
    sources: dict
    inner: dict
    representative: dict
    noise: dict
    targets: list[ObservedTarget]
    horizons: tuple | None
    shibata: ObservedTarget | None


# Cache input: matching, uniformly sampled r*Psi4 and retarded time.
def read_rpsi4_modes(case, radius_index, modes=None, *, cache_root=None):
    """Read only the selected cache columns, not raw radii or integrated strain.

    The shared generator owns restart merging, file-index offsets, retarded
    time and uniform interpolation. No FFI result enters this calculation.
    Never map raw samples onto an unrelated cached time array or extend times.
    """
    modes = MODES if modes is None else modes
    if isinstance(modes, str):
        if modes != "all":
            raise ValueError('MODES must be a sequence of (ell, m) pairs or "all"')
        modes = CACHED_MODES
    modes = tuple(tuple(mode) for mode in modes)
    label = str(radius_index + case.gw_psi4_file_index_offset)
    directory = (strain_cache_dir(case, label) if cache_root is None else
                 Path(cache_root) / f"{case.name}_psi4{label}")
    path = directory / "rpsi4_uniform.dat"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}; run generate_gw.py with GW_STRAIN_BACKEND='python'")
    metadata = json.loads((directory / "strain_cache.json").read_text())
    if metadata.get("backend") != "python" or str(metadata.get("source_label")) != label:
        raise ValueError(f"{directory}: unexpected backend or extraction label; regenerate the GW cache")
    if not np.isclose(float(metadata["madm"]), case.gw_madm, rtol=1e-10, atol=0):
        raise ValueError(f"{directory}: cached ADM mass differs from config; regenerate the GW cache")
    columns = [0] + [column for mode in modes for column in mode_columns(*mode)]
    data = np.loadtxt(path, usecols=columns, ndmin=2)
    # The generator's uniform retarded-time grid can overrun the last source
    # sample by a few steps; those rows are exact zero fill, not measured data.
    measured = np.flatnonzero(np.any(data[:, 1:] != 0.0, axis=1))
    if not measured.size:
        raise ValueError(f"{path}: selected Psi4 modes are identically zero")
    data = data[measured[0]:measured[-1] + 1]
    values = {mode: data[:, 1 + 2*i] + 1j * data[:, 2 + 2*i] for i, mode in enumerate(modes)}
    return data[:, 0], values


@dataclass(frozen=True)
class DetectorCurve:
    frequency: np.ndarray
    asd: np.ndarray

def active_detectors():
    if not ACTIVE_DETECTORS or set(ACTIVE_DETECTORS) - DETECTOR_BOUNDS.keys():
        raise ValueError(f"Choose at least one detector from {tuple(DETECTOR_BOUNDS)}")
    return ACTIVE_DETECTORS

def _read_detector_curve(filename: str, quantity: str, column=1) -> DetectorCurve:
    path = DETECTOR_CURVE_DIR / filename
    raw = np.loadtxt(path, comments="#")
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    frequency = np.asarray(raw[:, 0], dtype=float)
    values = np.asarray(raw[:, column], dtype=float)
    keep = np.isfinite(frequency) & np.isfinite(values) & (frequency > 0.0) & (values > 0.0)
    frequency = frequency[keep]
    values = values[keep]
    if frequency.size < 2:
        raise ValueError(f"Detector curve {path} has fewer than two positive samples")
    order = np.argsort(frequency)
    asd = np.sqrt(values) if quantity == "psd" else values
    return DetectorCurve(frequency[order], asd[order])

def load_detector_curves():
    return {name: _read_detector_curve(filename, quantity, column)
            for name, (filename, quantity, column) in DETECTOR_TABLES.items()}


# ----------------------------------------------------------------------
# 2. CALCULATION: time cuts -> window -> FFT -> polarization mean -> strain
# ----------------------------------------------------------------------
def tukey_window(size: int, alpha: float = TAPER_ALPHA) -> np.ndarray:
    """Return a symmetric Tukey window without requiring SciPy."""
    size = int(size)
    if size <= 1:
        return np.ones(max(size, 0), dtype=float)
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0:
        return np.ones(size, dtype=float)
    if alpha == 1.0:
        return np.hanning(size)

    x = np.linspace(0.0, 1.0, size)
    window = np.ones(size, dtype=float)
    left = x < alpha / 2.0
    right = x >= 1.0 - alpha / 2.0
    window[left] = 0.5 * (1.0 + np.cos(np.pi * (2.0 * x[left] / alpha - 1.0)))
    window[right] = 0.5 * (
        1.0 + np.cos(np.pi * (2.0 * x[right] / alpha - 2.0 / alpha + 1.0))
    )
    return window


def _clean_common_mode_series(time, modes):
    time = np.asarray(time, dtype=float)
    arrays = {tuple(mode): np.asarray(values, dtype=complex) for mode, values in modes.items()}
    if not arrays:
        raise ValueError("At least one Psi4 mode is required")
    if any(values.shape != time.shape for values in arrays.values()):
        raise ValueError("Every Psi4 mode must have the same shape as time")

    finite = np.isfinite(time)
    for values in arrays.values():
        finite &= np.isfinite(values.real) & np.isfinite(values.imag)
    time = time[finite]
    arrays = {mode: values[finite] for mode, values in arrays.items()}
    if time.size < 8:
        raise ValueError("Too few finite Psi4 samples")

    order = np.argsort(time, kind="mergesort")
    time = time[order]
    arrays = {mode: values[order] for mode, values in arrays.items()}
    _, reverse_indices = np.unique(time[::-1], return_index=True)
    keep = np.sort(time.size - 1 - reverse_indices)
    time = time[keep]
    arrays = {mode: values[keep] for mode, values in arrays.items()}
    if time.size < 8 or np.any(np.diff(time) <= 0.0):
        raise ValueError("Psi4 time samples are not strictly increasing")
    return time, arrays


def _uniform_common_mode_series(time, modes):
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Psi4 time step must be finite and positive")
    count = int(np.floor((time[-1] - time[0]) / dt)) + 1
    if count < 8:
        raise ValueError("Too few uniformly sampled Psi4 points")
    grid = time[0] + dt * np.arange(count)
    uniform = {
        mode: np.interp(grid, time, values.real) + 1j * np.interp(grid, time, values.imag)
        for mode, values in modes.items()
    }
    return grid, uniform


def _source_directions(theta_nodes: int, phi_nodes: int):
    theta_nodes = max(2, int(theta_nodes))
    phi_nodes = max(4, int(phi_nodes))
    cos_theta, cos_weights = np.polynomial.legendre.leggauss(theta_nodes)
    theta = np.arccos(cos_theta)
    phi = 2.0 * np.pi * np.arange(phi_nodes, dtype=float) / phi_nodes
    direction_theta = np.repeat(theta, phi_nodes)
    direction_phi = np.tile(phi, theta_nodes)
    # The normalized solid-angle weights sum to one.
    weights = np.repeat(cos_weights / (2.0 * phi_nodes), phi_nodes)
    return direction_theta, direction_phi, weights


def _harmonic_matrix(modes, theta, phi):
    harmonics = np.empty((theta.size, len(modes)), dtype=complex)
    for mode_index, (ell, emm) in enumerate(modes):
        harmonics[:, mode_index] = np.array(
            [
                spin_weighted_spherical_harmonic(-2, ell, emm, th, ph)
                for th, ph in zip(theta, phi)
            ],
            dtype=complex,
        )
    return harmonics


def _polarization_amplitude_from_mode_ffts(
    positive_modes,
    negative_modes,
    harmonics,
    weights,
    averaging,
    chunk_size=2048,
):
    n_frequency = positive_modes.shape[1]
    result = np.empty(n_frequency, dtype=float)
    for start in range(0, n_frequency, int(chunk_size)):
        stop = min(start + int(chunk_size), n_frequency)
        positive = harmonics @ positive_modes[:, start:stop]
        negative = harmonics @ negative_modes[:, start:stop]
        plus = 0.5 * (positive + np.conjugate(negative))
        cross = (positive - np.conjugate(negative)) / (2.0j)
        amplitude = np.sqrt(0.5 * (np.abs(plus) ** 2 + np.abs(cross) ** 2))
        if averaging == "mean":
            result[start:stop] = weights @ amplitude
        elif averaging == "rms":
            result[start:stop] = np.sqrt(weights @ np.square(amplitude))
        else:
            raise ValueError(f"Unknown source-direction averaging {averaging!r}")
    return result


def retained_psi4_modes(time, modes, bh_mass, transient_cutoff_mbh=TRANSIENT_CUTOFF_MBH,
                       trim_prearrival=True):
    """One retained measured grid for the spectrum and its join diagnostic."""
    time, modes = _clean_common_mode_series(time, modes)
    if trim_prearrival:
        arrived = time >= 0.0
        time = time[arrived]
        modes = {mode: values[arrived] for mode, values in modes.items()}
        if time.size < 8:
            raise ValueError("Too few Psi4 samples remain after the pre-arrival cut")
    cutoff = time[0] + max(float(transient_cutoff_mbh), 0.0) * bh_mass
    selected = time >= cutoff
    time = time[selected]
    modes = {mode: values[selected] for mode, values in modes.items()}
    if time.size < 8:
        raise ValueError("Too few Psi4 samples remain after the transient cut")
    return _uniform_common_mode_series(time, modes)


def direct_psi4_spectrum(
    time,
    rpsi4_modes: Mapping[tuple[int, int], np.ndarray],
    bh_mass: float,
    *,
    transient_cutoff_mbh: float = TRANSIENT_CUTOFF_MBH,
    taper_alpha: float = TAPER_ALPHA,
    zero_pad_factor: float = ZERO_PAD_FACTOR,
    low_frequency_cycles: float = LOW_FREQUENCY_CYCLES,
    theta_nodes: int = THETA_NODES,
    phi_nodes: int = PHI_NODES,
    averaging: str = SOURCE_AVERAGING,
    representative_direction: tuple[float, float] | None = None,
    trim_prearrival: bool = True,
    continuation=None,
):
    """Build a finite-duration strain spectrum directly from ``r*Psi4``.

    The dimensionless variables are ``tau=t/M_BH``, ``q=r*h/M_BH`` and
    ``p=M_BH*r*Psi4=d^2q/dtau^2``.  The Fourier-domain conversion is therefore
    ``q_tilde=-p_tilde/(2*pi*nu)^2``.  Source directions are combined with
    spin-weight -2 harmonics before plus/cross polarization averaging.
    """
    bh_mass = float(bh_mass)
    if not np.isfinite(bh_mass) or bh_mass <= 0.0:
        raise ValueError("bh_mass must be finite and positive")

    # 1. Retain measured post-transient samples on one uniform time grid.
    time, modes = retained_psi4_modes(time, rpsi4_modes, bh_mass,
                                      transient_cutoff_mbh, trim_prearrival)

    # 2. Use tau=t/M_BH and p=M_BH*r*Psi4; taper every mode identically.
    tau = (time - time[0]) / bh_mass
    dtau = float(np.median(np.diff(tau)))
    duration = float(tau[-1] - tau[0])
    window = tukey_window(tau.size, taper_alpha)
    ordered_modes = tuple(sorted(modes))
    mode_data = np.stack([bh_mass * modes[mode] * window for mode in ordered_modes])

    if continuation is not None and continuation.extended_strain is not None:
        if continuation.issues and not continuation.conditional:
            raise ValueError("A rejected temporal trial cannot extend Psi4")
        if (2, 2) not in modes:
            raise ValueError("The continuation requires measured mode (2,2)")
        extended_time, extended_psi4 = continued_psi4_mode(
            time/bh_mass, bh_mass*modes[2, 2], continuation)
        # All other modes keep exactly their measured Tukey termination. Only
        # (2,2) replaces that termination with the analytic model and final taper.
        mode_data = np.pad(mode_data, ((0, 0), (0, len(extended_time)-len(time))))
        half = len(time)//2
        extended_psi4[:half] *= window[:half]  # unchanged onset taper width
        mode_data[ordered_modes.index((2, 2))] = extended_psi4

    zero_pad_factor = max(float(zero_pad_factor), 1.0)
    count = mode_data.shape[1]
    nfft = max(count, int(np.ceil(zero_pad_factor * count)))
    padding = nfft - count
    pad_left = padding // 2
    pad_right = padding - pad_left
    mode_data = np.pad(mode_data, ((0, 0), (pad_left, pad_right)))
    # 3. Continuous-transform normalization: p_tilde = FFT(p) * d_tau.
    # A common window lets us FFT modes once, then reconstruct each direction.
    # This equals reconstructing in time first, without thousands of FFTs.
    mode_fft = np.fft.fft(mode_data, axis=1) * dtau
    frequency_full = np.fft.fftfreq(nfft, d=dtau)
    positive_indices = np.flatnonzero(frequency_full > 0.0)
    frequency = frequency_full[positive_indices]
    if duration <= 0.0:
        raise ValueError("Psi4 duration must be positive")
    frequency_floor = max(float(low_frequency_cycles), 0.0) / duration
    positive_indices = positive_indices[frequency >= frequency_floor]
    frequency = frequency_full[positive_indices]
    if frequency.size < 4:
        raise ValueError("Too few positive frequencies above the finite-duration floor")

    negative_indices = (-positive_indices) % nfft
    omega_squared = np.square(2.0 * np.pi * frequency)
    positive_modes = mode_fft[:, positive_indices]
    negative_modes = mode_fft[:, negative_indices]

    # 4. Reconstruct plus/cross in each source direction, then average amplitude.
    if representative_direction is None:
        theta, phi, weights = _source_directions(theta_nodes, phi_nodes)
        harmonics = _harmonic_matrix(ordered_modes, theta, phi)
        strain_ft = _polarization_amplitude_from_mode_ffts(
            positive_modes,
            negative_modes,
            harmonics,
            weights,
            averaging,
        )
        averaging_label = f"source-direction {averaging}"
        resolved_theta_nodes = max(2, int(theta_nodes))
        resolved_phi_nodes = max(4, int(phi_nodes))
    else:
        theta = np.array([float(representative_direction[0])])
        phi = np.array([float(representative_direction[1])])
        harmonics = _harmonic_matrix(ordered_modes, theta, phi)
        strain_ft = _polarization_amplitude_from_mode_ffts(
            positive_modes,
            negative_modes,
            harmonics,
            np.ones(1),
            "mean",
        )
        averaging_label = "representative direction"
        resolved_theta_nodes = 1
        resolved_phi_nodes = 1

    # 5. |q_tilde| = |p_tilde| / (2*pi*nu)^2. This common division commutes
    # with the amplitude average above; it is not fixed-frequency integration.
    # The Fourier second-derivative minus sign disappears in the magnitude.
    strain_ft /= omega_squared
    spectrum = DimensionlessSpectrum(
        frequency=np.asarray(frequency, dtype=float),
        strain_ft=np.asarray(strain_ft, dtype=float),
        averaging=averaging_label,
    )
    info = Psi4SpectrumInfo(
        modes=ordered_modes,
        samples=int(tau.size),
        duration_mbh=duration,
        dt_mbh=dtau,
        frequency_min=float(frequency[0]),
        frequency_max=float(frequency[-1]),
        theta_nodes=resolved_theta_nodes,
        phi_nodes=resolved_phi_nodes,
        taper_alpha=float(taper_alpha),
        zero_pad_factor=zero_pad_factor,
        low_frequency_cycles=float(low_frequency_cycles),
    )
    return spectrum, info


def characteristic_strain(frequency, strain_ft):
    """Moore et al. Eq. (17), using a continuous-transform normalization."""
    return 2.0 * np.asarray(frequency, dtype=float) * np.abs(strain_ft)


def cumulative_snr(frequency, strain_ft, noise_psd, fmin, fmax):
    frequency = np.asarray(frequency, dtype=float)
    strain_ft = np.asarray(strain_ft, dtype=complex)
    noise_psd = np.asarray(noise_psd, dtype=float)
    selected = (
        np.isfinite(frequency)
        & np.isfinite(strain_ft.real)
        & np.isfinite(strain_ft.imag)
        & np.isfinite(noise_psd)
        & (noise_psd > 0.0)
        & (frequency >= float(fmin))
        & (frequency <= float(fmax))
    )
    out = np.full(frequency.shape, np.nan, dtype=float)
    if np.count_nonzero(selected) < 2:
        return out

    f = frequency[selected]
    integrand = 4.0 * np.square(np.abs(strain_ft[selected])) / noise_psd[selected]
    area = 0.5 * (integrand[1:] + integrand[:-1]) * np.diff(f)
    snr2 = np.empty(f.size, dtype=float)
    snr2[0] = 0.0
    snr2[1:] = np.cumsum(area)
    out[selected] = np.sqrt(np.maximum(snr2, 0.0))
    return out


def wigner_small_d(ell: int, m_prime: int, m: int, theta: float) -> float:
    """Wigner small-d matrix for the low multipoles used by the GW files."""
    ell = int(ell)
    m_prime = int(m_prime)
    m = int(m)
    if abs(m) > ell or abs(m_prime) > ell:
        return 0.0

    prefactor = np.sqrt(
        factorial(ell + m)
        * factorial(ell - m)
        * factorial(ell + m_prime)
        * factorial(ell - m_prime)
    )
    k_min = max(0, m - m_prime)
    k_max = min(ell + m, ell - m_prime)
    cosine = np.cos(0.5 * theta)
    sine = np.sin(0.5 * theta)
    total = 0.0
    for k in range(k_min, k_max + 1):
        denominator = (
            factorial(ell + m - k)
            * factorial(k)
            * factorial(m_prime - m + k)
            * factorial(ell - m_prime - k)
        )
        sign = -1.0 if (k - m + m_prime) % 2 else 1.0
        total += (
            sign
            * prefactor
            / denominator
            * cosine ** (2 * ell + m - m_prime - 2 * k)
            * sine ** (m_prime - m + 2 * k)
        )
    return float(total)


def spin_weighted_spherical_harmonic(
    spin_weight: int,
    ell: int,
    emm: int,
    theta: float,
    phi: float,
) -> complex:
    """Evaluate _sY_lm using the Goldberg/Wigner-d convention."""
    normalization = np.sqrt((2 * int(ell) + 1) / (4.0 * np.pi))
    phase = np.exp(1j * int(emm) * float(phi))
    return (
        (-1.0) ** int(spin_weight)
        * normalization
        * wigner_small_d(int(ell), int(emm), -int(spin_weight), float(theta))
        * phase
    )


# Detector PSD/ASD conversion and response conventions.


@dataclass(frozen=True)
class BinnedSpectralPower:
    frequency: np.ndarray
    power_dfrequency: np.ndarray


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


def _log_interpolate_curve(curve: DetectorCurve, frequency) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=float)
    clipped = np.clip(frequency, curve.frequency[0], curve.frequency[-1])
    return np.power(
        10.0,
        np.interp(np.log10(clipped), np.log10(curve.frequency), np.log10(curve.asd)),
    )


def effective_detector_asd(detector: str, frequency, curves) -> np.ndarray:
    """Effective ASD for which 4*int h_res^2/S_eff df is the sky-averaged SNR^2.

    The detector sees F+ h+ + Fx hx. Averaging over sky position and
    polarization angle gives <F+^2> = <Fx^2> = 1/5 for a right-angle
    interferometer and <F+ Fx> = 0, so <|F+h+ + Fx hx|^2> = (2/5) h_res^2 with
    Eq. (8) h_res^2 = (|h+|^2+|hx|^2)/2. Hence S_eff = (5/2) S_instrument.
    The same SNR follows by applying sqrt(2/5) to h_res and using the raw PSD.
    Tests pin this averaged response directly; they do not reproduce Wessel's
    published numerical SNRs or establish its input-curve convention.
    """
    if detector == "decigo":
        return np.sqrt(2.5) * _decigo_instrument_asd(frequency)
    asd = _log_interpolate_curve(curves[detector], frequency)
    if detector in {"ligo", "ce"}:
        return np.sqrt(2.5) * asd
    if detector == "et":
        # CoBA's input is a single 90-degree equivalent. Three independent
        # 60-degree Michelsons give sqrt(3)*sin(60)=3/2 network amplitude gain.
        return np.sqrt(2.5) / 1.5 * asd
    if detector == "lisa":
        # SciRDv1 S_h is one Michelson channel averaged per polarization
        # (20/3; exactly twice Robson+2019 at low f). Two independent channels
        # (A, E) halve it, and the h_res convention halves it again.
        return 0.5 * asd
    raise ValueError(f"Unknown detector {detector!r}")


# Observer scaling: f_obs = nu / [(1+z) G M_BH/c^3]. M_BH is not M_ADM.

@dataclass(frozen=True)
class FlatLambdaCDM:
    """Small dependency-free flat-Lambda-CDM distance model.

    H0 and matter density use Planck18 values; radiation and neutrino evolution
    are omitted. This retained approximation is not the full Astropy Planck18.
    """

    h0_km_s_mpc: float = COSMOLOGY_H0_KM_S_MPC
    omega_m: float = COSMOLOGY_OMEGA_M
    z_max: float = 12.0
    samples: int = 24001

    @cached_property
    def table(self):
        z = np.linspace(0.0, self.z_max, self.samples)
        omega_lambda = 1.0 - self.omega_m
        inv_e = 1.0 / np.sqrt(self.omega_m * (1.0 + z) ** 3 + omega_lambda)
        dz = np.diff(z)
        integral = np.empty_like(z)
        integral[0] = 0.0
        integral[1:] = np.cumsum(0.5 * (inv_e[1:] + inv_e[:-1]) * dz)
        d_comoving = (LIGHT_SPEED_KM_S / self.h0_km_s_mpc) * integral
        return z, (1.0 + z) * d_comoving

    def luminosity_distance_mpc(self, redshift):
        z = np.asarray(redshift, dtype=float)
        if np.any(z < 0.0) or np.any(z > self.z_max):
            raise ValueError(f"redshift must lie in [0, {self.z_max:g}]")
        z_table, d_table = self.table
        values = np.interp(z, z_table, d_table)
        return float(values) if np.isscalar(redshift) else values

    def redshift_at_luminosity_distance(self, distance_mpc):
        distance = np.asarray(distance_mpc, dtype=float)
        if np.any(distance < 0.0):
            raise ValueError("luminosity distance must be nonnegative")
        z_table, d_table = self.table
        if np.any(distance > d_table[-1]):
            raise ValueError(
                f"distance exceeds z_max={self.z_max:g} cosmology table "
                f"({d_table[-1]:.3g} Mpc)"
            )
        values = np.interp(distance, d_table, z_table)
        return float(values) if np.isscalar(distance_mpc) else values


def observer_spectrum(
    spectrum: DimensionlessSpectrum,
    source_bh_mass_msun: float,
    luminosity_distance_mpc: float,
    redshift: float = 0.0,
):
    """Scale a dimensionless waveform into observer-frame Hz and strain/Hz."""
    source_mass = float(source_bh_mass_msun)
    distance = float(luminosity_distance_mpc)
    z = float(redshift)
    if source_mass <= 0.0 or distance <= 0.0 or z < 0.0:
        raise ValueError("source mass and distance must be positive and redshift nonnegative")

    redshifted_mass = source_mass * (1.0 + z)
    time_scale = redshifted_mass * SECONDS_PER_M_SUN
    amplitude_scale = redshifted_mass * METERS_PER_M_SUN / (distance * METERS_PER_MPC)
    frequency = np.asarray(spectrum.frequency, dtype=float) / time_scale
    strain_ft = np.asarray(spectrum.strain_ft, dtype=complex) * amplitude_scale * time_scale
    return frequency, strain_ft


# SNR integration and distance roots. Binning speeds up the mass/redshift scan.

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
    if not segment_frequency.size:
        return BinnedSpectralPower(np.array([]), np.array([]))
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


def _threshold_redshift(binned, detector, source_mass, curves, cosmology, *, refine=False):
    """Outermost sampled SNR crossing; optionally bisect its bracket for examples."""
    samples = max(24, int(HORIZON_REDSHIFT_SAMPLES))
    redshift = np.unique(
        np.concatenate(
            (
                np.geomspace(1.0e-6, 0.1, samples),
                np.linspace(0.1, HORIZON_REDSHIFT_MAX, samples),
            )
        )
    )
    distance = cosmology.luminosity_distance_mpc(redshift)
    snr = np.array([_snr_from_binned_power(binned, detector, source_mass * (1 + z), d, curves)
                    for z, d in zip(redshift, distance)])
    detected = np.flatnonzero(snr >= SNR_THRESHOLD)
    if not detected.size:
        return np.nan
    last = int(detected[-1])
    if last + 1 == redshift.size:
        if refine:
            raise ValueError("Example horizon exceeds HORIZON_REDSHIFT_MAX; increase the search bound")
        return float(redshift[last])
    lower, upper = redshift[last:last + 2]
    if refine:
        for _ in range(32):
            mid = (lower + upper) / 2
            value = _snr_from_binned_power(binned, detector, source_mass * (1 + mid),
                                           cosmology.luminosity_distance_mpc(mid), curves)
            if value >= SNR_THRESHOLD:
                lower = mid
            else:
                upper = mid
        return float((lower + upper) / 2)
    y0 = np.log(max(snr[last], np.finfo(float).tiny))
    y1 = np.log(max(snr[last + 1], np.finfo(float).tiny))
    fraction = np.clip((np.log(SNR_THRESHOLD) - y0) / (y1 - y0), 0., 1.) if not np.isclose(y0, y1) else 0.
    return float(lower + fraction * (upper - lower))


def horizon_curves(spectra, curves, cosmology):
    masses = np.logspace(
        np.log10(HORIZON_MASS_RANGE_MSUN[0]),
        np.log10(HORIZON_MASS_RANGE_MSUN[1]),
        int(HORIZON_MASS_SAMPLES),
    )
    # Include exact example masses so their crosses lie on the drawn curves.
    masses = np.unique(np.r_[masses, [mass for _, mass, _ in EXAMPLE_TARGETS
                                     if HORIZON_MASS_RANGE_MSUN[0] <= mass <= HORIZON_MASS_RANGE_MSUN[1]]])
    threshold_examples = {(detector, mass) for detector, mass, distance in EXAMPLE_TARGETS
                          if distance is None}
    result = {detector: {} for detector in active_detectors()}
    for sim_name, spectrum in spectra.items():
        binned = _bin_spectral_power(spectrum, HORIZON_SPECTRAL_BINS)
        for detector in active_detectors():
            horizon = np.full_like(masses, np.nan)
            for mass_index, source_mass in enumerate(masses):
                z_horizon = _threshold_redshift(
                    binned, detector, source_mass, curves, cosmology,
                    refine=(detector, source_mass) in threshold_examples)
                if np.isfinite(z_horizon):
                    horizon[mass_index] = cosmology.luminosity_distance_mpc(z_horizon)
            result[detector][sim_name] = horizon
    return masses, result


# ----------------------------------------------------------------------
# 3. COMPLETE WORKFLOW: cached modes -> spectra -> SNR -> distance reach
# ----------------------------------------------------------------------

def spectrum_for_case(case, time, modes, **overrides):
    options = dict(
        transient_cutoff_mbh=TRANSIENT_CUTOFF_MBH, taper_alpha=TAPER_ALPHA,
        zero_pad_factor=ZERO_PAD_FACTOR, low_frequency_cycles=LOW_FREQUENCY_CYCLES,
        theta_nodes=THETA_NODES, phi_nodes=PHI_NODES, averaging=SOURCE_AVERAGING,
    )
    options.update(overrides)
    return direct_psi4_spectrum(time, modes, case.mlittle, **options)


def observed_target(spectra, mass, distance, redshift, curves, detector=None):
    """Apply one shared physical source scale to every case."""
    strain, snr = {}, {}
    for name, spectrum in spectra.items():
        frequency, strain_ft = observer_spectrum(spectrum, mass, distance, redshift)
        strain[name] = frequency, characteristic_strain(frequency, strain_ft)
        power = _bin_spectral_power(spectrum, HORIZON_SPECTRAL_BINS)
        snr[name] = {
            detector_name: _snr_from_binned_power(power, detector_name, mass * (1 + redshift), distance, curves)
            for detector_name in active_detectors()
        }
    return ObservedTarget(mass, distance, redshift, strain, snr, detector)


def example_targets(spectra, curves, cosmology):
    """Use refined best-case detector reaches, or explicitly fixed distances."""
    targets = []
    for detector, mass, distance in EXAMPLE_TARGETS:
        if detector not in active_detectors():
            raise ValueError(f"Example detector {detector} is not in ACTIVE_DETECTORS")
        horizon_case = None
        if distance is None:
            candidates = []
            for name, spectrum in spectra.items():
                power = _bin_spectral_power(spectrum, HORIZON_SPECTRAL_BINS)
                z = _threshold_redshift(power, detector, mass, curves, cosmology, refine=True)
                if np.isfinite(z):
                    candidates.append((float(cosmology.luminosity_distance_mpc(z)), name, z))
            if not candidates:
                raise ValueError(f"No {detector} threshold example at {mass:g} solar masses")
            distance, horizon_case, redshift = max(candidates)
        else:
            redshift = cosmology.redshift_at_luminosity_distance(distance)
        target = observed_target(spectra, mass, distance, redshift, curves, detector)
        if horizon_case is not None:
            target.distance_policy = "best-case threshold"
            target.horizon_case = horizon_case
        targets.append(target)
    return targets


def analyze(names=None, *, compare_radius=False, compare_direction=False, compute_horizons=True,
            radius_index=None, cache_root=None, include_reference=False):
    """Run steps 1-7; return numerical results, without creating any figures."""
    print(f"Detectability: modes={MODES}, angular {SOURCE_AVERAGING}, "
          f"cut={TRANSIENT_CUTOFF_MBH:g} M_BH, Tukey alpha={TAPER_ALPHA:g}, "
          f"f_min={LOW_FREQUENCY_CYCLES:g}/duration; finite radius, no tail", flush=True)
    # Steps 1-4: modes -> windowed FFT -> angular average -> dimensionless strain.
    radius_index = OUTER_RADIUS_INDEX if radius_index is None else radius_index
    sources, representative, inner = {}, {}, {}
    for case in config.all_sim_configs(SIM_NAMES if names is None else names):
        try:
            # Reader: the cached columns already have a common retarded-time grid.
            time, modes = read_rpsi4_modes(case, radius_index, cache_root=cache_root)
            # Calculation: the function above contains the cut, window, FFT,
            # harmonic reconstruction, angular average and -(2*pi*f)^-2 step.
            spectrum, info = direct_psi4_spectrum(
                time, modes, case.mlittle, transient_cutoff_mbh=TRANSIENT_CUTOFF_MBH,
                taper_alpha=TAPER_ALPHA, zero_pad_factor=ZERO_PAD_FACTOR,
                low_frequency_cycles=LOW_FREQUENCY_CYCLES, theta_nodes=THETA_NODES,
                phi_nodes=PHI_NODES, averaging=SOURCE_AVERAGING)
            sources[case.name] = SourceSpectrum(case, spectrum, info)
            print(f"{case.name}: radius index {radius_index}, {info.samples} samples "
                  f"over {info.duration_mbh:.1f} M_BH", flush=True)
            if compare_direction:
                representative[case.name] = spectrum_for_case(
                    case, time, modes, representative_direction=(np.pi/2.34, 0.))[0]
            if compare_radius:
                inner_time, inner_modes = read_rpsi4_modes(case, FIRST_RADIUS_INDEX, cache_root=cache_root)
                inner_spectrum, inner_info = spectrum_for_case(case, inner_time, inner_modes)
                inner[case.name] = SourceSpectrum(case, inner_spectrum, inner_info)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{case.name}: {exc}") from exc
    if not sources:
        raise ValueError("No valid direct-Psi4 spectra were produced")
    spectra = {name: source.spectrum for name, source in sources.items()}

    # Step 5: detector noise. Use exactly the same effective ASD for h_n and SNR.
    curves = load_detector_curves()
    noise = {}
    for detector in active_detectors():
        low, high = DETECTOR_BOUNDS[detector]
        frequency = np.geomspace(low, high, 500)
        noise[detector] = frequency, np.sqrt(frequency) * effective_detector_asd(detector, frequency, curves)

    # Step 6: source choices; redshift follows the same cosmology as SNR.
    cosmology = FlatLambdaCDM(z_max=max(12.0, HORIZON_REDSHIFT_MAX + 0.5))
    targets = example_targets(spectra, curves, cosmology)

    shibata = None
    if include_reference:
        redshift = cosmology.redshift_at_luminosity_distance(SHIBATA_DISTANCE_MPC)
        shibata = observed_target(spectra, SHIBATA_BH_MASS_MSUN, SHIBATA_DISTANCE_MPC,
                                  redshift, curves)

    # Step 7: SNR^2 = integral (h_c/h_n)^2 d ln f; threshold luminosity distance.
    horizons = horizon_curves(spectra, curves, cosmology) if compute_horizons else None
    return DetectabilityResult(sources, inner, representative, noise, targets, horizons, shibata)


# ----------------------------------------------------------------------
# 4. PLOT AND FORMAT: distance panel above characteristic strain
# ----------------------------------------------------------------------
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, ScalarFormatter
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from matplotlib.textpath import TextPath
from helpers.plot_common import parser, setup, savefig, save_individual_fig
from helpers.style import (
    apply_paper_style, apply_styles, figure_size, format_paper_axes,
    ordered_sim_legend, COMPACT_LEGEND_KWARGS, COLOR_OPTS,
    AXES_LEGEND_BORDERAXESPAD, PAPER_SINGLE_PANEL_HEIGHT, PAPER_TWO_PANEL_HEIGHT,
    PAPER_THREE_PANEL_HEIGHT,
)

DETECTOR_LABELS = {
    "ligo": r"$\mathrm{LIGO\ A+}$", "ce": r"$\mathrm{Cosmic\ Explorer}$",
    "decigo": r"$\mathrm{DECIGO}$", "lisa": r"$\mathrm{LISA\ (SciRDv1)}$",
    "et": r"$\mathrm{ET\ (10\,km)}$",
}
DETECTOR_LINESTYLES = {"ligo": ":", "et": (0, (3, 1, 1, 1, 1, 1)),
                      "ce": "-.", "decigo": "-", "lisa": "--"}
DETECTOR_COLOR = "0.15"
DETECTOR_CURVE_LABELS = {"ce": r"$\mathrm{CE}$", "et": r"$\mathrm{ET}$", "lisa": r"$\mathrm{LISA}$"}
DETECTABILITY_HEIGHT = 5.4
COMBINED_FREQUENCY_LIMITS = (5e-4, 3e2)  # Display only; full bins enter SNR.
COMBINED_STRAIN_LIMITS = (1e-25, 1e-18)
EXAMPLE_LABEL_STYLE = "caption"  # Wessel-style default; "inline" or "key" optional.
EXAMPLE_KEY_HEIGHT = .7
EXAMPLE_KEY_FONT_SIZE = 10
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
                             distance_policy=target.distance_policy,
                             horizon_case=target.horizon_case))
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
    ax.text(0.97, 0.96, rf"$\mathrm{{SNR}}={SNR_THRESHOLD:g}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=SOURCE_LABEL_SIZE)
    _format_axes([ax])
    cosmology = FlatLambdaCDM()
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


def plot_combined(result, args, *, filename, note=None,
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
    bottom = .13
    gap = .42 if note else .32
    top = .97
    if example_labels == "key":
        bottom = .23 + .02 * extra_rows
    elif example_labels == "inline":
        gap = .66
        top = .90
    fig.subplots_adjust(left=.11, right=.89, bottom=bottom, top=top, hspace=gap)
    lower_top = axes[1].get_position().y1
    legend_offset = .26 if note and example_labels != "key" else .10
    legend_y = lower_top + legend_offset / height
    if example_labels == "inline":
        legend_y = .96
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


# ----------------------------------------------------------------------
# 5. OPTIONAL DIAGNOSTICS: validation, continuation model, and their drawings.
# The measured workflow and main figure above do not need this section.
# ----------------------------------------------------------------------

# Diagnostic choices. Scalar data and continuation fits are loaded only
# when the corresponding diagnostic is requested.
VALIDATION_RADIUS_INDICES = (FIRST_RADIUS_INDEX, 7, OUTER_RADIUS_INDEX)
VALIDATION_TAPER_ALPHAS = (0.02, TAPER_ALPHA, 0.10)
VALIDATION_TRANSIENT_CUTOFFS_MBH = (500.0, TRANSIENT_CUTOFF_MBH, 1500.0)
VALIDATION_COARSE_ANGULAR_GRID = (12, 24)
TAIL_MASS_SOURCE = "horizon_flux"  # Or "outside_mass" for the original Wessel-like fit.
TAIL_FIT_START_FRACTION = 0.60  # Last 40% of retained measured strain.
TAIL_WINDOW_FRACTIONS = (0.40, 0.50, 0.60, 0.70, 0.80)  # Report sensitivity; never pick the best fit.
TAIL_HOLDOUT_FRACTION = 0.25  # Withhold last quarter of that interval, then refit.
TAIL_REMAINING_MASS_FRACTION = 0.10  # Wessel: 90% of initial disk mass accreted.
TAIL_FIT_LIMITS = dict(mass=0.01, amplitude=0.25, phase=0.5, holdout=0.5)
TAIL_MAX_SAMPLES = 2_000_000  # Abort, never shorten a tail silently.
TAIL_JOIN_ORBITS, TAIL_TAPER_ORBITS = 1.0, 3.0
TAIL_REVIEW_CASES = ("A1",)  # Empty tuple disables conditional examples.


def orbital_spectrum(case, spectrum):
    orbital_frequency = float(case.gw_omega_orbital) * case.mlittle / (2 * np.pi)
    if not np.isfinite(orbital_frequency) or orbital_frequency <= 0:
        raise ValueError(f"{case.name}: positive orbital frequency is required")
    return spectrum.frequency / orbital_frequency, characteristic_strain(spectrum.frequency, spectrum.strain_ft)


def interpolated_spectral_ratio(reference_x, reference_y, test_x, test_y):
    lower = max(float(np.min(reference_x)), float(np.min(test_x)))
    upper = min(float(np.max(reference_x)), float(np.max(test_x)))
    keep = ((reference_x >= lower) & (reference_x <= upper)
            & np.isfinite(reference_y) & (reference_y > 0))
    x = reference_x[keep]
    return x, np.interp(x, test_x, test_y) / reference_y[keep]


def _significant_ratio(label, baseline_x, baseline_y, x, y):
    x, ratio = interpolated_spectral_ratio(baseline_x, baseline_y, x, y)
    significant = (np.isfinite(ratio)
                   & (np.interp(x, baseline_x, baseline_y) >= 1e-3 * np.max(baseline_y)))
    if np.any(significant):
        change = abs(ratio[significant] - 1)
        print(f"{label}: median fractional change={np.median(change):.3g}, "
              f"95th percentile={np.percentile(change, 95):.3g}", flush=True)
    return x[significant], ratio[significant]


def validation_data(sim_name):
    """Return radius, taper and transient comparisons as arrays ready to draw."""
    case = config.all_sim_configs([sim_name])[0]
    time, modes = read_rpsi4_modes(case, OUTER_RADIUS_INDEX)
    baseline, _ = spectrum_for_case(case, time, modes)
    baseline_x, baseline_y = orbital_spectrum(case, baseline)
    radial, tapers, cutoffs = [], [], []
    for index in dict.fromkeys(VALIDATION_RADIUS_INDICES):
        try:
            if index == OUTER_RADIUS_INDEX:
                spectrum = baseline
            else:
                other_time, other_modes = read_rpsi4_modes(case, index)
                spectrum, _ = spectrum_for_case(case, other_time, other_modes)
        except (OSError, ValueError, KeyError) as exc:
            print(f"{sim_name}: radius index {index} unavailable: {exc}", flush=True)
            continue
        x, y = orbital_spectrum(case, spectrum)
        radial.append((index, x, y))
        if index != OUTER_RADIUS_INDEX:
            _significant_ratio(f"{sim_name} radius index {index}/outer", baseline_x, baseline_y, x, y)

    for alpha in dict.fromkeys(VALIDATION_TAPER_ALPHAS):
        spectrum = baseline if alpha == TAPER_ALPHA else spectrum_for_case(case, time, modes, taper_alpha=alpha)[0]
        x, ratio = _significant_ratio(
            f"{sim_name} Tukey alpha={alpha:g}/baseline", baseline_x, baseline_y, *orbital_spectrum(case, spectrum))
        tapers.append((alpha, x, ratio))
    for cutoff in dict.fromkeys(VALIDATION_TRANSIENT_CUTOFFS_MBH):
        spectrum = baseline if cutoff == TRANSIENT_CUTOFF_MBH else spectrum_for_case(
            case, time, modes, transient_cutoff_mbh=cutoff)[0]
        x, ratio = _significant_ratio(
            f"{sim_name} transient={cutoff:g} M_BH/baseline", baseline_x, baseline_y, *orbital_spectrum(case, spectrum))
        cutoffs.append((cutoff, x, ratio))
    coarse, _ = spectrum_for_case(case, time, modes,
                                 theta_nodes=VALIDATION_COARSE_ANGULAR_GRID[0],
                                 phi_nodes=VALIDATION_COARSE_ANGULAR_GRID[1])
    _significant_ratio(f"{sim_name} coarse/baseline angular grid", baseline_x, baseline_y, *orbital_spectrum(case, coarse))
    return radial, tapers, cutoffs


@dataclass
class WesselFit:
    start: float
    end: float
    gamma: float
    omega: float
    amplitude: float
    phase: float
    mass_end: float
    errors: dict

    def strain(self, time):
        return self.amplitude * np.exp((-self.gamma + 1j*self.omega)
                                      * (np.asarray(time) - self.end) + 1j*self.phase)

    def mass(self, time):
        return self.mass_end * np.exp(-self.gamma * (np.asarray(time) - self.end))


@dataclass
class WesselTrial:
    time: np.ndarray
    strain: np.ndarray
    mass_time: np.ndarray
    mass: np.ndarray
    initial_mass: float
    fit: WesselFit
    holdout_fit: WesselFit
    holdout_error: float
    issues: list[str]
    extended_time: np.ndarray | None = None
    extended_strain: np.ndarray | None = None
    finite_spectrum: DimensionlessSpectrum | None = None
    extended_spectrum: DimensionlessSpectrum | None = None
    conditional: bool = False
    density_time: np.ndarray | None = None
    density_modes: np.ndarray | None = None
    window_fits: tuple[WesselFit, ...] = ()
    mass_source: str = "outside_mass"
    orbital_period: float = 0.0


def fit_wessel_mode(time, strain, mass_time, mass, start, end):
    """Eq. (10): mass -> gamma, unwrapped phase -> omega, amplitude -> B.

    All times are in M_BH, strain is r*h_22/M_BH. Fit the mass itself in
    linear space (not log mass). Uniformly sampled waveform and scalar data
    have equal sample weights. No absolute value or positivity bound on gamma
    is imposed: a growing mass history must remain a failed decay hypothesis.
    """
    from scipy.optimize import curve_fit

    wave = (time >= start) & (time <= end)
    disk = (mass_time >= start) & (mass_time <= end)
    if wave.sum() < 16 or disk.sum() < 16 or end <= start:
        raise ValueError("Need at least 16 waveform and mass samples in the fit interval")
    h, x = strain[wave], time[wave] - end
    y, xm = mass[disk], (mass_time[disk] - end) / (end - start)
    if np.any(y <= 0) or np.any(np.abs(h) == 0):
        raise ValueError("Mass and complex-mode amplitude must be positive for this fit")
    scale = np.mean(y)
    guess = np.polyfit(xm, np.log(y / scale), 1)
    pars, _ = curve_fit(lambda u, b, g: b * np.exp(-g*u), xm, y/scale,
                        p0=(np.exp(guess[1]), -guess[0]), maxfev=10000)
    mass_end, gamma = float(pars[0]*scale), float(pars[1]/(end-start))
    phase = np.unwrap(np.angle(h))
    omega, phase_end = np.polyfit(x, phase, 1)
    envelope = np.exp(-gamma*x)
    amplitude = float(np.dot(envelope, np.abs(h)) / np.dot(envelope, envelope))
    rms = lambda a: float(np.sqrt(np.mean(np.abs(a)**2)))
    errors = dict(mass=rms(y-mass_end*np.exp(-gamma*(mass_time[disk]-end)))/rms(y),
                  amplitude=rms(np.abs(h)-amplitude*envelope)/rms(h),
                  phase=rms(phase-(omega*x+phase_end)))
    return WesselFit(float(start), float(end), gamma, float(omega), amplitude,
                     float(phase_end), mass_end, errors)


def wessel_trial(time, strain, mass_time, mass, orbital_period, *,
                 fit_start_fraction=TAIL_FIT_START_FRACTION,
                 holdout_fraction=TAIL_HOLDOUT_FRACTION,
                 allow_poor_waveform_fit=False):
    """Fit Eq. (10), retaining diagnostics even for conditional review models.

    The optional review override waives only waveform-residual limits, not
    mass-decay or sampling checks. It must not be interpreted as validation.

    Source-coordinate scalar time is compared with retarded GW source time,
    both t/M_BH with the same simulation origin. This is the usual approximate
    source-time identification, not a gauge-invariant time mapping.
    """
    time, strain = np.asarray(time, float), np.asarray(strain, complex)
    mass_time, mass = np.asarray(mass_time, float), np.asarray(mass, float)
    if not (0 <= fit_start_fraction < 1 and 0 < holdout_fraction < 1):
        raise ValueError("Fit and holdout fractions must define nonempty intervals")
    if not np.isfinite(orbital_period) or orbital_period <= 0:
        raise ValueError("Orbital period must be finite and positive")
    for t, y in ((time, strain), (mass_time, mass)):
        if t.ndim != 1 or t.size < 32 or y.shape != t.shape or not np.all(np.isfinite(y)) \
                or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("Need finite, increasing waveform and mass series")
    if mass[0] <= 0 or time[0] < mass_time[0] or time[-1] > mass_time[-1]:
        raise ValueError("Positive initial disk mass and scalar coverage of the waveform are required")
    if not np.allclose(np.diff(time), np.median(np.diff(time)), rtol=1e-5):
        raise ValueError("Use the uniformly sampled strain cache")
    start = time[0] + fit_start_fraction*np.ptp(time)
    holdout_start = time[-1] - holdout_fraction*(time[-1]-start)
    fit = fit_wessel_mode(time, strain, mass_time, mass, start, time[-1])
    held_fit = fit_wessel_mode(time, strain, mass_time, mass, start, holdout_start)
    held = time > holdout_start
    held_error = float(np.linalg.norm(strain[held]-held_fit.strain(time[held]))
                       / np.linalg.norm(strain[held]))
    issues = []
    if fit.gamma <= 0:
        issues.append("disk mass does not decay")
    elif fit.gamma*(fit.end-fit.start) <= 3*fit.errors["mass"]:
        issues.append("mass decay unresolved above fit residuals")
    if fit.gamma*orbital_period/(2*np.pi) >= 0.1:
        issues.append("decay is not slow compared with orbital motion")
    if abs(fit.omega)*(fit.end-fit.start)/(2*np.pi) < 3:
        issues.append("fewer than three fitted wave cycles")
    if not np.isfinite(fit.errors["mass"]) or fit.errors["mass"] > TAIL_FIT_LIMITS["mass"]:
        issues.append("mass residual exceeds trial limit")
    blocking = bool(issues)
    for key, error in {**fit.errors, "holdout": held_error}.items():
        if key == "mass":
            continue
        if not np.isfinite(error) or error > TAIL_FIT_LIMITS[key]:
            issues.append(f"{key} residual exceeds trial limit")
            blocking |= not np.isfinite(error)
    trial = WesselTrial(time, strain, mass_time, mass, float(mass[0]), fit,
                         held_fit, held_error, issues, orbital_period=orbital_period)
    if blocking or (issues and not allow_poor_waveform_fit):
        return trial

    # The paper does not specify a join
    # function: quintic blending over the last orbit is our explicit choice.
    stop = fit.end + np.log(fit.mass_end/(TAIL_REMAINING_MASS_FRACTION*mass[0]))/fit.gamma
    join_width, taper_width = TAIL_JOIN_ORBITS*orbital_period, TAIL_TAPER_ORBITS*orbital_period
    dt = float(np.median(np.diff(time)))
    count = int(np.ceil((stop+taper_width-time[0])/dt)) + 1
    if stop <= fit.end or fit.end-join_width < fit.start:
        issues.append("no future mass threshold or insufficient join interval")
        return trial
    if count > TAIL_MAX_SAMPLES:
        issues.append("tail exceeds sample budget; no truncated substitute")
        return trial
    extended_time = time[0] + dt*np.arange(count)
    extended = fit.strain(extended_time)
    blend = np.clip((time-(fit.end-join_width))/join_width, 0, 1)
    blend = blend**3*(10-15*blend+6*blend**2)
    extended[:time.size] = (1-blend)*strain + blend*extended[:time.size]
    taper = np.clip((extended_time-stop)/taper_width, 0, 1)
    extended *= 1-taper**3*(10-15*taper+6*taper**2)
    trial.extended_time, trial.extended_strain = extended_time, extended
    # Identical FFI input, FFT grid and measured-duration frequency floor.
    # Only the termination differs. Keep the onset taper's physical width
    # fixed; a full-length Tukey would re-window the measured signal.
    off_window = tukey_window(time.size, TAPER_ALPHA)
    on_window = np.ones(count)
    half = time.size//2
    on_window[:half] = off_window[:half]
    nfft = int(np.ceil(ZERO_PAD_FACTOR*count))
    floor = LOW_FREQUENCY_CYCLES/np.ptp(time)
    trial.finite_spectrum = _single_22_spectrum(strain*off_window, dt, nfft, floor)
    trial.extended_spectrum = _single_22_spectrum(extended*on_window, dt, nfft, floor)
    trial.conditional = bool(issues)
    return trial


def _single_22_spectrum(strain, dt, nfft, frequency_floor):
    """Same polarization/orientation statistic as production, for one mode.

    For a single mode the angular factor separates exactly, avoiding a large
    direction-by-frequency array for long tails. No extra 1/f^2: this is h.
    """
    fft = np.fft.fft(strain, n=nfft)*dt
    freq = np.fft.fftfreq(nfft, dt)
    pos = np.flatnonzero((freq > 0) & (freq >= frequency_floor))
    theta, phi, weights = _source_directions(THETA_NODES, PHI_NODES)
    harmonics = np.abs(_harmonic_matrix(((2, 2),), theta, phi)[:, 0])
    if SOURCE_AVERAGING == "mean":
        factor = weights @ harmonics
    elif SOURCE_AVERAGING == "rms":
        factor = np.sqrt(weights @ harmonics**2)
    else:
        raise ValueError(f"Unknown source averaging {SOURCE_AVERAGING}")
    amplitude = 0.5*factor*np.sqrt(np.abs(fft[pos])**2+np.abs(fft[(-pos)%nfft])**2)
    return DimensionlessSpectrum(freq[pos], amplitude, f"source-direction {SOURCE_AVERAGING}")


def disk_mass_for_tail(scalar, source):
    """Outside-AH mass or a flux-only remaining-rest-mass estimate.

    bhns.don columns 2/3 are total/inside-AH M0; column 13 is the recorded
    positive cumulative inward flux. Subtract its initial offset, not abs().
    Fitting M0(0)-F(t) to an exponential is equivalent to fitting the implied
    accumulated flux M0(0)-B exp(-gamma*t), with the initial mass fixed.
    """
    scalar = np.asarray(scalar, float)
    outside = scalar[:, 1]-scalar[:, 2]
    if source == "outside_mass":
        mass = outside
    elif source == "horizon_flux":
        mass = outside[0]-(scalar[:, 12]-scalar[0, 12])
    else:
        raise ValueError("TAIL_MASS_SOURCE must be horizon_flux or outside_mass")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Temporal mass history must remain finite and positive")
    return mass


def continued_psi4_mode(time, psi4, trial):
    """Continue p=M_BH*r*Psi4 using q''=lambda^2*q, q=r*h/M_BH.

    The Wessel strain model supplies lambda=-gamma+i*omega. Analytic
    differentiation is exact for that MODEL, not validation of the data fit.
    Blend measured p into model p over the last orbit; taper p at the same
    mass threshold as the strain trial. These join/window choices are ours.
    The measured direct-Psi4 baseline never uses this routine.
    """
    time, psi4 = np.asarray(time), np.asarray(psi4)
    dt = float(np.median(np.diff(time)))
    # Join at the measured Psi4 end; allow for one sample of grid rounding.
    if trial.extended_time is None or abs(time[-1]-trial.fit.end) > 1.5*dt:
        raise ValueError("Psi4 and temporal trial endpoints differ by more than one sample")
    orbital_period = trial.orbital_period
    if orbital_period <= 0:
        raise ValueError("A positive orbital period is required for the continuation")
    stop = trial.fit.end+np.log(trial.fit.mass_end/
           (TAIL_REMAINING_MASS_FRACTION*trial.initial_mass))/trial.fit.gamma
    end = stop+TAIL_TAPER_ORBITS*orbital_period
    count = int(np.ceil((end-time[0])/dt))+1
    if count > TAIL_MAX_SAMPLES:
        raise ValueError("Psi4 tail exceeds the sample budget")
    extended_time = time[0]+dt*np.arange(count)
    lam = -trial.fit.gamma+1j*trial.fit.omega
    extended = lam**2*trial.fit.strain(extended_time)
    blend = np.clip((time-(time[-1]-TAIL_JOIN_ORBITS*orbital_period))/
                    (TAIL_JOIN_ORBITS*orbital_period), 0, 1)
    blend = blend**3*(10-15*blend+6*blend**2)
    extended[:len(time)] = (1-blend)*psi4+blend*extended[:len(time)]
    taper = np.clip((extended_time-stop)/(TAIL_TAPER_ORBITS*orbital_period), 0, 1)
    extended *= 1-taper**3*(10-15*taper+6*taper**2)
    return extended_time, extended


def temporal_trial_for_case(case):
    """Reuse the existing GW cache and restart-aware scalar reader; no raw writes."""
    from helpers.reader import DiskSim

    label = str(OUTER_RADIUS_INDEX + case.gw_psi4_file_index_offset)
    cached = read_strain_cache(strain_cache_dir(case, label), case.data_path/"Psi4_rad.mon")
    if cached.backend != "python" or str(cached.metadata.get("source_label")) != label \
            or not np.isclose(cached.metadata.get("madm", np.nan), case.gw_madm, rtol=1e-10, atol=0):
        raise ValueError(f"{case.name}: regenerate inconsistent strain cache")
    hp, hc = cached.hplus_hcross(2, 2)
    time = cached.time/case.mlittle
    # Undo the legacy Fortran conjugation for the natural spin -2 complex mode.
    strain = (hp-1j*hc)/case.mlittle
    sim = DiskSim(case)
    _, scalar, mass_time, _ = sim.loaddata("bhns.don")
    disk_mass = disk_mass_for_tail(scalar, TAIL_MASS_SOURCE)
    mass_time = mass_time/case.mlittle
    if mass_time[0] != 0:
        raise ValueError(f"{case.name}: initial disk mass at t=0 is unavailable")
    # Stop at the last measured Psi4 sample; later FFI rows integrate zero fill.
    measured_end = read_rpsi4_modes(case, OUTER_RADIUS_INDEX, ((2, 2),))[0][-1]/case.mlittle
    keep = ((time >= TRANSIENT_CUTOFF_MBH) & (time >= mass_time[0]) & (time <= mass_time[-1])
            & (time <= measured_end + 1e-9))
    trial = wessel_trial(time[keep], strain[keep], mass_time, disk_mass, case.Pc/case.mlittle,
                         allow_poor_waveform_fit=case.name in TAIL_REVIEW_CASES)
    trial.mass_source = TAIL_MASS_SOURCE
    sim.load_modes()
    valid = (sim.modes[:, 0].real > 0) & np.all(np.isfinite(sim.modes), axis=1)
    trial.density_time = sim.modes_t[valid]/case.mlittle
    trial.density_modes = np.abs(sim.modes[valid, 1:])/np.abs(sim.modes[valid, :1])
    # C0 is density-cut in the Illinois diagnostic, unlike the full M0 integral.
    # Use C1/C0 to assess structure, never substitute C0 for disk rest mass.
    trial.window_fits = tuple(
        fit_wessel_mode(trial.time, trial.strain, mass_time, disk_mass,
                        trial.time[0]+fraction*np.ptp(trial.time), trial.time[-1])
        for fraction in TAIL_WINDOW_FRACTIONS)
    return trial


BATCH_PLOTS = ("radius", "averaging", "combined", "reference", "extrapolation")
AVAILABLE_PLOTS = ("combined", "strain", "distance", "radius", "averaging", "reference", "extrapolation")
OUTPUT_FILENAME_CHARACTERISTIC_STRAIN = "gw/gw_detectability_characteristic_strain.png"
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
        x, hc = orbital_spectrum(source.config, source.spectrum)
        inner_x, inner_hc = orbital_spectrum(source.config, result.inner[name].spectrum)
        keep = _spectrum_plot_mask(x, hc)
        axes[0].plot(x[keep], hc[keep], color=source.color, linestyle=source.linestyle, label=source.legend_name)
        ratio_x, ratio = interpolated_spectral_ratio(x, hc, inner_x, inner_hc)
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
        x, hc = orbital_spectrum(source.config, source.spectrum)
        keep = _spectrum_plot_mask(x, hc)
        axes[0].plot(x[keep], hc[keep], color=source.color, linestyle=source.linestyle, label=source.legend_name)
        representative = result.representative.get(name)
        if representative is not None:
            _, rep_hc = orbital_spectrum(source.config, representative)
            ratio = np.divide(hc, rep_hc, out=np.full_like(hc, np.nan), where=rep_hc > 0)
            mask = keep & np.isfinite(ratio)
            axes[1].plot(x[mask], ratio[mask], color=source.color, linestyle=source.linestyle)
    axes[0].set_ylabel(r"$r h_c/M_{\mathrm{BH}}$")
    axes[0].set_yscale("log")
    average_label = r"\langle h_c\rangle_\Omega" if SOURCE_AVERAGING == "mean" else r"h_{c,\mathrm{RMS}}"
    axes[1].set_ylabel(rf"${average_label}/h_c(\pi/2.34)$")
    axes[1].set_xlabel(r"$f/f_{\mathrm{orbit}}$")
    axes[1].axhline(1, color="0.45", linewidth=0.8, linestyle=":")
    _format_axes(axes)
    ordered_sim_legend(axes[0], ncols=3, loc="upper right", **COMPACT_LEGEND_KWARGS)
    fig.subplots_adjust(left=0.16, right=0.98, bottom=0.10, top=0.98, hspace=0.08)
    savefig(fig, args, OUTPUT_FILENAME_METHOD_COMPARISON)


def _plot_characteristic_strain(result, args, *, fixed_source=False, filename=None, note=None):
    height = SHIBATA_HEIGHT if fixed_source else TARGETS_HEIGHT
    fig, ax = plt.subplots(figsize=figure_size("double", height))
    _draw_characteristic_strain(ax, result, fixed_source=fixed_source, note=note)
    fig.subplots_adjust(left=0.13, right=0.98, bottom=0.18, top=0.98)
    for detector, (frequency, hn) in result.noise.items():
        _label_noise_curve(ax, frequency, hn, detector)
    savefig(fig, args, filename or (OUTPUT_FILENAME_SHIBATA if fixed_source else OUTPUT_FILENAME_CHARACTERISTIC_STRAIN))


def _plot_horizon(result, args, *, filename=OUTPUT_FILENAME_HORIZON, note=None):
    fig, ax = plt.subplots(figsize=figure_size("double", HORIZON_HEIGHT))
    _draw_horizon(ax, result, note=note)
    fig.subplots_adjust(left=0.13, right=0.87, bottom=0.18, top=0.98)
    savefig(fig, args, filename)


def _plot_extrapolation(result, args):
    """Keep the direct-Psi4 baseline; modify only the (2,2) termination."""
    spectra, reports, extended, conditional = {}, {}, [], []
    for name, source in result.sources.items():
        trial = temporal_trial_for_case(source.config)
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
        time, modes = read_rpsi4_modes(source.config, OUTER_RADIUS_INDEX)
        spectra[name], _ = spectrum_for_case(source.config, time, modes, continuation=trial)
        extended.append(name)
        if trial.conditional:
            conditional.append(name)
        reports[name]["extended"] = True
        reports[name]["total_duration_over_measured"] = float(np.ptp(trial.extended_time)/np.ptp(trial.time))
        save_individual_fig(plot_temporal_transition(source.config, trial), args,
                            name, "gw_temporal_transition.png")
        time, modes = retained_psi4_modes(time, modes, source.config.mlittle)
        tau = time/source.config.mlittle
        p = source.config.mlittle*modes[2, 2]
        te, pe = continued_psi4_mode(tau, p, trial)
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
    curves, cosmology = load_detector_curves(), FlatLambdaCDM()
    targets = example_targets(spectra, curves, cosmology)
    updated = replace(result, targets=targets)
    # This diagnostic always draws the two panels together. Its spectra and
    # threshold examples use the extrapolated model; the measured result stays
    # available separately, with its own unchanged output name.
    updated.horizons = horizon_curves(spectra, curves, cosmology)
    plot_combined(
        updated, args, filename=f"gw/detectability/radius_{args.radius_index}_conditional.png", note=note)
    report = dict(input="direct Psi4 FFT for both off and on", modes=MODES,
                  extended_mode=[2, 2], cases=reports,
                  off=[dict(mass=t.mass, distance=t.distance, snr=t.snr) for t in result.targets],
                  on=[dict(mass=t.mass, distance=t.distance, snr=t.snr) for t in targets])
    if not args.no_save:
        path = Path(args.outdir)/"gw/gw_detectability_time_comparison.json"
        path.write_text(json.dumps(report, indent=2)+"\n")
    return report


def plot_validation(sim_name):
    radial, tapers, cutoffs = validation_data(sim_name)
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=figure_size("double", VALIDATION_HEIGHT))
    for index, x, y in radial:
        endpoint = index in {FIRST_RADIUS_INDEX, OUTER_RADIUS_INDEX}
        label = "Outer" if index == OUTER_RADIUS_INDEX else "Inner" if index == FIRST_RADIUS_INDEX else "Intermediate"
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
    join_start = max(trial.time[0], end-TAIL_JOIN_ORBITS*period)
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
    args = parser("Test Wessel's disk-mass-driven (2,2) strain continuation.").parse_args(argv)
    setup(args)
    cases = config.all_sim_configs(args.sims or SIM_NAMES) if cases is None else cases
    reports = {}
    curves, cosmology = load_detector_curves(), FlatLambdaCDM()
    for case in cases:
        trial = temporal_trial_for_case(case)
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
        mass, distance = SHIBATA_BH_MASS_MSUN, SHIBATA_DISTANCE_MPC
        z = cosmology.redshift_at_luminosity_distance(distance)
        result = observed_target({"Off": trial.finite_spectrum, "On": trial.extended_spectrum},
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


def validation_main(argv=None):
    args = parser("Validate direct-Psi4 detectability for individual simulations.").parse_args(argv)
    setup(args)
    for name in args.sims or SIM_NAMES:
        fig = plot_validation(name)
        save_individual_fig(fig, args, name, f"gw_detectability_method_validation_{name}.png")


# 6. COMMAND: select inputs -> calculate what is requested -> draw and save.
def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Measured GW spectra, SNR=8 reach and combined figure.")
    p.add_argument("--sims", nargs="+", default=list(SIM_NAMES))
    p.add_argument("--outdir", type=Path, default=config.PLOTS_DIR)
    p.add_argument("--show", action="store_true")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--radius-index", type=int, choices=range(9), default=OUTER_RADIUS_INDEX,
                   help="Zero-based parfile extraction index (0-8); A-case file label = index + 1")
    p.add_argument("--cache-dir", type=Path, default=config.GW_WORK_ROOT,
                   help="Directory containing A1_psi4*, A2_psi4*, A3_psi4* caches")
    p.add_argument("--example-labels", choices=("caption", "key", "inline"), default=EXAMPLE_LABEL_STYLE)
    p.add_argument("--plots", nargs="+", choices=AVAILABLE_PLOTS,
                   default=("combined",), metavar="NAME",
                   help="Choose from %(choices)s; default: %(default)s")
    args = p.parse_args(argv)
    requested = set(args.plots)
    if "extrapolation" in requested and (args.radius_index != OUTER_RADIUS_INDEX
                                         or args.cache_dir != config.GW_WORK_ROOT):
        p.error("Continuation diagnostics require the configured outer-radius and scalar caches")

    # 1. Check and print the actual inputs before attempting any expensive work.
    cases = config.all_sim_configs(args.sims)
    unknown = set(name.upper() for name in args.sims) - {case.name for case in cases}
    if unknown:
        p.error("Unknown simulation(s): " + ", ".join(sorted(unknown)))
    inputs = []
    for case in cases:
        label = args.radius_index + case.gw_psi4_file_index_offset
        folder = args.cache_dir / f"{case.name}_psi4{label}"
        inputs.append(dict(case=case.name, radius_index=args.radius_index,
                           psi4_file_label=label, cache=str(folder.resolve())))
        print(f"{case.name}: parfile index {args.radius_index} -> Psi4_rad.mon.{label} -> {folder}", flush=True)
        for name in ("rpsi4_uniform.dat", "strain_cache.json"):
            if not (folder/name).is_file():
                p.error(f"Missing {folder/name}. GW caches are not included in GitHub. "
                        "Copy the selected caches from Anvil or use --cache-dir PATH. "
                        "See notes/run_combined_gw_plot.md.")

    # 2. Compute only the requested comparisons; the default needs measured
    #    waveforms at one radius and never loads the scalar/continuation model.
    try:
        result = analyze(
            args.sims, radius_index=args.radius_index, cache_root=args.cache_dir,
            compare_radius="radius" in requested, compare_direction="averaging" in requested,
            compute_horizons=bool(requested & {"combined", "distance"}),
            include_reference="reference" in requested)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    apply_styles(list(result.sources.values()))

    # 3. Apply the usual TeX/Palatino style and draw the existing polished layout.
    apply_paper_style()
    args.outdir.mkdir(parents=True, exist_ok=True)
    filename = f"gw/detectability/radius_{args.radius_index}.png"
    if "combined" in requested:
        plot_combined(result, args, filename=filename, example_labels=args.example_labels)
    if "strain" in requested:
        _plot_characteristic_strain(result, args)
    if "distance" in requested:
        _plot_horizon(result, args)
    if "radius" in requested and result.inner:
        _plot_radius_comparison(result, args)
    if "averaging" in requested:
        _plot_method_comparison(result, args)
    if "reference" in requested:
        _plot_characteristic_strain(result, args, fixed_source=True)
    if "extrapolation" in requested:
        _plot_extrapolation(result, args)

    # 4. Save the radius/input list alongside the figure, caption and example SNRs.
    if "combined" in requested and not args.no_save:
        path = args.outdir / filename
        path.with_name(path.stem+"_inputs.json").write_text(json.dumps(inputs, indent=2)+"\n")
        print(f"Figure: {path.resolve()} (and .pdf)", flush=True)
        print(f"Caption and source parameters: {path.with_suffix('').resolve()}_caption.tex / _examples.json", flush=True)
    return result


if __name__ == "__main__":
    main()
