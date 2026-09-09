"""Unit-safe GW detectability utilities.

The reusable spectral object is normalized by the configured initial
central-BH mass scale: tau=t/M_BH and q=r*h/M_BH. A target source-frame BH
mass and cosmological redshift are applied only when constructing an observed
waveform.  The total ADM mass is intentionally absent from these conversions.
"""
from __future__ import annotations

import config

from dataclasses import dataclass
from functools import cached_property
from math import factorial
from typing import Mapping

import numpy as np


SECONDS_PER_M_SUN = 4.92549095e-6
METERS_PER_M_SUN = 1476.6250385
METERS_PER_MPC = 3.085677581491367e22
LIGHT_SPEED_KM_S = 299792.458


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


def retarded_time_from_cache(coordinate_time, cached_retarded_time):
    """Map every raw Psi4 row onto a legacy cached retarded-time axis.

    A legacy cache generated before a simulation continuation can be a strict prefix
    of the subsequently merged Psi4 table. In that case, preserve every cached
    value and extend only the time mapping with the robust late-time slope.
    No cached strain amplitude is used.
    """
    coordinate = np.asarray(coordinate_time, dtype=float)
    retarded = np.asarray(cached_retarded_time, dtype=float)
    if coordinate.ndim != 1 or retarded.ndim != 1:
        raise ValueError("coordinate and cached retarded times must be one-dimensional")
    if coordinate.size < 8 or retarded.size < 8:
        raise ValueError("too few coordinate or cached retarded-time samples")
    if not np.all(np.isfinite(coordinate)) or not np.all(np.isfinite(retarded)):
        raise ValueError("coordinate and cached retarded times must be finite")
    if np.any(np.diff(coordinate) <= 0.0) or np.any(np.diff(retarded) <= 0.0):
        raise ValueError("coordinate and cached retarded times must be strictly increasing")
    if coordinate.size == retarded.size:
        return retarded.copy(), "cached-exact"
    if retarded.size > coordinate.size:
        raise ValueError(
            f"cached retarded-time table has {retarded.size} rows but Psi4 has only {coordinate.size}"
        )

    overlap = coordinate[: retarded.size]
    tail_samples = min(256, retarded.size - 1)
    coordinate_dt = np.diff(overlap[-(tail_samples + 1) :])
    retarded_dt = np.diff(retarded[-(tail_samples + 1) :])
    slope = float(np.median(retarded_dt / coordinate_dt))
    if not np.isfinite(slope) or slope <= 0.0 or not 0.5 <= slope <= 1.5:
        raise ValueError(f"cached retarded-time continuation slope is implausible: {slope!r}")

    mapped = np.empty_like(coordinate)
    mapped[: retarded.size] = retarded
    mapped[retarded.size :] = retarded[-1] + slope * (
        coordinate[retarded.size :] - overlap[-1]
    )
    return mapped, f"cached-prefix+linear-extension({coordinate.size - retarded.size} rows)"


def prepared_rpsi4_modes(strain_result, psi4_file, modes):
    """Return direct-Psi4 inputs, preferring the maintained uniform cache.

    The cached values are still ``r Psi4`` rather than integrated strain. This
    lets detectability apply its own window and frequency-domain conversion
    while sharing the ordinary waveform workflow's retarded-time correction
    and interpolation.
    """
    requested = tuple(modes)
    if strain_result.rpsi4_uniform is not None:
        time, values = strain_result.rpsi4_modes(requested)
        return time, values, f"{strain_result.backend}-uniform-rPsi4"

    time, method = retarded_time_from_cache(psi4_file.time, strain_result.time)
    values = {
        mode: psi4_file.psi4(ell=mode[0], emm=mode[1], multiply_by_r=True)
        for mode in requested
    }
    return time, values, method


@dataclass(frozen=True)
class FlatLambdaCDM:
    """Small dependency-free flat-Lambda-CDM distance model.

    Defaults match the Astropy Planck18 realization to the shown precision.
    Radiation is negligible over the redshift range used for these plots.
    """

    h0_km_s_mpc: float = 67.66
    omega_m: float = 0.30966
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


def characteristic_strain(frequency, strain_ft):
    return 2.0 * np.asarray(frequency, dtype=float) * np.abs(strain_ft)


def tukey_window(size: int, alpha: float = 0.05) -> np.ndarray:
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


def direct_psi4_spectrum(
    time,
    rpsi4_modes: Mapping[tuple[int, int], np.ndarray],
    bh_mass: float,
    *,
    transient_cutoff_mbh: float = 1000.0,
    taper_alpha: float = 0.05,
    zero_pad_factor: float = 2.0,
    low_frequency_cycles: float = 3.0,
    theta_nodes: int = 24,
    phi_nodes: int = 48,
    averaging: str = "mean",
    representative_direction: tuple[float, float] | None = None,
    trim_prearrival: bool = True,
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

    time, modes = _clean_common_mode_series(time, rpsi4_modes)
    if trim_prearrival and np.any(time >= 0.0):
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
    time, modes = _uniform_common_mode_series(time, modes)

    tau = (time - time[0]) / bh_mass
    dtau = float(np.median(np.diff(tau)))
    duration = float(tau[-1] - tau[0])
    window = tukey_window(tau.size, taper_alpha)
    ordered_modes = tuple(sorted(modes))
    mode_data = np.stack([bh_mass * modes[mode] * window for mode in ordered_modes])

    zero_pad_factor = max(float(zero_pad_factor), 1.0)
    nfft = max(tau.size, int(np.ceil(zero_pad_factor * tau.size)))
    padding = nfft - tau.size
    pad_left = padding // 2
    pad_right = padding - pad_left
    mode_data = np.pad(mode_data, ((0, 0), (pad_left, pad_right)))
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
    positive_modes = -mode_fft[:, positive_indices] / omega_squared
    negative_modes = -mode_fft[:, negative_indices] / omega_squared

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


# Detector response and SNR.
DETECTOR_CURVE_DIR = config.REPOSITORY_ROOT / "detector_curves"

DETECTOR_BOUNDS = {
    "ligo": (5.0, 2.5e3),
    "ce": (5.0, 5.0e3),
    "decigo": (1.0e-2, 10.0),
    "lisa": (1.0e-4, 1.0),
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
    return tuple(name for name in config.DETECTABILITY_ACTIVE_DETECTORS if name in DETECTOR_BOUNDS)


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


def _horizon_curves(spectra, curves, cosmology):
    masses = np.logspace(
        np.log10(config.DETECTABILITY_HORIZON_MASS_RANGE_MSUN[0]),
        np.log10(config.DETECTABILITY_HORIZON_MASS_RANGE_MSUN[1]),
        int(config.DETECTABILITY_HORIZON_MASS_SAMPLES),
    )
    samples = max(24, int(config.DETECTABILITY_HORIZON_REDSHIFT_SAMPLES))
    redshift = np.unique(
        np.concatenate(
            (
                np.geomspace(1.0e-6, 0.1, samples),
                np.linspace(0.1, config.DETECTABILITY_HORIZON_REDSHIFT_MAX, samples),
            )
        )
    )
    distance = cosmology.luminosity_distance_mpc(redshift)
    result = {detector: {} for detector in active_detectors()}
    for sim_name, spectrum in spectra.items():
        binned = _bin_spectral_power(spectrum, config.DETECTABILITY_HORIZON_SPECTRAL_BINS)
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
                detected = np.flatnonzero(snr >= config.DETECTABILITY_SNR_THRESHOLD)
                if detected.size == 0:
                    continue
                last = int(detected[-1])
                z_horizon = redshift[last]
                if last + 1 < redshift.size and snr[last + 1] < config.DETECTABILITY_SNR_THRESHOLD:
                    y0 = np.log(max(snr[last], np.finfo(float).tiny))
                    y1 = np.log(max(snr[last + 1], np.finfo(float).tiny))
                    target = np.log(config.DETECTABILITY_SNR_THRESHOLD)
                    if not np.isclose(y0, y1):
                        fraction = np.clip((target - y0) / (y1 - y0), 0.0, 1.0)
                        z_horizon += fraction * (redshift[last + 1] - redshift[last])
                horizon[mass_index] = cosmology.luminosity_distance_mpc(z_horizon)
            result[detector][sim_name] = horizon
    return masses, result
