"""Finite-signal detectability: the human seven-step workflow in one place.

1. Read the selected r*Psi4 modes on their matching retarded-time grid.
2. Tukey-window and FFT; reconstruct the two polarizations with spin -2 Y_lm.
3. Take the solid-angle mean (or RMS) of the polarization amplitude.
4. Divide by (2*pi*f)^2 and form h_c = 2*f*abs(h_tilde).
5. Read detector PSD/ASD and form h_n = sqrt(f*S_n).
6. Apply source BH mass, redshift and luminosity-distance scaling.
7. Integrate SNR and solve for the chosen detection threshold.

Moore, Cole & Berry (2014), equations 16-21:
https://arxiv.org/html/1408.0740v2#S2.SS2
Wessel et al., Sections III.3-III.5 (transients, polarization and detectors):
https://arxiv.org/abs/2011.04077
The window, retained interval, source-orientation average and detector response
are additional choices; Moore's amplitude definitions do not prescribe them.
The production path uses no radial extrapolation, synthetic tail or FFI.
The separate Wessel trial below tests a mass-driven tail on cached FFI strain.
The optional time-on chart differentiates that analytic model back to Psi4;
it never replaces the direct-Psi4 measured baseline with an FFI spectrum.
Formatting and all figure creation live in wip_plots/gw_detectability_all.py.

The reusable spectral object is normalized by the configured initial
central-BH mass scale: tau=t/M_BH and q=r*h/M_BH. A target source-frame BH
mass and cosmological redshift are applied only when constructing an observed
waveform.  The total ADM mass is intentionally absent from these conversions.
"""
from __future__ import annotations

import config
import json

from dataclasses import dataclass
from functools import cached_property
from math import factorial
from typing import Mapping

import numpy as np
from gw import MODES as CACHED_MODES, mode_columns, read_strain_cache, strain_cache_dir


# ANALYSIS CHOICES: edit this block, not the functions below.
# Reviewed on 2026-09-11 for six disks at the fixed test targets in the notes.
# The displayed examples use fixed round source scales, not optimized horizons.
# See the measured
# sensitivity table in notes/KNOWLEDGE.md; these are not universal error bounds.
# Window/time/frequency bounds below refer to the default four-mode subset.
# Simulation paths and mass/orbit metadata stay in config.py.

# 1. Observable and extraction sphere.
# Deliberately the non-axisymmetric quadrupole subset, NOT total GW emission.
# Adding (2,0) changes SNR by factors 20-164: its origin is not established,
# so exclusion is an explicit scope choice, not a claim to remove only noise.
SIM_NAMES = ("A1", "A2", "A3")  # Detectability figures; other workflows keep A/B.
MODES = ((2, 2), (2, 1), (2, -2), (2, -1))  # Or "all" for every cached mode (ell=2..4).
FIRST_RADIUS_INDEX = config.GW_FIRST_WAVEZONE_PARFILE_INDEX
OUTER_RADIUS_INDEX = config.GW_OUTERMOST_PARFILE_INDEX
# Use the outer finite sphere as the central result, inner/intermediate as
# checks. Radius changes still shift SNR appreciably; no justified infinity fit
# exists here. Keep the measured finite signal; do not invent a decaying tail.

# 2. Retained time and Tukey window before FFT.
# Remove t_ret<0, then this duration from the first arrived sample. Wessel
# Sec. III.5 motivates 1000 M_BH by initial relaxation; 500-1500 changes our
# tested SNRs by <=3.2%. This is a sensitivity check, not proof all junk is gone.
TRANSIENT_CUTOFF_MBH = 1000.0
# Symmetric 5% taper (2.5% at each end), as in the collaborator's script.
# Avoid discarding more measured signal without evidence: alpha=.02-.10
# changes SNR by -15.0% to +8.0%; alpha=.20 suppresses it by up to 27.2%.
TAPER_ALPHA = 0.05

# 3. Polarization amplitude sqrt((|FFT(Re)|^2+|FFT(Im)|^2)/2), then solid-angle
# mean. This follows the collaborator's observable; "rms" is also available.
# Mean-spectrum SNR is NOT an orientation-averaged SNR (RMS raises it 5-7%).
# The optional direction comparison uses Wessel's fixed (theta,phi)=(pi/2.34,0).
SOURCE_AVERAGING = "mean"

# 4. Direct Psi4/(2*pi*f)^2 -> h_c=2*f*|h_tilde| (Moore Eqs. 16-21).
# Reject DC and the first few poorly resolved cycles: f_min=3/duration.
# Provisional: reducing 3 to 1 changes SNR <=3.2%, but raising it to 6 lowers
# CE SNR by up to 35.5%. Do not call the low-frequency band converged.
# Keep all remaining positive FFT bins below Nyquist, with no extra high-pass,
# low-pass, detrending or amplitude-based trimming in the SNR calculation.
LOW_FREQUENCY_CYCLES = 3.0

# 5. Noise model and integration support in observed Hz, NOT source cutoffs.
# Input quantities are explicit. With Eq. 8 h_res, the sky/polarization-averaged
# response of a right-angle interferometer is (2/5) h_res^2: instrument ASD
# times sqrt(5/2). Wessel footnote 4 states sqrt(10) for a raw instrument ASD;
# with the same Eq. (8) waveform, that curve would halve the computed SNR.
# Same effective noise for plots and SNR; see effective_detector_asd.
ACTIVE_DETECTORS = ("ligo", "et", "ce", "decigo", "lisa")
DETECTOR_CURVE_DIR = config.REPOSITORY_ROOT / "detector_curves"
DETECTOR_TABLES = {
    "ligo": ("AplusDesign.txt", "asd", 1),
    "et": ("ET10kmcolumns.txt", "psd", 3),  # CoBA 2023, 90-degree equivalent.
    "ce": ("CE2_40km_strain.txt", "asd", 1),
    "lisa": ("LISA_Alloc_Sh.txt", "psd", 1),
}  # DECIGO uses the analytic Yagi-Seto Eq. (5) below.
DETECTOR_BOUNDS = {
    "ligo": (5.0, 2.5e3), "et": (1.0, 1.0e4), "ce": (5.0, 5.0e3),
    "decigo": (1.0e-2, 10.0), "lisa": (1.0e-4, 1.0),
}

# 6. Fixed illustrative sources: (detector band, source M_BH/Msun, D_L/Mpc).
# Stellar, intermediate and massive BH scales; shared by the selected simulations.
# The ground-based example uses 100 Mpc, matching the Shibata reference below.
# These are fixed comparison choices, not known sources, detection guarantees
# or rate forecasts. Do not retune them when new data arrive.
# Rescale using the initial BH mass, NOT ADM mass: f_obs ~ 1/[M_BH(1+z)].
EXAMPLE_TARGETS = (
    ("ce", 50.0, 100.0),
    ("decigo", 1.0e3, 500.0),
    ("lisa", 1.0e5, 50.0),
)
# Fixed reference scale from Shibata et al. (2021), arXiv:2101.05440.
# Not GW190521's inferred parameters or a waveform/model fit to that event.
SHIBATA_BH_MASS_MSUN = 50.0
SHIBATA_DISTANCE_MPC = 100.0
# Retained flat matter+Lambda approximation; no radiation/neutrino evolution.
# These Planck18 parameter values do NOT make this the full Planck18 model.
COSMOLOGY_H0_KM_S_MPC = 67.66
COSMOLOGY_OMEGA_M = 0.30966

# 7. SNR^2 = integral (h_c/h_n)^2 d ln f; conventional threshold as in Wessel.
# Report a threshold distance for the selected observable, not a rate or a
# total-signal detection guarantee. Horizon masses/redshifts are search bounds.
SNR_THRESHOLD = 8.0
HORIZON_MASS_RANGE_MSUN = (1.0, 1.0e7)
HORIZON_REDSHIFT_MAX = 10.0

# Numerical resolution and comparison choices.
ZERO_PAD_FACTOR = 2.0  # Denser frequency sampling, not longer physical duration.
THETA_NODES, PHI_NODES = 24, 48  # Halving both changes tested SNRs by <0.02%.
HORIZON_MASS_SAMPLES = 44
HORIZON_REDSHIFT_SAMPLES = 64
HORIZON_SPECTRAL_BINS = 2048  # Tested SNR differs from full-grid integral by <1%.
VALIDATION_RADIUS_INDICES = (FIRST_RADIUS_INDEX, 7, OUTER_RADIUS_INDEX)
VALIDATION_TAPER_ALPHAS = (0.02, TAPER_ALPHA, 0.10)
VALIDATION_TRANSIENT_CUTOFFS_MBH = (500.0, TRANSIENT_CUTOFF_MBH, 1500.0)
VALIDATION_COARSE_ANGULAR_GRID = (12, 24)

# Optional Wessel Eq. (10) trial, separate from the direct-Psi4 results above.
# Only complex (2,2); no inferred negative-m partner or other-mode tails.
# These fit-window/acceptance choices are ours, not prescribed by the paper.
TAIL_MASS_SOURCE = "horizon_flux"  # Or "outside_mass" for the original Wessel-like fit.
# Flux proxy = initial outside-AH M0 minus recorded cumulative inward rest-mass
# flux. This neglects non-AH losses/sources and is NOT a BH-energy measurement.
TAIL_FIT_START_FRACTION = 0.60  # Last 40% of retained measured strain.
TAIL_WINDOW_FRACTIONS = (0.40, 0.50, 0.60, 0.70, 0.80)  # Report sensitivity; never pick the best fit.
TAIL_HOLDOUT_FRACTION = 0.25  # Withhold last quarter of that interval, then refit.
TAIL_REMAINING_MASS_FRACTION = 0.10  # Wessel: 90% of initial disk mass accreted.
TAIL_JOIN_ORBITS, TAIL_TAPER_ORBITS = 1.0, 3.0
TAIL_FIT_LIMITS = dict(mass=0.01, amplitude=0.25, phase=0.5, holdout=0.5)
# Fractional RMS errors except phase (radians); not statistical confidence.
TAIL_MAX_SAMPLES = 2_000_000  # Abort, never shorten a tail silently.
# Explicit conditional example, NOT a validated prediction. A1 has a persistent
# late m=1 structure and positive fitted mass decay, but a poor coherent fit.
# Retain/report those waveform failures; never waive mass or sampling checks.
TAIL_REVIEW_CASES = ("A1",)  # Empty tuple disables conditional examples.


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


@dataclass
class DetectabilityResult:
    sources: dict
    inner: dict
    representative: dict
    noise: dict
    targets: list[ObservedTarget]
    horizons: tuple | None
    shibata: ObservedTarget


def analyze(names=None, *, compare_radius=False, compare_direction=False, compute_horizons=True):
    """Run steps 1-7; return numerical results, without creating any figures."""
    print(f"Detectability: modes={MODES}, angular {SOURCE_AVERAGING}, "
          f"cut={TRANSIENT_CUTOFF_MBH:g} M_BH, Tukey alpha={TAPER_ALPHA:g}, "
          f"f_min={LOW_FREQUENCY_CYCLES:g}/duration; finite radius, no tail", flush=True)
    # Steps 1-4: modes -> windowed FFT -> angular average -> dimensionless strain.
    sources, representative = source_spectra(names, OUTER_RADIUS_INDEX, compare_direction)
    if not sources:
        raise ValueError("No valid direct-Psi4 spectra were produced")
    inner = source_spectra(names, FIRST_RADIUS_INDEX)[0] if compare_radius else {}
    spectra = {name: source.spectrum for name, source in sources.items()}

    # Step 5: detector noise. Use exactly the same effective ASD for h_n and SNR.
    curves = load_detector_curves()
    noise = {}
    for detector in active_detectors():
        low, high = DETECTOR_BOUNDS[detector]
        frequency = np.geomspace(low, high, 500)
        noise[detector] = frequency, np.sqrt(frequency) * effective_detector_asd(detector, frequency, curves)

    # Step 6: fixed source choices; redshift follows the same cosmology as SNR.
    cosmology = FlatLambdaCDM(z_max=max(12.0, HORIZON_REDSHIFT_MAX + 0.5))
    targets = []
    for detector, mass, distance in EXAMPLE_TARGETS:
        if detector not in active_detectors():
            raise ValueError(f"Example detector {detector} is not in ACTIVE_DETECTORS")
        redshift = cosmology.redshift_at_luminosity_distance(distance)
        targets.append(observed_target(spectra, mass, distance, redshift, curves, detector))

    redshift = cosmology.redshift_at_luminosity_distance(SHIBATA_DISTANCE_MPC)
    shibata = observed_target(spectra, SHIBATA_BH_MASS_MSUN, SHIBATA_DISTANCE_MPC,
                             redshift, curves)

    # Step 7: SNR^2 = integral (h_c/h_n)^2 d ln f; threshold luminosity distance.
    horizons = horizon_curves(spectra, curves, cosmology) if compute_horizons else None
    return DetectabilityResult(sources, inner, representative, noise, targets, horizons, shibata)


def observed_target(spectra, mass, distance, redshift, curves, detector=None):
    """Apply one shared physical source scale to every case, without tuning it."""
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


# 1. Read matching, uniformly sampled r*Psi4 and retarded time.
def read_rpsi4_modes(case, radius_index, modes=None):
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
    directory = strain_cache_dir(case, label)
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


def spectrum_for_case(case, time, modes, **overrides):
    options = dict(
        transient_cutoff_mbh=TRANSIENT_CUTOFF_MBH, taper_alpha=TAPER_ALPHA,
        zero_pad_factor=ZERO_PAD_FACTOR, low_frequency_cycles=LOW_FREQUENCY_CYCLES,
        theta_nodes=THETA_NODES, phi_nodes=PHI_NODES, averaging=SOURCE_AVERAGING,
    )
    options.update(overrides)
    return direct_psi4_spectrum(time, modes, case.mlittle, **options)


def source_spectra(names=None, radius_index=OUTER_RADIUS_INDEX, representative=False):
    sources, directions = {}, {}
    for case in config.all_sim_configs(SIM_NAMES if names is None else names):
        try:
            time, modes = read_rpsi4_modes(case, radius_index)
            spectrum, info = spectrum_for_case(case, time, modes)
            sources[case.name] = SourceSpectrum(case, spectrum, info)
            if representative:
                directions[case.name] = spectrum_for_case(
                    case, time, modes, representative_direction=(np.pi / 2.34, 0.0)
                )[0]
            print(f"{case.name}: radius index {radius_index}, {len(info.modes)} modes, "
                  f"{info.samples} samples over {info.duration_mbh:.1f} M_BH", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            print(f"{case.name}: skipping detectability; {exc}", flush=True)
    return sources, directions


# 2-4. Common preprocessing, windowed FFT, polarization/orientation and strain.
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

    time, modes = retained_psi4_modes(time, rpsi4_modes, bh_mass,
                                      transient_cutoff_mbh, trim_prearrival)

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

    # Step 4: the common division commutes with the amplitude averages above.
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


# 5. Detector PSD/ASD conversion.
@dataclass(frozen=True)
class DetectorCurve:
    frequency: np.ndarray
    asd: np.ndarray


@dataclass(frozen=True)
class BinnedSpectralPower:
    frequency: np.ndarray
    power_dfrequency: np.ndarray


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
    return {name: _read_detector_curve(filename, quantity, column)
            for name, (filename, quantity, column) in DETECTOR_TABLES.items()}


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
    Wessel footnote 4 instead describes multiplying the 5*S_n curve by 2
    (sqrt(10) in ASD). For the same raw PSD and Eq. (8) waveform, that stated
    curve would halve the computed SNR. Tests pin our averaged response directly.
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


# 6. Cosmology and mass/distance scaling. M_BH is not M_ADM.
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


# 7. SNR integration and threshold horizons.
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
    result = {detector: {} for detector in active_detectors()}
    for sim_name, spectrum in spectra.items():
        binned = _bin_spectral_power(spectrum, HORIZON_SPECTRAL_BINS)
        for detector in active_detectors():
            horizon = np.full_like(masses, np.nan)
            for mass_index, source_mass in enumerate(masses):
                z_horizon = _threshold_redshift(binned, detector, source_mass, curves, cosmology)
                if np.isfinite(z_horizon):
                    horizon[mass_index] = cosmology.luminosity_distance_mpc(z_horizon)
            result[detector][sim_name] = horizon
    return masses, result


# Numerical comparisons use the same analysis functions, never a second pipeline.
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


# Optional Wessel trial: isolated from the production seven-step path above.
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
