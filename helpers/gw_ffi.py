"""Pure NumPy implementation of the established Psi4-to-strain workflow.

The numerical conventions intentionally follow
``psi4_hlm_ref/ccc_ffi_hplus_hcross_ejkick.f90``. In particular, the input is
converted to gauge-corrected retarded time, resampled with local four-point
polynomial interpolation, and integrated twice with the mode-dependent
fixed-frequency cutoff of Reisswig and Pollney (2011).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FFIProducts:
    """Arrays written to the reusable GW cache."""

    rhphc: np.ndarray
    rhphcdot: np.ndarray
    omega22: np.ndarray
    ejv_gw: np.ndarray
    times: np.ndarray
    rpsi4_uniform: np.ndarray
    fft_size: int
    dt: float


def _padded_fft_size(sample_count: int) -> int:
    """Match the deliberately generous power-of-two padding in the Fortran."""
    if sample_count < 2:
        raise ValueError("Psi4 conversion requires at least two samples")
    exponent = int(np.log(float(sample_count)) / np.log(2.0) - 1.0e-10) + 3
    return 2**exponent


def _retarded_time(data: np.ndarray, madm: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the legacy uniform coordinate-time grid, retarded time, and dt."""
    sample_count = data.shape[0]
    if sample_count < 4:
        raise ValueError("Psi4 conversion requires at least four samples")

    t0 = float(data[0, 0])
    tend = float(data[-1, 0])
    dt = (tend - t0) / (sample_count - 1)
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("Psi4 coordinate times must span a positive interval")
    coordinate_time = t0 + dt * np.arange(sample_count, dtype=float)

    radius = np.asarray(data[:, -4], dtype=float)
    gtt = np.asarray(data[:, -3], dtype=float)
    gtr = np.asarray(data[:, -2], dtype=float)
    grr = np.asarray(data[:, -1], dtype=float)
    discriminant = gtr * gtr - gtt * grr
    lapse_factor = 1.0 - 2.0 * madm / radius
    if (
        np.any(~np.isfinite(radius))
        or np.any(radius <= 2.0 * madm)
        or np.any(discriminant < 0.0)
        or np.any(gtt == 0.0)
        or np.any(lapse_factor == 0.0)
    ):
        raise ValueError("Invalid radius or inverse-metric data in Psi4 input")

    dtsch_dt = (gtr - np.sqrt(discriminant)) / gtt / lapse_factor
    schwarzschild_time = np.empty(sample_count, dtype=float)
    schwarzschild_time[0] = t0
    schwarzschild_time[1:] = t0 + np.cumsum(
        0.5 * (dtsch_dt[:-1] + dtsch_dt[1:]) * dt
    )
    rstar = radius + 2.0 * madm * np.log(radius / (2.0 * madm) - 1.0)
    tret = schwarzschild_time - rstar
    if np.any(~np.isfinite(tret)) or np.any(np.diff(tret) <= 0.0):
        raise ValueError("Gauge-corrected retarded time is not finite and increasing")
    times = np.column_stack((coordinate_time, schwarzschild_time, tret, rstar))
    return times, tret, dt


def _four_point_interpolate(
    source_time: np.ndarray,
    source_values: np.ndarray,
    target_time: np.ndarray,
) -> np.ndarray:
    """Local cubic interpolation matching the Fortran ``polint`` stencil."""
    if source_time.size < 4:
        raise ValueError("Four-point interpolation requires at least four samples")
    interval = np.searchsorted(source_time, target_time, side="right") - 1
    starts = np.clip(interval - 1, 0, source_time.size - 4)
    indices = starts[:, None] + np.arange(4)[None, :]
    x = source_time[indices]
    y = source_values[indices]

    weights = np.ones_like(x)
    for column in range(4):
        for other in range(4):
            if column != other:
                weights[:, column] *= (
                    (target_time - x[:, other])
                    / (x[:, column] - x[:, other])
                )
    return np.sum(weights[:, :, None] * y, axis=1)


def _fixed_frequency_integrate(
    rpsi4: np.ndarray,
    dt: float,
    modes: tuple[tuple[int, int], ...],
    omega_orbital: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate complex ``r Psi4`` once and twice using the legacy FFI filter."""
    sample_count = rpsi4.shape[0]
    spectrum = np.fft.ifft(rpsi4, axis=0) * sample_count
    index = np.arange(sample_count)
    signed_index = np.where(index < sample_count // 2, index, index - sample_count)
    omega = 2.0 * np.pi * signed_index / (sample_count * dt)

    hdot_spectrum = np.empty_like(spectrum)
    h_spectrum = np.empty_like(spectrum)
    for mode_index, (_, emm) in enumerate(modes):
        cutoff = max(abs(emm) * omega_orbital, omega_orbital)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError("FFI orbital-frequency cutoff must be positive")
        effective_abs = np.maximum(np.abs(omega), cutoff)
        effective_signed = np.copysign(effective_abs, omega)
        effective_signed[0] = cutoff
        hdot_spectrum[:, mode_index] = 1j * spectrum[:, mode_index] / effective_signed
        h_spectrum[:, mode_index] = -spectrum[:, mode_index] / effective_abs**2

    hdot = np.fft.fft(hdot_spectrum, axis=0) / sample_count
    strain = np.fft.fft(h_spectrum, axis=0) / sample_count
    return strain, hdot


def _mode_table(time: np.ndarray, values: np.ndarray) -> np.ndarray:
    table = np.empty((time.size, 1 + 2 * values.shape[1]), dtype=float)
    table[:, 0] = time
    table[:, 1::2] = values.real
    table[:, 2::2] = values.imag
    return table


def _omega22(
    time: np.ndarray,
    rpsi4: np.ndarray,
    strain: np.ndarray,
    strain_dot: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Reproduce the three legacy estimates of the (2,2) angular frequency."""
    rows = []
    phase = np.angle(strain[:, 0])
    for i in range(time.size):
        if abs(rpsi4[i, 0]) == 0.0:
            continue
        next_phase = np.angle(strain[i + 1, 0])
        previous_phase = phase[0] if i == 0 else phase[i - 1]
        delta = next_phase - previous_phase
        if delta < -np.pi:
            delta += 2.0 * np.pi
        if delta > np.pi:
            delta -= 2.0 * np.pi
        if i > 0:
            delta *= 0.5
        with np.errstate(divide="ignore", invalid="ignore"):
            estimate_a = -1j * rpsi4[i, 0] / strain_dot[i, 0]
            estimate_b = -1j * strain_dot[i, 0] / strain[i, 0]
        rows.append(
            (
                time[i],
                delta / dt,
                estimate_a.real,
                estimate_a.imag,
                estimate_b.real,
                estimate_b.imag,
            )
        )
    return np.asarray(rows, dtype=float).reshape((-1, 6))


def _radiated_diagnostics(
    time: np.ndarray,
    strain: np.ndarray,
    strain_dot: np.ndarray,
    modes: tuple[tuple[int, int], ...],
    madm: float,
    dt: float,
) -> np.ndarray:
    """Port the legacy energy, angular-momentum, and recoil accumulation."""
    mode_index = {mode: index for index, mode in enumerate(modes)}
    factor = dt / (32.0 * np.pi)
    lmax = max(ell for ell, _ in modes) + 1
    sample_count = time.size
    zero = np.zeros(sample_count, dtype=complex)

    def series(ell, emm):
        index = mode_index.get((ell, emm))
        return zero if index is None else strain_dot[:, index]

    mode_power = np.sum(np.abs(strain_dot) ** 2, axis=1)
    angular_density = np.zeros(sample_count, dtype=float)
    xy_density = np.zeros(sample_count, dtype=complex)
    z_density = np.zeros(sample_count, dtype=float)
    for ell in range(2, lmax):
        for emm in range(-ell, ell + 1):
            index = mode_index.get((ell, emm))
            if index is None:
                continue
            hd = strain_dot[:, index]
            angular_density += emm * np.imag(np.conj(hd) * strain[:, index])

            c_plus = -np.sqrt(
                (ell + emm + 1.0)
                * (ell + emm + 2.0)
                * (ell - 1.0)
                * (ell + 3.0)
                / ((2.0 * ell + 1.0) * (2.0 * ell + 3.0))
            ) / (ell + 1.0)
            c_zero = 2.0 * np.sqrt((ell - emm) * (ell + emm + 1.0)) / (ell * (ell + 1.0))
            c_minus = np.sqrt(
                (ell - emm - 1.0)
                * (ell - emm)
                * (ell - 2.0)
                * (ell + 2.0)
                / ((2.0 * ell - 1.0) * (2.0 * ell + 1.0))
            ) / ell
            d_plus = np.sqrt(
                (ell - emm + 1.0)
                * (ell + emm + 1.0)
                * (ell - 1.0)
                * (ell + 3.0)
                / ((2.0 * ell + 1.0) * (2.0 * ell + 3.0))
            ) / (ell + 1.0)
            d_zero = 2.0 * emm / (ell * (ell + 1.0))

            xy_density += np.conj(hd) * (
                c_plus * series(ell + 1, emm + 1)
                + c_zero * series(ell, emm + 1)
                + c_minus * series(ell - 1, emm + 1)
            )
            z_density += (
                2.0 * d_plus * np.real(np.conj(hd) * series(ell + 1, emm))
                + d_zero * np.abs(hd) ** 2
            )

    def trapezoidal_cumulative(density):
        return factor * np.cumsum(density[:-1] + density[1:])

    energy = trapezoidal_cumulative(mode_power)
    angular_momentum = -trapezoidal_cumulative(angular_density)
    px = trapezoidal_cumulative(xy_density.real)
    py = trapezoidal_cumulative(xy_density.imag)
    pz = trapezoidal_cumulative(z_density)
    velocity_scale = 299792.458 / (madm - energy)
    vx, vy, vz = px * velocity_scale, py * velocity_scale, pz * velocity_scale
    flux = mode_power[:-1] / (16.0 * np.pi)
    return np.column_stack(
        (
            time[1:],
            energy,
            angular_momentum,
            np.sqrt(vx * vx + vy * vy + vz * vz),
            vx,
            vy,
            vz,
            flux,
        )
    )


def reconstruct_strain(
    data: np.ndarray,
    modes: tuple[tuple[int, int], ...],
    omega_orbital: float,
    madm: float,
    t_start: float | None = None,
    t_end: float | None = None,
) -> FFIProducts:
    """Create the complete set of legacy-compatible cached waveform products."""
    data = np.asarray(data, dtype=float)
    expected_columns = 1 + 2 * len(modes) + 4
    if data.ndim != 2 or data.shape[1] != expected_columns:
        raise ValueError(f"Psi4 data must have shape (N, {expected_columns})")
    if np.any(~np.isfinite(data)):
        raise ValueError("Psi4 input contains non-finite values")

    times, tret, dt = _retarded_time(data, madm)
    coordinate_time = times[:, 0]
    if t_start is None:
        t_start = float(coordinate_time[0])
    if t_end is None or t_end >= coordinate_time[-1]:
        t_end = float(coordinate_time[-1] - 0.5 * dt)
    start_index = int(np.clip(np.floor((t_start - coordinate_time[0]) / dt), 0, data.shape[0] - 1))
    end_index = int(np.clip(np.ceil((t_end - coordinate_time[0]) / dt), 0, data.shape[0] - 1))
    if end_index < start_index:
        raise ValueError("Requested Psi4 time interval is empty")

    fft_size = _padded_fft_size(data.shape[0])
    uniform_tret = tret[0] + dt * np.arange(fft_size, dtype=float)
    active = (uniform_tret >= tret[start_index]) & (uniform_tret <= tret[end_index])
    raw_modes = data[:, 1 : 1 + 2 * len(modes) : 2] + 1j * data[:, 2 : 1 + 2 * len(modes) : 2]
    # The legacy routine integrates Re(r Psi4) - i Im(r Psi4), then stores its
    # real and imaginary parts as r h_plus and r h_cross.
    legacy_rpsi4 = data[:, -4, None] * np.conj(raw_modes)
    uniform_legacy = np.zeros((fft_size, len(modes)), dtype=complex)
    uniform_legacy[active] = _four_point_interpolate(tret, legacy_rpsi4, uniform_tret[active])

    strain, strain_dot = _fixed_frequency_integrate(
        uniform_legacy,
        dt,
        modes,
        omega_orbital,
    )
    output_time = uniform_tret[: data.shape[0]]
    output_strain = strain[: data.shape[0]]
    output_strain_dot = strain_dot[: data.shape[0]]
    output_rpsi4 = uniform_legacy[: data.shape[0]]

    times_output = np.column_stack(
        (
            times,
            legacy_rpsi4[:, 0].real,
            -legacy_rpsi4[:, 0].imag,
        )
    )
    return FFIProducts(
        rhphc=_mode_table(output_time, output_strain),
        rhphcdot=_mode_table(output_time, output_strain_dot),
        omega22=_omega22(output_time, uniform_legacy, strain, strain_dot, dt),
        ejv_gw=_radiated_diagnostics(output_time, output_strain, output_strain_dot, modes, madm, dt),
        times=times_output,
        rpsi4_uniform=_mode_table(output_time, np.conj(output_rpsi4)),
        fft_size=fft_size,
        dt=dt,
    )


def write_products(
    products: FFIProducts,
    workdir: Path,
    metadata: dict[str, object],
) -> dict[str, object]:
    """Write legacy-compatible products plus reusable preprocessing metadata."""
    workdir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "rhphc.dat": products.rhphc,
        "rhphcdot.dat": products.rhphcdot,
        "omega22.dat": products.omega22,
        "ejv_GW.dat": products.ejv_gw,
        "times.dat": products.times,
        "rpsi4_uniform.dat": products.rpsi4_uniform,
    }
    for name, values in outputs.items():
        np.savetxt(workdir / name, values, fmt="%25.15E")
    manifest = dict(metadata)
    manifest.update(
        {
            "format_version": 1,
            "backend": "python",
            "fft_size": products.fft_size,
            "dt": products.dt,
        }
    )
    (workdir / "strain_cache.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
