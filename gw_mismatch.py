"""Independent complex-(2,2) mismatch and a public-API kuibit cross-check.

This opt-in diagnostic reads FFI strain caches; it is separate from the
direct-Psi4, orientation-averaged detectability calculation.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal.windows import tukey


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_pair(times1, strain1, times2, strain2, start=1000., end=None,
                 alpha=0.05):
    """Common physical interval, common uniform grid, demean, taper, zero-pad.

    Inputs use t/M_BH and r(h_plus-i h_cross)/M_BH. No time stretching,
    amplitude fitting, extrapolation, or synthetic continuation is performed.
    """
    arrays = []
    for time, strain in ((times1, strain1), (times2, strain2)):
        time, strain = np.asarray(time, float), np.asarray(strain, complex)
        if (time.ndim != 1 or strain.shape != time.shape or len(time) < 16
                or not np.all(np.isfinite(time))
                or not np.all(np.isfinite(strain)) or np.any(np.diff(time) <= 0)):
            raise ValueError("Need finite, increasing time arrays and matching strains")
        arrays.append((time, strain))
    if not 0 <= alpha <= 1 or not np.isfinite(start):
        raise ValueError("Invalid taper or start time")
    lo = max(start, times1[0], times2[0])
    hi = min(times1[-1], times2[-1], np.inf if end is None else end)
    step = max(np.max(np.diff(t)) for t, _ in arrays)
    count = int(np.floor((hi - lo) / step)) + 1
    if count < 16:
        raise ValueError("Common retained interval has fewer than 16 samples")
    time = np.linspace(lo, hi, count)
    dt = float(time[1] - time[0])
    # At least 2N-1 samples prevents circular wrap in the linear-lag range.
    nfft = 1 << (2 * count - 2).bit_length()
    window = tukey(count, alpha)
    values = []
    for t, h in arrays:
        resampled = np.interp(time, t, h)
        values.append(np.pad((resampled - resampled.mean()) * window,
                             (0, nfft - count)))
    return dt * np.arange(nfft), values[0], values[1], {
        "start_MBH": float(lo), "end_MBH": float(hi),
        "duration_MBH": float(hi - lo), "dt_MBH": dt,
        "retained_samples": count, "fft_samples": nfft,
        "tukey_alpha": alpha, "demean_before_taper": True,
        "interpolation": "linear, no finer than coarsest input step",
    }


class ManualOverlap:
    """Two-sided complex Fourier overlap, independent of kuibit internals.

    An even PSD makes this exactly equivalent to the sum of positive-frequency
    inner products for two equally weighted, orthogonal polarization channels.
    """

    def __init__(self, h1, h2, dt, fmin, fmax, psd=None):
        h1, h2 = np.asarray(h1, complex), np.asarray(h2, complex)
        if (h1.shape != h2.shape or h1.ndim != 1 or len(h1) < 16
                or not np.all(np.isfinite(h1)) or not np.all(np.isfinite(h2))
                or not np.isfinite(dt) or dt <= 0
                or not 0 < fmin < fmax < 0.5 / dt):
            raise ValueError("Invalid strains, sampling, or frequency band")
        freq = np.fft.fftfreq(len(h1), dt)
        keep = (np.abs(freq) >= fmin) & (np.abs(freq) <= fmax)
        if np.count_nonzero(keep) < 4:
            raise ValueError("Frequency band contains fewer than two positive bins")
        self.frequency = freq[keep]
        noise = np.ones(keep.sum()) if psd is None else psd(np.abs(self.frequency))
        noise = np.asarray(noise, float)
        if noise.shape != self.frequency.shape or np.any(~np.isfinite(noise)) or np.any(noise <= 0):
            raise ValueError("PSD must be finite and positive on every retained bin")
        a, b = np.fft.fft(h1)[keep], np.fft.fft(h2)[keep]
        norm = np.sqrt(np.sum(np.abs(a)**2 / noise) * np.sum(np.abs(b)**2 / noise))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Zero or invalid band-limited waveform norm")
        self.product = a * b.conj() / noise / norm
        self.full_product = np.zeros(len(h1), complex)
        self.full_product[keep] = self.product
        self.dt = dt

    def correlation(self, shifts):
        shifts = np.atleast_1d(shifts)
        result = np.empty(len(shifts), complex)
        for first in range(0, len(shifts), 32):
            last = first + 32
            phase = np.exp(2j * np.pi * shifts[first:last, None] * self.frequency)
            result[first:last] = phase @ self.product
        return result

    def grid(self, shifts, phase_parameters):
        corr = self.correlation(shifts)
        # kuibit 1.6.1 applies exp(i*pi*p), with p on [0,2*pi].
        actual_phases = np.pi * np.asarray(phase_parameters)
        overlaps = np.abs(np.real(np.exp(-1j * actual_phases[:, None]) * corr))
        ip, it = np.unravel_index(np.argmax(overlaps), overlaps.shape)
        return {"mismatch": float(1 - overlaps[ip, it]),
                "time_shift_MBH": float(shifts[it]),
                "rotation_rad": float(actual_phases[ip] % (2*np.pi))}, np.abs(corr)

    def optimum(self, limit):
        # Search every sample lag across the entire allowed interval first,
        # then refine each of the strongest local maxima by direct quadrature.
        corr = np.fft.ifft(self.full_product) * len(self.full_product)
        shifts = np.fft.fftfreq(len(corr)) * len(corr) * self.dt
        keep = np.abs(shifts) <= limit
        x, y = shifts[keep], np.abs(corr[keep])
        order = np.argsort(x)
        x, y = x[order], y[order]
        peaks = np.r_[0, np.flatnonzero((y[1:-1] >= y[:-2]) & (y[1:-1] >= y[2:])) + 1, len(y)-1]
        candidates = list(zip(x[peaks], y[peaks]))
        candidates.extend((edge, abs(self.correlation([edge])[0])) for edge in (-limit, limit))
        for t, _ in sorted(candidates, key=lambda item: item[1], reverse=True)[:16]:
            a, b = max(-limit, t-self.dt), min(limit, t+self.dt)
            result = minimize_scalar(lambda lag: -abs(self.correlation([lag])[0]),
                                     bounds=(a, b), method="bounded",
                                     options={"xatol": 1e-8})
            candidates.append((float(result.x), float(-result.fun)))
        lag, match = max(candidates, key=lambda item: item[1])
        rotation = np.angle(self.correlation([lag])[0])
        return {"mismatch": float(1-match), "time_shift_MBH": float(lag),
                "rotation_rad": float(rotation),
                "near_search_boundary": bool(abs(lag) >= 0.98*limit)}, (x, y)


def kuibit_check(time, h1, h2, fmin, fmax, psd, start, end, ntime, nphase):
    """Call the actual public API; no monkey-patching or internal core calls."""
    try:
        from kuibit import gw_mismatch as kgm
        from kuibit.frequencyseries import FrequencySeries
        from kuibit.timeseries import TimeSeries
    except ImportError as exc:
        raise RuntimeError("Install kuibit: uv run --with kuibit==1.6.1 python -m wip_plots.gw_mismatch") from exc
    noises = None
    if psd is not None:
        frequency = np.fft.fftfreq(len(time), time[1]-time[0])
        frequency = np.sort(frequency[frequency >= 0])
        # Supply exact FFT-bin PSD values: no additional interpolation choices.
        values = psd(np.maximum(frequency, fmin))
        noise = FrequencySeries(frequency, values)
        noises = [noise, noise]
    mismatch, (p, lag) = kgm.mismatch_from_strains(
        TimeSeries(time, h1), TimeSeries(time, h2), fmin=fmin, fmax=fmax,
        noises=noises, antenna_patterns=[(0., 1.), (1., 0.)],
        num_time_shifts=ntime, num_polarization_shifts=nphase,
        time_shift_start=start, time_shift_end=end, force_numba=False,
    )
    return {"mismatch": float(mismatch), "time_shift_MBH": float(lag),
            "phase_parameter": float(p), "rotation_rad": float(np.pi*p % (2*np.pi)),
            "kuibit_version": importlib.metadata.version("kuibit"),
            "kuibit_source_sha256": sha256(kgm.__file__)}


def compare(time, h1, h2, fmin, fmax, limit, psd=None, ntime=257, nphase=128):
    """Independent broad and refined grid checks, plus manual phase maximization."""
    if ntime < 5 or nphase < 4 or not np.isfinite(limit) or limit <= 0:
        raise ValueError("Need positive shift limit, >=5 times and >=4 phases")
    manual = ManualOverlap(h1, h2, time[1]-time[0], fmin, fmax, psd)
    optimum, curve = manual.optimum(limit)
    runs = []
    start, end = -limit, limit
    for label, npol in (("broad", nphase), ("refined", 2*nphase)):
        shifts = np.r_[0., np.linspace(start, end, ntime-1)]
        phases = np.linspace(0., 2*np.pi, npol)
        independent, _ = manual.grid(shifts, phases)
        library = kuibit_check(time, h1, h2, fmin, fmax, psd, start, end, ntime, npol)
        difference = abs(independent["mismatch"] - library["mismatch"])
        if difference > 1e-9:
            raise RuntimeError(f"Manual/kuibit grid disagreement {difference:.3g}; inspect conventions")
        runs.append({"search": label, "time_start_MBH": start, "time_end_MBH": end,
                     "num_time_shifts": ntime, "num_phase_shifts": npol,
                     "manual_same_grid": independent, "kuibit": library,
                     "absolute_difference": difference})
        spacing = (end-start)/(ntime-2)
        # Independently found coarse kuibit maximum centers its own refinement.
        start = max(-limit, library["time_shift_MBH"] - 2*spacing)
        end = min(limit, library["time_shift_MBH"] + 2*spacing)
    return {"manual_optimized": optimum, "grid_checks": runs,
            "kuibit_minus_manual_optimized": runs[-1]["kuibit"]["mismatch"]-optimum["mismatch"]}, curve
