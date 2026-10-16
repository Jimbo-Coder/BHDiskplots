#!/usr/bin/env python3
"""Compare two stored GW strain modes, manually and with kuibit.

Run: python gw_mismatch.py
     python gw_mismatch.py --sims A2 A3 --radius-index 6

Read in order: settings and file reader; waveform preparation and the two
mismatch calculations; the complete analysis; then plotting and CLI options.
The default uses simulation code units and white noise. No mass, distance,
or redshift is assigned. Equations and choices: notes/gw_mismatch.md.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.signal.windows import tukey

import config


# 1. SETTINGS AND INPUTS
# Edit these for a normal rerun; CLI overrides are defined at the very bottom.
SIM_NAMES = ("A1", "A3")
RADIUS_INDEX = config.GW_OUTERMOST_PARFILE_INDEX
START_TIME = 50.0                 # Stored retarded time, in code units.
END_TIME = None                   # None: end of the shorter waveform.
TAPER_ALPHA = 0.05
MIN_FREQUENCY = None              # None: 3 / retained duration, in 1/code time.
MAX_FREQUENCY = None              # None: 95% of the Nyquist frequency.
SHIFT_LIMIT = None                # None: half the retained duration, in code time.
DETECTOR = "white"                # Default: no physical detector or mass scale.
SOURCE_BH_MASS_MSUN = None         # Required only for detector-shaped weighting.
REDSHIFT = 0.0
TIME_SEARCH_POINTS = 257          # Kuibit grid; manual also finds a refined peak.
PHASE_SEARCH_POINTS = 128


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_strain_pair(names, cache_root, radius_index):
    """Read stored time and r*(h_plus-i*h_cross)_22 in the requested pair order."""
    cases = {case.name: case for case in config.all_sim_configs()}
    unknown = set(names) - cases.keys()
    if unknown:
        raise ValueError("Unknown simulation(s): " + ", ".join(sorted(unknown)))
    waves, inputs = [], []
    for name in names:
        case = cases[name]
        label = radius_index + case.gw_psi4_file_index_offset
        folder = Path(cache_root) / f"{name}_psi4{label}"
        path = folder / "rhphc.dat"
        manifest = folder / "strain_cache.json"
        for required in (path, manifest):
            if not required.is_file():
                raise FileNotFoundError(
                    f"Missing {required}. Copy the selected GW cache from Anvil "
                    "or run generate_gw.py; see notes/gw_mismatch.md.")
        print(f"{name}: radius index {radius_index} -> Psi4_rad.mon.{label} -> {path}", flush=True)
        data = np.loadtxt(path, usecols=(0, 1, 2), ndmin=2)
        # These are the original code-unit columns: no division by BH mass.
        waves.append((data[:, 0], data[:, 1] - 1j * data[:, 2]))
        inputs.append({
            "case": name,
            "cache": str(path.resolve()),
            "sha256": sha256(path),
            "initial_BH_mass_code": case.mlittle,
            "cache_manifest_sha256": sha256(manifest),
            "cache_manifest": json.loads(manifest.read_text()),
        })
    return waves, inputs


# 2. CALCULATION: common interval -> taper -> Fourier overlap -> best alignment
# Both methods below receive exactly the same prepared complex signals.
def prepare_pair(times1, strain1, times2, strain2, start=0., end=None,
                 alpha=0.05):
    """Common physical interval, common uniform grid, demean, taper, zero-pad.

    Inputs use a common time unit and complex h_plus-i h_cross. No time stretching,
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
        "start": float(lo), "end": float(hi),
        "duration": float(hi - lo), "dt": dt,
        "retained_samples": count, "fft_samples": nfft,
        "tukey_alpha": alpha, "demean_before_taper": True,
        "interpolation": "linear, no finer than coarsest input step",
    }


class ManualOverlap:
    """Two-sided complex Fourier overlap, independent of kuibit internals.

    An even PSD makes this exactly equivalent to the sum of positive-frequency
    inner products for two equally weighted, orthogonal polarization channels.
    With H1=FFT(h1), H2=FFT(h2), the normalized correlation is

        C(lag) = sum[H1 * conj(H2) / PSD * exp(2*pi*i*f*lag)] / (N1*N2).

    The match is max_lag |C(lag)|; mismatch = 1 - match. Taking the absolute
    value maximizes over a constant complex-mode phase analytically.
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
        fft1 = np.fft.fft(h1)[keep]
        fft2 = np.fft.fft(h2)[keep]
        norm1_squared = np.sum(np.abs(fft1)**2 / noise)
        norm2_squared = np.sum(np.abs(fft2)**2 / noise)
        norm = np.sqrt(norm1_squared * norm2_squared)
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Zero or invalid band-limited waveform norm")
        # Common FFT, df and response prefactors cancel against the norms.
        self.product = fft1 * fft2.conj() / noise / norm
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
                "time_shift": float(shifts[it]),
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
        return {"mismatch": float(1-match), "time_shift": float(lag),
                "rotation_rad": float(rotation),
                "near_search_boundary": bool(abs(lag) >= 0.98*limit)}, (x, y)


def kuibit_check(time, h1, h2, fmin, fmax, psd, start, end, ntime, nphase):
    """Call the actual public API; no monkey-patching or internal core calls."""
    try:
        from kuibit import gw_mismatch as kgm
        from kuibit.frequencyseries import FrequencySeries
        from kuibit.timeseries import TimeSeries
    except ImportError as exc:
        raise RuntimeError("Install kuibit: uv run --with kuibit==1.6.1 python gw_mismatch.py") from exc
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
    return {"mismatch": float(mismatch), "time_shift": float(lag),
            "phase_parameter": float(p), "rotation_rad": float(np.pi*p % (2*np.pi)),
            "kuibit_version": importlib.metadata.version("kuibit"),
            "kuibit_source_sha256": sha256(kgm.__file__)}


def compare(time, h1, h2, fmin, fmax, limit, psd=None, ntime=257, nphase=128):
    """Run the manual optimum and an independently centered kuibit search.

    The manual result also evaluates kuibit's exact grids: this distinguishes
    a convention error from the finite time/phase resolution of the search.
    """
    if ntime < 5 or nphase < 4 or not np.isfinite(limit) or limit <= 0:
        raise ValueError("Need positive shift limit, >=5 times and >=4 phases")
    manual = ManualOverlap(h1, h2, time[1]-time[0], fmin, fmax, psd)
    optimum, curve = manual.optimum(limit)
    runs = []
    start, end = -limit, limit
    for label, npol in (("broad", nphase), ("refined", 2*nphase)):
        # Include zero exactly, matching kuibit's grid construction.
        shifts = np.r_[0., np.linspace(start, end, ntime-1)]
        phases = np.linspace(0., 2*np.pi, npol)
        independent, _ = manual.grid(shifts, phases)
        library = kuibit_check(
            time, h1, h2, fmin, fmax, psd, start, end, ntime, npol)
        difference = abs(independent["mismatch"] - library["mismatch"])
        if difference > 1e-9:
            raise RuntimeError(f"Manual/kuibit grid disagreement {difference:.3g}; inspect conventions")
        runs.append({"search": label, "time_start": start, "time_end": end,
                     "num_time_shifts": ntime, "num_phase_shifts": npol,
                     "manual_same_grid": independent, "kuibit": library,
                     "absolute_difference": difference})
        spacing = (end-start)/(ntime-2)
        # Independently found coarse kuibit maximum centers its own refinement.
        start = max(-limit, library["time_shift"] - 2*spacing)
        end = min(limit, library["time_shift"] + 2*spacing)
    return {"manual_optimized": optimum, "grid_checks": runs,
            "kuibit_minus_manual_optimized": runs[-1]["kuibit"]["mismatch"]-optimum["mismatch"]}, curve


# 3. COMPLETE ANALYSIS: read the selected pair, prepare it, run both methods

def analyze(args):
    if len(args.sims) != 2:
        raise ValueError("Choose exactly two simulations, e.g. --sims A1 A3")
    if args.mass is not None and (args.mass <= 0 or not np.isfinite(args.mass)):
        raise ValueError("BH mass must be finite and positive")
    if args.redshift < 0 or not np.isfinite(args.redshift):
        raise ValueError("Redshift must be finite and nonnegative")
    if args.tag is not None and (Path(args.tag).name != args.tag or args.tag in (".", "..")):
        raise ValueError("--tag must be one folder name")
    if args.detector != "white" and args.mass is None:
        raise ValueError("Detector weighting needs an explicit --mass; white noise uses code units")

    # Read: original (2,2) FFI strain and its provenance, never raw simulation data.
    waves, inputs = read_strain_pair(args.sims, args.cache_dir, args.radius_index)

    # Prepare: crop to the shared interval, resample, remove means, taper and pad.
    time, h1, h2, selection = prepare_pair(
        *waves[0], *waves[1], start=args.start, end=args.end, alpha=args.taper)
    duration = selection["duration"]
    dt = selection["dt"]
    fmin = 3 / duration if args.fmin is None else args.fmin
    fmax = 0.95 * 0.5 / dt if args.fmax is None else args.fmax
    limit = duration / 2 if args.shift_limit is None else args.shift_limit
    if limit > duration:
        raise ValueError("--shift-limit must not exceed the retained duration")

    # Weight: a constant PSD is the default. Only this optional branch assigns
    # a physical mass and moves the code-unit frequencies onto a detector PSD.
    psd = None
    weighting = {"detector": args.detector, "channels": "equal, orthogonal plus/cross"}
    if args.detector != "white":
        from gw_detectability import (
            DETECTOR_BOUNDS, SECONDS_PER_M_SUN,
            effective_detector_asd, load_detector_curves,
        )
        code_masses = {row["initial_BH_mass_code"] for row in inputs}
        if len(code_masses) != 1:
            raise ValueError("Detector weighting requires the same code BH mass for both signals")
        code_mass = code_masses.pop()
        # One code-time unit in seconds, including cosmological time dilation.
        scale = SECONDS_PER_M_SUN * args.mass * (1 + args.redshift) / code_mass
        low, high = DETECTOR_BOUNDS[args.detector]
        curves = load_detector_curves()
        if args.detector in curves:
            low = max(low, curves[args.detector].frequency[0])
            high = min(high, curves[args.detector].frequency[-1])
        fmin = max(fmin, low * scale)
        fmax = min(fmax, high * scale)

        def psd(frequency):
            return effective_detector_asd(args.detector, frequency / scale, curves)**2

        weighting.update(
            source_BH_mass_Msun=args.mass, redshift=args.redshift,
            mass_time_seconds=scale, band_Hz=[fmin / scale, fmax / scale])

    # Compare: independent Fourier calculation and the installed kuibit package.
    print(f"Interval: [{selection['start']:.9g}, {selection['end']:.9g}] code time; "
          f"band: [{fmin:.9g}, {fmax:.9g}] inverse code time; {args.detector} weighting", flush=True)
    result, (lags, overlaps) = compare(
        time, h1, h2, fmin, fmax, limit, psd,
        ntime=args.num_times, nphase=args.num_phases)
    result.update(
        schema_version=2, status="WIP diagnostic, specified waveform and interval",
        waveform="stored FFI r(h_plus-i h_cross)_22, no mass rescaling",
        units="simulation code time and stored r*h amplitude",
        inputs=inputs, selection=selection, weighting=weighting,
        band_frequency_code=[fmin, fmax], shift_limit=limit,
        search="broad kuibit grid, independent kuibit-centered refinement; manual all-sample lags plus local refinement",
        implementation_sha256={
            "gw_mismatch.py": sha256(__file__),
        })
    manual = result["manual_optimized"]["mismatch"]
    kuibit = result["grid_checks"][-1]["kuibit"]["mismatch"]
    print(f"{' / '.join(args.sims)} mismatch: manual={manual:.10f}; kuibit={kuibit:.10f}", flush=True)
    if result["kuibit_minus_manual_optimized"] > 1e-3:
        print("WARNING: kuibit search not converged; increase --num-times/--num-phases", flush=True)
    if result["manual_optimized"]["near_search_boundary"]:
        print("WARNING: overlap maximum near the allowed time-shift boundary", flush=True)
    return result, (time, h1, h2, lags, overlaps)


# 4. PLOT AND FORMAT: waveform histories above the overlap versus time shift

def plot_comparison(args, result, arrays, output):
    import matplotlib.pyplot as plt
    from helpers.plot_common import setup, savefig

    setup(args)
    time, h1, h2, lags, overlaps = arrays
    selection = result["selection"]
    count = selection["retained_samples"]
    fig, (wave, overlap) = plt.subplots(2, 1, figsize=(7., 5.8))
    fig.subplots_adjust(left=.13, right=.97, bottom=.11, top=.94, hspace=.42)
    for values, color, label in zip((h1, h2), ("#0072B2", "#D55E00"), args.sims):
        # Display normalization only; the original prepared strains enter compare.
        wave.plot(time[:count] + selection["start"],
                  values[:count].real / np.max(np.abs(values)),
                  color=color, lw=.85, label=label)
    wave.set(xlabel=r"$t_{\mathrm{ret}}$ (code units)",
             ylabel=r"$\mathrm{Re}(h_{22})/\max|h_{22}|$")
    wave.legend(loc="upper left", ncol=2, fontsize=10)
    title = " versus ".join(args.sims) + r": finite $(2,2)$ FFI strain; "
    wave.set_title(title + args.detector + " weighting", fontsize=11)
    overlap.plot(lags, overlaps, color="black", lw=1., label="Manual: phase maximized")
    for run, color, marker in zip(result["grid_checks"], ("#009E73", "#D55E00"), ("o", "x")):
        for method in ("manual_same_grid", "kuibit"):
            point = run[method]
            label = None
            if method == "manual_same_grid":
                label = run["search"].capitalize() + ": manual / kuibit"
            overlap.plot(point["time_shift"], 1 - point["mismatch"], marker=marker,
                         ms=7 if method == "manual_same_grid" else 4, mfc="none",
                         color=color, linestyle="none", label=label)
    optimum = result["manual_optimized"]
    ymax = min(1.04, max(.05, 1.65 * (1 - optimum["mismatch"])))
    limit = result["shift_limit"]
    overlap.set(xlabel=r"$\Delta t$ (code units)", ylabel="Normalized overlap",
                ylim=(0, ymax), xlim=(-limit, limit))
    overlap.legend(loc="upper right", fontsize=9)
    overlap.text(.02, .97, f"Manual mismatch: {optimum['mismatch']:.6f}\n"
                 f"Kuibit mismatch: {result['grid_checks'][-1]['kuibit']['mismatch']:.6f}",
                 transform=overlap.transAxes, va="top", fontsize=10)
    if not args.no_save:
        fig.savefig((args.outdir / output).with_suffix(".pdf"), bbox_inches=None)
    savefig(fig, args, output.with_suffix(".png"))


# CLI overrides live last. Normal scientific choices are in the settings above.
def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sims", nargs=2, default=SIM_NAMES, metavar=("FIRST", "SECOND"))
    parser.add_argument("--radius-index", type=int, choices=range(9), default=RADIUS_INDEX,
                        help="Zero-based parfile index; A-case raw file label = index + 1")
    parser.add_argument("--cache-dir", type=Path, default=config.GW_WORK_ROOT)
    parser.add_argument("--outdir", type=Path, default=config.PLOTS_DIR)
    parser.add_argument("--tag", help="Optional output subfolder for a parameter variant")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--no-plot", action="store_true", help="Save numbers without requiring TeX")
    selection = parser.add_argument_group("interval and frequency band (simulation code units)")
    selection.add_argument("--start", type=float, default=START_TIME)
    selection.add_argument("--end", type=float, default=END_TIME)
    selection.add_argument("--taper", type=float, default=TAPER_ALPHA)
    selection.add_argument("--fmin", type=float, default=MIN_FREQUENCY)
    selection.add_argument("--fmax", type=float, default=MAX_FREQUENCY)
    selection.add_argument("--shift-limit", type=float, default=SHIFT_LIMIT)
    detector = parser.add_argument_group("optional detector weighting (requires a physical mass)")
    detector.add_argument("--detector", choices=("white", "ligo", "et", "ce", "decigo", "lisa"), default=DETECTOR)
    detector.add_argument("--mass", type=float, default=SOURCE_BH_MASS_MSUN)
    detector.add_argument("--redshift", type=float, default=REDSHIFT)
    search = parser.add_argument_group("kuibit search resolution")
    search.add_argument("--num-times", type=int, default=TIME_SEARCH_POINTS)
    search.add_argument("--num-phases", type=int, default=PHASE_SEARCH_POINTS)
    args = parser.parse_args(argv)
    args.sims = [name.upper() for name in args.sims]

    try:
        result, arrays = analyze(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    # Names and stored arrays follow the requested pair, including reversed order.
    pair = "_".join(args.sims)
    output = Path("wip/gw_mismatch") / (args.tag or args.detector) / pair
    if not args.no_save:
        destination = args.outdir / output
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
        time, h1, h2, lags, overlaps = arrays
        np.savez_compressed(
            destination.with_name(pair + "_inputs.npz"),
            time_code=time, lags_code=lags, optimized_overlap=overlaps,
            **{args.sims[0]: h1, args.sims[1]: h2})
        print(f"Results: {destination.with_suffix('.json')}", flush=True)
    if not args.no_plot:
        plot_comparison(args, result, arrays, output)
    return result


if __name__ == "__main__":
    main()
