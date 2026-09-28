"""Exploratory density-mode growth and matter/GW frequency comparisons.

The growth rate is a coordinate-time diagnostic, not an instability eigenvalue.
The spectra test frequency correspondence, not a decomposition of GW sources.
The frequency comparison follows the diagnostic logic of Wessel et al.,
Phys. Rev. D 103, 043013 (2021), Figs. 6-9; it is not their dataset or fit.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal.windows import tukey

from config import GW_OUTERMOST_PARFILE_INDEX, REPOSITORY_ROOT, all_sim_configs
from gw_detectability import read_rpsi4_modes
from helpers.plot_common import parser, savefig, setup
from helpers.reader import DiskSim
from helpers.style import apply_sim_style, figure_size


# Analysis choices. C_m/C_0 follows paper_plots/modes_all.py; C_0 is the
# density-cut diagnostic integral, not the total disk rest mass. Staged local
# files are used only when the configured live roots are absent.
CASE_NAMES = ("A1", "A2", "A3", "B1", "B2", "B3")
MODE_NUMBERS = (1, 2, 3, 4, 5)
LOCAL_MODE_ROOT = REPOSITORY_ROOT / "data" / "mode_diagnostics"
EARLIEST_GROWTH_TIME_PC = 1.0
SMOOTH_WIDTH_PC = 0.35
GROWTH_FRACTIONS_OF_PEAK = (0.10, 0.5)
MIN_RISE_SPAN_PC = 0.75
SPECTRUM_START_PC = 10.0  # Late-time comparison, not the detectability transient cut.
SPECTRUM_TAPER_ALPHA = 0.10
SPECTRUM_FREQUENCY_RANGE = (0.35, 4.0)  # f / f_orb = f * P_c
SPECTRUM_MODES = (1, 2)
GW_MODES = ((2, 1), (2, 2))
GW_RADIUS_INDEX = GW_OUTERMOST_PARFILE_INDEX

GROWTH_FILENAME = "wip/mode_growth_summary.png"
RISE_WINDOW_FILENAME = "wip/mode_growth_rise_windows.png"
FREQUENCY_FILENAME = "wip/mode_gw_frequency.png"


@dataclass(frozen=True)
class GrowthMeasure:
    peak: float
    rate: float | None
    rise_start: float | None
    rise_end: float | None
    r2: float | None
    peak_at_end: bool


def load_cases(names=CASE_NAMES):
    cases = []
    for config in all_sim_configs(names):
        if config.name not in names:
            continue
        if not any((root / "bhns-dens_mode.con").is_file() for root in config.data_roots):
            staged = LOCAL_MODE_ROOT / config.data_path.name
            if not (staged / "bhns-dens_mode.con").is_file():
                raise FileNotFoundError(f"No density modes for {config.name} in live or local roots")
            config = replace(config, data_roots=(staged,))
        sim = apply_sim_style(DiskSim(config))
        sim.load_modes()
        print(f"{config.name}: density modes from {sim.modepath}")
        cases.append(sim)
    return cases


def density_amplitude(sim, mode):
    c0 = np.abs(sim.modes[:, 0])
    amplitude = np.divide(
        np.abs(sim.modes[:, mode]), c0,
        out=np.full_like(c0, np.nan), where=c0 > 0,
    )
    return sim.modes_t / sim.config.Pc, amplitude


def _smoothed_amplitude(time, amplitude, width_pc=SMOOTH_WIDTH_PC):
    valid = np.isfinite(time) & np.isfinite(amplitude) & (amplitude > 0)
    time = np.asarray(time[valid], dtype=float)
    amplitude = np.asarray(amplitude[valid], dtype=float)
    if len(time) < 5 or np.any(np.diff(time) <= 0):
        raise ValueError("Density-mode times must be unique, increasing, and nonempty")
    width = max(3, int(round(width_pc / np.median(np.diff(time))))) | 1
    return time, median_filter(amplitude, size=width, mode="nearest")


def measure_growth(time, amplitude):
    """Measure the first 10%-to-50% rise toward the sustained observed peak.

    The rate is ln(5) divided by elapsed t/Pc, not an eigenmode fit. R2 of
    a log-linear fit to that interval is reported separately for inspection.
    """
    time, smooth = _smoothed_amplitude(time, amplitude)
    active = np.flatnonzero(time >= EARLIEST_GROWTH_TIME_PC)
    if active.size < 5:
        raise ValueError("No density-mode history remains after the startup cut")
    peak_index = active[np.argmax(smooth[active])]
    peak = float(smooth[peak_index])
    peak_at_end = bool(peak_index >= len(time) - 1 - max(1, int(0.5 / np.median(np.diff(time)))))
    invalid = GrowthMeasure(peak, None, None, None, None, peak_at_end)
    low, high = (fraction * peak for fraction in GROWTH_FRACTIONS_OF_PEAK)
    prior = active[active < peak_index]
    starts = prior[smooth[prior] >= low]
    if starts.size == 0:
        return invalid
    start = starts[0]
    if start == active[0]:
        return invalid  # The 10% onset was already present at the startup cut.
    ends = prior[(prior > start) & (smooth[prior] >= high)]
    if ends.size == 0:
        return invalid
    end = ends[0]
    if time[end] - time[start] < MIN_RISE_SPAN_PC:
        return invalid
    x = time[start:end + 1]
    y = np.log(smooth[start:end + 1])
    slope, intercept = np.polyfit(x, y, 1)
    residual = np.sum((y - (slope*x + intercept))**2)
    spread = np.sum((y - np.mean(y))**2)
    r2 = 1 - residual/spread if spread > 0 else -np.inf
    rate = np.log(high/low)/(time[end] - time[start])
    if slope <= 0 or rate <= 0:
        return invalid
    return GrowthMeasure(peak, float(rate), float(x[0]), float(x[-1]), float(r2), peak_at_end)


def plot_growth_summary(cases, measures):
    fig, axes = plt.subplots(2, 1, figsize=figure_size("double", 5.8), sharex=True,
                             gridspec_kw={"hspace": 0.31})
    offsets = {name: (index - 2.5)*0.07 for index, name in enumerate(CASE_NAMES)}
    for sim in cases:
        name = sim.config.name
        for mode in MODE_NUMBERS:
            result = measures[name, mode]
            x = mode + offsets[name]
            style = dict(marker=sim.markerstyle, color=sim.color,
                         markerfacecolor=sim.color if name.startswith("A") else "none",
                         linestyle="none", markersize=6, markeredgewidth=1.3)
            if result.rate is not None:
                axes[0].plot(x, result.rate, **style)
            axes[1].plot(x, result.peak, **style)
    axes[0].set_ylabel(r"Rise rate $\ln 5/\Delta(t/P_c)$")
    axes[1].set_ylabel(r"Observed peak $|C_m|/C_0$")
    axes[1].set_xlabel(r"Density-mode number $m$")
    axes[1].set_yscale("log")
    axes[1].set_xticks(MODE_NUMBERS)
    axes[1].set_xlim(0.55, 5.45)
    for ax in axes:
        ax.grid(axis="y", alpha=0.35)
    handles = [Line2D([], [], marker=sim.markerstyle, linestyle="none", color=sim.color,
                      markerfacecolor=sim.color if sim.config.name.startswith("A") else "none",
                      label=sim.legend_name, markersize=6)
               for sim in cases]
    if len(handles) == 6:
        handles = [handles[i] for i in (0, 3, 1, 4, 2, 5)]
    fig.legend(handles=handles, ncol=3, loc="center", bbox_to_anchor=(0.59, 0.55))
    fig.subplots_adjust(left=0.17, right=0.98, top=0.98, bottom=0.12)
    return fig


def plot_fit_checks(cases, measures):
    fig, axes = plt.subplots(2, 3, figsize=figure_size("double", 6.1), sharex=True,
                             sharey=True, gridspec_kw={"hspace": 0.22, "wspace": 0.20})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for ax, sim in zip(axes.flat, cases):
        for mode in MODE_NUMBERS:
            time, amplitude = density_amplitude(sim, mode)
            time, smooth = _smoothed_amplitude(time, amplitude)
            result = measures[sim.config.name, mode]
            color = colors[mode - 1]
            ax.plot(time, smooth, color=color, lw=0.75, alpha=0.7,
                    label=rf"$m={mode}$" if sim is cases[0] else None)
            if result.rate is not None:
                selected = (time >= result.rise_start) & (time <= result.rise_end)
                ax.plot(time[selected], smooth[selected], color=color, lw=2.5)
        ax.set_title(sim.config.name)
        ax.set_yscale("log")
        ax.set_xlim(0, min(35, time[-1]))
        ax.grid(alpha=0.25)
    fig.supxlabel(r"$t/P_c$")
    fig.supylabel(r"Smoothed $|C_m|/C_0$")
    axes.flat[0].legend(ncol=2, fontsize=8, loc="lower right")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.95, bottom=0.10)
    return fig


def normalized_spectrum(time, values, start, end, *, psi4_to_hc=False):
    """Two-sided complex FFT folded by absolute frequency; no sign convention.

    For a single Psi4 mode, characteristic-strain shape is proportional to
    |Psi4_tilde|/f. Constants cancel when each curve is normalized to its peak.
    """
    selected = (time >= start) & (time <= end) & np.isfinite(values)
    time = np.asarray(time[selected], dtype=float)
    values = np.asarray(values[selected], dtype=complex)
    if len(time) < 16 or np.any(np.diff(time) <= 0):
        raise ValueError("Insufficient unique samples in the common spectral interval")
    dt = float(np.median(np.diff(time)))
    count = int(np.floor((time[-1] - time[0])/dt)) + 1
    grid = time[0] + dt*np.arange(count)
    samples = np.interp(grid, time, values.real) + 1j*np.interp(grid, time, values.imag)
    samples -= np.mean(samples)
    fft_size = 1 << (4*count - 1).bit_length()
    transform = np.fft.fft(samples*tukey(count, SPECTRUM_TAPER_ALPHA), n=fft_size)
    frequency = np.fft.fftfreq(fft_size, d=dt)
    indices = np.flatnonzero(frequency > 0)
    amplitude = np.hypot(np.abs(transform[indices]), np.abs(transform[-indices]))
    if psi4_to_hc:
        amplitude = amplitude/frequency[indices]
    band = (frequency[indices] >= SPECTRUM_FREQUENCY_RANGE[0]) & (
        frequency[indices] <= SPECTRUM_FREQUENCY_RANGE[1])
    if not np.any(band) or np.max(amplitude[band]) <= 0:
        raise ValueError("No spectral power in the displayed frequency band")
    return frequency[indices][band], amplitude[band]/np.max(amplitude[band])


def plot_frequency(cases):
    fig, axes = plt.subplots(2, 3, figsize=figure_size("double", 5.4), sharex=True,
                             sharey=True, gridspec_kw={"hspace": 0.17, "wspace": 0.12})
    styles = ((1, "#2456b8", "-"), (2, "#c43c30", "-"),
              ((2, 1), "#2456b8", "--"), ((2, 2), "#c43c30", "--"))
    for ax, sim in zip(axes.flat, cases):
        gw_time, gw = read_rpsi4_modes(sim.config, GW_RADIUS_INDEX, GW_MODES)
        matter_time = sim.modes_t/sim.config.Pc
        gw_time = gw_time/sim.config.Pc
        start = max(SPECTRUM_START_PC, matter_time[0], gw_time[0])
        end = min(matter_time[-1], gw_time[-1])
        if end <= start:
            raise ValueError(f"{sim.config.name}: no common matter/GW interval")
        peaks = []
        series = {}
        for mode in SPECTRUM_MODES:
            c0 = sim.modes[:, 0]
            series[mode] = np.divide(sim.modes[:, mode], c0,
                                     out=np.full(len(c0), np.nan + 0j), where=np.abs(c0) > 0)
        series.update(gw)
        for mode, color, line_style in styles:
            time = matter_time if isinstance(mode, int) else gw_time
            freq, amplitude = normalized_spectrum(
                time, series[mode], start, end, psi4_to_hc=not isinstance(mode, int))
            label = rf"$C_{mode}$" if isinstance(mode, int) else rf"$h_{{c,{mode[0]}{mode[1]}}}$"
            ax.plot(freq, amplitude, color=color, linestyle=line_style, lw=1.25,
                    label=label if sim is cases[0] else None)
            peaks.append(f"{label}={freq[np.argmax(amplitude)]:.2f}")
        print(f"{sim.config.name}: spectra over t/Pc={start:.2f}-{end:.2f}; "
              + ", ".join(peaks))
        ax.set_title(sim.config.name)
        ax.set_xlim(*SPECTRUM_FREQUENCY_RANGE)
        ax.set_xticks((1, 2, 3, 4))
        ax.set_ylim(0, 1.07)
        ax.set_yticks((0, 0.5, 1))
        ax.grid(alpha=0.25)
    fig.supxlabel(r"$f/f_{\rm orb,0}=fP_c$")
    fig.supylabel("Spectrum / own peak")
    fig.legend(*axes.flat[0].get_legend_handles_labels(), ncol=4,
               loc="upper center", bbox_to_anchor=(0.57, 1.0))
    fig.subplots_adjust(left=0.10, right=0.99, top=0.91, bottom=0.11)
    return fig


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    selected = CASE_NAMES if args.sims is None else tuple(name.upper() for name in args.sims)
    if len(selected) != len(CASE_NAMES) or set(selected) != set(CASE_NAMES):
        raise ValueError("The six-panel mode appendix requires A1 A2 A3 B1 B2 B3")
    setup(args)
    cases = load_cases(CASE_NAMES)
    measures = {}
    for sim in cases:
        for mode in MODE_NUMBERS:
            time, amplitude = density_amplitude(sim, mode)
            result = measure_growth(time, amplitude)
            measures[sim.config.name, mode] = result
            interval = "none" if result.rate is None else f"{result.rise_start:.2f}-{result.rise_end:.2f}"
            print(f"{sim.config.name} m={mode}: peak={result.peak:.3g}, "
                  f"rise_rate={result.rate}, rise_window={interval}, "
                  f"log_linear_R2={result.r2}, peak_near_end={result.peak_at_end}")
    savefig(plot_growth_summary(cases, measures), args, GROWTH_FILENAME)
    savefig(plot_fit_checks(cases, measures), args, RISE_WINDOW_FILENAME)
    savefig(plot_frequency(cases), args, FREQUENCY_FILENAME)


if __name__ == "__main__":
    main()
