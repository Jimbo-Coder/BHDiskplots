"""Linear accretion and a separate mass-budget check (Wessel 2021, Fig. 10).

Use the existing scalar reader; do not change the paper M0dot plot or fit a
decay model here. Positive accretion is -M0dot_BH, not its absolute value.
The rate uses the fixed initial-data disk mass, as in the paper plot. Time
starts at initial data, NOT at an automatically guessed PPI saturation time.

The budget compares the measured outside-AH mass loss with the signed flux
integral. They need not agree if there are boundary losses, atmosphere/source
terms, or numerical/diagnostic errors. The escaping-mass columns are nested
mass inventories, not cumulative boundary fluxes; do not add them to the budget.
"""
from pathlib import Path
import sys
import json

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.integrate import cumulative_trapezoid

from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import (
    COMPACT_LEGEND_KWARGS,
    PAPER_TWO_PANEL_HEIGHT,
    PAPER_SINGLE_PANEL_HEIGHT,
    figure_size,
    ordered_sim_legend,
)
from helpers.time_units import MILLISECONDS_PER_SOLAR_MASS

# Physical choices. Rescale the same dimensionless simulation to this BH mass.
SOURCE_BH_MASS_MSUN = 10.0
CASE_GROUPS = (("A1", "A2", "A3"), ("B1", "B2", "B3"))

# Presentation. A/B have separate linear rate axes so weak cases stay visible.
FIG_HEIGHT = PAPER_TWO_PANEL_HEIGHT
RATE_FILENAME = "wip/accretion_rate.png"
BUDGET_FILENAME = "wip/accretion_mass_budget.png"
LATE_BUDGET_FILENAME = "wip/accretion_late_mass_budget.png"


def accretion_history(data, config, source_bh_mass=SOURCE_BH_MASS_MSUN):
    """Compute both diagnostics from restart-cleaned bhns.don (code units)."""
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[0] < 2 or data.shape[1] < 13:
        raise ValueError("bhns.don needs at least two rows and thirteen columns")
    time = data[:, 0]
    if not np.all(np.isfinite(data[:, [0, 1, 2, 11, 12]])):
        raise ValueError("nonfinite time, mass, or accretion diagnostic")
    if np.any(np.diff(time) <= 0):
        raise ValueError("accretion history needs strictly increasing, restart-cleaned time")
    scales = [source_bh_mass, config.mlittle, config.Pc, config.disk_rest_mass_code]
    if not np.all(np.isfinite(scales)) or min(scales) <= 0:
        raise ValueError("BH mass, orbital period, and disk mass must be positive")

    outside_mass = data[:, 1] - data[:, 2]
    if np.any(outside_mass <= 0):
        raise ValueError("outside-AH mass must be positive")
    inward_flux = -data[:, 11]
    # G=c=M_sun=1 originally. Scaling BH mass from mlittle to the chosen
    # physical mass multiplies the code time unit by source_bh_mass/mlittle.
    seconds_per_code_time = (
        source_bh_mass / config.mlittle * MILLISECONDS_PER_SOLAR_MASS / 1000
    )
    integrated_flux = cumulative_trapezoid(inward_flux, time, initial=0)
    recorded_flux = data[:, 12] - data[0, 12]
    # A missing AH writes zero but leaves both the integration time and previous
    # flux unchanged. Only mark zero runs whose next integral step matches
    # that skipped-sample formula (allow for rounding in the ASCII output).
    unavailable = np.zeros(time.size, dtype=bool)
    edges = np.diff(np.r_[False, inward_flux == 0, False].astype(int))
    for first, stop in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if first == 0 or stop == time.size:
            continue
        previous = first - 1
        if inward_flux[previous] <= 0 or inward_flux[stop] <= 0:
            continue
        if np.any(recorded_flux[first:stop] != recorded_flux[previous]):
            continue
        expected_step = 0.5 * (inward_flux[previous] + inward_flux[stop]) * (
            time[stop] - time[previous]
        )
        next_step = recorded_flux[stop] - recorded_flux[previous]
        if np.isclose(next_step, expected_step, rtol=1e-5, atol=0):
            unavailable[first:stop] = True
    initial_mass = outside_mass[0]
    return {
        "initial_outside_mass": initial_mass,
        "time_pc": time / config.Pc,
        "rate_per_second": inward_flux / config.disk_rest_mass_code / seconds_per_code_time,
        "unavailable_flux": unavailable,
        "mass_loss": initial_mass - outside_mass,
        "accreted_mass": integrated_flux,
        "recorded_accreted_mass": recorded_flux,
        "initial_scalar_to_fixed_mass": initial_mass / config.disk_rest_mass_code,
    }


def plot_rate(sims, histories):
    groups = [tuple(sim for sim in sims if sim.config.name in names) for names in CASE_GROUPS]
    groups = [group for group in groups if group]
    if not groups:
        raise ValueError("no configured A/B cases available for accretion plot")
    fig, axes = plt.subplots(
        len(groups), 1, sharex=True, squeeze=False,
        figsize=figure_size("double", FIG_HEIGHT),
    )
    axes = axes[:, 0]
    for ax, group in zip(axes, groups):
        for sim in group:
            history = histories[sim.config.name]
            # Draw unavailable samples as gaps, not zero accretion or interpolated data.
            rate = np.where(history["unavailable_flux"], np.nan, history["rate_per_second"])
            ax.plot(history["time_pc"], rate,
                    color=sim.color, linestyle=sim.linestyle, label=sim.legend_name)
        ax.set_ylabel(r"$\dot M_{\rm acc}/M_{0,\rm disk}\ [\mathrm{s}^{-1}]$")
        ax.axhline(0, color="0.65", linewidth=0.7)
        ax.grid(which="major", alpha=0.25)
        ax.margins(x=0.01, y=0.10)
        ordered_sim_legend(ax, ncols=3, loc="upper left", **COMPACT_LEGEND_KWARGS)
    axes[0].text(0.97, 0.94, rf"$M_{{\rm BH}}={SOURCE_BH_MASS_MSUN:g}\,M_\odot$",
                 transform=axes[0].transAxes, ha="right", va="top")
    axes[-1].set_xlabel(r"$t/P_c$")
    axes[-1].set_xlim(left=0)
    fig.align_ylabels(axes)
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.11, top=0.98, hspace=0.13)
    return fig


def plot_mass_budget(sims, histories, *, groups=CASE_GROUPS, late=False):
    """Raw code-unit mass changes; positive means lost/accreted/gained."""
    by_name = {sim.config.name: sim for sim in sims}
    height = FIG_HEIGHT if len(groups) > 1 else PAPER_SINGLE_PANEL_HEIGHT
    fig, axes = plt.subplots(len(groups), 3, sharex=len(groups) > 1 and not late,
                             figsize=figure_size("double", height), squeeze=False)
    populated = []
    for ax, name in zip(axes.flat, sum(groups, ())):
        if name not in by_name:
            ax.set_visible(False)
            continue
        history = histories[name]
        ax.plot(history["time_pc"], history["mass_loss"],
                color="tab:blue", label="Disk mass loss")
        ax.plot(history["time_pc"], history["recorded_accreted_mass"],
                color="tab:orange", linestyle="--", label="AH rest-mass flux")
        if "bh_time_pc" in history:
            ax.plot(history["bh_time_pc"], history["bh_mass_gain"],
                    color="0.25", linestyle=":", label="BH mass change")
        ax.axhline(0, color="0.65", linewidth=0.7)
        ax.text(0.07, 0.92, name, transform=ax.transAxes, ha="left", va="top")
        ax.grid(which="major", alpha=0.25)
        ax.margins(x=0.02, y=0.18)
        ax.locator_params(axis="y", nbins=4)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0),
                            useOffset=False, useMathText=True)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        populated.append(ax)
    if not populated:
        plt.close(fig)
        raise ValueError("no configured A/B cases available for mass-budget plot")
    if len(groups) > 1 and not late:
        populated[0].set_xlim(left=0)
    handles, labels = populated[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.54, 0.99),
               ncols=3, **COMPACT_LEGEND_KWARGS)
    fig.supxlabel(r"$t/P_c$", y=0.015)
    fig.supylabel(r"$\Delta M\,[M_\odot]$", x=0.015)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.16 if len(groups) == 1 else 0.11,
                        top=0.82 if len(groups) == 1 else 0.88,
                        hspace=0.32 if late else 0.18, wspace=0.38)
    return fig


def late_mass_budget(sim, history):
    """Rebase all three quantities over the SAME GW fit interval, no extrapolation."""
    import gw_detectability as analysis

    trial = analysis.temporal_trial_for_case(sim.config)
    start, end = np.array([trial.fit.start, trial.fit.end])*sim.config.mlittle/sim.config.Pc
    t, tb = history["time_pc"], history["bh_time_pc"]
    if start < max(t[0], tb[0]) or end > min(t[-1], tb[-1]):
        raise ValueError(f"{sim.config.name}: BH/scalar data do not cover the GW fit interval")
    sample = np.r_[start, t[(t > start) & (t < end)], end]
    result = dict(time_pc=sample, bh_time_pc=sample)
    for key in ("mass_loss", "recorded_accreted_mass", "bh_mass_gain"):
        values = np.interp(sample, tb if key == "bh_mass_gain" else t, history[key])
        result[key] = values-values[0]
    return result


def main(argv=None):
    args = parser(__doc__.splitlines()[0]).parse_args(argv)
    setup(args)
    names = args.sims or sum(CASE_GROUPS, ())
    sims = load_sims(names=names)
    histories = {}
    report = {}
    for sim in sims:
        if sim.config.name not in sum(CASE_GROUPS, ()):
            continue
        _, data, time, _ = sim.loaddata("bhns.don")
        if time[0] != 0:
            print(f"{sim.config.name}: first mass-budget sample is t={time[0]:g}, not t=0")
        history = accretion_history(data, sim.config)
        histories[sim.config.name] = history
        discrepancy = np.max(np.abs(
            history["accreted_mass"] - history["recorded_accreted_mass"]
        ))
        # Same Christodoulou mass as the existing spin/tripleM plots. Never
        # compare extrapolated AH area/spin samples outside their common support.
        sim.load_spin_parameter()
        common = ((sim.J_t >= max(time[0], sim.Rs_t[0])) &
                  (sim.J_t <= min(time[-1], sim.Rs_t[-1])))
        bh_time, bh_mass = sim.J_t[common], sim.Mbh[common]
        if len(bh_time) < 2 or not np.all(np.isfinite(bh_mass)):
            raise ValueError(f"{sim.config.name}: insufficient common BH/scalar coverage")
        history["bh_time_pc"] = bh_time/sim.config.Pc
        history["bh_mass_gain"] = bh_mass-bh_mass[0]
        flux_common = np.interp(bh_time, time, data[:, 12])
        report[sim.config.name] = dict(
            mass_units="G=c=M_sun=1; no mass normalization or astrophysical rescaling",
            first_bh_time=float(bh_time[0]), last_bh_time=float(bh_time[-1]),
            bh_mass_change=float(bh_mass[-1]-bh_mass[0]),
            recorded_accreted_mass=float(flux_common[-1]-flux_common[0]),
            outside_mass_loss=float(history["mass_loss"][-1]),
            flux_integral_discrepancy=float(discrepancy),
        )
        print(f"{sim.config.name}: measured loss={history['mass_loss'][-1]:.6g}; "
              f"integrated AH flux={history['accreted_mass'][-1]:.6g}; "
              f"max |integral - int_M0dot|={discrepancy:.6g} (code mass units); "
              f"{np.count_nonzero(history['unavailable_flux'])} missing-AH-pattern plot gaps")
    savefig(plot_rate(sims, histories), args, RATE_FILENAME)
    savefig(plot_mass_budget(sims, histories), args, BUDGET_FILENAME)
    late = {}
    for sim in sims:
        if sim.config.name not in histories:
            continue
        try:
            late[sim.config.name] = late_mass_budget(sim, histories[sim.config.name])
        except (OSError, ValueError) as exc:
            print(f"{sim.config.name}: late GW-interval comparison unavailable: {exc}")
            continue
        values = late[sim.config.name]
        report[sim.config.name]["gw_fit_interval_comparison"] = dict(
            interval_pc=values["time_pc"][[0, -1]].tolist(),
            mass_loss=float(values["mass_loss"][-1]),
            accreted_mass=float(values["recorded_accreted_mass"][-1]),
            bh_mass_gain=float(values["bh_mass_gain"][-1]),
        )
    if late:
        savefig(plot_mass_budget([s for s in sims if s.config.name in late], late,
                                late=True), args, LATE_BUDGET_FILENAME)
    elif not args.no_save:
        (Path(args.outdir)/LATE_BUDGET_FILENAME).unlink(missing_ok=True)
    if not args.no_save:
        (Path(args.outdir)/"wip/accretion_mass_budget.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
