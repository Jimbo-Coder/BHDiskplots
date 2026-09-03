#!/usr/bin/env python3
"""Make multi-simulation 2D panels from the shared 2D slice machinery."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_mplconfigdir = Path(tempfile.gettempdir()) / f"bhdiskplot_mpl_{os.getuid()}"
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from config import PLOTS_DIR
from helpers.style import (
    DOUBLE_COLUMN_WIDTH,
    PAPER_AXIS_LABEL_SIZE,
    PAPER_FONT_SIZE,
)
from helpers.plot_common import savefig, setup
from helpers.reader import load_sims
from helpers.time_units import ORBITAL_PERIOD_LATEX
from paper_plots.rho2d_individual import (
    OUTPUT_PREFIX,
    RHO2D_PLANE,
    RHO2D_X_LIMITS,
    RHO2D_Y_LIMITS,
    first_rho2d_iteration_info,
    last_rho2d_iteration_info,
    load_rho2d_slice,
    _set_colorbar_label,
    plot_rho2d_on_axis,
    rho2d_colorbar_label,
    rho2d_coordinate_label,
    snapshot_requests,
    valid_rho2d_iteration_infos,
)

# Critical knobs.
FAMILY_GROUPS = {
    "A": ("A1", "A2", "A3"),
    "B": ("B1", "B2", "B3"),
}
PAIR_GROUPS = (("A1", "B1"), ("A2", "B2"), ("A3", "B3"))

# Critical knobs.
MATCH_TBYPC = True
# Same selectors as rho2d_individual:
#   integer = exact valid-frame index
#   float in [0, 1] = fraction of the applicable valid time interval
#   matching BY_TBYPC entry 1 = interpret the value as a target t/P_c
# If the first entry is a t/P_c target, fractions run from that target to each
# simulation's final valid frame. These defaults select t/P_c=2, halfway, last.
SNAPSHOT_VALUES = [2.0, 0.5, -1]
SNAPSHOT_BY_TBYPC = [1, 0, 0]
DEFAULT_LAYOUT = "all"
USE_SHARED_COLUMN_TIME_LABELS = True
SHARED_COLUMN_TIME_DECIMALS = 2
SHOW_HORIZON_DEBUG = True

# Presentation knobs. Sized directly for a 7-inch double-column figure.
PANEL_FIGURE_WIDTH = DOUBLE_COLUMN_WIDTH
PANEL_ROW_HEIGHT = 2.02
PANEL_FONT_SIZE = PAPER_FONT_SIZE
PANEL_TITLE_SIZE = PANEL_FONT_SIZE
PANEL_LABEL_SIZE = PAPER_AXIS_LABEL_SIZE
PANEL_TICK_SIZE = PANEL_FONT_SIZE
PANEL_TICK_DECIMALS = 2
PANEL_TITLE_PAD = 3
PANEL_ROW_LABEL_X = -0.27
PANEL_XLABEL_Y = -0.01
PANEL_YLABEL_X = -0.065

# Presentation knobs.
PANEL_COLORBAR_PAD = 0.012
PANEL_COLORBAR_WIDTH = 0.022
PANEL_COMPACT_WSPACE = -0.035
PANEL_COMPACT_HSPACE = 0.012
PANEL_AXIS_DECIMAL = 3
PANEL_SHOW_DATA_LIMITS_IN_TITLE = False
PANEL_LEFT_MARGIN = 0.065
PANEL_RIGHT_MARGIN = 0.90
PANEL_BOTTOM_MARGIN = 0.045
PANEL_TOP_MARGIN = 0.985

# Operational knobs.
SHOW_PANEL_SUPTITLE = False
SHOW_TIME_SUBTITLES = True

def sim_lookup(names):
    sims = load_sims([], names=names)
    return {sim.config.name: sim for sim in sims}


def panel_time_bounds(sim, timeline=None):
    if timeline is None:
        first = first_rho2d_iteration_info(sim)
        last = last_rho2d_iteration_info(sim)
    else:
        first = timeline[0] if timeline else None
        last = timeline[-1] if timeline else None
    if first is None or last is None:
        raise ValueError(f"{sim.config.name}: no plottable 2D data in configured sources")
    return first, last


def pc_for_sim(sim):
    pc = float(getattr(sim.config, "Pc", np.nan))
    if not np.isfinite(pc) or pc <= 0.0:
        raise ValueError(f"{sim.config.name}: invalid Pc={pc!r}")
    return pc


def time_match_value(t_code, sim, normalize_by_pc):
    t_code = float(t_code)
    if normalize_by_pc:
        return t_code / pc_for_sim(sim)
    return t_code


def code_time_from_match_value(value, sim, normalize_by_pc):
    value = float(value)
    if normalize_by_pc:
        return value * pc_for_sim(sim)
    return value


def time_match_label(normalize_by_pc):
    return f"t/{ORBITAL_PERIOD_LATEX}" if normalize_by_pc else "t"


def time_match_latex(normalize_by_pc):
    return f"t/{ORBITAL_PERIOD_LATEX}" if normalize_by_pc else "t"


def panel_snapshot_requests():
    return snapshot_requests(
        SNAPSHOT_VALUES,
        SNAPSHOT_BY_TBYPC,
    )


def panel_shared_time_bounds(
    sims_by_name,
    case_names,
    normalize_by_pc,
    requests,
    timelines_by_name=None,
):
    bounds = {
        name: panel_time_bounds(
            sims_by_name[name],
            None if timelines_by_name is None else timelines_by_name[name],
        )
        for name in case_names
    }
    start_value = max(
        time_match_value(first[1], sims_by_name[name], normalize_by_pc)
        for name, (first, _) in bounds.items()
    )
    stop_value = min(
        time_match_value(last[1], sims_by_name[name], normalize_by_pc)
        for name, (_, last) in bounds.items()
    )
    if start_value > stop_value:
        ranges = ", ".join(
            f"{name}=["
            f"{time_match_value(first[1], sims_by_name[name], normalize_by_pc):.2f}, "
            f"{time_match_value(last[1], sims_by_name[name], normalize_by_pc):.2f}]"
            for name, (first, last) in bounds.items()
        )
        raise ValueError(f"No overlapping 2D {time_match_label(normalize_by_pc)} range for panel {case_names}: {ranges}")

    first_value, first_by_tbypc = requests[0]
    if first_by_tbypc:
        if not normalize_by_pc:
            raise ValueError(
                "A leading t/P_c panel selector requires MATCH_TBYPC=True"
            )
        anchored_start = float(first_value)
        if not start_value <= anchored_start <= stop_value:
            raise ValueError(
                f"Leading t/P_c={anchored_start:g} is outside the shared valid "
                f"interval [{start_value:g}, {stop_value:g}]"
            )
        start_value = anchored_start
    return start_value, stop_value


def panel_target_values(requests, shared_bounds):
    start_value, stop_value = shared_bounds
    targets = []
    for value, by_tbypc in requests:
        if by_tbypc:
            target = float(value)
        elif isinstance(value, (float, np.floating)):
            target = start_value + float(value) * (stop_value - start_value)
        elif int(value) == 0:
            target = start_value
        elif int(value) == -1:
            target = stop_value
        else:
            raise ValueError(
                "Panel integer selectors support only 0 and -1; use a float "
                "fraction or mark the entry as t/P_c"
            )
        if not start_value <= target <= stop_value:
            raise ValueError(
                f"Panel target {target:g} is outside the shared valid interval "
                f"[{start_value:g}, {stop_value:g}]"
            )
        targets.append(target)
    return np.asarray(targets, dtype=float)


def panel_iteration_infos(sim, timeline, target_values, normalize_by_pc):
    if not timeline:
        raise ValueError(f"{sim.config.name}: no valid 2D data")
    infos = []
    for target_value in target_values:
        target_time = code_time_from_match_value(
            target_value, sim, normalize_by_pc
        )
        info = min(timeline, key=lambda item: abs(item[1] - target_time))
        infos.append(info)
    return infos


def _iter_axis_limits(sim, info):
    iteration, _, start_byte = info
    load_rho2d_slice(sim, iteration=iteration, start_byte=start_byte)
    axis0 = getattr(sim.rho2d, sim.rho2d.axis_names[0])
    axis1 = getattr(sim.rho2d, sim.rho2d.axis_names[1])
    x = np.asarray(axis0, dtype=float)
    y = np.asarray(axis1, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return None, None
    return (float(np.nanmin(x)), float(np.nanmax(x))), (float(np.nanmin(y)), float(np.nanmax(y)))


def _collect_global_axis_limits(case_names, sims_by_name, infos_by_name):
    x_min = np.inf
    x_max = -np.inf
    y_min = np.inf
    y_max = -np.inf
    for name in case_names:
        sim = sims_by_name[name]
        for info in infos_by_name[name]:
            xlim, ylim = _iter_axis_limits(sim, info)
            if xlim is None or ylim is None:
                continue
            x_min = min(x_min, xlim[0])
            x_max = max(x_max, xlim[1])
            y_min = min(y_min, ylim[0])
            y_max = max(y_max, ylim[1])

    if not (np.isfinite(x_min) and np.isfinite(x_max) and np.isfinite(y_min) and np.isfinite(y_max)):
        return None, None

    # Mild rounding to smooth tiny per-panel floating-point drift.
    decimals = int(PANEL_AXIS_DECIMAL)
    x_min = np.floor(x_min * 10**decimals) / 10**decimals
    x_max = np.ceil(x_max * 10**decimals) / 10**decimals
    y_min = np.floor(y_min * 10**decimals) / 10**decimals
    y_max = np.ceil(y_max * 10**decimals) / 10**decimals
    return (x_min, x_max), (y_min, y_max)


def shared_column_time_labels(target_values, normalize_by_pc):
    return [
        rf"${time_match_latex(normalize_by_pc)}={value:.{SHARED_COLUMN_TIME_DECIMALS}f}$"
        for value in target_values
    ]


def format_panel_axis(ax, row, col, nrows, ncols, compact_labels=False):
    ax.tick_params(axis="both", which="both", labelsize=PANEL_TICK_SIZE)
    tick_format = f"%.{PANEL_TICK_DECIMALS}f"
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(tick_format))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(tick_format))
    ax.title.set_fontsize(PANEL_TITLE_SIZE)
    ax.xaxis.label.set_size(PANEL_LABEL_SIZE)
    ax.yaxis.label.set_size(PANEL_LABEL_SIZE)
    ax.set_xlabel("")
    ax.set_ylabel("")
    if row < nrows - 1:
        ax.tick_params(axis="x", labelbottom=False)
    if col > 0:
        ax.tick_params(axis="y", labelleft=False)


def add_row_labels(axes, case_names):
    for row, name in enumerate(case_names):
        axes[row, 0].annotate(
            rf"$\mathrm{{{name}}}$",
            xy=(PANEL_ROW_LABEL_X, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=PANEL_TITLE_SIZE,
            annotation_clip=False,
        )


def add_panel_colorbar(fig, axes, mesh):
    """Add one colorbar at a fixed location shared by all panel layouts."""
    positions = [ax.get_position() for ax in np.ravel(axes)]
    bottom = min(position.y0 for position in positions)
    top = max(position.y1 for position in positions)
    right = max(position.x1 for position in positions)
    cax = fig.add_axes(
        [right + PANEL_COLORBAR_PAD, bottom, PANEL_COLORBAR_WIDTH, top - bottom]
    )
    cbar = fig.colorbar(mesh, cax=cax)
    _set_colorbar_label(cbar, rho2d_colorbar_label())
    cbar.ax.tick_params(labelsize=PANEL_TICK_SIZE)
    return cbar


def plot_panel(case_names, args, filename, suptitle):
    sims_by_name = sim_lookup(case_names)
    missing = [name for name in case_names if name not in sims_by_name]
    if missing:
        raise ValueError(f"Missing simulations for panel: {', '.join(missing)}")

    requests = panel_snapshot_requests()
    ncols = len(requests)
    nrows = len(case_names)
    timelines_by_name = {
        name: valid_rho2d_iteration_infos(sims_by_name[name])
        for name in case_names
    }
    shared_bounds = panel_shared_time_bounds(
        sims_by_name,
        case_names,
        normalize_by_pc=MATCH_TBYPC,
        requests=requests,
        timelines_by_name=timelines_by_name,
    )
    target_values = panel_target_values(requests, shared_bounds)
    infos_by_name = {
        name: panel_iteration_infos(
            sims_by_name[name],
            timelines_by_name[name],
            target_values,
            normalize_by_pc=MATCH_TBYPC,
        )
        for name in case_names
    }
    global_x = RHO2D_X_LIMITS
    global_y = RHO2D_Y_LIMITS
    if global_x is None or global_y is None:
        collected_x, collected_y = _collect_global_axis_limits(case_names, sims_by_name, infos_by_name)
        if global_x is None:
            global_x = collected_x
        if global_y is None:
            global_y = collected_y

    column_labels = None
    if args.show_time_subtitles and USE_SHARED_COLUMN_TIME_LABELS:
        column_labels = shared_column_time_labels(target_values, MATCH_TBYPC)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(PANEL_FIGURE_WIDTH, PANEL_ROW_HEIGHT * nrows),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(
        left=PANEL_LEFT_MARGIN,
        right=PANEL_RIGHT_MARGIN,
        bottom=PANEL_BOTTOM_MARGIN,
        top=PANEL_TOP_MARGIN,
        wspace=PANEL_COMPACT_WSPACE,
        hspace=PANEL_COMPACT_HSPACE,
    )
    mesh = None
    for row, name in enumerate(case_names):
        sim = sims_by_name[name]
        for col, (iteration, _, start_byte) in enumerate(infos_by_name[name]):
            ax = axes[row, col]
            load_rho2d_slice(
                sim,
                iteration=iteration,
                start_byte=start_byte,
                xlim=global_x,
                ylim=global_y,
            )
            title = column_labels[col] if column_labels is not None and row == 0 else ""
            mesh = plot_rho2d_on_axis(
                sim,
                ax,
                xlim=global_x,
                ylim=global_y,
                title=title,
                title_pad=PANEL_TITLE_PAD,
                show_data_limits_in_title=PANEL_SHOW_DATA_LIMITS_IN_TITLE,
                horizon_debug=SHOW_HORIZON_DEBUG,
            )
            format_panel_axis(ax, row, col, nrows, ncols, compact_labels=True)

    add_row_labels(axes, case_names)
    fig.text(
        0.5,
        PANEL_XLABEL_Y,
        rho2d_coordinate_label({"xy": "x", "xz": "x", "yz": "y"}[RHO2D_PLANE]),
        ha="center",
        va="top",
        fontsize=PANEL_LABEL_SIZE,
    )
    fig.text(
        PANEL_YLABEL_X,
        0.5,
        rho2d_coordinate_label({"xy": "y", "xz": "z", "yz": "z"}[RHO2D_PLANE]),
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=PANEL_LABEL_SIZE,
    )

    if mesh is not None:
        add_panel_colorbar(fig, axes, mesh)
    if SHOW_PANEL_SUPTITLE:
        fig.suptitle(suptitle, fontsize=PANEL_TITLE_SIZE + 2)
    savefig(fig, args, filename)


def run_class_panels(args):
    for family, cases in FAMILY_GROUPS.items():
        plot_panel(cases, args, f"{OUTPUT_PREFIX}_panel_{family}.png", f"{family} cases")


def run_crossclass_panels(args):
    for cases in PAIR_GROUPS:
        plot_panel(cases, args, f"{OUTPUT_PREFIX}_panel_{cases[0]}_{cases[1]}.png", f"{cases[0]} vs {cases[1]}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Make 2D rho panel plots.")
    parser.add_argument(
        "layout",
        nargs="?",
        choices=("class", "crossclass", "all"),
        default=DEFAULT_LAYOUT,
        help="Panel set to make: class gives A/B 3-row panels, crossclass gives A1/B1 etc., all does both.",
    )
    parser.add_argument("--outdir", type=Path, default=PLOTS_DIR, help="Directory for saved figures.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    parser.add_argument("--no-save", action="store_true", help="Do not save figures.")
    parser.add_argument(
        "--time-subtitles",
        dest="show_time_subtitles",
        action="store_true",
        help="Show a compact time-only subtitle on each panel cell.",
    )
    parser.add_argument(
        "--no-time-subtitles",
        dest="show_time_subtitles",
        action="store_false",
        help="Only label each panel cell by case name.",
    )
    parser.set_defaults(show_time_subtitles=SHOW_TIME_SUBTITLES)
    args = parser.parse_args(argv)
    setup(args)

    if args.layout in ("class", "all"):
        run_class_panels(args)
    if args.layout in ("crossclass", "all"):
        run_crossclass_panels(args)


if __name__ == "__main__":
    main()
