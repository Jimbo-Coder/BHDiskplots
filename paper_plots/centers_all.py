from pathlib import Path
import sys
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from helpers.plot_common import parser, savefig, setup
from helpers.reader import load_sims
from helpers.style import COMPACT_LEGEND_KWARGS, ordered_sim_legend

# Critical knobs.
CENTER_GROUPS = [
    ("A_ML", ("A1", "A2", "A3", "ML")),
    ("B_ML", ("B1", "B2", "B3", "ML")),
]
CENTER_SYMMETRIC_LIMITS = True
CENTER_NORMALIZE_BY_ADM_MASS = True
OUTPUT_TEMPLATE = "centers_{group}.png"

# Presentation knobs.
CENTER_FIGSIZE = (4.0, 4.0)
CENTER_LINEWIDTH = 0.8
CENTER_LEGEND_NCOLS = 2
CENTER_LEGEND_LOC = "lower left"
CENTER_SCI_EXPONENT = -2
CENTER_DIRECTION_FRACTIONS = (0.25, 0.50, 0.75)
CENTER_ARROW_LENGTH_FRACTION = 0.018
CENTER_ARROW_SCALE = 7
CENTER_MARKER_SIZE = 3.5
CENTER_LIMIT_PAD = 0.04
CENTER_SHOW_MAJOR_GRID = True
CENTER_GRID_COLOR = "0.82"
CENTER_GRID_LINEWIDTH = 0.6
CENTER_GRID_ALPHA = 0.7


def _center_coordinates(sim):
    x = np.asarray(sim.Rsdata[:, 2], dtype=float)
    y = np.asarray(sim.Rsdata[:, 3], dtype=float)
    if not CENTER_NORMALIZE_BY_ADM_MASS:
        return x, y

    adm_mass = float(getattr(sim.config, "gw_madm", np.nan))
    if not np.isfinite(adm_mass) or adm_mass <= 0.0:
        raise ValueError(
            f"{sim.config.name}: invalid ADM-mass coordinate divisor {adm_mass!r}"
        )
    return x / adm_mass, y / adm_mass


def _direction_segment(x, y, fraction, arrow_length):
    finite = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x)[finite]
    y = np.asarray(y)[finite]
    if len(x) < 2:
        return None

    distance = np.hypot(np.diff(x), np.diff(y))
    cumulative = np.concatenate(([0.0], np.cumsum(distance)))
    total = cumulative[-1]
    if not np.isfinite(total) or total <= arrow_length:
        return None

    target = fraction * total
    start_distance = max(0.0, target - 0.5 * arrow_length)
    end_distance = min(total, target + 0.5 * arrow_length)
    start = max(np.searchsorted(cumulative, start_distance), 0)
    end = min(np.searchsorted(cumulative, end_distance), len(x) - 1)
    if end <= start:
        start = max(0, end - 1)
    if end <= start or (x[start] == x[end] and y[start] == y[end]):
        return None
    return x[start], y[start], x[end], y[end]


def plot(sims):
    fig, ax = plt.subplots(figsize=CENTER_FIGSIZE)
    trajectories = []
    for sim in sims:
        x, y = _center_coordinates(sim)
        finite_indices = np.flatnonzero(np.isfinite(x) & np.isfinite(y))
        if len(finite_indices) == 0:
            continue
        trajectories.append((sim, x, y, finite_indices))
        ax.plot(
            x,
            y,
            color=sim.color,
            linestyle=sim.linestyle,
            linewidth=CENTER_LINEWIDTH,
            label=sim.legend_name,
        )
        ax.plot(
            x[finite_indices[0]],
            y[finite_indices[0]],
            marker="o",
            markersize=CENTER_MARKER_SIZE,
            markerfacecolor="white",
            markeredgecolor=sim.color,
            markeredgewidth=CENTER_LINEWIDTH,
            linestyle="none",
            zorder=3,
        )
        ax.plot(
            x[finite_indices[-1]],
            y[finite_indices[-1]],
            marker="X",
            markersize=CENTER_MARKER_SIZE,
            markerfacecolor=sim.color,
            markeredgecolor=sim.color,
            linestyle="none",
            zorder=3,
        )
    if trajectories:
        x_all = np.concatenate([x[index] for _, x, _, index in trajectories])
        y_all = np.concatenate([y[index] for _, _, y, index in trajectories])
        display_span = max(np.ptp(x_all), np.ptp(y_all))
        arrow_length = CENTER_ARROW_LENGTH_FRACTION * display_span
        for sim, x, y, _ in trajectories:
            for fraction in CENTER_DIRECTION_FRACTIONS:
                segment = _direction_segment(x, y, fraction, arrow_length)
                if segment is not None:
                    x0, y0, x1, y1 = segment
                    ax.annotate(
                        "",
                        xy=(x1, y1),
                        xytext=(x0, y0),
                        arrowprops={
                            "arrowstyle": "-|>",
                            "color": sim.color,
                            "linewidth": CENTER_LINEWIDTH,
                            "mutation_scale": CENTER_ARROW_SCALE,
                            "shrinkA": 0,
                            "shrinkB": 0,
                        },
                        zorder=4,
                    )
        if CENTER_SYMMETRIC_LIMITS:
            x_limit = (1.0 + CENTER_LIMIT_PAD) * np.max(np.abs(x_all))
            y_limit = (1.0 + CENTER_LIMIT_PAD) * np.max(np.abs(y_all))
            ax.set_xlim(-x_limit, x_limit)
            ax.set_ylim(-y_limit, y_limit)

    for axis in (ax.xaxis, ax.yaxis):
        formatter = mticker.ScalarFormatter(useMathText=True)
        if CENTER_NORMALIZE_BY_ADM_MASS:
            formatter.set_powerlimits((-3, 3))
        else:
            formatter.set_scientific(True)
            formatter.set_powerlimits((CENTER_SCI_EXPONENT, CENTER_SCI_EXPONENT))
        axis.set_major_formatter(formatter)
    if CENTER_NORMALIZE_BY_ADM_MASS:
        ax.set_xlabel(r"$x\ [M]$")
        ax.set_ylabel(r"$y\ [M]$")
    else:
        ax.set_xlabel(r"$x\ [M_\odot]$")
        ax.set_ylabel(r"$y\ [M_\odot]$")
    ax.set_aspect("equal", adjustable="box")
    if CENTER_SHOW_MAJOR_GRID:
        ax.set_axisbelow(True)
        ax.grid(
            which="major",
            color=CENTER_GRID_COLOR,
            linewidth=CENTER_GRID_LINEWIDTH,
            alpha=CENTER_GRID_ALPHA,
        )
    ordered_sim_legend(
        ax,
        ncols=CENTER_LEGEND_NCOLS,
        loc=CENTER_LEGEND_LOC,
        **COMPACT_LEGEND_KWARGS,
    )
    fig.tight_layout()
    return fig

def main(argv=None):
    args = parser("Plot BH centers.").parse_args(argv)
    setup(args)
    sims = load_sims(["Rs"], names=args.sims)
    sim_by_name = {sim.config.name: sim for sim in sims}
    for filename_tag, names in CENTER_GROUPS:
        group = [sim_by_name[name] for name in names if name in sim_by_name]
        if group:
            savefig(
                plot(group),
                args,
                OUTPUT_TEMPLATE.format(group=filename_tag),
            )


if __name__ == "__main__":
    main()
