"""Shared paper plotting style and legend helpers."""
from __future__ import annotations

import os
import shlex
import shutil
import subprocess

import matplotlib as mpl
import matplotlib.lines as mlines
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import numpy as np

DOUBLE_COLUMN_WIDTH = 7.0
SINGLE_COLUMN_WIDTH = 3.375
PAPER_FONT_SIZE = 12
PAPER_AXIS_LABEL_SIZE = 15
PAPER_LEGEND_SIZE = 13
USE_LATEX_TEXT = True
TEXLIVE_MODULE = "texlive/20200406"
REQUIRED_TEX_FONT = "pplr7t.tfm"
TICK_MAJOR_LENGTH = 10
TICK_MAJOR_WIDTH = 1.5
TICK_MINOR_LENGTH = 5
TICK_MINOR_WIDTH = 1.0
AXIS_LABELPAD = 3
TICK_LABELPAD = 5
SPANNING_LEGEND_BBOX = (0.98, 0.5)
LEGEND_BORDERPAD = 0.18
LEGEND_LABELSPACING = 0.15
AXES_LEGEND_BORDERAXESPAD = 0.9
FIGURE_LEGEND_BORDERAXESPAD = 0.0
LEGEND_HANDLEHEIGHT = 0.6
COMPACT_LEGEND_KWARGS = {
    "fontsize": PAPER_LEGEND_SIZE,
    "handlelength": 1.2,
    "handletextpad": 0.45,
    "columnspacing": 0.75,
}
GW_LEGEND_KWARGS = {
    "fontsize": PAPER_LEGEND_SIZE,
    "handlelength": 1.2,
    "handletextpad": 0.45,
    "columnspacing": 0.8,
}
INTERPANEL_LEGEND_INSET = 0.012
PAPER_SINGLE_PANEL_HEIGHT = 3.8
PAPER_TWO_PANEL_HEIGHT = 5.8
PAPER_THREE_PANEL_HEIGHT = 7.2


def jet_white_low_cmap():
    jet = mpl.colormaps["jet"].resampled(256)
    jet_trunc = jet(np.linspace(0.2, 1.0, 220))
    white_to_cyan = np.linspace([1, 1, 1, 1], jet(0.2), 36)
    colors = np.vstack((white_to_cyan, jet_trunc))
    return LinearSegmentedColormap.from_list("jet_white_low", colors)


JET_WHITE_LOW_CMAP = jet_white_low_cmap()


def _tex_file_available(filename):
    kpsewhich = shutil.which("kpsewhich")
    if kpsewhich is None:
        return False
    result = subprocess.run(
        [kpsewhich, filename],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _load_module_environment(module_name):
    command = (
        f"module load {shlex.quote(module_name)} >/dev/null && env -0"
    )
    result = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        check=True,
    )
    for entry in result.stdout.split(b"\0"):
        if b"=" not in entry:
            continue
        key, value = entry.split(b"=", 1)
        os.environ[os.fsdecode(key)] = os.fsdecode(value)


def ensure_tex_path():
    if not USE_LATEX_TEXT or _tex_file_available(REQUIRED_TEX_FONT):
        return
    try:
        _load_module_environment(TEXLIVE_MODULE)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Could not load the {TEXLIVE_MODULE} module required by Matplotlib"
        ) from error
    if not _tex_file_available(REQUIRED_TEX_FONT):
        raise RuntimeError(
            f"The {TEXLIVE_MODULE} module does not provide {REQUIRED_TEX_FONT}"
        )


PAPER_PLOT_STYLE = {
    "text.usetex": USE_LATEX_TEXT,
    "font.family": "freeserif",
    "font.serif": ["palatino", "Palatino", "FreeSerif", "Times New Roman", "DejaVu Serif"],
    "font.weight": "black",
    "mathtext.fontset": "cm",
    "mathtext.default": "sf",
    "axes.labelsize": PAPER_AXIS_LABEL_SIZE,
    "axes.titlesize": PAPER_FONT_SIZE,
    "axes.labelpad": AXIS_LABELPAD,
    "axes.linewidth": 1.0,
    "legend.fontsize": PAPER_LEGEND_SIZE,
    "legend.frameon": True,
    "legend.framealpha": 1.0,
    "legend.fancybox": False,
    "legend.edgecolor": "black",
    "legend.handlelength": 1.0,
    "legend.handleheight": LEGEND_HANDLEHEIGHT,
    "legend.handletextpad": 0.7,
    "legend.columnspacing": 1.0,
    "legend.borderpad": LEGEND_BORDERPAD,
    "legend.labelspacing": LEGEND_LABELSPACING,
    "legend.borderaxespad": FIGURE_LEGEND_BORDERAXESPAD,
    "xtick.labelsize": PAPER_FONT_SIZE,
    "ytick.labelsize": PAPER_FONT_SIZE,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": TICK_MAJOR_LENGTH,
    "ytick.major.size": TICK_MAJOR_LENGTH,
    "xtick.major.width": TICK_MAJOR_WIDTH,
    "ytick.major.width": TICK_MAJOR_WIDTH,
    "xtick.major.pad": TICK_LABELPAD,
    "ytick.major.pad": TICK_LABELPAD,
    "xtick.minor.pad": TICK_LABELPAD,
    "ytick.minor.pad": TICK_LABELPAD,
    "xtick.minor.size": TICK_MINOR_LENGTH,
    "ytick.minor.size": TICK_MINOR_LENGTH,
    "xtick.minor.width": TICK_MINOR_WIDTH,
    "ytick.minor.width": TICK_MINOR_WIDTH,
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "figure.dpi": 200,
    "savefig.dpi": 800,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
}

COLOR_OPTS = ["b", "g", "r", "c"]
LINESTYLE_OPTS = ["-", "--"]
MARKER_OPTS = ["o", "^"]
SIM_LEGEND_ORDER = ["A1", "A2", "A3", "A4", "B1", "B2", "B3", "B4", "ML", "WL"]
# Two complete, centered dash-dot cycles fit the shared 1.2-em legend handle.
# Starting midway through the final gap avoids clipped partial dashes at either end.
ML_LEGEND_DASHDOT = (4.7, (2.4, 1.0, 0.8, 1.0))
ML_LEGEND_LINEWIDTH = 1.5


def _massless_legend_items(handles, labels):
    updated_handles = list(handles)
    for index, (handle, label) in enumerate(zip(updated_handles, labels)):
        # Keep the dashed ML symbol distinct without touching A/B or WL legend lines.
        if not (
            _clean_legend_label(label).upper() == "ML" and isinstance(handle, mlines.Line2D)
        ):
            continue
        updated_handles[index] = mlines.Line2D(
            [0, 1],
            [0, 0],
            color=handle.get_color(),
            # The normal "-." cycle is longer than our compact legend handle
            # and can look solid. Use a centered, complete dash-dot pattern in
            # the legend while leaving the plotted ML curve unchanged.
            linestyle=ML_LEGEND_DASHDOT,
            linewidth=ML_LEGEND_LINEWIDTH,
            dash_capstyle="butt",
            marker=handle.get_marker(),
            markersize=handle.get_markersize(),
        )

    return updated_handles


def apply_paper_style():
    ensure_tex_path()
    mpl.rcParams.update(PAPER_PLOT_STYLE)


def format_paper_axes(ax):
    ax.minorticks_on()
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
    return ax


def figure_size(kind="double", height=3.8):
    width = DOUBLE_COLUMN_WIDTH if kind == "double" else SINGLE_COLUMN_WIDTH
    return (width, height)


def apply_sim_style(sim):
    if sim.config.legend:
        sim.legend_name = rf"$\mathrm{{{sim.config.name}}}$"
    else:
        sim.legend_name = None
    name = sim.config.name
    if name and name[0] in {"A", "B"}:
        family_index = 0 if name[0] == "A" else 1
        color_index = int(name[1]) - 1
        sim.linestyle = LINESTYLE_OPTS[family_index]
        sim.markerstyle = MARKER_OPTS[family_index]
        sim.color = COLOR_OPTS[color_index % len(COLOR_OPTS)]
    else:
        sim.linestyle = "-."
        sim.markerstyle = "o"
        sim.color = "k"
    return sim


def apply_styles(sims):
    return [apply_sim_style(sim) for sim in sims]


def _clean_legend_label(label):
    if label is None:
        return ""
    clean = str(label).replace("$", "").strip()
    # Detectability labels retain the radial-extrapolation status in math mode,
    # e.g. ``\mathrm{A1}^{(\mathrm{i})}``.  Recover the simulation name so the
    # same A-row/B-row ordering is used by every combined plot.
    for name in SIM_LEGEND_ORDER:
        if rf"\mathrm{{{name}}}" in clean:
            return name
    for suffix in (" (i)", " (f)"):
        if clean.endswith(suffix):
            clean = clean[: -len(suffix)].strip()
    if clean.startswith(r"\mathrm{") and clean.endswith("}"):
        clean = clean[len(r"\mathrm{"):-1]
    return clean


def _ordered_sim_items(items, ncols=4):
    clean_to_item = {_clean_legend_label(label): (handle, label) for handle, label in items}
    present = set(clean_to_item)

    a_names = [name for name in ["A1", "A2", "A3", "A4"] if name in present]
    b_names = [name for name in ["B1", "B2", "B3", "B4"] if name in present]
    other_names = [name for name in ["ML", "WL"] if name in present]
    remaining = [
        _clean_legend_label(label)
        for _, label in items
        if _clean_legend_label(label) not in set(a_names + b_names + other_names)
    ]

    if other_names and a_names and b_names:
        ncols = max(ncols, max(len(a_names), len(b_names)) + 1)

    first_family_names = a_names if a_names else b_names
    later_family_names = b_names if a_names else []

    if len(first_family_names) < ncols and other_names:
        fill_count = ncols - len(first_family_names)
        first_row_others = other_names[:fill_count]
        other_names = other_names[fill_count:]
    else:
        first_row_others = []

    display_order = first_family_names + first_row_others + later_family_names + other_names + remaining
    ordered = [clean_to_item[name] for name in display_order if name in clean_to_item]
    ncols = max(1, min(ncols, len(ordered)))
    nrows = int(np.ceil(len(ordered) / ncols))
    draw_order = []
    for col in range(ncols):
        for row in range(nrows):
            index = row * ncols + col
            if index < len(ordered):
                draw_order.append(ordered[index])
    return draw_order, ncols


def _legend_items_from_axis(ax):
    handles, labels = ax.get_legend_handles_labels()
    items = [(handle, label) for handle, label in zip(handles, labels) if label and not str(label).startswith("_")]
    return items


def ordered_sim_legend(ax, ncols=4, loc="best", **kwargs):
    items = _legend_items_from_axis(ax)
    if not items:
        return None
    draw_order, ncols = _ordered_sim_items(items, ncols=ncols)
    ordered_handles, ordered_labels = zip(*draw_order)
    ordered_handles = _massless_legend_items(ordered_handles, ordered_labels)
    kwargs.setdefault("borderaxespad", AXES_LEGEND_BORDERAXESPAD)
    return ax.legend(ordered_handles, ordered_labels, ncols=ncols, loc=loc, **kwargs)


def ordered_sim_fig_legend(fig, ax, ncols=4, loc="upper center", **kwargs):
    items = _legend_items_from_axis(ax)
    if not items:
        return None
    draw_order, ncols = _ordered_sim_items(items, ncols=ncols)
    ordered_handles, ordered_labels = zip(*draw_order)
    ordered_handles = _massless_legend_items(ordered_handles, ordered_labels)
    kwargs.setdefault("borderaxespad", FIGURE_LEGEND_BORDERAXESPAD)
    return fig.legend(ordered_handles, ordered_labels, ncols=ncols, loc=loc, **kwargs)


def ordered_sim_spanning_legend(fig, ax, axes=None, ncols=4, loc="center right", **kwargs):
    if "bbox_to_anchor" not in kwargs and axes is not None:
        axes_list = list(np.ravel(axes))
        if len(axes_list) >= 2:
            positions = sorted((axis.get_position() for axis in axes_list), key=lambda pos: pos.y0, reverse=True)
            gap_midpoint = 0.5 * (positions[0].y0 + positions[1].y1)
            kwargs["bbox_to_anchor"] = (SPANNING_LEGEND_BBOX[0], gap_midpoint)
    kwargs.setdefault("bbox_to_anchor", SPANNING_LEGEND_BBOX)
    kwargs.setdefault("borderaxespad", FIGURE_LEGEND_BORDERAXESPAD)
    return ordered_sim_fig_legend(fig, ax, ncols=ncols, loc=loc, **kwargs)


def _interpanel_bbox(axes, x):
    axes_list = list(np.ravel(axes))
    if len(axes_list) >= 2:
        positions = sorted((axis.get_position() for axis in axes_list), key=lambda pos: pos.y0, reverse=True)
        gap_midpoint = 0.5 * (positions[0].y0 + positions[1].y1)
        return (x, gap_midpoint)
    return (x, 0.5)


def _inset_interpanel_x(x, loc):
    loc_text = str(loc).lower()
    if "right" in loc_text:
        return x - INTERPANEL_LEGEND_INSET
    if "left" in loc_text:
        return x + INTERPANEL_LEGEND_INSET
    return x


def ordered_sim_interpanel_legend(fig, ax, axes, ncols=4, loc="center left", x=0.12, **kwargs):
    kwargs.setdefault("bbox_to_anchor", _interpanel_bbox(axes, _inset_interpanel_x(x, loc)))
    kwargs.setdefault("borderaxespad", FIGURE_LEGEND_BORDERAXESPAD)
    return ordered_sim_fig_legend(fig, ax, ncols=ncols, loc=loc, **kwargs)


def interpanel_legend(fig, ax, axes, ncols=2, loc="center left", x=0.12, **kwargs):
    items = _legend_items_from_axis(ax)
    if not items:
        return None
    handles, labels = zip(*items)
    handles = _massless_legend_items(handles, labels)
    kwargs.setdefault("bbox_to_anchor", _interpanel_bbox(axes, _inset_interpanel_x(x, loc)))
    kwargs.setdefault("borderaxespad", FIGURE_LEGEND_BORDERAXESPAD)
    return fig.legend(handles, labels, ncols=ncols, loc=loc, **kwargs)


def _format_scientific_power(value, _position=None):
    if not np.isfinite(value):
        return ""
    if value == 0:
        return r"$0$"
    exponent = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10.0**exponent)
    if np.isclose(mantissa, 1.0):
        return rf"$10^{{{exponent}}}$"
    if np.isclose(mantissa, -1.0):
        return rf"$-10^{{{exponent}}}$"
    return rf"${mantissa:.2g}\times10^{{{exponent}}}$"


def _nice_step_at_least(value):
    if not np.isfinite(value) or value <= 0:
        return 1.0
    exponent = int(np.floor(np.log10(value)))
    fraction = value / (10.0**exponent)
    for nice in (1.0, 2.0, 4.0, 5.0, 8.0, 10.0):
        if fraction <= nice:
            return nice * (10.0**exponent)
    return 10.0 ** (exponent + 1)


def set_symmetric_gw_ticks(axes, values, ticks_per_side=2):
    arrays = [np.asarray(value) for value in values if np.asarray(value).size]
    if not arrays:
        return
    ymax = max(np.nanmax(np.abs(array)) for array in arrays)
    if not np.isfinite(ymax) or ymax <= 0:
        return
    step = _nice_step_at_least(ymax / ticks_per_side)
    tick_max = ticks_per_side * step
    ticks = np.arange(-ticks_per_side, ticks_per_side + 1) * step
    for ax in axes:
        ax.set_ylim(-tick_max, tick_max)
        ax.yaxis.set_major_locator(mticker.FixedLocator(ticks))


def format_shared_gw_yaxes(axes, nbins=5):
    for ax in axes:
        if not isinstance(ax.yaxis.get_major_locator(), mticker.FixedLocator):
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nbins, prune=None))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(_format_scientific_power))
        ax.yaxis.get_offset_text().set_visible(False)


def legend_key_handles():
    return [mlines.Line2D([0], [0], color=COLOR_OPTS[i], linestyle="-", lw=2, label=f"{i+1}") for i in range(len(COLOR_OPTS))]
