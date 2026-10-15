#!/usr/bin/env python3
"""Recreate Shibata et al. (2021), Fig. 5, and overlay our initial disks.

Reference coordinates: Table I, https://arxiv.org/abs/2101.05440v2.
Types follow Fig. 5 (M13 and H15 are Iw there; the text calls them borderline).
Our coordinates/masses: Initial_data/BHT/<case>/bhtphyseq.dat on Anvil,
matching the sequence number to sol_XX, not the last sequence in the file.
Columns used: Outer point (width), Rest, proper mass (first), Black hole mass.
Both axes use the initial BH mass, never the total ADM mass.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from config import all_sim_configs
from helpers.plot_common import parser, savefig, setup
from helpers.style import (
    AXES_LEGEND_BORDERAXESPAD, COLOR_OPTS, COMPACT_LEGEND_KWARGS,
    PAPER_FONT_SIZE, figure_size, format_paper_axes,
)

# Scientific knobs. Extended boundaries are visual extrapolations of Fig. 5.
REFERENCE_XLIM = (20, 55)
COMPARISON_XLIM = (18, 158)
REFERENCE_YLIM = (0.2, 1.1)
COMPARISON_YLIM = (-0.035, 1.1)
BH_MASS_REFERENCE_MSUN = 50.0
BH_MASS_CODE = 0.05

# name, r_out/M_BH, disk mass in solar masses, published waveform class
REFERENCE = [
    ("L11", 21.8, 15.1, "Iw"), ("L12", 27.9, 15.1, "II"),
    ("J12", 25.4, 20.0, "Iw"), ("J13", 32.8, 19.9, "II"),
    ("K13", 29.7, 25.0, "Iw"), ("K14", 38.6, 25.0, "II"),
    ("M12", 21.1, 30.1, "Is"), ("M13", 27.0, 30.0, "Iw"),
    ("M14", 35.0, 29.9, "Iw"), ("M15", 45.6, 30.1, "II"),
    ("H14", 28.9, 40.0, "Is"), ("H15", 37.4, 40.0, "Iw"),
    ("H16", 49.0, 40.0, "Iw"), ("E15", 31.0, 50.1, "Is"),
    ("E16", 40.0, 50.2, "Is"), ("E17", 52.6, 50.0, "Iw"),
]
# name, outer coordinate radius and initial disk rest mass in code units.
# Full-precision ID values retained here for reproducibility of this comparison.
OUR_DISKS = [
    ("A1", 2.326250000000002, 9.369608118348474e-4),
    ("A2", 4.769999999999994, 9.910278743792066e-3),
    ("A3", 6.185897093303726, 3.010755255514919e-2),
    ("B1", 2.441250000000002, 5.354429876098988e-4),
    ("B2", 5.062023879535146, 6.058632527696616e-3),
    ("B3", 7.494811790650769, 3.325546186065383e-2),
]

# Presentation knobs.
FIG_HEIGHT = 4.5
COMPARISON_FIGSIZE = (3.75, 3.0)
SEQUENCE_FIGSIZE = (3.75, 4.2)
INCLUDE_SEQUENCES = True
SEQUENCE_CASES = {"A": {1: "A1", 5: "A2", 7: "A3"},
                  "B": {28: "B1", 32: "B2", 35: "B3"}}
COMPARISON_FONT_SIZE = 8
REGION_FONT_SIZE = 8
TYPE_STYLES = {"Is": ("+", "#ae28df"), "Iw": ("x", "#21a58a"), "II": ("*", "#36a5e8")}


@dataclass(frozen=True)
class SequenceModel:
    number: int
    outer_radius: float
    disk_rest_mass: float
    bh_mass: float

    @property
    def coordinates(self):
        return self.outer_radius / self.bh_mass, self.disk_rest_mass / self.bh_mass


def read_cocal_sequence(path):
    """Read every numbered model, using rest mass and its own initial BH mass."""
    blocks = re.split(r"==\s*Sequence number\s*==\s*(\d+)", Path(path).read_text())
    if len(blocks) == 1:
        raise ValueError(f"No COCAL sequence records in {path}")
    models = []
    for number, block in zip(blocks[1::2], blocks[2::2]):
        values = []
        for label in ("Outer point (width)", "Rest, proper mass", "Black hole mass (M)"):
            match = re.search(re.escape(label) + r"\s*=\s*(\S+)", block)
            if match is None:
                raise ValueError(f"{path}, sequence {number}: missing {label}")
            value = float(match[1].replace("D", "E").replace("d", "e"))
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{path}, sequence {number}: invalid {label}")
            values.append(value)
        models.append(SequenceModel(int(number), *values))
    if len({model.number for model in models}) != len(models):
        raise ValueError(f"Duplicate sequence numbers in {path}")
    # Retain sequence order: the high-mass branches turn back in radius.
    return models


def load_sequences():
    configs = {case.name: case for case in all_sim_configs()}
    evolved = {name: (radius / BH_MASS_CODE, mass / BH_MASS_CODE)
               for name, radius, mass in OUR_DISKS}
    sequences = {}
    for family, selected in SEQUENCE_CASES.items():
        case = configs[next(iter(selected.values()))]
        models = read_cocal_sequence(case.initial_data_path / "bhtphyseq.dat")
        by_number = {model.number: model for model in models}
        for number, name in selected.items():
            if number not in by_number or not np.allclose(
                    by_number[number].coordinates, evolved[name], rtol=1e-12, atol=1e-14):
                raise ValueError(f"{name}: selected model disagrees with the COCAL sequence")
        sequences[family] = models
    return sequences


def _draw_sequences(ax, sequences):
    for family, models in sequences.items():
        coordinates = np.array([model.coordinates for model in models])
        ax.plot(*coordinates.T, color="0.55", alpha=.55, linewidth=.8,
                linestyle="-" if family == "A" else "--",
                marker="o" if family == "A" else "^", markersize=2.3,
                markerfacecolor="white", markeredgewidth=.5,
                label=f"Sequence {family}", zorder=1)


def plot(include_ours=False, include_sequences=INCLUDE_SEQUENCES, sequence_overview=False):
    figsize = COMPARISON_FIGSIZE if include_ours else figure_size("double", FIG_HEIGHT)
    overview = None
    if include_ours and include_sequences and sequence_overview:
        sequences = load_sequences()
        fig = plt.figure(figsize=SEQUENCE_FIGSIZE, layout="constrained")
        grid = fig.add_gridspec(2, 1, height_ratios=(1, 2.35))
        ax = fig.add_subplot(grid[1])
        overview = fig.add_subplot(grid[0])
        for axis in (overview, ax):
            _draw_sequences(axis, sequences)
        xy = np.array([model.coordinates for models in sequences.values() for model in models])
        overview.set(xlim=(min(REFERENCE_XLIM[0], xy[:, 0].min()) - 3, COMPARISON_XLIM[1]),
                     ylim=(-.1, xy[:, 1].max() * 1.08), yticks=(0, 2, 4))
        overview.text(.02, .95, "Full COCAL sequences", transform=overview.transAxes,
                      va="top", fontsize=7)
        overview.legend(loc="upper right", fontsize=6.5, frameon=False, handlelength=2)
        format_paper_axes(overview)
        overview.tick_params(which="major", labelsize=7, length=3, width=.6, pad=3)
        overview.tick_params(which="minor", length=1.5, width=.5)
        for spine in overview.spines.values():
            spine.set_linewidth(.7)
    else:
        fig, ax = plt.subplots(figsize=figsize, layout="constrained")
        if include_ours and include_sequences:
            _draw_sequences(ax, load_sequences())
    for kind, (marker, color) in TYPE_STYLES.items():
        rows = [row for row in REFERENCE if row[3] == kind]
        ax.scatter([r[1] for r in rows], [r[2] / BH_MASS_REFERENCE_MSUN for r in rows],
                   marker=marker, color=color, s=24 if include_ours else 48,
                   linewidths=0.9 if include_ours else 1.2,
                   label=f"Type {kind}", zorder=3)
    x = np.linspace(*(COMPARISON_XLIM if include_ours else REFERENCE_XLIM), 300)
    for intercept in (3, -10):
        ax.plot(x, (x + intercept) / BH_MASS_REFERENCE_MSUN,
                color="#ae28df", linestyle="--", linewidth=1)
    if include_ours:
        for xtext, ytext, description, fontsize in (
            (21, .89, "Type Is\nShort\nburst", 7),
            (77, .46, "Type II\nSustained oscillations", REGION_FONT_SIZE),
        ):
            ax.text(xtext, ytext, description, fontsize=fontsize,
                    ha="left", va="center", linespacing=1.2)
        # Point into the narrow band without obscuring its boundaries or data.
        ax.annotate("Type Iw\nFew-cycle burst", xy=(46, .88), xytext=(69, .93),
                    fontsize=REGION_FONT_SIZE, ha="left", va="center",
                    linespacing=1.2,
                    arrowprops=dict(arrowstyle="->", color="0.25", linewidth=0.9,
                                    shrinkA=5, shrinkB=2,
                                    connectionstyle="arc3,rad=0"))
        for name, radius, mass in OUR_DISKS:
            color = COLOR_OPTS[int(name[1]) - 1]
            ax.scatter(radius / BH_MASS_CODE, mass / BH_MASS_CODE, s=36,
                       marker="o" if name[0] == "A" else "^",
                       facecolors=color if name[0] == "A" else "white",
                       edgecolors=color, linewidths=1.1, zorder=5)
            if overview is not None:
                overview.scatter(radius / BH_MASS_CODE, mass / BH_MASS_CODE, s=16,
                                 marker="o" if name[0] == "A" else "^",
                                 facecolors=color if name[0] == "A" else "white",
                                 edgecolors=color, linewidths=.8, zorder=5)
            offset = (-12, 8) if name == "A1" else (10, 8) if name == "B1" else (0, 7)
            ax.annotate(rf"$\mathrm{{{name}}}$", (radius / BH_MASS_CODE, mass / BH_MASS_CODE),
                        xytext=offset, textcoords="offset points", ha="center", color=color,
                        fontsize=COMPARISON_FONT_SIZE)
        legend_style = {**COMPACT_LEGEND_KWARGS, "fontsize": 7.5}
        handles, labels = ax.get_legend_handles_labels()
        reference_handles = [(handle, label) for handle, label in zip(handles, labels)
                             if label.startswith("Type ")]
        ax.legend(*zip(*reference_handles), loc="upper right", title="Shibata et al. (2021)",
                  title_fontsize=7.5,
                  borderaxespad=AXES_LEGEND_BORDERAXESPAD, **legend_style)
    else:
        for xtext, ytext, text in ((30, .9, "Type Is"), (40, .76, "Type Iw"), (40, .48, "Type II")):
            ax.text(xtext, ytext, text, fontsize=PAPER_FONT_SIZE)
    ax.set(xlim=COMPARISON_XLIM if include_ours else REFERENCE_XLIM,
           ylim=COMPARISON_YLIM if include_ours else REFERENCE_YLIM,
           xlabel=r"$r_{\mathrm{out}}/M_{\mathrm{BH},0}$",
           ylabel=r"$M_{\mathrm{disk}}/M_{\mathrm{BH},0}$")
    format_paper_axes(ax)
    if include_ours:
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.tick_params(which="major", labelsize=COMPARISON_FONT_SIZE,
                       length=4, width=0.8, pad=3)
        ax.tick_params(which="minor", length=2, width=0.6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
    return fig


def main(argv=None):
    argument_parser = parser(__doc__)
    argument_parser.add_argument("--sequences", action=argparse.BooleanOptionalAction,
                                default=INCLUDE_SEQUENCES,
                                help="Overlay COCAL sequence curves (default: enabled)")
    argument_parser.add_argument("--overview", action=argparse.BooleanOptionalAction,
                                default=False, help="Include the full-sequence overview above the comparison")
    args = argument_parser.parse_args(argv)
    if args.overview and not args.sequences:
        argument_parser.error("--overview requires --sequences")
    setup(args)
    selected = "sequences_overview" if args.overview else "sequences_zoom" if args.sequences else "selected_only"
    variants = ((False, False, False, "reference"), (True, False, False, "selected_only"),
                (True, True, False, "sequences_zoom"), (True, True, True, "sequences_overview"))
    for include_ours, sequences, overview, name in variants:
        fig = plot(include_ours, include_sequences=sequences, sequence_overview=overview)
        try:
            # Keep the comparison's exported width exactly 3.75 inches.
            with plt.rc_context({"savefig.bbox": None} if include_ours else {}):
                if not args.no_save:
                    path = args.outdir / "wip" / "compactness" / f"{name}.pdf"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    fig.savefig(path)
                    print(f"saved {path}")
                savefig(fig, args, f"wip/compactness/{name}.png")
                if not args.no_save and name in ("reference", selected):
                    alias = "disk_compactness_reference" if name == "reference" else "disk_compactness_comparison"
                    for suffix in ("png", "pdf"):
                        shutil.copyfile(args.outdir / "wip" / "compactness" / f"{name}.{suffix}",
                                        args.outdir / "wip" / f"{alias}.{suffix}")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
