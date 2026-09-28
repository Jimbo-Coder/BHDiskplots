#!/usr/bin/env python3
"""Recreate Shibata et al. (2021), Fig. 5, and overlay our initial disks.

Reference coordinates: Table I, https://arxiv.org/abs/2101.05440v2.
Types follow Fig. 5 (M13 and H15 are Iw there; the text calls them borderline).
Our coordinates/masses: Initial_data/BHT/<case>/bhtphyseq.dat on Anvil,
matching the sequence number to sol_XX, not the last sequence in the file.
Columns used: Outer point (width), Rest, proper mass (first), Black hole mass.
Both axes use the initial BH mass, never the total ADM mass.
"""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
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
TYPE_STYLES = {"Is": ("+", "#ae28df"), "Iw": ("x", "#21a58a"), "II": ("*", "#36a5e8")}


def plot(include_ours=False):
    fig, ax = plt.subplots(figsize=figure_size("double", FIG_HEIGHT), layout="constrained")
    for kind, (marker, color) in TYPE_STYLES.items():
        rows = [row for row in REFERENCE if row[3] == kind]
        ax.scatter([r[1] for r in rows], [r[2] / BH_MASS_REFERENCE_MSUN for r in rows],
                   marker=marker, color=color, s=48, linewidths=1.2,
                   label=f"Type {kind}", zorder=3)
    x = np.linspace(*(COMPARISON_XLIM if include_ours else REFERENCE_XLIM), 300)
    for intercept in (3, -10):
        ax.plot(x, (x + intercept) / BH_MASS_REFERENCE_MSUN,
                color="#ae28df", linestyle="--", linewidth=1)
    if include_ours:
        for xtext, ytext, description in (
            (21, .89, "Type Is\nShort burst"),
            (77, .46, "Type II\nSustained oscillations"),
        ):
            ax.text(xtext, ytext, description, fontsize=PAPER_FONT_SIZE,
                    ha="left", va="center", linespacing=1.2)
        ax.text(49, .91, "Type Iw\nFew-cycle\nburst", fontsize=PAPER_FONT_SIZE,
                ha="center", va="center", linespacing=1.1)
        for name, radius, mass in OUR_DISKS:
            color = COLOR_OPTS[int(name[1]) - 1]
            ax.scatter(radius / BH_MASS_CODE, mass / BH_MASS_CODE, s=62,
                       marker="o" if name[0] == "A" else "^",
                       facecolors=color if name[0] == "A" else "white",
                       edgecolors=color, linewidths=1.5, zorder=5)
            offset = (-15, 10) if name == "A1" else (12, 9) if name == "B1" else (0, 10)
            ax.annotate(rf"$\mathrm{{{name}}}$", (radius / BH_MASS_CODE, mass / BH_MASS_CODE),
                        xytext=offset, textcoords="offset points", ha="center", color=color,
                        fontsize=PAPER_FONT_SIZE)
        ax.legend(loc="upper right", title="Shibata et al. (2021)",
                  title_fontsize=PAPER_FONT_SIZE, borderaxespad=AXES_LEGEND_BORDERAXESPAD,
                  **COMPACT_LEGEND_KWARGS)
    else:
        for xtext, ytext, text in ((30, .9, "Type Is"), (40, .76, "Type Iw"), (40, .48, "Type II")):
            ax.text(xtext, ytext, text, fontsize=PAPER_FONT_SIZE)
    ax.set(xlim=COMPARISON_XLIM if include_ours else REFERENCE_XLIM,
           ylim=COMPARISON_YLIM if include_ours else REFERENCE_YLIM,
           xlabel=r"$r_{\mathrm{out}}/M_{\mathrm{BH},0}$",
           ylabel=r"$M_{\mathrm{disk}}/M_{\mathrm{BH},0}$")
    format_paper_axes(ax)
    return fig


def main(argv=None):
    args = parser(__doc__).parse_args(argv)
    setup(args)
    for include_ours, name in ((False, "disk_compactness_reference"), (True, "disk_compactness_comparison")):
        fig = plot(include_ours)
        try:
            if not args.no_save:
                path = args.outdir / "wip" / f"{name}.pdf"
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path)
                print(f"saved {path}")
            savefig(fig, args, f"wip/{name}.png")
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
