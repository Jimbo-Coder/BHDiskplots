"""Common command-line helpers for plot scripts."""
from __future__ import annotations

import argparse
from pathlib import Path

from config import PLOTS_DIR
from .reader import load_sims
from .style import apply_paper_style


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--sims", nargs="+", default=None, help="Simulation names to include, e.g. A1 A2 B1")
    p.add_argument("--outdir", type=Path, default=PLOTS_DIR, help="Directory for saved figures")
    p.add_argument("--show", action="store_true", help="Show figure interactively after saving")
    p.add_argument("--no-save", action="store_true", help="Do not save figure")
    return p


def setup(args):
    apply_paper_style()
    args.outdir.mkdir(parents=True, exist_ok=True)


def savefig(fig, args, filename):
    import matplotlib.pyplot as plt

    try:
        if not args.no_save:
            path = args.outdir / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path)
            print(f"saved {path}")
        if args.show:
            plt.show()
    finally:
        plt.close(fig)


def save_individual_fig(fig, args, sim, filename):
    """Save a per-simulation figure under the selected output root."""
    sim_name = sim if isinstance(sim, str) else sim.config.name
    savefig(fig, args, Path(sim_name) / filename)
