#!/usr/bin/env python3
"""Make just the polished luminosity-distance + characteristic-strain figure.

    python plot_gw_detectability.py
    python plot_gw_detectability.py --radius-index 6

Requires cached rPsi4 files, not raw simulation directories or FFI strain.
See notes/run_combined_gw_plot.md for inputs and the few places to edit.
"""
import json
from pathlib import Path

import gw_detectability as analysis
from config import GW_WORK_ROOT, all_sim_configs
from helpers.plot_common import parser, setup
from helpers.style import apply_styles
from wip_plots.gw_detectability_all import plot_combined

# Edit these defaults, or override the first two on the command line.
SIMULATIONS = ("A1", "A2", "A3")
RADIUS_INDEX = 8  # Zero-based parfile index: 6 -> Psi4_rad.mon.7 for A cases.
EXAMPLE_LABELS = "caption"  # "caption", "key", or "inline".


def main(argv=None):
    p = parser(__doc__)
    p.set_defaults(sims=list(SIMULATIONS))
    p.add_argument("--radius-index", type=int, choices=range(9), default=RADIUS_INDEX,
                   help="Zero-based parfile extraction index (0-8); A-case file label = index + 1")
    p.add_argument("--cache-dir", type=Path, default=GW_WORK_ROOT,
                   help="Directory containing A1_psi4*, A2_psi4*, A3_psi4* caches")
    p.add_argument("--example-labels", choices=("caption", "key", "inline"), default=EXAMPLE_LABELS)
    args = p.parse_args(argv)

    # 1. Check and print the actual inputs before attempting any expensive work.
    cases = all_sim_configs(args.sims)
    unknown = set(name.upper() for name in args.sims) - {case.name for case in cases}
    if unknown:
        p.error("Unknown simulation(s): " + ", ".join(sorted(unknown)))
    inputs = []
    for case in cases:
        label = args.radius_index + case.gw_psi4_file_index_offset
        folder = args.cache_dir / f"{case.name}_psi4{label}"
        inputs.append(dict(case=case.name, radius_index=args.radius_index,
                           psi4_file_label=label, cache=str(folder.resolve())))
        print(f"{case.name}: parfile index {args.radius_index} -> Psi4_rad.mon.{label} -> {folder}", flush=True)
        for name in ("rpsi4_uniform.dat", "strain_cache.json"):
            if not (folder/name).is_file():
                p.error(f"Missing {folder/name}. GW caches are not included in GitHub. "
                        "Copy the selected caches from Anvil or use --cache-dir PATH. "
                        "See notes/run_combined_gw_plot.md.")

    # 2. Compute measured spectra and SNR=8 reaches at THIS extraction radius.
    #    No tail fits, other radii, scalar data, or ancillary plots are requested.
    try:
        result = analysis.analyze(args.sims, radius_index=args.radius_index,
                                  cache_root=args.cache_dir, strict=True)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    apply_styles(list(result.sources.values()))

    # 3. Apply the usual TeX/Palatino style and draw the existing polished layout.
    setup(args)
    filename = f"gw/detectability/radius_{args.radius_index}.png"
    plot_combined(result, args, filename=filename, example_labels=args.example_labels)

    # 4. Save the radius/input list alongside the figure, caption and example SNRs.
    if not args.no_save:
        path = args.outdir / filename
        path.with_name(path.stem+"_inputs.json").write_text(json.dumps(inputs, indent=2)+"\n")
        print(f"Figure: {path.resolve()} (and .pdf)", flush=True)
        print(f"Caption and source parameters: {path.with_suffix('').resolve()}_caption.tex / _examples.json", flush=True)
    return result


if __name__ == "__main__":
    main()
