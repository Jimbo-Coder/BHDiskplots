"""Opt-in A1/A3 complex-(2,2) mismatch: manual calculation versus kuibit."""
from pathlib import Path
import json
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from config import GW_WORK_ROOT, GW_OUTERMOST_PARFILE_INDEX, all_sim_configs
from gw_mismatch import compare, prepare_pair, sha256
from helpers.plot_common import parser, setup, savefig


def main(argv=None):
    p = parser(__doc__)
    p.add_argument("--cache-root", type=Path, default=GW_WORK_ROOT)
    p.add_argument("--start", type=float, default=1000., help="Retarded time / initial BH mass")
    p.add_argument("--end", type=float, default=None, help="Default: shorter waveform's end, in M_BH")
    p.add_argument("--radius-index", type=int, default=GW_OUTERMOST_PARFILE_INDEX)
    p.add_argument("--taper", type=float, default=.05)
    p.add_argument("--shift-limit", type=float, default=None, help="Maximum |time shift| in M_BH; default half retained duration")
    p.add_argument("--fmin", type=float, default=None, help="Dimensionless M_BH f; default 3 / retained duration")
    p.add_argument("--fmax", type=float, default=None, help="Dimensionless M_BH f; default 0.95 Nyquist")
    p.add_argument("--detector", choices=("white", "ligo", "et", "ce", "decigo", "lisa"), default="white")
    p.add_argument("--mass", type=float, default=50., help="Source-frame BH mass in solar masses, for detector weighting")
    p.add_argument("--redshift", type=float, default=0.)
    p.add_argument("--num-times", type=int, default=257)
    p.add_argument("--num-phases", type=int, default=128)
    p.add_argument("--tag", default=None, help="Optional output subfolder for a parameter variant")
    p.add_argument("--no-plot", action="store_true", help="Compute and save JSON/NPZ without requiring TeX")
    args = p.parse_args(argv)
    if args.sims not in (None, ["A1", "A3"]):
        p.error("This diagnostic compares A1 and A3; omit --sims")
    if args.mass <= 0 or args.redshift < 0 or not np.isfinite(args.mass + args.redshift):
        p.error("Need finite positive mass and nonnegative redshift")
    if args.tag is not None and (Path(args.tag).name != args.tag or args.tag in (".", "..")):
        p.error("--tag must be one folder name")
    inputs, waves = [], []
    for cfg in all_sim_configs(["A1", "A3"]):
        label = args.radius_index + cfg.gw_psi4_file_index_offset
        folder = args.cache_root / f"{cfg.name}_psi4{label}"
        path = folder / "rhphc.dat"
        data = np.loadtxt(path, usecols=(0, 1, 2))
        waves.append((data[:, 0]/cfg.mlittle,
                      (data[:, 1]-1j*data[:, 2])/cfg.mlittle))
        manifest = folder / "strain_cache.json"
        inputs.append({"case": cfg.name, "cache": str(path.resolve()),
                       "sha256": sha256(path), "initial_BH_mass_code": cfg.mlittle,
                       "cache_manifest_sha256": sha256(manifest),
                       "cache_manifest": json.loads(manifest.read_text())})
    time, h1, h3, selection = prepare_pair(*waves[0], *waves[1], start=args.start,
                                         end=args.end, alpha=args.taper)
    duration = selection["duration_MBH"]
    dt = selection["dt_MBH"]
    fmin = 3/duration if args.fmin is None else args.fmin
    fmax = .95*.5/dt if args.fmax is None else args.fmax
    psd = None
    weighting = {"detector": args.detector, "channels": "equal, orthogonal plus/cross"}
    if args.detector != "white":
        from gw_detectability import (load_detector_curves, effective_detector_asd,
                                      DETECTOR_BOUNDS, SECONDS_PER_M_SUN)
        scale = SECONDS_PER_M_SUN * args.mass * (1+args.redshift)
        low, high = DETECTOR_BOUNDS[args.detector]
        curves = load_detector_curves()
        if args.detector in curves:
            low = max(low, curves[args.detector].frequency[0])
            high = min(high, curves[args.detector].frequency[-1])
        fmin, fmax = max(fmin, low*scale), min(fmax, high*scale)
        def psd(frequency):
            return effective_detector_asd(args.detector, frequency/scale, curves)**2
        weighting.update(source_BH_mass_Msun=args.mass, redshift=args.redshift,
                         mass_time_seconds=scale, band_Hz=[fmin/scale, fmax/scale])
    limit = duration/2 if args.shift_limit is None else args.shift_limit
    if limit > duration:
        p.error("--shift-limit must not exceed the retained duration")
    print(f"A1/A3: {selection}; weighting={weighting}; band M_BH f={[fmin, fmax]}", flush=True)
    result, (lags, overlaps) = compare(time, h1, h3, fmin, fmax, limit, psd,
                                      args.num_times, args.num_phases)
    result.update(schema_version=1, status="WIP diagnostic, specified waveform and interval",
                  waveform="cached FFI r(h_plus-i h_cross)_22 / initial M_BH",
                  inputs=inputs, selection=selection, weighting=weighting,
                  band_MBH_f=[fmin, fmax], shift_limit_MBH=limit,
                  search="broad kuibit grid, independent kuibit-centered refinement; manual all-sample lags plus local refinement",
                  implementation_sha256={name: sha256(Path(__file__).resolve().parents[1]/name)
                                         for name in ("gw_mismatch.py", "wip_plots/gw_mismatch.py")})
    print(json.dumps({key: result[key] for key in ("manual_optimized", "grid_checks", "kuibit_minus_manual_optimized")}, indent=2), flush=True)
    # A coarse kuibit search can miss a narrow peak. Report it rather than
    # making the library agree by centering its search on the manual answer.
    if result["kuibit_minus_manual_optimized"] > 1e-3:
        print("WARNING: kuibit search not converged; increase --num-times/--num-phases", flush=True)
    if result["manual_optimized"]["near_search_boundary"]:
        print("WARNING: overlap maximum near the allowed time-shift boundary", flush=True)
    root = Path("wip/gw_mismatch") / (args.tag or args.detector)
    if not args.no_save:
        destination = args.outdir/root
        destination.mkdir(parents=True, exist_ok=True)
        (destination/"A1_A3.json").write_text(json.dumps(result, indent=2)+"\n")
        np.savez_compressed(destination/"A1_A3_inputs.npz", time_MBH=time,
                            A1=h1, A3=h3, lags_MBH=lags, optimized_overlap=overlaps)
    if args.no_plot:
        return result
    setup(args)
    fig, (wave, overlap) = plt.subplots(2, 1, figsize=(7., 5.8))
    fig.subplots_adjust(left=.13, right=.97, bottom=.11, top=.94, hspace=.42)
    count = selection["retained_samples"]
    for values, color, label in ((h1, "#0072B2", "A1"), (h3, "#D55E00", "A3")):
        wave.plot(time[:count]+selection["start_MBH"], values[:count].real / np.max(np.abs(values)),
                  color=color, lw=.85, label=label)
    wave.set(xlabel=r"$t_{\mathrm{ret}}/M_{\mathrm{BH}}$",
             ylabel=r"$\mathrm{Re}(h_{22})/\max|h_{22}|$")
    wave.legend(loc="upper left", ncol=2, fontsize=10)
    wave.set_title(r"A1 versus A3: finite $(2,2)$ FFI strain; " + args.detector + " weighting", fontsize=11)
    overlap.plot(lags, overlaps, color="black", lw=1., label="Manual: phase maximized")
    for run, color, marker in zip(result["grid_checks"], ("#009E73", "#D55E00"), ("o", "x")):
        for method in ("manual_same_grid", "kuibit"):
            r = run[method]
            overlap.plot(r["time_shift_MBH"], 1-r["mismatch"], marker=marker,
                         ms=7 if method=="manual_same_grid" else 4, mfc="none",
                         color=color, linestyle="none",
                         label=(run["search"].capitalize()+": manual / kuibit") if method=="manual_same_grid" else None)
    opt = result["manual_optimized"]
    ymax = min(1.04, max(.05, 1.65*(1-opt["mismatch"])))
    overlap.set(xlabel=r"$\Delta t/M_{\mathrm{BH}}$", ylabel="Normalized overlap", ylim=(0, ymax), xlim=(-limit, limit))
    overlap.legend(loc="upper right", fontsize=9)
    overlap.text(.02, .97, f"Manual mismatch: {opt['mismatch']:.6f}\n"
                 f"Kuibit mismatch: {result['grid_checks'][-1]['kuibit']['mismatch']:.6f}",
                 transform=overlap.transAxes, va="top", fontsize=10)
    # Shared savefig closes the figure, so save the PDF first with the same style.
    if not args.no_save:
        fig.savefig(args.outdir/root/"A1_A3.pdf", bbox_inches=None)
    savefig(fig, args, root/"A1_A3.png")
    return result


if __name__ == "__main__":
    main()
