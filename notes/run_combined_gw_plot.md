# Run or edit the combined GW figure

The short entry point is **`plot_gw_detectability.py`** in the repository root.
It produces only the polished Wessel-style figure: luminosity-distance reach
above characteristic strain. It reads cached, measured rPsi4, computes the
spectra and detection distances, and draws the existing layout. No
time-extrapolation fits, scalar files, extra radii or other figures are needed.

## Run it on Anvil

Use the usual plotting environment, then:

```bash
cd /anvil/projects/x-mca99s008/maxwork/BHDiskplots
python plot_gw_detectability.py
python plot_gw_detectability.py --radius-index 6
python plot_gw_detectability.py --radius-index 7
```

Each radius has its own output names, so comparisons do not overwrite one
another or the existing paper export:

```text
figures/gw/detectability/radius_6.png
figures/gw/detectability/radius_6.pdf
figures/gw/detectability/radius_6_caption.tex
figures/gw/detectability/radius_6_examples.json
figures/gw/detectability/radius_6_inputs.json
```

The caption and example JSON contain masses, distances, redshifts and SNRs.
The inputs JSON records the chosen index, file labels and cache directories.
For another output root use `--outdir PATH`. Run `--help` for all options.

## Which radius does 6 mean?

`--radius-index` is the **zero-based index in the extraction-radius list in
the simulation parfile**. It is not a radius in mass units, and is not the
number at the end of the raw filename. The A-case mapping is:

| Parfile index / command option | Raw Psi4 file | Cache directories |
| --- | --- | --- |
| 6 | `Psi4_rad.mon.7` | `A1_psi47`, `A2_psi47`, `A3_psi47` |
| 7 | `Psi4_rad.mon.8` | `A1_psi48`, `A2_psi48`, `A3_psi48` |
| 8 (current paper default) | `Psi4_rad.mon.9` | `A1_psi49`, `A2_psi49`, `A3_psi49` |

If by "radius 6" you mean **`Psi4_rad.mon.6`**, use `--radius-index 5` for
the A cases. The script prints the actual mapping before calculating.
Both panels are recomputed at the chosen radius, including the SNR=8 example
distances. The script does not change the global default or mix radii.

## Run it from a clone on another computer

The code, detector curves and plotting dependencies are in GitHub. The
simulation-derived **GW caches are ignored by Git**. A fresh clone alone
therefore does not contain enough data to reproduce the signal curves.

For each selected case/radius, copy these two files from Anvil:

```text
gw_work/A1_psi47/rpsi4_uniform.dat
gw_work/A1_psi47/strain_cache.json
```

Repeat for A2 and A3; use `psi48` for index 7 or `psi49` for index 8. Preserve
those directory names. You do not need the raw simulation directories,
`rhphc.dat`, or the complete `gw_work` tree for this figure. The input cache's
recorded source path may refer to Anvil; this plotting command does not open
that path. It reads the two local cache files instead.

Then, from the checkout:

```bash
uv sync
uv run python plot_gw_detectability.py --radius-index 6
# Or point at a copy of the cache tree elsewhere:
uv run python plot_gw_detectability.py --radius-index 6 --cache-dir /path/to/gw_work
```

The usual TeX installation with Palatino fonts is required for the established
paper style. On Anvil the existing style helper loads the TeX module. On
another machine, TeX/Palatino must already be installed and on PATH. The
script reports a missing cache by its full path before running the calculation;
it stops rather than silently dropping a requested case.

The old `wip_plots/gw_detectability_all.py` is also directly runnable, but runs
the larger diagnostic suite, including conditional tails. It is not the
recommended entry point for editing this one figure. Without the collaborator's
actual command/error, we cannot identify which dependency caused their failure.

## Where to edit

1. **Cases and extraction index:** the three short defaults at the top of
   `plot_gw_detectability.py`, or `--sims A1 A2 A3` and `--radius-index 6`.
   `--example-labels key` puts the source-parameter key on the figure;
   `caption` is the polished default and `inline` is also available.
2. **Scientific choices:** the numbered settings block at the top of
   `gw_detectability.py`: `MODES`, `TRANSIENT_CUTOFF_MBH`, `TAPER_ALPHA`,
   `SOURCE_AVERAGING`, `LOW_FREQUENCY_CYCLES`, detector choices, example
   masses and `SNR_THRESHOLD`. Its `analyze` function calls the steps in order.
3. **Plot appearance:** `plot_combined` in
   `wip_plots/gw_detectability_all.py` draws this figure. The display limits,
   detector line styles, label sizes and figure height are near the top of
   that same file. `_draw_horizon` and `_draw_characteristic_strain` draw the
   individual panels. Shared TeX/fonts remain in `helpers/style.py`.

The driver is deliberately short: check inputs, compute, draw, print/save
outputs. The mathematical routines are reused so changes can be checked
against the existing numerical tests. The usual `run_wip.py detectability`
workflow and its paper-facing output names keep working.
