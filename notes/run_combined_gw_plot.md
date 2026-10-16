# Run or edit the combined GW figure

The complete measured workflow is **`gw_detectability.py`** in the repository
root. It contains the cache readers, spectral calculation, detector weighting,
SNR/reach calculation, and all detectability figures in that order.
With no `--plots` option it produces the polished Wessel-style figure:
luminosity-distance reach above characteristic strain. It reads cached, measured rPsi4, computes the
spectra and detection distances, and draws the existing layout. No
time-extrapolation fits, scalar files, extra radii or other figures are needed.

## Run it on Anvil

Use the usual plotting environment, then:

```bash
cd /anvil/projects/x-mca99s008/maxwork/BHDiskplots
python gw_detectability.py
python gw_detectability.py --radius-index 6
python gw_detectability.py --radius-index 7
```

Each radius has its own output names, so comparisons do not overwrite one
another:

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

This repository includes the code, detector curves and dependency list. The
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
uv run python gw_detectability.py --radius-index 6
# Or point at a copy of the cache tree elsewhere:
uv run python gw_detectability.py --radius-index 6 --cache-dir /path/to/gw_work
```

The usual TeX installation with Palatino fonts is required for the established
paper style. On Anvil the existing style helper loads the TeX module. On
another machine, TeX/Palatino must already be installed and on PATH. The
script reports a missing cache by its full path before running the calculation;
it stops rather than silently dropping a requested case.

## Read and edit the workflow

These are the usual changes; each has one editing location:

| Change | Edit or command |
| --- | --- |
| Extraction radius | `--radius-index 6` (both panels recompute at that radius) |
| Simulations | `--sims A1 A3`, or `SIM_NAMES` in section 1 |
| Modes, transient cut, taper, averaging | Section 1: `MODES`, `TRANSIENT_CUTOFF_MBH`, `TAPER_ALPHA`, `SOURCE_AVERAGING` |
| Illustrative BH masses/distances | Section 1: `EXAMPLE_TARGETS`; distance `None` solves the SNR threshold |
| Detectors and SNR threshold | Section 1: `ACTIVE_DETECTORS`, `SNR_THRESHOLD` |
| Numerical resolution and cosmology | Section 1, immediately below the physical choices |
| Axis limits, labels, line styles | Section 4, beside `plot_combined` and its two panel functions |
| Legend placement and panel spacing | `plot_combined` in section 4 |

Read `gw_detectability.py` in order:

1. **Settings and inputs:** a compact physical-choice block, the cache column
   reader, and the detector-table reader. Radius index 6 maps to file label 7.
2. **Calculation:** time cuts, Tukey window, FFT, harmonic reconstruction,
   plus/cross amplitude, solid-angle mean, conversion from Psi4 to strain,
   observer mass/redshift scaling, PSD weighting and SNR/distance roots.
   These are actual equations and array operations in this file.
3. **Complete workflow:** `analyze` shows the case loop, reads each selected
   cache, calls the spectral calculation with explicit choices, constructs noise,
   solves the example scales, and computes distance-reach curves.
4. **Plot and format:** `plot_combined` and its two shared panel functions
   contain the Matplotlib code. Limits, line styles and label placement sit
   beside the drawing code. The renderer is not imported from a WIP module.

The command-line parser is at the bottom. The few presentation options remain
available, including `--example-labels key`, but they are outside the path
through the calculation. The standard font/style and cache-generation readers
remain shared with other figures. Optional validation sweeps and continuation
fits are labelled in section 2 of this file; they do not run for the default command.
The batch runner calls this same file directly; the additional diagnostic
drawings are in section 5 and the single command is in section 6.

## Run a specific diagnostic

Use the same file for diagnostic subsets:

```bash
python gw_detectability.py --plots radius averaging --sims A1 A3
python gw_detectability.py --plots strain distance
```

| Plot name | Calculation and output |
| --- | --- |
| `combined` | Measured characteristic strain and distance reach together |
| `strain` | Measured characteristic strain on its own |
| `distance` | Measured SNR-threshold distance on its own |
| `radius` | Outer/inner extraction-radius comparison |
| `averaging` | Source-direction average versus one representative direction |
| `reference` | Measured strain at the fixed 50 solar mass, 100 Mpc reference source |
| `extrapolation` | Combined figure with the optional fitted continuation beyond the measured end; requires scalar data |

The direct command defaults to `combined`. `run_wip.py detectability` requests
`BATCH_PLOTS`, including the labelled extrapolation. Both entry points use
the same radius-specific filenames; there is no batch-only naming behavior.
A selected measured diagnostic does not run that fit or load
unrequested radius/direction comparisons. Only `combined` and `distance` need
the full measured mass-versus-distance scan. Illustrative strain sources still
solve their own threshold distances, as described above.

For waveform histories, the five data-preparation branches are together in
`waveform()` in `wip_plots/gw_waveforms.py`. For example:

```bash
python -m wip_plots.gw_waveforms --kinds strain --radius-index 6 --sims A1 A3
```

Other views use the same file: `--kinds strain_panel` gives the six-case
polarization grid; `--kinds strain_observer` gives physical face-on strain;
`--kinds gw_radiated` gives emitted energy, angular momentum and phase frequency.
The modal families are `psi4`, `strain`, `psi4_minus_ml`, `strain_minus_ml`
and `strain_from_psi4_minus_ml`. The last two remain distinct: subtracting two
FFI strains and integrating a Psi4 difference use different caches. The reader,
subtraction, and normalization calls are visible in each branch. Shared units
and labels remain in `helpers/gw_units.py`. Both parsers are at the bottom of
their scripts, below the calculations and plots.
