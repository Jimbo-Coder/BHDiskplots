# BHDiskplots

Plotting and GW post-processing for the BHDisk simulations on Anvil.

## Layout

- `paper_plots/`: plots currently approved for the paper.
- `wip_plots/`: preserved and actively developed plots that are not yet part
  of the paper workflow.
- `config.py`: simulation metadata, input roots, shared units, and run settings.
- `gw.py`: extraction-file reading, restart merging, FFI, and waveform caches.
- `helpers/`: scalar/2D readers, shared units/style, and separate detectability physics.
- `data/initial_profiles/`: small initial-profile inputs used by the paper plot.
- `cache/2d_indices/`: persistent text indices for the large 2D ASCII sources.
- `generate_gw.py`: the sole standard entry point for the reusable GW
  time-series cache.
- `psi4_hlm_ref/`: preserved Fortran regression reference and supporting scripts.
- `wip_plots/detectability_ref/`: collaborator-supplied historical method reference.
- `run_logs/`: ignored nohup logs and PID files; runtime state belongs here,
  never beside the top-level workflow scripts.

Plot-specific scientific and presentation choices remain near the top of each
plot file. Shared helpers are used only where several plots require exactly the
same behavior.

## Quick Start on Anvil

Activate the plotting environment, enter the repository, then run:

```bash
./generate_gw.py
./run_paper.py
```

For an isolated local environment, run `uv sync` once and prefix the same
commands with `uv run`.

Run the complete non-movie workflow with:

```bash
./run_all.py
```

For a detached run, keep its runtime files contained:

```bash
mkdir -p run_logs
nohup ./run_all.py > run_logs/run_all.log 2>&1 < /dev/null &
echo $! > run_logs/run_all.pid
```

The workflow block in `config.py` controls the cache, paper, WIP, and individual
stages. With the defaults, the GW cache is left alone and every figure is
rewritten. When the cache stage is enabled, existing products are rebuilt so
newly appended Psi4 data are included.

Run selected paper plots by short name:

```bash
./run_paper.py modes phase rhomax
```

Every plot remains directly runnable, for example:

```bash
./paper_plots/rhomax_all.py
```

Each plot keeps its scientific selectors, output filename template, and
presentation knobs in that plot file. Shared modules are limited to input
parsing, physical normalization, and presentation conventions that must remain
identical across figures. For example, the six-case polarization panel is:

```bash
./run_wip.py strain_panel
```

Generated figures are not tracked by Git. The `figures/` root contains only
paper figures with stable LaTeX-facing names (`modes.png`, `phase.png`,
`rhomax.png`, and so on). Combined exploratory plots live in `figures/wip/`,
combined waveform and detectability plots in `figures/gw/`, disk-minus-ML
products in `figures/gw/difference/`, and individual products in
`figures/A1/` through `figures/B3/`. GW filenames encode only the mode and
loaded extraction radius needed to distinguish simultaneous products. The
filename constant or template is kept beside the scientific knobs in each
plot file.

## Data Safety

Simulation directories are read-only inputs. Each simulation has an ordered
`data_roots` tuple in `config.py`; later roots replace overlapping restart data
in memory. In particular, the Massless case merges the project copy and its
scratch restart without modifying either location.

Only two machine-specific paths are configured: `MILTON_DATA_ROOT` for the
main simulation outputs and `SUPPLEMENTAL_DATA_ROOT` for the older Massless
output and recovered 2D slices. Figures, GW work, the 2D index, the Fortran
source/executable, and initial-profile inputs are all relative to the checkout.

All 2D consumers use the persistent text indices under `cache/2d_indices/`.
The first panel or movie scan records only
iteration, time, and source byte offsets. Unchanged sources reuse that index;
append-only growth rescans from the previous final iteration. The cache is
lock-protected, safe to rebuild, and never stores simulation arrays. The text
indices are tracked so collaborators on Anvil can reuse the expensive initial
scan; when a source grows, the updated index is an ordinary reviewable Git
change.

Populate every disk index ahead of plotting with:

```bash
./generate_2d_indices.py
```

Its top-of-file case, worker, and regeneration knobs control the complete 2D
cache-and-plot pass.

## GW Workflows

Mass conventions: `disk_to_bh_mass_ratio` stores the dimensionless initial
disk rest mass divided by the initial BH mass. `disk_rest_mass_code` is derived
by multiplying that ratio by `mlittle`, the initial BH mass in code units
(`G = c = M_sun = 1`). Use the ratio for mass-ratio comparison plots and the
code mass for normalizing raw waveforms, accretion rates, and coordinates.
`gw_madm` is the separate total ADM mass. The supplied initial-data ratios
agree with the evolution's initial `bhns.don` mass ratios within about 1-3%;
they are preserved rather than silently replaced by the grid diagnostic.

The standard waveform workflow is intentionally simple:

```text
Psi4 -> gauge-corrected t_ret -> fixed-frequency integration -> h_+, h_cross
```

The same route is used for disk-minus-Massless Psi4 differences. Plot scripts
read the completed cache and never run the Fortran executable implicitly. A
missing product fails with the exact cache path and asks for `generate_gw.py`.
`generate_gw.py` writes the reusable waveform products to `gw_work/`. The
default NumPy backend is a parity-tested implementation of the established
`psi4_hlm_ref/rhphc` algorithm. It preserves the legacy `.dat` layouts and also
writes `rpsi4_uniform.dat` plus `strain_cache.json`, which record the uniformly
sampled intermediate and its numerical provenance for later analysis.

`GW_STRAIN_BACKEND` in `config.py` selects `"python"` or
`"fortran"`. The
Fortran implementation is retained as a regression reference; when selected,
`generate_gw.py` builds it with an available `ifort` or `gfortran`. The source
contains one narrowly repaired padding loop so bounds-checked compilers produce
the same defined result as the NumPy backend.

`GW_TIME_SCALE` in `config.py` controls every time-domain GW x axis. Its four
choices are `"M_BH"`, `"M_ADM"` (displayed as `M`), `"P_c"`, and `"code"`;
the current default is `"M_BH"`. Raw code time also adds the physical-time
axis in milliseconds.

Edit ordinary waveform figures in `wip_plots/gw_waveforms.py`: `waveform`
selects a named series function, `plot` draws combined figures, and `plot_individual`
draws radius/mode comparisons. The 2x3 polarization layout remains in
`wip_plots/gw_strain_polarization_panel.py`. Modes and shared radii are selected
once in `config.py`; loading a different mode does not reread the cache.

For analysis, `GWRun(config).at_radius(index)` returns a radius-specific
`Waveform` containing raw `psi4` and cached `strain_result`, each with all modes.
Requesting another radius does not change an earlier result. `read_strain_cache`
and `read_difference` only read; `generate_strain` only generates. The single
rebuild/reuse decision belongs in `generate_gw.py`, not in plotting code.

```bash
./run_wip.py psi4 strain
./run_wip.py psi4_minus_ml strain_minus_ml strain_from_psi4_minus_ml
./wip_plots/run_individual.py A1 B1
```

The difference plots deliberately distinguish `h(Psi4_disk)-h(Psi4_ML)` from
`h(Psi4_disk-Psi4_ML)`. Both are retained for comparison, over shared retarded
times without extrapolation. All previous figure filenames are preserved.

Detectability plots remain separate in `wip_plots/`; numerical transforms,
detector response, and SNR calculations live in `helpers/gw_detectability.py`.
Their settings are in the detectability block of `config.py`. Direct-Psi4 transforms,
source averaging, physical scaling, windowing, and detector conventions are
documented and tested independently of ordinary waveform plots. It consumes
the cached, uniformly sampled `rpsi4_uniform.dat` when available, so the
retarded-time correction and interpolation are shared without routing the
detectability calculation through time-domain strain.
The colleague-supplied historical implementation is retained verbatim under
`wip_plots/detectability_ref/`; it is reference material, not an executable workflow.

## Plot Promotion

New plots begin in `wip_plots/`. Once the method and presentation are agreed
upon, move the file to `paper_plots/` and add it to `PAPER_PLOTS` in
`run_paper.py`.

Movie scripts are currently local and ignored. Movie frames, videos, figures,
logs, and derived GW products are also untracked.
