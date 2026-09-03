# BHDiskplots

Plotting and GW post-processing for the BHDisk simulations on Anvil.

## Layout

- `paper_plots/`: plots currently approved for the paper.
- `wip_plots/`: preserved and actively developed plots that are not yet part
  of the paper workflow.
- `helpers/`: shared data readers, units, plotting style, and GW adapters.
- `config.py`: simulation constants and the two external input roots.
- `plot_settings.py`: shared display-unit choices used across plot scripts.
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

Its small top-of-file knob block controls the cache, paper, WIP, and individual
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

`GW_STRAIN_BACKEND` near the top of `generate_gw.py` selects `"python"` or
`"fortran"`. The
Fortran implementation is retained as a regression reference; when selected,
`generate_gw.py` builds it with an available `ifort` or `gfortran`. The source
contains one narrowly repaired padding loop so bounds-checked compilers produce
the same defined result as the NumPy backend.

`GW_TIME_SCALE` in `plot_settings.py` controls every time-domain GW x axis. Its four
choices are `"M_BH"`, `"M_ADM"` (displayed as `M`), `"P_c"`, and `"code"`;
the current default is `"M_BH"`. Raw code time also adds the physical-time
axis in milliseconds.

There are two intentionally distinct disk-minus-Massless strain products in
`wip_plots/`:

- `gw_strain_from_psi4_minus_massless_all.py` is the canonical route:
  subtract aligned Psi4 first, then run the same configured FFI converter.
- `gw_strain_minus_massless_all.py` preserves the older post-hoc strain
  subtraction for comparison only; it is not generated by the standard GW
  workflow.

Detectability remains separate in `wip_plots/`. Its direct-Psi4 transforms,
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
