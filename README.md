# BHDiskplots

Plotting and GW post-processing for the BHDisk simulations on Anvil.

## Layout

- `paper_plots/`: plots currently approved for the paper.
- `wip_plots/`: preserved and actively developed plots that are not yet part
  of the paper workflow.
- `config.py`: simulation metadata, input roots, shared units, and run settings.
- `gw.py`: extraction-file reading, restart merging, FFI, and waveform caches.
- `helpers/`: scalar/2D readers and shared units/style.
- `gw_detectability.py`: the seven-step direct-Psi4 detectability analysis and its settings.
- `data/initial_profiles/`: small initial-profile inputs used by the paper plot.
- `cache/2d_indices/`: persistent text indices for the large 2D ASCII sources.
- `generate_gw.py`: the sole standard entry point for the reusable GW
  time-series cache.
- `psi4_hlm_ref/`: preserved Fortran regression reference and supporting scripts.
- `notes/`: human feedback, running knowledge/work logs, and source references.
- `run_logs/`: ignored nohup logs and PID files; runtime state belongs here,
  never beside the top-level workflow scripts.

Plot-specific scientific and presentation choices remain near the top of each
plot file. Shared helpers are used only where several plots require exactly the
same behavior.

## Notes and Working Copies

Use this checkout, not the retired dated sync folders. The canonical paths are
`/Users/mrizzo/phys_research/AnvilSyncing/BHDiskplots` on the Mac and
`/anvil/projects/x-mca99s008/maxwork/BHDiskplots` on Anvil.

- [Human GW method proposal](notes/human_gw_method.md)
- [Current work and unresolved decisions](notes/WORKING.md)
- [Established conventions and dated numerical evidence](notes/KNOWLEDGE.md)

Keep editing the `human*` files in `notes/`; they are preserved verbatim when
syncing. Completed small plot notes are recorded in `WORKING.md` and removed;
the larger GW method note remains for collaboration.
Before local work, compare and pull Anvil's uncommitted edits as well
as checking the Git commit. Before syncing back, check for new Anvil changes
again. Never mirror `.git/`, environments, generated outputs, logs or the local
temporary review products between machines, and do not use a broad
`rsync --delete` on the repo.

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
stages. By default, the GW cache is rebuilt first, including newly appended
Psi4 data, followed by paper, WIP, and all individual plots with extras.
Plots only read the cache; they do not repeatedly regenerate it. A failed
cache-generation stage stops the workflow before plotting. Set `RUN_GW_CACHE`
to `False` only for an intentional plot-only rerun.

Run selected paper plots by short name:

```bash
./run_paper.py modes phase rhomax
```

The exploratory C3–C5 density-mode comparison is `./run_wip.py modes_345`;
it writes `figures/wip/modes_345.png` with the same $|C_m|/C_0$ normalization.

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

For Shibata Fig. 4-style **face-on time-domain strain** at 100 Mpc:

```bash
./run_wip.py strain_observer
```

`wip_plots/gw_strain_observer.py` contains its source mass (default 50 solar
masses), distance, radius, mode subset, initial-time cut, and layout settings.
It makes `figures/gw/strain_observer_A.png` and `strain_observer_B.png`, with
one case per row and both polarizations. These reuse the normal FFI cache;
they do not invert the detectability spectrum. At the rotation axis only
`m=2` contributes, so the default `(2,+/-1),(2,+/-2)` subset reduces to `(2,2)`
times `sqrt(5/(4*pi))`. Times are observer seconds relative to each retained
amplitude peak; the existing cosmology adds a 2.2% redshift correction at
100 Mpc (optional). No new window, integration cutoff or extrapolation is
applied. The first 1000 BH-mass units are excluded, as for detectability.
Our cached orbital-frequency FFI cutoff is retained, not the paper's 8 Hz;
its observer-frame value is printed for each case when making these figures.

The Shibata Fig. 5 recreation and BHDisk comparison are also included in the
WIP/full workflow. Run just those with `./run_wip.py shibata`; PNG and PDF
outputs are `figures/wip/disk_compactness_reference.*` and
`figures/wip/disk_compactness_comparison.*`.

The Wessel-style accretion check is `./run_wip.py accretion`, also included in
WIP/full runs. `wip_plots/accretion.py` owns the two plots and its BH-mass knob.
It saves `figures/wip/accretion_rate.png` (linear, A/B separated, scaled to a
10-solar-mass BH) and `accretion_mass_budget.png` (measured outside-AH mass loss
versus integrated inward flux). Time stays relative to initial data, not an
inferred saturation time. Missing-horizon-pattern flux samples are gaps, not
interpolated values. The paper M0dot plot and GW extrapolation are unchanged.

The initial Toomre diagnostic is `./run_wip.py toomre`, also included in the
WIP/full workflow. It saves `figures/wip/toomre.png`. Calculation and useful
knobs live together in `wip_plots/toomre.py`; it reads the matching COCAL
`emdg_xz*.txt`, `omeg_xp*.txt`, and `peos_parameter_output.dat` beside the
existing equatorial initial profiles. No HDF5 or evolution scan is needed.
It plots **Newtonian** `Q_N` against cylindrical `R/M_ADM`: coordinate column
density, polytropic adiabatic sound speed, and the measured rotation-gradient
epicycle. The `Q=1` reference is not a GR/thick-disk or global-PPI criterion.
The same command also saves `toomre_romeo.png` and `toomre_meidt.png` in
`figures/wip/`, each comparing the original Q with one alternative. Romeo's
single-gas thickness correction defaults to isotropic support (`T=1.5`;
`ROMEO_SIGMA_Z_OVER_SIGMA_R` controls the assumption). Meidt's separate 3D
midplane diagnostic uses `kappa^2/(4 pi rho_mid)`, not surface density.
These are exploratory Newtonian comparisons, not validated GR stability tests;
the original `toomre.png` is unchanged. References are beside the calculation.
`toomre_cases.png` puts all three methods on each case's axes in one six-panel
figure, with a shared method legend; it does not create per-case files.

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

Detectability has two files to edit:

- `gw_detectability.py`: scientific settings at the top, followed by `analyze()`
  and the seven-step calculation: read cached modes, window/FFT, orientation
  average, characteristic strain, detector noise, physical scaling, SNR/horizons.
  Each scientific choice has its rationale and measured limitation beside it;
  the sensitivity table is in `notes/KNOWLEDGE.md`.
- `wip_plots/gw_detectability_all.py`: combined and individual figures, output
  selection, labels and layout. Run `./run_wip.py detectability`; individual
  numerical-choice checks remain part of `./run_individual.py A1 --extra`.

The default uses the human note's four modes `(2,2), (2,1), (2,-2), (2,-1)`.
Set `MODES = "all"` in `gw_detectability.py` for every cached mode (21 modes,
ell=2 through 4), or supply another tuple/list of pairs. Both choices use the
same calculation and figure filenames; rerunning replaces the selected outputs.
Mean amplitude and RMS are distinct selectable source-orientation averages.
Window/cut choices are explicit, not implied by Moore's strain definitions.
The uniformly sampled `rpsi4_uniform.dat` and its own retarded-time column are
required; rebuild missing caches with `generate_gw.py`. This shares the prepared
Psi4 data without routing detectability through integrated time-domain strain.
Standard waveform plots and their FFI calculation are unchanged.
The CE/DECIGO/LISA examples share one characteristic-strain axis and fixed
source scales: 50 Msun / 10 Mpc, 1000 Msun / 500 Mpc, 100000 Msun / 50 Mpc.
Edit `EXAMPLE_TARGETS` at the top of the analysis file. They represent stellar,
intermediate and massive BH scales, with round distances near the sensitivity
range of our signals, not optimized or automatically retuned thresholds.
All six cases share each source pair. The separate luminosity-distance plot
`figures/gw/gw_detectability_horizon.png` shows SNR=8 reach versus source-frame
BH mass. Source annotations use LaTeX scientific notation; redshift is still
computed from the same cosmology but is not redundantly printed beside D_L.
Plot limits only trim the displayed range, never the SNR integration.
The separate `gw_detectability_shibata.png` uses a fixed 50 Msun BH and 100 Mpc
luminosity distance, with redshift from the same cosmology. This matches the
reference source scale in Shibata et al. (2021), not GW190521's inferred
parameters. It retains our selected modes and orientation average; no reference
waveform is overlaid. `SHIBATA_BH_MASS_MSUN` and `SHIBATA_DISTANCE_MPC` control
this comparison; `PLOT_SHIBATA_COMPARISON` enables it in the plotting file.
The colleague-supplied historical implementation is retained verbatim under
`notes/reference/collaborator_detectability.py.txt`; it is reference material,
not an executable workflow.

The **experimental** Wessel disk-mass-driven `(2,2)` continuation has individual
diagnostics via `./run_wip.py temporal_trial`. These individual diagnostics are
excluded from default/full runs. `TAIL_*` settings in `gw_detectability.py` control the fit, held-out check,
join and termination. Each case gets `gw_temporal_fit.png`: measured disk mass,
strain amplitude and unwrapped phase against their fits over the fitted interval.
Phase has an arbitrary constant removed from both data and fit. Matter-mode
evolution, fit-window sensitivity and fit scores remain in the JSON report. Passing cases and
explicit `TAIL_REVIEW_CASES` get `gw_temporal_on_off.png` (spectral comparison)
and `gw_temporal_transition.png` (waveform and amplitude at the data/tail join).
The fit figure contains no future data. The transition figure separates the
unmodified simulation, the last-orbit blend and the future model; the measured
curve remains visible throughout the blend. Plots contain no pass/fail verdicts.
Review cases retain their waveform-fit warnings and **conditional** status in the report. Mass-decay
and sampling checks are never waived. Numerical results are in
`figures/gw/temporal_trial.json`. This compares cached FFI strain with/without
a modeled tail, not direct-Psi4 versus FFI. No production figure is overwritten.
The September 15 strict test rejected all six cases. A1 now has a conditional
example: the continuation is strongly fit-window dependent. See KNOWLEDGE.md
before interpreting it. Passing the numerical criteria would still not prove that
the disk structure persists throughout a long extrapolation.
The three-rescaling `gw_detectability_characteristic_strain.png` remains the
measured-only, multi-mode direct-Psi4 result. Detectability now defaults to
A1-A3 and includes the ET 10 km triangular-network curve alongside A+, CE,
DECIGO and LISA. Change `SIM_NAMES` / `ACTIVE_DETECTORS` in `gw_detectability.py`.

The same command also writes `gw_detectability_characteristic_strain_time_on.png`
and `gw_detectability_horizon_time_on.png`. `PLOT_TIME_COMPARISON` at the top of
the plot script controls these additional figures. Both use direct-Psi4 FFTs:
only the (2,2) termination is replaced with the analytic second derivative of
the fitted strain model. The measured baseline, source rescalings, frequency
floor and other measured modes are unchanged. A1 is explicitly conditional;
A2/A3 remain measured only because their waveform fits fail the existing checks.
The JSON `gw_detectability_time_comparison.json` records fits and off/on SNRs.
The actual Psi4 join is shown in `A1/gw_temporal_psi4_transition.png`.

`TAIL_MASS_SOURCE="horizon_flux"` tests the requested remaining-mass proxy:
initial outside-AH M0 minus the recorded cumulative inward rest-mass flux.
Set it to `"outside_mass"` for the previous volume-mass fit. The flux proxy
assumes no other losses/sources; it does not fix the poor waveform fits or
prove the disk will persist for the modeled duration. It is a documented
variation on Wessel's mass-driven model, not their exact measured-mass method.
`run_wip.py accretion` now compares disk mass loss, recorded AH flux and BH
mass change, with an additional A-case view over the exact GW fit intervals.
BH gravitational mass and accreted rest mass need not be identical.

`./run_wip.py restmass` restores the former tripleM rest-mass panel as
`figures/wip/restmass.png`, also included in default WIP/full nonmovie runs.
It shows total conserved rest mass from `bhns.don` column 2, normalized to its
first sample (normally t=0), using the shared time axis and A/B/ML styling.
It includes matter inside the AH; it is not the outside-AH mass used in the
temporal fit. The paper tripleM figure is unchanged.

## Plot Promotion

New plots begin in `wip_plots/`. Once the method and presentation are agreed
upon, move the file to `paper_plots/` and add it to `PAPER_PLOTS` in
`run_paper.py`.

Movie scripts are currently local and ignored. Movie frames, videos, figures,
logs, and derived GW products are also untracked.
