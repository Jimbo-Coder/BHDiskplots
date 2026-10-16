# BHDiskplots

Plotting and GW post-processing for the BHDisk simulations. Run commands from
this repository's root in the usual Anvil plotting environment (`condactivate`).
Locally, run `uv sync` once and prefix Python commands with `uv run`.

```text
read-only simulation data
  -> scalar/2D readers -> paper_plots/ and wip_plots/ -> figures/
  -> generate_gw.py -> gw_work/ (uniform r*Psi4, FFI strain and derivatives)
       -> gw_detectability.py       spectra, SNR, distance and optional models
       -> wip_plots/gw_waveforms.py  waveform, polarization, energy/frequency views
       -> gw_mismatch.py            independent manual and kuibit overlap methods
```

## Run and edit a GW figure

Each family owns its calculations and drawing. Physical choices and output
names are grouped beside the relevant section; parsers are at the bottom.
Shared cache readers, units and TeX/Palatino style remain shared.

| Task | Command | Where to edit |
| --- | --- | --- |
| Combined distance + strain figure | `python gw_detectability.py` | Section 1: physical choices; `analyze`: workflow; section 4: Matplotlib |
| Different extraction radius | `python gw_detectability.py --radius-index 6` | The same workflow recomputes both panels at that radius |
| Radius/averaging checks | `python gw_detectability.py --plots radius averaging --sims A1 A3` | Labelled validation calculations and diagnostic drawings in the same file |
| Modal strain histories | `python -m wip_plots.gw_waveforms --kinds strain --sims A1 A3` | `waveform`, then `plot` |
| Polarization grid | `python -m wip_plots.gw_waveforms --kinds strain_panel` | `plot_polarization_panel` |
| Face-on observer strain | `python -m wip_plots.gw_waveforms --kinds strain_observer` | `observer_waveform`, then `plot_observer_group` |
| Radiated E, J and phase frequency | `python -m wip_plots.gw_waveforms --kinds gw_radiated` | `radiated`, `frequency_22`, then `plot_radiated` |
| Two-waveform mismatch | `python gw_mismatch.py --sims A1 A3` | Settings, reader, manual/kuibit calculations, then diagnostic plot |
| Rebuild waveform caches | `python generate_gw.py` | `config.py` selects inputs/backend; `gw.py` implements reading and FFI |

For the combined figure, read settings -> `analyze()` -> `plot_combined()`.
The measured calculation and main figure come first. Optional diagnostics
and continuation fits follow in section 5 of the same file. Time-history
views live together in `wip_plots/gw_waveforms.py`, which reads each simulation
once and reuses its radius caches.

Detectability `--plots` accepts `combined`, `strain`, `distance`, `radius`,
`averaging`, `reference`, and `extrapolation`. Its default is measured
`combined`. Optional fits are calculated only when requested. The WIP batch
requests `BATCH_PLOTS`, which includes the labelled extrapolation.

Waveform `--kinds` accepts `psi4`, `strain`, `psi4_minus_ml`, `strain_minus_ml`,
`strain_from_psi4_minus_ml`, `strain_panel`, `strain_observer`, and `gw_radiated`.
With no selection it draws all views. `--radius-index` chooses one extraction;
otherwise modal/polarization views use indices 4 and 8, and observer/radiated
views use index 8. The individual runner adds per-case radius/mode comparisons.

Outputs are under `figures/`:

| Family | Products |
| --- | --- |
| Combined detectability | `gw/detectability/radius_<index>.{png,pdf}` |
| Optional continuation | `gw/detectability/radius_<index>_conditional.{png,pdf}` |
| Companion information | `_caption.tex`, `_examples.json`; measured `_inputs.json` |
| Other detectability checks | `gw/gw_detectability_*` and per-case diagnostic folders |
| Modal/polarization/observer histories | `gw/`; differences in `gw/difference/` |
| Radiated E/J/frequency | `wip/gw_radiated.{png,json}` |
| Mismatch | `wip/gw_mismatch/<weighting>/<first>_<second>.*` |

The combined figure is 7 inches wide. Its examples are ordered by BH mass:
10/50 solar masses in CE, 1000 in DECIGO, and 100000 in LISA. Their distances
solve the largest SNR=8 reach among the plotted cases. The caption/JSON records
BH mass, distance, redshift, SNR and the winning case/detector. Use
`--example-labels key` or `inline` to change the presentation.

GW caches are ignored by Git. The [combined-figure guide](notes/run_combined_gw_plot.md)
lists the two files needed per case/radius for a separate clone and explains
index 6 versus raw `Psi4_rad.mon.7`. Waveform histories also need the configured
raw extraction files; observer-only and mismatch runs read strain caches.
Both standalone GW analyses accept `--cache-dir PATH` for copied caches.
The [mismatch guide](notes/gw_mismatch.md) gives its inputs and equations.
Mismatch needs kuibit, installed in the normal Anvil environment; locally use
`uv run --with kuibit==1.6.1 python gw_mismatch.py`.

## Compactness comparison

The complete COCAL sequences are tracked in `data/initial_profiles/`; this
figure needs no simulation output or GW cache.

```bash
python -m wip_plots.disk_compactness_comparison               # sequences, no overview
python -m wip_plots.disk_compactness_comparison --overview    # overview above the zoom
python -m wip_plots.disk_compactness_comparison --no-sequences
```

Each command saves `reference` and the requested comparison as PNG/PDF in
`figures/wip/compactness/`. The comparison names are `sequences_zoom`,
`sequences_overview`, and `selected_only`. Width is 3.75 inches; height is
3 inches, or 4.2 with the overview. Normal TeX/Palatino rendering is retained.

`read_cocal_sequence` reads the first `Rest, proper mass` value (rest mass),
the first `Outer point (width)` coordinate, and each model's `Black hole mass (M)`.
Both coordinates use the initial BH mass, not total ADM mass. Sequence order
is preserved through radius turnbacks, and evolved coordinates are checked
against full-precision reference values. The overview includes all 17 A and
43 B models. A1/A2/A3 select sequence models 1/5/7; B1/B2/B3 select 28/32/35.
Light circles/solid lines denote A; triangles/dashed lines denote B.

## Batch commands and other plots

```bash
./run_all.py                      # rebuild GW caches; paper, WIP and individual plots
./run_paper.py modes phase         # selected approved figures
./run_wip.py detectability toomre  # selected exploratory families
./wip_plots/run_individual.py A1 --extra
./generate_2d_indices.py
```

`run_all.py` stops if cache generation fails. Only `generate_gw.py` writes GW
caches; plotting does not integrate or regenerate them. Set `RUN_GW_CACHE=False`
in `config.py` only for an intentional plot-only run. Detached runs can use
`nohup ./run_all.py > run_logs/run_all.log 2>&1 < /dev/null &`.

| WIP name | Purpose |
| --- | --- |
| `detectability` | Measured combined figure, radius/averaging/reference checks and labelled continuation |
| `temporal_trial` | Explicit opt-in fit/join/on-off diagnostics |
| `gw_mismatch` | Explicit opt-in manual/kuibit comparison |
| `psi4`, `strain`, `*_minus_ml`, `strain_panel`, `strain_observer`, `gw_radiated` | Views in the waveform family |
| `compactness` | Compactness comparison with sequences |
| `accretion`, `restmass` | Horizon flux/mass budget and total rest mass |
| `mode_appendix`, `modes_345` | Matter modes, frequency comparisons and higher-mode growth |
| `toomre` | Newtonian Q and Romeo/Meidt comparisons |
| `displacement`, `horizon`, `irreducible_mass`, `radii` | BH displacement and horizon diagnostics |

Approved figures are registered in `run_paper.py`; exploratory families in
`run_wip.py`. Paper filenames stay in `figures/`, with individual outputs in
`figures/<case>/`. Generated figures/movies and local movie scripts are ignored.

## Scientific conventions and methods

- `mlittle` is the initial BH mass in simulation units; `gw_madm` is total ADM
  mass, used for retarded time. Physical detectability scales with BH mass.
- FFI histories and direct-Psi4 detectability are separate calculations.
  The NumPy FFI is parity-tested against `psi4_hlm_ref`; the reference Fortran
  backend remains available for that calculation.
- Detectability uses the four l=2, m!=0 modes, a 1000 BH-mass transient cut,
  5% Tukey taper, source-orientation mean amplitude and SNR threshold 8.
  This amplitude statistic is not a population-averaged SNR.
- Compare `h_c` with `h_n=sqrt(f S_eff)` and apply detector response once.
  Noise-table conventions are in [detector_curves/README.md](detector_curves/README.md).
- Radii 4/8 are finite-radius checks; no extrapolation to infinity is applied.
  `GW_TIME_SCALE` selects BH mass, ADM mass, orbital-period or code-time axes.
- Subtracting two FFI strains and integrating the Psi4 difference are distinct
  views with separate caches. Optional future-time fits are labelled conditional.

[GW paper methods](notes/gw_paper_methods.md) gives the equations, averaging
conventions and LaTeX algorithm. The [detailed walkthrough](notes/gw_workflow_explained.md)
maps equations to code. Dated measurements and limitations are in
[KNOWLEDGE.md](notes/KNOWLEDGE.md); pending work/history is in
[WORKING.md](notes/WORKING.md). Published Wessel SNRs have not been reproduced
from the available reference inputs.

## Development and data safety

Run `uv run python -m unittest discover -s tests -t .` for the normal suite;
include `--with kuibit==1.6.1` after `uv run` for both mismatch methods locally.

Simulation directories are read-only. Ordered `data_roots` merge restarts in
memory, with later roots replacing overlaps. Shared paths/metadata live in
`config.py`; cache/FFI infrastructure is in `gw.py`; common readers, units and
style are in `helpers/`. Scientific `human*` notes are kept as supplied.

The canonical checkouts are this Mac repository and
`/anvil/projects/x-mca99s008/maxwork/BHDiskplots`. Compare Anvil edits before
working or syncing. Sync changed sources only, preserving unrelated dirty
work; never mirror `.git`, environments, simulation data or generated caches,
and never use `rsync --delete`. Promote an agreed plot family from
`wip_plots/` to `paper_plots/` and register it in `run_paper.py`.
