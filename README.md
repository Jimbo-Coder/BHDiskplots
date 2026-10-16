# BHDiskplots

Plotting and GW post-processing for the BHDisk simulations on Anvil.

```text
read-only simulation data (scalars, Psi4, 2D slices)
  -> generate_gw.py -> gw_work/   (retarded time, uniform r*Psi4, FFI strain)
  -> paper_plots/, wip_plots/, gw_detectability.py   (read caches only)
  -> figures/
```

## Where things live

| Path | Contents |
| --- | --- |
| `config.py` | Simulation metadata, data roots, shared units, workflow switches |
| `gw.py` | Psi4 reading, restart merging, FFI strain, waveform caches |
| `generate_gw.py` | The only entry point that writes the GW cache (`gw_work/`) |
| `gw_detectability.py` | Direct-Psi4 detectability analysis; settings at the top |
| `paper_plots/` | Approved paper figures, registered in `run_paper.py` |
| `wip_plots/` | Exploratory figures, registered in `run_wip.py` |
| `helpers/` | Scalar/2D readers, units, shared style |
| `detector_curves/` | Noise tables and their conventions (see its README) |
| `data/initial_profiles/` | Small COCAL initial-data inputs |
| `cache/2d_indices/` | Tracked byte-offset indices for the large 2D ASCII files |
| `psi4_hlm_ref/` | Fortran FFI regression reference |
| `notes/` | `WORKING.md` (log, open decisions), `KNOWLEDGE.md` (conventions, numbers), `gw_paper_methods.md`, `human*` notes, `reference/` |
| `run_logs/` | Ignored nohup logs and PID files |

Scientific choices, output filenames and layout knobs sit at the top of each
plot file. Shared modules hold only behavior several figures must share.

## Running

On Anvil, activate the plotting environment (`condactivate`), then:

```bash
./run_all.py                      # GW cache, then paper + WIP + individual (--extra)
./generate_gw.py                  # GW cache only
./run_paper.py modes phase        # selected paper plots
./run_wip.py detectability toomre # selected WIP plots
./wip_plots/run_individual.py A1 --extra
./generate_2d_indices.py          # pre-build 2D indices
```

Detached: `nohup ./run_all.py > run_logs/run_all.log 2>&1 < /dev/null &`.
Locally: `uv sync` once, then prefix commands with `uv run`. Tests:

```bash
uv run python -m unittest discover -s tests -t .
```

`run_all.py` stops if cache generation fails; plots only read caches. Set
`RUN_GW_CACHE = False` in `config.py` only for an intentional plot-only rerun.

## WIP plots (`./run_wip.py <name>`)

| Name | Output (under `figures/`) | What it shows |
| --- | --- | --- |
| `detectability` | `gw/gw_detectability_*` | h_c vs noise, SNR=8 reach, Shibata 50 Msun/100 Mpc case, time-on variants |
| `temporal_trial` | `<case>/gw_temporal_*`, `gw/temporal_trial.json` | Opt-in Wessel mass-driven (2,2) tail; not in default runs |
| `gw_mismatch` | `wip/gw_mismatch/<weighting>/A1_A3.*` | Opt-in A1/A3 complex (2,2) mismatch: independent NumPy and public kuibit calculations; [settings and equations](notes/gw_mismatch.md) |
| `gw_radiated` | `wip/gw_radiated.{png,json}` | Radiated E, J_z (detectability modes) and (2,2) frequency over f_orb |
| `psi4`, `strain`, `*_minus_ml` | `gw/`, `gw/difference/` | Waveforms and disk-minus-ML differences |
| `strain_panel`, `strain_observer` | `gw/` | Polarization panel; face-on strain at 100 Mpc |
| `accretion`, `restmass` | `wip/` | AH flux vs mass budget; total rest mass |
| `toomre` | `wip/toomre*.png` | Newtonian Q, plus Romeo and Meidt variants |
| `mode_appendix`, `modes_345` | `wip/mode_*`, `wip/modes_345.png` | Mode growth, matter/GW frequency match, C3-C5 |
| `shibata` | `wip/disk_compactness_*` | Shibata Fig. 5 recreation and comparison |
| `displacement`, `horizon`, `irreducible_mass`, `radii` | `wip/` | BH displacement and horizon diagnostics |

Paper figures keep stable LaTeX-facing names in `figures/`; individual
products go to `figures/A1/` through `figures/B3/`. Nothing under `figures/`
is tracked.

## Compactness comparison

The comparison is reproducible from this checkout: the full COCAL
`bhtphyseq.dat` inputs are already included under `data/initial_profiles/`.
No Anvil simulation output or GW cache is required. From the repository root:

```bash
uv sync
uv run python -m wip_plots.disk_compactness_comparison --sequences
# Same overlay plus a full-sequence overview above it:
uv run python -m wip_plots.disk_compactness_comparison --sequences --overview
# Original comparison without sequence overlays/overview:
uv run python -m wip_plots.disk_compactness_comparison --no-sequences
```

On Anvil, use the normal plotting environment and `python` instead of
`uv run python`. The registered `run_wip.py shibata` command includes sequences
by default; `INCLUDE_SEQUENCES` at the top of the plot module controls that
default. Existing TeX/Palatino rendering is retained and requires the normal
TeX Live installation, including Palatino fonts.

Every run saves PNG/PDF variants together in `figures/wip/compactness/`:
`reference`, `selected_only`, `sequences_zoom`, and `sequences_overview`.
The stable `figures/wip/disk_compactness_comparison.{png,pdf}` paper-facing files
use the selected options (default: the sequence overlay without an overview).
The comparison remains **3.75 inches wide** and 3 inches high. With `--overview`
it is 4.2 inches high: a small overview shows all 17 A and 43 B models, and the
main comparison retains its low-mass zoom. Light circles/solid
lines identify sequence A; light triangles/dashed lines identify sequence B.
Coloured labelled markers identify the evolved models: A1/A2/A3 are sequence
models 1/5/7, and B1/B2/B3 are models 28/32/35.

The parser reads the first value on `Rest, proper mass` (rest mass, not proper
mass), the first coordinate on `Outer point (width)`, and each model's own
`Black hole mass (M)`. It preserves sequence order, including the high-mass
branches that turn back in radius, and verifies the six evolved coordinates
against the existing full-precision comparison values. Both plotted coordinates
are normalized by the initial BH mass, not total ADM mass.

## GW conventions

For **just the polished combined distance + characteristic-strain figure**, use
the short directly runnable script (in the normal plotting environment):

```bash
python plot_gw_detectability.py                   # current default: radius index 8
python plot_gw_detectability.py --radius-index 6  # compare another extraction
```

Outputs are `figures/gw/detectability/radius_<index>.{png,pdf}`, with caption,
example parameters and input paths alongside. Each index gets separate files.
For A cases index 6 means raw `Psi4_rad.mon.7`; index 7 means `.8`.
Cached GW inputs are not included in GitHub. See
[the short run/edit guide](notes/run_combined_gw_plot.md) for the two required
files per case/radius and exactly where to edit the scientific and plot settings.

The combined detectability figure is 7 inches wide, with distance reach above
characteristic strain, exterior legends and lettered source examples following
Wessel Fig. 10. Related exports live together in `figures/gw/detectability/` as
`measured` and `conditional` PNG/PDF variants, while the existing
`figures/gw/gw_detectability_combined{,_time_on}` filenames remain available.
Their generated `_caption.tex` and `_examples.json` companions
give masses, luminosity distances, redshifts and SNRs. All four examples solve
the uppermost SNR=8 distances among the plotted cases at fixed source masses:
(a)/(b) use 10/50 Msun in CE; (c)/(d) use 1000/1e5 Msun in DECIGO/LISA.
Letters follow increasing source BH mass in both panels and their companions.
Exact example masses are included in the reach curves. Conditional-tail figures
solve their own threshold distances; they are explicitly labelled as conditional.

- Mass: `disk_to_bh_mass_ratio` is M0_disk/M_BH; `mlittle` is the initial BH
  mass in code units (G = c = M_sun = 1); `gw_madm` is the total ADM mass,
  used only for retarded time. Detectability rescales by M_BH, not M_ADM.
- Waveforms: `Psi4 -> gauge-corrected t_ret -> fixed-frequency integration ->
  h_+, h_x`. The NumPy backend is parity-tested against `psi4_hlm_ref`
  (`GW_STRAIN_BACKEND = "fortran"` uses the reference). The cache also stores
  `rpsi4_uniform.dat`, which detectability reads directly, so it never uses FFI.
- Time axes follow `GW_TIME_SCALE` (`"M_BH"`, `"M_ADM"`, `"P_c"`, `"code"`).
- Radii: index 8 (outermost) is the default, index 4 the wave-zone check.
  There is no extrapolation to infinity.
- Detectability: the four l=2, m!=0 modes, 1000 M_BH transient cut, 5% Tukey
  taper, source-orientation mean of the Wessel Eq. (8) amplitude, SNR=8.
  Detector response gives the exact sky- and polarization-averaged SNR
  (A+/CE/DECIGO x sqrt(5/2); ET and LISA in `detector_curves/README.md`).
  The response factor follows a direct antenna average for this source
  convention. Wessel footnote 4 uses different sensitivity-curve language;
  its numerical SNRs have not been reproduced from the published inputs.
- Difference plots keep `h(Psi4_disk) - h(Psi4_ML)` and `h(Psi4_disk - Psi4_ML)`
  distinct.

For paper writing, `notes/gw_paper_methods.md` gives every equation the code
uses, plus a compiled LaTeX `algorithm` block (Sec. 9). Measured sensitivities
and dated numbers are in `notes/KNOWLEDGE.md`.

## Data safety and syncing

- Simulation directories are read-only. Each case has ordered `data_roots`;
  later roots replace overlapping restart data in memory only. The only
  machine-specific paths are `MILTON_DATA_ROOT` and `SUPPLEMENTAL_DATA_ROOT`.
- Canonical copies: this checkout on the Mac and
  `/anvil/projects/x-mca99s008/maxwork/BHDiskplots` on Anvil.
- Before local work, pull Anvil's uncommitted edits and compare. Before syncing
  back, check again. Copy changed source files only: never mirror `.git/`,
  environments, `gw_work/`, `figures/` or logs, and never `rsync --delete`.
- `human*` notes are preserved verbatim.

## Plot promotion

New plots start in `wip_plots/`. Once the method and presentation are agreed,
move the file to `paper_plots/` and add it to `PAPER_PLOTS` in `run_paper.py`.
Movie scripts are local and ignored.
