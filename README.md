# BHDiskplots

Plotting and GW post-processing for the BHDisk simulations on Anvil.

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

## GW conventions

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
  (A+/CE/DECIGO x sqrt(5/2); ET and LISA in `detector_curves/README.md`). This
  deliberately differs from Wessel footnote 4, which halves SNRs.
- Difference plots keep `h(Psi4_disk) - h(Psi4_ML)` and `h(Psi4_disk - Psi4_ML)`
  distinct.

Method details and measured sensitivities: `notes/gw_paper_methods.md` and
`notes/KNOWLEDGE.md`.

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
