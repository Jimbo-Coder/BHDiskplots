# BHDiskplots Knowledge

Organization checked: 2026-09-11. The numerical results below are dated
historical evidence, not a new scientific validation.

Keep established conventions here and pending work in [WORKING.md](WORKING.md).
The [human GW proposal](human_gw_method.md) is preserved verbatim. Its seven
steps and four-mode selection now organize the implementation; numerical
window/cut and detector choices still require scientific review.

## Face-On Strain at 100 Mpc (2026-09-14)

- `./run_wip.py strain_observer` makes separate A/B three-row figures,
  `figures/gw/strain_observer_{A,B}.png`. Settings and the short conversion
  live in `wip_plots/gw_strain_observer.py`; no new integration pipeline.
- Shibata et al. https://arxiv.org/pdf/2101.05440 Figs. 3-4 show plus/cross
  strain viewed along the rotation axis, at 100 Mpc with initial BH mass
  50 Msun. They shift the amplitude peak to roughly zero and use independent
  vertical scales. These are not sky-averaged spectra or detector responses.
- Use cached `rhphc.dat` and its own retarded time, at outer radius index 8.
  At the north pole, _{-2}Y_lm=sqrt((2*l+1)/(4*pi))*delta_m2. The selected
  (2,+/-1),(2,+/-2) modes therefore reduce exactly to (2,2). The `"all"`
  option adds (3,2),(4,2); other m values vanish at this viewing angle.
  Keep the cache's plus/cross sign convention; no extra factor of two.
- q = projected(r*h)/M_BH_code; h=q*G*M_BH_source*(1+z)/(c^2*D_L).
  t_obs=(t_ret/M_BH_code - peak)*G*M_BH_source*(1+z)/c^3. Code BH mass is
  0.05, not 1 and not M_ADM. The existing cosmology gives z=0.0221938 at
  100 Mpc; its small 2.2% correction is on by default and can be disabled.
- Retain t_ret/M_BH >= 1000, matching the detectability relaxation exclusion.
  Peak alignment uses hypot(h_plus,h_cross) on that retained interval. No
  synthetic tail, extra taper, filtering, sky average, or forced-zero endpoint.
  Our cached FFI cutoff remains max(|m|*Omega_orb, Omega_orb), unlike the
  paper's fixed 8 Hz for m=2. Do not claim an identical extraction method.
  At the default physical scale the observer-frame m=2 cutoffs are
  A1/A2/A3=24.389/21.641/18.961 Hz and B1/B2/B3=22.996/19.319/16.284 Hz.
- Fresh Anvil outer-radius caches give peak hypot(h_plus,h_cross):
  A1=1.71655e-24, A2=2.94374e-24, A3=4.00477e-24;
  B1=1.04766e-24, B2=1.46977e-24, B3=1.94590e-24.
  These are peaks of retained finite-radius FFI waveforms, not validated
  astrophysical burst classifications. A3's peak is near its final sample.
- Tests verify analytic face-on normalization/sign, all-mode selection,
  source/code mass, distance and redshift scaling, and no time extrapolation.

## Detectability Example Distance (2026-09-22)

- The ground-based example is now M_BH=50 Msun, D_L=100 Mpc, matching the
  separate Shibata comparison. This replaces the optimistic 10 Mpc choice
  in BOTH the measured and conditional-tail characteristic-strain charts.
- The existing cosmology recomputes redshift from the new distance; this is
  not just a vertical plot shift. DECIGO/LISA examples, intrinsic spectra,
  detector curves, horizon scans and all analysis choices are unchanged.
- The earlier 10 Mpc SNR values below are historical, not current predictions.
- With the refreshed Sep22 cache, measured CE SNRs at 100 Mpc are
  A1=0.6818, A2=2.4620, A3=1.7400; ET gives 0.3277, 1.0462, 0.8734.
  The conditional A1 tail gives CE=0.7610 and ET=0.3563; A2/A3 stay measured.

## Detectability Figures (2026-09-14, Historical Source Choices)

- `EXAMPLE_TARGETS` fixes (detector band, M_BH/Msun, D_L/Mpc) at
  (CE,50,10), (DECIGO,1000,500), (LISA,100000,50). Stellar/intermediate/massive
  BH scales with round, illustrative near-threshold distances; not known events.
  They replace the earlier automatically optimized A1 examples. All cases use
  the same three pairs, and future cache changes do not retune them.
- A1 SNRs for those detector/source pairs: 6.73789, 6.65119, 6.50695 with the
  inspected local cache. Redshifts come from the unchanged flat-LambdaCDM model.
- `gw_detectability_characteristic_strain.png` uses one axis for all three
  rescalings and the noise curves, with one case legend. Source labels use
  scientific notation; D_L suffices without redundant z and SNR labels.
- `gw_detectability_horizon.png` combines detector groups on one luminosity
  distance versus source-frame BH mass axis; SNR threshold is labeled.
  The fixed Shibata reference remains a separate optional figure.
- Plot y limits only control visibility; no new cut, smoothing, mode selection,
  time/radial extrapolation, noise change, or SNR integration change was made.
  All six dimensionless spectra and all horizon arrays exactly match pre-edit
  results. The observer conversion is unchanged; only example M_BH/D_L changed.
- The original human proposal still exists on both hosts at
  `notes/human_gw_method.md`; the new `wip_plots/simple_gwworkflow.txt` on Anvil
  contains only a restoration reminder, not a different scientific method.

## Initial Toomre Diagnostic (2026-09-12)

- Entry point: `./run_wip.py toomre`; included in full WIP runs. One plot file,
  `wip_plots/toomre.py`, owns reading, analysis and plotting; shared config/style
  are reused. Reads matching density/rotation profiles without requiring the
  unrelated `ell_xp` diagnostic. Output: `wip/toomre.png`.
- Read-only source: `/anvil/projects/x-mca99s008/Initial_data/BHT/<case>/`.
  Matching converged suffixes: A1=71, A2=69, A3=78, B1=70, B2=72, B3=90, ML=70.
  `emdg_xz`, `omeg_xp`, and EOS tables are copied into repo `data/initial_profiles`.
- COCAL `IO_output_2D_general` writes two meridians, radius varying fastest;
  `emdg=P/rho_0`, verified against `peos_q2hprho` and the ID scalar table.
  The 2D equator matches existing `emdg_xp`; Omega at its peak matches every
  configured initial orbital frequency. No assumed mass rescaling is added.
- Newtonian coordinate diagnostic: Sigma=integral rho_0 dz (both halves),
  c_s^2=Gamma*emdg, kappa^2=R^-3*d(R^4*Omega^2)/dR, Q_N=c_s*kappa/(pi*Sigma).
  Plot radius is labeled r/M (r=R=x on the positive equatorial ray, M=M_ADM).
  No relativistic metric correction or thickness factor;
  Q=1 is only a razor-thin/local/axisymmetric reference, not a PPI criterion.
- Native-grid linear interpolation; 2049 vertical samples. Atmosphere emdg
  <=1e-12 of peak is vacuum. Isolated nonfinite samples surrounded by atmosphere
  are reported; nonfinite samples adjacent to resolved matter fail. Both
  meridians are checked for axisymmetry. Midplane density >=1e-3 peak defines
  the displayed disk interval. Nonpositive kappa^2 is never replaced by abs().
- Six-case minima: A1=10.8816, A2=1.18347, A3=0.477399, B1=30.1671,
  B2=3.67655, B3=0.879423. All retained kappa^2 values are positive.
  2049->4097 vertical samples change Sigma by <=0.047%; halving radial samples
  changes interior kappa^2 by <=0.68%. These are postprocessing sensitivity
  checks, not convergence of the underlying GR simulation or Toomre theory.
- ML density exports were missing; derived locally from `rnsflu_3D.las` and
  `rnsgrids_3D.las` with `toomre.extract_cocal_profiles`. No source-data edits.
  Source directory and hashes are in the ML profile folder's README.
  Raw Omega matches `omeg_xp70`; q_max=0.00340363620916 and Omega_peak=0.441137056342
  match sequence 1 in `bhtphyseq.dat` and configured ML orbital frequency.
  ML Q_N,min=707988 at r/M=16.625; all 35 retained kappa^2 values positive.
  2049->4097 integration samples change Sigma by <0.002%. The same atmosphere
  checks accept 1804 nonfinite vacuum-neighbor samples across the two meridians.
  Very large ML Q expands the log-axis range; no arbitrary curve rescaling.

## Canonical Workspace

- Mac: `/Users/mrizzo/phys_research/AnvilSyncing/BHDiskplots`.
- Anvil: `/anvil/projects/x-mca99s008/maxwork/BHDiskplots`.
- These are the two working copies of the same repository, not independent
  workflows. Refresh Anvil changes before editing locally. Compare contents
  before sending changes back; Git HEAD alone misses uncommitted notes.
- Edit locally, test there where possible, sync only the intended working
  changes. Never change source simulation outputs during post-processing.
- `config.py` owns simulation metadata and shared run choices; `paper_plots/`
  and `wip_plots/` own plots; `gw.py` owns ordinary GW processing; `helpers/`
  owns shared readers and numerical/display utilities.
- `gw_detectability.py` owns all direct-Psi4 detectability analysis/settings;
  `wip_plots/gw_detectability_all.py` owns all its figure rendering.
- Figures, waveform caches, movies and logs are local products. Superseded
  Mac sync/review copies and their temporary recovery archive were removed;
  retain useful references here rather than accumulating old working trees.
- The Shibata Fig. 5 comparison lives in
  `wip_plots/disk_compactness_comparison.py` and runs as `run_wip.py shibata`
  or as part of the full non-movie workflow. Outputs belong in `figures/wip/`.

## Inputs and Units

- Code units are `G = c = M_sun = 1`. `M` in normalized paper coordinates
  denotes total ADM mass, not the solar-mass code unit or the BH mass.
- [reference/restmass.txt](reference/restmass.txt) contains supplied initial
  disk/BH rest-mass ratios. `config.py` converts these to code mass using the
  initial BH code mass; the file is provenance, not a second runtime config.
- Source roots, including the Massless restart and supplemental data, remain
  in `config.py`. Preserve restart precedence and physical-radius matching.
- [reference/massless.par](reference/massless.par) and the collaborator's two
  `rho_*_parallel.py` scripts are historical input/plotting references only.

## GW Record

The sections below preserve the earlier GW method and its verification record.
Read their dates and the pending decisions in `WORKING.md` before reuse.

## Provenance

- The verbatim collaborator reference script is
  [reference/collaborator_detectability.py.txt](reference/collaborator_detectability.py.txt).
- SHA-256: `021e0dc8c1c7e16c39448f49a143f069eedcbd4b5f4e7d3983a7b5041a52990f`.
- The script came from a BH-cluster analysis intended to reproduce the type of
  characteristic-strain plot shown in the bottom panel of Wessel et al.
  (2021), Fig. 11.
- It is preserved as evidence and a methodological reference. It is not a
  drop-in BHDisk implementation and contains project-specific paths, masses,
  constants, and curve conventions.

## Collaborator Recommendation

For this BHDisk problem, form the detectability spectrum directly from
`Psi_4` in the frequency domain. This avoids choosing the uncertain
fixed-frequency-integration cutoff needed to construct time-domain strain:

1. Reconstruct the complex radiation field from spin-weight `-2` modes.
2. Use `r Psi_4`, since the leading wave-zone quantity is approximately
   invariant with extraction radius.
3. Remove pre-arrival data using retarded time, apply a Tukey window, pad only
   to sample the frequency grid, and Fourier transform.
4. Convert after the FFT:

   `r h_tilde(f) = r Psi4_tilde(f) / (2 pi f)^2`.

5. Construct either:

   `r h_c(f) = 2 f |r h_tilde(f)|`,

   or signal ASD:

   `sqrt(S_h(f)) = 2 sqrt(f) |h_tilde(f)|`.

6. Apply source mass, redshift, and luminosity-distance scaling consistently.
7. Compare `h_c` with `h_n = sqrt(f S_n)`, or signal ASD with detector ASD.

## Source-Orientation Averages

The collaborator script implements two distinct quantities:

- Mean amplitude:
  `(1/4pi) integral |r Psi4_tilde(theta, phi)| dOmega`.
- RMS amplitude:
  `sqrt((1/4pi) integral |r Psi4_tilde(theta, phi)|^2 dOmega)`.

The collaborator ultimately used the mean. These are not interchangeable and
must be labeled explicitly. A representative viewing angle is a third,
separate convention.

## Windowing and Frequency Treatment

- The reference script uses a Tukey window with `alpha=0.05`.
- Zero padding changes frequency sampling, not physical spectral resolution.
- Low-frequency-bin removal, maximum-frequency cuts, the retained time
  interval, and window strength are analysis choices requiring robustness
  checks.
- Direct `Psi_4` conversion still divides by `f^2`; therefore the low-frequency
  result remains sensitive to leakage and windowing even though it avoids a
  time-domain FFI cutoff.

## Physical Scaling

- Keep simulation ADM mass, black-hole mass, disk rest mass, and chosen target
  astrophysical mass distinct.
- Use a documented cosmology, preferably Astropy Planck18 or a numerically
  verified equivalent, for redshift and luminosity distance.
- Frequency scales inversely with redshifted source mass. Amplitude scaling
  must consistently include source mass, `(1+z)`, and luminosity distance.
- The preserved script's `m_BH=0.5`, mass dictionary, and hand-written physical
  constants are specific to its original problem and are not BHDisk defaults.

## Detector Conventions

- 2026-09-28, supersedes our former sqrt(10) scaling recorded below and all
  earlier SNR/horizon numbers: effective ASD = sqrt(5/2) x instrument ASD for
  A+/CE/DECIGO, sqrt(5/2)/(3/2) for ET, and 0.5 x sqrt(PSD) for the
  single-channel SciRDv1 LISA table (two channels). This makes
  4*int h_res^2/S_eff df equal the sky- and polarization-angle-averaged
  SNR^2 exactly (quadrature test in tests/test_gw_detectability.py). Old
  ground/DECIGO SNRs were 2x low and LISA 2*sqrt(2)x low; horizons scale
  similarly at low z.
  This corrects our earlier use of the raw input PSD with Eq. (8) h_res. It does
  not establish an error in Wessel's numerical horizons; the published input
  curves and SNR implementation have not been independently reproduced.

- Determine whether every imported detector curve is PSD, ASD, or
  characteristic noise before plotting or integrating it.
- Determine whether each curve already includes source-sky, detector-sky, and
  polarization response averaging before applying another response factor.
- The reference script uses `sqrt(1/5)` for an L-shaped detector,
  `sqrt(2/5)*(3/2)` for a triangular detector, and `sqrt(2)` factors for LISA
  and DECIGO. These must be checked against the exact curve definitions.
- SNR is valid only when one-sided PSD/ASD, polarization, Fourier-transform,
  and response conventions are mutually consistent.

## BHDisk Production Method

- Combined detectability runs through `wip_plots/gw_detectability_all.py`,
  registered in `run_wip.py` and the configured `run_all.py` workflow.
- Per-simulation numerical validation uses `individual_main` in that same
  plotting file, called by `run_individual.py CASE --extra`.
- The production observable is the finite outer-radius (`r=170`) `r Psi_4`
  signal. The first extraction sphere treated as wave-zone data (`r=120`) is
  always processed as a finite-radius systematic check. One intermediate
  radius is retained in the per-simulation validation figure, but is not mixed
  into the production central value.
  It uses the maintained waveform cache's gauge-corrected retarded-time grid
  and uniformly sampled `r Psi_4`, removes pre-arrival samples (`t_ret<0`),
  then discards the first `1000 M_BH` of the arrived waveform. It uses every
  selected mode `(2,2), (2,1), (2,-2), (2,-1)`, applies a Tukey window with `alpha=0.05`, and
  zero-pads only to sample the frequency axis. Detectability does not consume
  the cache's integrated strain.
- With `tau=t/M_BH`, `q=r h/M_BH`, and `p=M_BH r Psi_4`, the conversion is
  `q_tilde=-p_tilde/(2 pi nu)^2`.
- Complex modes are combined with spin-weight `-2` harmonics before separate
  plus/cross Fourier amplitudes are formed. The production source quantity is
  the collaborator's mean amplitude over solid angle, evaluated with
  Gauss-Legendre quadrature in `cos(theta)` and a nonduplicated uniform `phi`
  grid.
- The production path intentionally performs no radial extrapolation and no
  temporal tail extrapolation. Those earlier experimental paths and their
  paper-facing outputs were superseded rather than mixed into the finite
  signal.
- A low-frequency reliability floor of three cycles over the retained time
  interval is imposed before division by frequency squared. The central
  choices are the Wessel transient cut `t>=1000 M_BH` and the collaborator's
  Tukey `alpha=0.05`; per-simulation diagnostics bracket both choices rather
  than silently treating them as exact.

## Production Outputs

- `gw_detectability_characteristic_strain.png`: three Wessel-style finite
  source examples with all six simulations and all active detector curves.
- `gw_detectability_horizon.png`: SNR=8 luminosity-distance horizons versus
  source-frame BH mass.
- `gw_detectability_method_comparison.png`: source-direction mean versus the
  Wessel representative angle.
- `gw_detectability_radius_comparison.png`: the outer-radius spectrum and the
  first-wave-zone/outer spectral ratio for all six simulations.
- `gw_detectability_method_validation_<SIM>.png`: first-wave-zone,
  intermediate, and outer radii plus Tukey-window, transient-cut, and
  angular-quadrature checks for an individual simulation.

## Verified Numerical Invariants

- Synthetic periodic data recover the expected strain Fourier amplitude after
  direct `Psi_4/(2 pi f)^2` conversion.
- The angular quadrature weights integrate to one and reproduce the expected
  single-mode source-direction mean.
- Doubling source mass halves observed frequency and doubles characteristic
  strain at fixed luminosity distance.
- The dependency-free flat-Lambda-CDM distance table round-trips redshift over
  the plotted range in the focused tests.
- The cache reader uses `rpsi4_uniform.dat`'s own time column and only the
  selected mode columns. Missing caches must be regenerated. The old fallback
  associating raw samples with cached strain times and extending missing times
  was removed on 2026-09-11; it is not part of the current method.
- Moore's frequency and log-frequency SNR integrals agree in the focused tests.

## 2026-09-11 Four-Mode Review

Current displayed examples are selected from our A1 horizon scan, not borrowed
from Wessel: CE (62 Msun, 9.2799 Mpc), DECIGO (3800 Msun, 1062.7620 Mpc),
LISA (340000 Msun, 103.2403 Mpc). The sampled greatest-reach mass is rounded
to two significant digits, then distance/redshift is refined to A1 SNR=8.
This is a representative sampled optimum, not a continuous global optimization.
Every disk uses the same pair within each panel. Stars identify the examples
on the horizon curves. These use the existing four-mode, finite-radius,
finite-duration calculation; selecting a distance does not validate the mode
subset or demonstrate total-signal detectability. The old fixed pairs below
remain inputs to the numerical sensitivity study, not current figure defaults.

- The top `ANALYSIS CHOICES` block in `gw_detectability.py` is the current
  human-facing method specification. It gives each choice, rationale and
  limitation in the seven-step workflow order. Detector file quantities,
  integration bounds and cosmology parameters are visible there too.
- All 18 case/radius combinations agree with the preceding implementation to
  relative tolerance `2e-13` when both receive the same four modes. This is
  implementation parity, not independent scientific validation.
- Four modes replace the previous 12-mode default. The historical SNRs below
  do not apply to these new spectra. At `10 M_sun`, `150 Mpc`, CE SNRs now
  span approximately `0.086-0.268` across the six cases.
- Finite-radius dependence remains substantial: for A1 the significant-band
  median inner/outer fractional change is about `1.08`, and next-inner/outer
  about `0.24`. Do not interpret the curves as precision predictions.
- The implemented flat-Lambda-CDM model uses Planck18 H0/Omega_m values but
  omits radiation/neutrino evolution. It is not full Astropy Planck18.
- Wessel Eq. 8 and footnote 4 explicitly motivate the retained polarization
  amplitude and extra `sqrt(2)` effective-noise convention. Moore defines
  `h_c`, `h_n` and their SNR integral, not those response choices. The SNR of
  a mean-amplitude source spectrum is not the orientation average of SNR.
- Individual ratio panels use `h_c,base` for the top-of-file baseline window
  and transient cut; their legends identify the varied parameter.

### Default Choices and Measured Sensitivity

Choose the four-mode non-axisymmetric quadrupole observable requested in the
human note, the outer finite extraction sphere, mean orientation amplitude,
1000 M_BH transient exclusion, Tukey alpha=0.05 and f_min=3/retained duration.
Keep the finite measured duration and no radial extrapolation. These choices
do not estimate total GW emission or establish that excluded content is noise.

Local cache study: all six disks, 20 alternatives per disk, changing one factor
at a time. The relevant detector/source pairs were A+/CE at 10 M_sun and
150 Mpc, DECIGO at 1000 M_sun and 40000 Mpc, and LISA at 200000 M_sun and
7000 Mpc. The following are extrema over these 24 case/target pairs, not
confidence intervals and not convergence bounds over the entire horizon scan.

| Change from default | SNR ratio range | Decision |
| --- | --- | --- |
| Transient 500-1500 M_BH | 0.968-1.017 | Keep 1000; relatively stable and Wessel-motivated |
| Tukey alpha=0.02-0.10 | 0.850-1.080 | Keep collaborator's 0.05; include taper sensitivity |
| Tukey alpha=0.20 | 0.728-0.997 | No evidence for discarding more endpoint signal |
| Frequency floor 1/duration | 1.000-1.032 | Keep 3 as provisional low-bin exclusion |
| Frequency floor 6/duration | 0.645-1.000 | Low-frequency contribution remains important |
| Add (2,0) to default subset | 19.77-163.41 | Subset is not a total-signal approximation |
| All 21 cached modes | 29.85-238.71 | Explicit comparison option, not a cleaner default |
| RMS instead of mean amplitude | 1.053-1.066 | Different observable, not a numerical error |
| Angular grid 12x24 instead of 24x48 | 1.00000-1.00014 | Retain 24x48 |
| Inner extraction instead of outer | 0.740-1.950 | Retain outer; report finite-radius uncertainty |
| Next-inner instead of outer | 0.572-1.122 | No justified infinity extrapolation |

For CE, 6-60% of retained SNR squared lies below twice the baseline frequency
floor, depending on case. No claim of low-frequency convergence is warranted.
No automatic percentile/power cut was added. High-frequency bins above a quarter
of Nyquist contributed below the precision of this cumulative SNR assessment
at these targets; this is not a universal high-frequency cutoff prescription.
The production 2048-bin SNR approximation differed from full-grid trapezoidal
integration by at most 0.95% over the 120 tested spectra and four target pairs.

Evidence: `run_logs/detectability_choices_20260911.json` records the analysis
hash, spectra/modes/durations, per-target SNR ratios and cumulative 5-95% SNR
squared bands in f/f_orbit. Reproduce with `read_rpsi4_modes`, one-at-a-time
overrides to `spectrum_for_case`, then `observer_spectrum` and `cumulative_snr`
using the same `effective_detector_asd` and `DETECTOR_BOUNDS` as production.
This evaluates numerical-choice sensitivity, not detector calibration or
independent physical validation of the extraction data.

## 2026-08-30 Real-Data Verification

- All six A1-A3/B1-B3 cases produced direct spectra from 12 modes through
  `ell=3`. Retained durations after pre-arrival and transient cuts were about
  `6200-7550 M_BH`.
- At the Wessel finite targets, the six-case SNR ranges were:
  - `10 M_sun`, `150 Mpc`: A+ `0.61-0.64`, CE `10.8-11.0`.
  - `1000 M_sun`, `40000 Mpc`: DECIGO `13.1-13.5`.
  - `2e5 M_sun`, `7000 Mpc`: LISA `7.51-7.76`.
- Maximum SNR=8 horizons across the six cases were about `13.2-13.7 Mpc`
  (A+), `661-698 Mpc` (CE), `7.33e4-7.58e4 Mpc` (DECIGO), and
  `7040-7330 Mpc` (LISA). None is pinned to the configured `z=10` ceiling.
- The `12x24` versus `24x48` angular grids changed significant-band amplitude
  by only `5.5e-5-7.2e-5` in the median and at most about `1.5e-4` at the 95th
  percentile.
- The finite-radius systematic is not negligible: next-inner versus outer
  extraction changed significant-band amplitude by `0.35-0.43` in the median
  and up to `0.87-1.36` at the 95th percentile.
- Tukey-window variation was also visible: relative to `alpha=0.05`, median
  significant-band changes ranged from about `0.10` to `0.32`; the largest
  differences occur in spectral notches and the suppressed high-frequency
  tail.

## 2026-09-15 Wessel Temporal Trial (Local, Not Adopted)

Reference: accepted Wessel et al. PRD 103, 043013, Sec. III.E, Eq. (10),
[printed page 12](https://link.aps.org/accepted/10.1103/PhysRevD.103.043013).
Use the [local accepted-manuscript PDF](reference/Wessel_2021_PRD103_043013_accepted.pdf)
for future reading instead of repeatedly fetching the paper. Downloaded from
the APS accepted-manuscript URL on 2026-09-15, unchanged, 17 PDF pages.
Its cover sheet offsets the page numbers: Fig. 12 fits are on printed page 11
(PDF page 12); Eq. (10), extrapolation assumptions, and the 50%/90% endpoint
comparison are on printed page 12 (PDF page 13).
SHA-256: `f09d0d9cf4bfa1559bdd463bf707e27c9c5b37c205b2ee5ead9c4dbd19fe13b2`.

The model fits only complex (2,2): gamma comes from late disk-mass decay,
omega from linear least squares on unwrapped phase, then B from amplitude
least squares at fixed gamma. Their endpoint is 90% of initial mass accreted,
followed by a smooth taper over a few orbits. Their exact joining function and
fit-window selection are unspecified. The paper does not explicitly settle
how other measured modes are retained in the extended Fig. 13 signal.

Our isolated trial uses only measured (2,2) cached FFI strain, not the main
direct-Psi4 spectrum or a synthetic negative-m partner. Natural complex strain
is `hp - i*hc` after undoing the legacy cache conjugation. Times and amplitudes
are `t_ret/M_BH` and `rh/M_BH`. Disk input is bhns.don column 2 minus column 3,
as labeled M0 and M0 inside AH. Initial mass is that diagnostic at t=0.
The existing scalar reader merges restarts; there is no new reader. Source
scalar coordinate time is identified approximately with retarded source time,
using the same origin. This is not a gauge-invariant alignment.

Implementation choices, not paper-prescribed requirements:
- Fit the last 40% of the retained measured waveform, excluding the first
  1000 M_BH. Fit disk mass in linear space, not log mass; do not constrain a
  negative gamma into a positive one. All current scalar series are uniform.
- Withhold the last quarter of the fitting interval, predict that observed
  strain from the earlier fit, then refit the entire late interval.
- Trial limits: 1% mass RMS, 25% amplitude RMS, 0.5 rad phase RMS, and 50%
  held-out complex-strain relative RMS. Require positive resolved decay,
  gamma/Omega_c < 0.1, and at least three fitted wave cycles. These are
  conservative screening choices, not uncertainties or proof of validity.
- Accepted fits blend measured strain to the model over the last orbital
  period with a quintic smoothstep, then continue until 10% initial mass
  remains and taper over three orbital periods. This changes the last orbit
  explicitly, not invisibly. An excessive sample count aborts rather than
  clipping the physical endpoint. Long extrapolations are not justified merely
  by a small residual; persistence of disk structure remains unverified.
- Off/on both use the same cached FFI input, measured-duration frequency floor
  and FFT grid. Onset taper width is fixed; the finite off case retains the
  existing Tukey termination. The added duration does not lower the cutoff or
  change the initial window. No comparison with direct-Psi4 is mislabeled as a
  pure time-extension effect. Standard source scaling/cosmology/SNR are reused.

Default-window results (gamma in inverse BH-mass time):

| Case | gamma | phase RMS [rad] | held-out relative RMS |
| --- | ---: | ---: | ---: |
| A1 | 1.99e-6 | 1.81 | 1.63 |
| A2 | -3.02e-7 | 1.01 | 2.69 |
| A3 | -1.41e-8 | 5.28 | 1.07 |
| B1 | -1.53e-6 | 3.70 | 1.03 |
| B2 | -2.76e-7 | 3.60 | 1.42 |
| B3 | 5.86e-7 | 3.03 | 1.44 |

In the initial strict trial, no real-case tail or enhanced SNR was produced. Fitting starts at fractions
0.4/0.5/0.6/0.7/0.8 gives 30 rejected trials. A3's last-20% fit has better phase
(0.16 rad) and holdout error (0.38), but amplitude error is 0.30: the envelope
is still growing rather than following the very slow fitted mass decay.
These observations concern current finite-radius FFI inputs, not every
possible late regime or proof of incorrect simulation data. M0 evolves weakly
and sometimes increases; its relationship to integrated horizon flux should be
confirmed from the simulation diagnostic implementation before interpreting
gamma as a reliable accretion timescale. Do not silently substitute an integral
of accretion flux or force an exponential decay.

Evidence: `figures/review/temporal_20260915_2/gw/temporal_trial.json`, the six
four-row figures beside it, and `run_logs/temporal_fit_windows_20260915.json`.
Inputs were downloaded into /tmp/bhdisk_temporal_20260915 from the configured
six bhns.don paths and gw_work/<case>_psi49/{rhphc.dat,strain_cache.json}; no
remote files were changed. Local synthetic tests cover parameter recovery,
held-out failure, join preservation, endpoint, sample budget, FFT angular
normalization and accepted-branch plotting. Six production direct-Psi4 spectra
were array-identical to those computed with the incoming Anvil source.

### Physical-Assumption Follow-Up and Conditional A1 Example

Same-day follow-up, still local and not adopted in production. Refreshed the six
`bhns-dens_mode.con` files, scalar histories and outer strain caches into
`/tmp/bhdisk_wessel_physics_20260915`. The five refreshed upstream source files
were unchanged from the earlier incoming Anvil copy. Nothing on Anvil was written.

Checked the project Illinois build source under
`/anvil/projects/x-mca99s008/BHDisk_illinois_maxjamie/illinois_grmhd/configs/giggle/build/`:
- `bhns/diagnostics.f90:1557-1562` writes M0 and inside-AH mass as the respective
  volume integrals times SymmFactor. Lines 1603-1608 separately integrate the
  negative inward flux; this is NOT added to the M0 output.
- `diagnostics_mhd/vol_integrand-rest_mass.f90` uses `rho_star` throughout the
  grid interior, with the existing AMR reduction in `Integrate_vol_integrand.f90`.
- `vol_integrand-rest_mass_inside_AH.f90` uses the excision/AH mask.
- `vol_integrand-meq0_density.f90` and `vol_integrand-meq12_density.f90` require
  `rho_b/rho_b_max > rhob_cutoff`. C0 is therefore a density-cut integral, not
  the same total mass as M0. Do not substitute its steeper decline as a more
  favorable accretion fit. The density-mode angle is measured about the CoM.
These are inspected project build sources, not a binary-provenance audit of
every simulation restart. Source checksums are in the assessment JSON. The
near-constant M0 versus nonzero integrated accretion still needs a mass-budget
assessment before its small slope is interpreted as a precise physical lifetime.

Matter-mode means in four successive quarters of the default GW fit interval:

| Case | normalized m=1 means |
| --- | --- |
| A1 | 0.0244, 0.0201, 0.0254, 0.0296 |
| A2 | 0.0177, 0.0135, 0.0121, 0.0074 |
| A3 | 0.0010, 0.0029, 0.0066, 0.0095 |
| B1 | 0.0036, 0.0029, 0.0037, 0.0017 |
| B2 | 0.0002, 0.0003, 0.0004, 0.0009 |
| B3 | 0.0001, 0.0002, 0.0002, 0.0006 |

A1 is the most plausible *conditional example*, not a demonstrated stationary
source. Its later available matter data (beyond the last retarded GW source
time) also rise, with m=1 quarter means 0.0341, 0.0443, 0.0433, 0.0478. Its
mass model predicts that later scalar endpoint to about 0.055%, but good mass
prediction alone does not validate stationary structure or coherent phase.

`TAIL_REVIEW_CASES=("A1",)` allows an explicitly conditional comparison despite
the retained amplitude/phase/heldout warnings. It cannot override nondecay,
unresolved mass decay, insufficient cycles, nonfinite fits or sample limits.
This does NOT reclassify a failed fit as accepted. Wessel's physical assumptions
and our numerical screening thresholds must not be conflated.

For this example we use exactly the Eq. (10) fitting sequence, the same fitted
mass gamma and measured cached (2,2) strain, and our previously documented
smooth join/taper. No gamma floor, C0 substitution, additional modes or arbitrary
finite lifetime. At source BH mass 50 Msun, D_L=100 Mpc (z=0.0221938), the
single-mode CE SNR is 0.203627 off and 0.725204 on (full frequency-grid integral).
This is NOT the main four-mode direct-Psi4 result or an orientation-averaged SNR.

The fitted mass drops only 0.566% during the late fit. Reaching 10% of initial
mass produces a total waveform 163.25 times the measured post-transient duration.
Changing fit-start fraction from 0.4/0.5/0.6/0.7 gives CE SNR
1.296/0.867/0.725/0.645; the 0.8 window has unresolved mass decay and stays
unextended. The sharp spectral line comes from assuming a coherent long-lived
oscillator, not newly simulated evidence. Do not present it as an improved
prediction or select the window that produces the largest SNR.

Following Wessel's endpoint check, 50% versus 90% accreted gives CE SNR
0.6391 versus 0.7252, a ratio 0.881. FFT padding factors 1/2/4 agree in the
full-grid SNR to 3e-8 relative; shared 2048-bin power integration differs by
less than 0.08% across tested windows. Numerical convergence is much better
than model uncertainty. Other source, detector, FFI and finite-radius systematics
are not included in this test.

Evidence: `run_logs/wessel_physics_assessment_20260915.json`, updated six-case
`figures/review/temporal_20260915_3/gw/temporal_trial.json`, physical/fit figures
per case, and A1's conditional join/spectrum figure. Local tests: 73 pass,
including conditional warnings retained and mass-growth override prohibited.
No production setting/figure has been replaced, and nothing has been synced.

## 2026-09-16 Accretion Convention and Mass-Budget Investigation

Wessel's accretion figure is Fig. 10 in the saved accepted manuscript (printed
p. 9, PDF p. 10), not the frequency-domain Fig. 9. It uses a linear fractional
accretion-rate axis scaled to a 10 Msun BH, and repeats the data against t/M_BH
and t/t_orbit. Printed p. 6 defines time zero at the first maximum of the m=1
density mode (PPI saturation). Our paper plot is logarithmic and starts at
initial data. These presentation differences do not demonstrate a physics bug.

`wip_plots/accretion.py` now makes the separate linear rate comparison and a
mass-budget diagnostic; `./run_wip.py accretion` runs both. Positive inward
accretion is `-bhns.don[:,11]`, NOT abs(). The rate retains the paper plot's
fixed initial-data rest-mass normalization from config. For a source BH mass
M_target, one code time unit is `(M_target / config.mlittle) * G M_sun / c^3`.
There is no ADM-mass factor or cosmological time dilation in this source-frame
diagnostic. Time stays t/P_c from initial data; no saturation time is guessed.

Inputs: the Sep15 staged six-case `bhns.don` files and Illinois source under
`/tmp/bhdisk_wessel_physics_20260915/diagnostic_source_ref/`. Checked raw headers
and writer, horizon-flux routine, volume-integrands and volume reductions.
Columns 2/3 are total/interior-AH mass; column 12 is signed horizon flux;
column 13 accumulates its negative with a trapezoid. The flux routine already
accounts for equatorial symmetry. Escaping-mass columns 4-7 are nested volume
inventories, NOT cumulative boundary losses. No extra factor, sign or sum of
these inventories was applied. Source inspection is not verification of the
binary or scheduling/parameters used for every evolution.

Mass-budget line definitions (one-based bhns.don column numbers): blue is
100 [M_out(t_ref)-M_out(t)]/M_out(t_ref), where M_out=column 2-column 3;
orange is 100 integral[-column 12] dt / M_out(t_ref), trapezoidal integration
of the restart-cleaned scalar samples in code time. t_ref is the first
available scalar sample, not a fitted/extrapolated t=0 value. These are rest
masses, not changes in the BH gravitational mass. M_out includes all recorded
matter outside the AH within the grid, not just gravitationally bound disk
material. Both denominators differ intentionally from the fixed initial-data
mass used by the separate rate plot. Column 13 is an integration cross-check,
not either plotted line. The inspected nonstatic-BH flux routine interpolates
rho_star and velocity onto the AH and includes center motion and dR/dt.
Agreement would require matching diagnostic surfaces/times and no unaccounted
external boundary flux or mass sources/sinks; the plot is not itself a complete
conservation audit. A negative blue value means outside-AH mass increased.

At the last available sample, percentages relative to the first scalar
outside-AH mass (positive measured loss means M_out decreased):

| Case | Measured mass loss [%] | Integrated inward flux [%] | Recorded int_M0dot change [%] |
| --- | ---: | ---: | ---: |
| A1 | 0.8428 | 6.7928 | 6.8071 |
| A2 | -1.2465 | 2.6798 | 2.6838 |
| A3 | -0.0316 | 1.6114 | 1.6126 |
| B1 | -1.3957 | 0.4645 | 0.4652 |
| B2 | -0.0032 | 0.0741 | 0.0742 |
| B3 | 0.5970 | 0.0182 | 0.0182 |

The independent flux integral agrees with the recorded accumulated flux within
0.22% of final accreted mass. Rejecting every old appended restart branch from
the newer branch's start gives identical rows/results for A1-A3/B1-B2. For B3
it drops 25 rows and changes the accumulated flux by only 0.000058 percentage
points (including its slightly earlier retained endpoint). Thus duplicate
handling does not explain the budget mismatch. Scalar versus fixed initial
mass differs by 1.4-3.0%; that also cannot explain the mismatch.

The linear view revealed brief exact-zero flux runs. For 16/14/12/10/16/13
samples in A1/A2/A3/B1/B2/B3, respectively, int_M0dot is unchanged during the
zero(s) and its next increment matches a trapezoid across the whole gap using
the surrounding nonzero fluxes (relative error below 2e-7). This is precisely
the source's `HorizonWasFound != 1` branch: return zero, skip accumulation,
and retain previous integration time/flux. Only samples satisfying this pattern
are displayed as gaps, never filled or smoothed. The raw flux integral stays
unchanged for the independent budget check. Other zeros are not suppressed.

Interpretation: A2 has a declining post-peak accretion rate; A1 remains strongly
variable, and the cases do not share an established exponential depletion
phase. A declining rate alone does not validate Wessel's mass-decay model.
The budget discrepancy requires checking the outside-AH volume diagnostic
(including its mask/scheduling), atmosphere/source terms, numerical mass
conservation and boundary losses. Current scalar data do not determine the
cause or justify calling the simulation bad. Do not replace measured disk
mass by integrated flux to force a desired GW decay rate.

## 2026-09-16 Thick-Disk Stability WIP Comparisons

- [Romeo & Wiegert 2011](https://arxiv.org/abs/1101.4519), Sec. 2.3, Eq. (8):
  finite thickness increases a component's effective Q by T approximately
  0.8 + 0.7 sigma_z/sigma_R for anisotropy ratios 0.5-1. A single isotropic
  gas component gives T=1.5. This stabilizes the conventional planar mode.
- [Romeo & Falstad 2013](https://arxiv.org/abs/1302.4291), Eq. (18), extends
  the anisotropy approximation; Sec. 3 explicitly requires local wavelengths
  kR >> 1 and uses k_max approximately kappa/sigma for its diagnostic.
- [Meidt 2022](https://arxiv.org/abs/2208.01888), Eq. (43): the separate
  midplane 3D diagnostic is Q_M = kappa^2/(4 pi G rho_mid). Its threshold 1
  concerns allowed Jeans-like midplane modes, not the same planar mode as
  thickness-corrected Q. Mapping to Q_T approximately 2 depends on vertical
  structure, anisotropy and self-gravity (Eqs. 44-45); not a universal cutoff.
  The reference model is Newtonian, approximately isothermal and radially
  extended. It is not a GR torus/PPI stability theorem.

Calculations on our existing COCAL profiles (same Newtonian kappa,
sound speed, coordinate density and surface density as wip_plots/toomre.py):

| Case | min Q_N | min (1.5 Q_N) | min Q_M | kappa R/c_s at min Q_N |
| --- | ---: | ---: | ---: | ---: |
| A1 | 10.8816 | 16.3223 | 1.5105 | 0.734 |
| A2 | 1.1835 | 1.7752 | 0.1010 | 0.359 |
| A3 | 0.4774 | 0.7161 | 0.0338 | 0.282 |
| B1 | 30.1671 | 45.2507 | 8.1554 | 1.564 |
| B2 | 3.6766 | 5.5148 | 0.7816 | 0.920 |
| B3 | 0.8794 | 1.3191 | 0.1424 | 0.621 |

Each Q minimum is taken independently over the existing resolved-density
selection; Q_M's minimum need not be at the Q_N minimum. These are exploratory
numbers, not adopted instability classifications. Isotropic thermal pressure
motivates trying T=1.5, but the approximation is not validated for these tori.
At Q_N minima the coordinate thickness proxy Sigma/(2 rho_mid R) is 0.35-0.55;
the local-wavelength estimate kappa R/c_s is not >>1. The EOS is Gamma=1.6,
not isothermal. These limitations remain even after multiplying Q by T.
The user requested two companion comparisons, keeping the original figure.
`./run_wip.py toomre` now also makes `wip/toomre_romeo.png` and
`wip/toomre_meidt.png`: original Q_N above the corresponding alternative,
shared r/M_ADM axis, standard A/B legend. Romeo's panels share y limits to
make its modest factor explicit; Meidt's distinct parameter autoscales.
All calculations/plotting remain in wip_plots/toomre.py. Profiles are read
once per case; Romeo's anisotropy is a top-of-file knob, default 1 (T=1.5).
Both alternatives use the same resolved-density selection as Q_N. Undefined
or nonpositive kappa^2/density samples are gaps, not folded to positive values.
Sound speed and rotation were already sampled at the midplane, but Q_N uses
the vertically integrated Sigma; Meidt instead uses the midplane gas density.
The original six-case numerical arrays and local rendered toomre.png were
verified exactly unchanged against a pre-edit baseline. Four new formula/domain
tests pass with the full 89-test local suite. These WIP plots do not change
the paper plot, GW pipeline, or the scientific applicability limitations above.
The same command now also writes `wip/toomre_cases.png`, a six-panel
same-case overlay of all three diagnostics with common axes and one method
legend. This avoids individual-file sprawl. Anvil LaTeX output inspected;
90 local tests and 12 focused Anvil tests pass after this addition.

## Combined Temporal Comparison: Withdrawn Attempt (2026-09-16)

The established three-rescaling baseline is the direct-Psi4 characteristic
strain. A new cached-FFI-strain off/on pair was implemented and tested, but
the user correctly objected that its off spectrum changed that baseline.
Windowing strain and FFTing it is not the same finite-duration operation as
windowing Psi4, FFTing it and dividing by omega squared; FFI also imposes its
own cutoff. Keeping mass, distance and orientation choices fixed does not
make those spectra identical. The attempted pair and its new helpers were
removed, restoring the pre-change analysis and renderer exactly.

Do not silently replace the baseline to add extrapolation. The combined
time-on request is still unfinished and needs an agreed consistent method.
Existing individual cached-FFI tail diagnostics remain separate, with the
same conditional A1 status and fit limitations documented above. Their
successful numerical checks do not establish a direct-Psi4 continuation.

The restored `wip/restmass.png` uses the former tripleM panel's definition:
bhns.don column 2, total conserved M0, normalized to its first recorded value
(t=0 for current inputs). It includes mass inside the AH. It is not the
column-2-minus-column-3 outside-AH mass driving the temporal decay fit.

## 2026-09-21 Local Notes: ET, Flux Fits and Baseline-Preserving Tails

Source: the user's local todo.txt. Anvil source refresh found no upstream
differences. All work used locally staged copies; no simulation files changed.

Detectability now shows A1-A3 only. The added ET curve is the collaboration's
public 10 km CoBA table ET-0304B-22 (28 March 2023), fourth column PSD:
https://apps.et-gw.eu/tds/?r=18213 . Its single-90-degree-equivalent ASD is
converted to three independent 60-degree Michelsons with amplitude gain 3/2,
then uses the existing Wessel sky/polarization convention. Effective ASD is
sqrt(10)/(3/2) times sqrt(PSD), used identically for plots and SNR. This is a
specified design/network assumption, not an operating-detector sensitivity.
Other detector curves, source masses/distances, units and cosmology are unchanged.

The new requested trial fits M_flux(t)=M_out(0)-[int_M0dot(t)-int_M0dot(0)]
instead of the volume-integrated M_out(t). Columns 2-3 define initial M_out;
column 13 is the native cumulative inward rest-mass flux. This proxy assumes
no other mass losses/sources, so it is a variation on Wessel's measured-mass
fit, not an identity or a repair of the volume diagnostic. TAIL_MASS_SOURCE
can be horizon_flux or outside_mass. Signed flux is retained, never abs().

| Case | gamma M_BH, flux fit | GW fit interval t/P_c | Delta M_BH / integrated rest-mass flux in that interval |
| --- | ---: | --- | ---: |
| A1 | 8.9363e-6 | 16.1891-24.9342 | 0.2197 |
| A2 | 5.7819e-6 | 14.7148-22.7083 | 0.9091 |
| A3 | 7.1143e-7 | 13.9561-21.6679 | 0.9046 |

BH mass is the existing Christodoulou estimate from AH areal radius and spin,
not irreducible mass alone. Compare only common scalar/AH/spin time coverage;
rebase all curves at the exact fit-interval start for the late view. BH energy
gain and rest-mass accretion need not be equal (binding/internal/kinetic energy,
radiation and numerical horizon errors matter). A2/A3 show close late-time
tracking, while the t=0 comparison is dominated by initial BH relaxation and
drift. A1 remains discrepant. The volume-mass budget discrepancy is not solved.
The independently integrated instantaneous flux still agrees with the recorded
integral within 0.22% of accumulated mass. The full budget now plots column 13
and BH mass change; the Python integral remains a cross-check. The previous
Sep16 line definitions above are historical, not the current orange-line source.

Combined temporal charts now stay in Psi4 for BOTH calculations. The measured
off path is numerically identical for every A case to the saved pre-edit arrays.
For an allowed trial, q=B exp[(-gamma+i omega)(tau-t_end)+i phase] gives the
analytic dimensionless Psi4 model p=q''=(-gamma+i omega)^2 q. Blend this p into
measured (2,2) over the last orbit using the existing quintic; terminate at the
same 90%-depleted mass threshold and taper p over three orbits. Keep the measured
onset-taper width and original measured-duration lower frequency floor. Other
modes keep their measured Tukey termination; sum complex modes BEFORE averaging.
No negative-m tail, radial extrapolation, new rescaling, or new cutoff is inferred.
The strain-model fit still depends on the existing cached FFI convention. Its
analytic derivative and a smooth join do not establish that it fits Psi4 well.

Only A1 has the existing conditional-review waiver. Its waveform errors remain
large (amplitude 0.557, phase RMS 1.813 rad, held-out relative error 1.629;
Psi4 model residual 1.021). A2/A3 still fail waveform checks and have no tail.
The A1 flux trial lasts 36.75 measured durations, not a prediction validated by
the simulation. At the fixed CE/DECIGO/LISA example scales, A1 SNR off -> on is
6.738 -> 7.513, 6.651 -> 7.077, 6.507 -> 6.921. Binned versus full-grid SNR differs
by <0.010% for these three tail spectra. This tests integration, not tail physics.

Outputs: the canonical characteristic-strain/horizon files are measured only;
their *_time_on.png partners contain the labeled conditional tail. The JSON
gw_detectability_time_comparison.json records case status, residuals and SNRs.
A1/gw_temporal_psi4_transition.png shows the ACTUAL Psi4 join used in that chart;
the separate strain transition remains available. accretion_mass_budget.png
shows all six cases; accretion_late_mass_budget.png shows A1-A3 over their GW
fit intervals, with numerical provenance in accretion_mass_budget.json.

Writing guidance retained from the user's note: describe measured waveforms
and their timing relative to matter-mode growth/saturation before detectability.
PPI-generated nonaxisymmetry provides a changing quadrupole; angular-momentum
redistribution can limit growth (Wessel Sec. III). Present that as the physical
interpretation, not as a causal inference from a rescaled FFT or synthetic tail.
Peak coincidence and transport claims for OUR runs need support from the
matter-mode and angular-momentum diagnostics; no universal saturation time has
been assigned here. The paper prose remains an author task in todo.txt.

### Raw-unit budget follow-up, September 21

The full and late budgets now show raw changes in G=c=M_sun=1, with scientific
tick notation and no mass normalization or astrophysical mass rescaling. The
10-M_sun rescaling in the separate accretion RATE plot is unchanged and is NOT
applied to these budgets. Previous percent plots divided all three curves in
each case by the SAME initial outside-AH mass: that did not cause their agreement.
Panel y ranges remain independent and are explicitly numbered.

The late figure includes all A/B cases, each using its own existing late GW-fit
interval. Every curve is rebased to zero at that interval's start. Raw net changes:

| Case | t/P_c interval | integrated AH rest mass | BH mass change |
| --- | --- | ---: | ---: |
| A1 | 16.1891-24.9342 | 2.40414e-5 | 5.28081e-6 |
| A2 | 14.7148-22.7083 | 1.69957e-4 | 1.54507e-4 |
| A3 | 13.9561-21.6679 | 1.02940e-4 | 9.31188e-5 |
| B1 | 14.0164-21.4290 | 8.35261e-7 | -2.24316e-5 |
| B2 | 12.9527-19.9660 | 1.78055e-6 | -1.53098e-5 |
| B3 | 12.1435-18.8711 | 2.38946e-6 | -1.56148e-5 |

Checked the existing Christodoulou calculation and common-support interpolation.
For B1/B2/B3, Delta M_irr is +1.99806e-4/+2.30028e-4/+3.70961e-4, whereas
Delta(M_BH-M_irr) is -2.22238e-4/-2.45338e-4/-3.86575e-4. Thus their decreasing
net BH estimate is already in the area/spin diagnostics, not an artifact of
plot normalization. A1 likewise cancels +2.16912e-4 against -2.11631e-4.
This decomposition identifies the cancellation, NOT the physical/numerical
origin of the spin/area evolution. Rest-mass flux is not an energy flux; do not
claim a conservation-law violation or interpret these ratios as efficiencies.

ET distance curves now use dashed lines while CE remains solid, retaining case
colors and both detector families. This is a presentation change only.

## References

- Wessel et al., Phys. Rev. D 103, 043013 (2021).
- Moore, Cole, and Berry (2014), sensitivity curves and characteristic noise.
- C. P. L. Berry (2020), gravitational-wave data-analysis guides and FFT
  windowing discussion.
- Detector-specific current documentation for LVK/A+, CE, ET, LISA, DECIGO,
  and PTA curves.
