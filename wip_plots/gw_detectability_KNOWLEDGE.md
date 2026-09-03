# GW Detectability Knowledge Log

Last updated: 2026-09-02

This file records stable scientific and implementation knowledge for the
BHDisk GW detectability work. Keep tentative plans and unresolved work in
`WORKING.md`.

## Provenance

- The verbatim collaborator reference script is
  `colleague_bh_cluster_detectability_20260830.py`.
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

- Combined detectability runs through `plots/gw_detectability_all.py` from
  `run_all.py --extra`.
- Per-simulation numerical validation runs through
  `plots/gw_detectability_diagnostics_individual.py`.
- The production observable is the finite outer-radius (`r=170`) `r Psi_4`
  signal. The first extraction sphere treated as wave-zone data (`r=120`) is
  always processed as a finite-radius systematic check. One intermediate
  radius is retained in the per-simulation validation figure, but is not mixed
  into the production central value.
  It uses the maintained waveform cache's gauge-corrected retarded-time grid
  and uniformly sampled `r Psi_4`, removes pre-arrival samples (`t_ret<0`),
  then discards the first `1000 M_BH` of the arrived waveform. It uses every
  available mode through `ell=3`, applies a Tukey window with `alpha=0.05`, and
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
- A legacy cache without `rpsi4_uniform.dat` may be a strict prefix of a later
  merged Psi4 table.
  In that case every cached retarded-time value is preserved and only the
  missing time coordinates are extended using the measured late-time
  `dt_ret/dt` slope. No waveform amplitude is extrapolated. This supplied 51
  missing A3 times and 103 missing B3 times in the 2026-08-30 production run.

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

## References

- Wessel et al., Phys. Rev. D 103, 043013 (2021).
- Moore, Cole, and Berry (2014), sensitivity curves and characteristic noise.
- C. P. L. Berry (2020), gravitational-wave data-analysis guides and FFT
  windowing discussion.
- Detector-specific current documentation for LVK/A+, CE, ET, LISA, DECIGO,
  and PTA curves.
