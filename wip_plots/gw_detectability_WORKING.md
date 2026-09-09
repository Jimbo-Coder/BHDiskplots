# GW Detectability Working Log

Last updated: 2026-09-09
Status: direct detectability is implemented and unit-tested. The shared NumPy
waveform preprocessing backend is parity-tested against the repaired Fortran
reference locally and on Anvil. The full Anvil cache and combined
detectability figures were regenerated successfully on 2026-09-02.

This file records the current objective, open decisions, and next work. Move
settled conclusions into `KNOWLEDGE.md`.

## Code Cleanup

- Second pass: `GWRun.at_radius(index)` returns a stable radius-specific
  `Waveform`, not a mutable current selection. All modes remain in that result.
  Removed unused per-simulation mode/radius-list settings and reader accessors.
- Cache reads and generation are separate functions; only `generate_gw.py`
  chooses rebuild versus reuse. The numerical FFI functions are unchanged.
  Radius-to-LaTeX formatting lives with display helpers. Waveform plots select
  explicit named series functions rather than inferring behavior from names.
- Second-pass local verification: 42 tests, including generation/readback of
  case and difference caches without source edits. The same 24 synthetic
  figures match in all 344 recorded data/layout arrays and label text.
- Second pass synced after source checksum checks: Anvil also passed all 42
  tests and rendered 24 real-data figures; A1's direct spectrum again had 6316
  bins. No production figure regeneration, commit, or push was performed.

- `gw.py` now owns extraction reading, restart merging, the unchanged FFI
  equations, and cache access. Scalar/2D readers no longer own GW conversion.
- Nine ordinary waveform plotting scripts are consolidated in
  `wip_plots/gw_waveforms.py`. Combined jobs load each radius once for all modes
  and plot types; individual jobs reuse loaded extraction files and strain.
- Shared run, unit, and detectability settings are in `config.py`. Detector
  response and SNR calculations moved out of the plot into
  `helpers/gw_detectability.py`; no response/window/cutoff convention changed.
- Local verification: 41 tests; 24 old/new synthetic figures agree in plotted
  data, axis bounds, panel positions, and labels. Detector curves, binned power,
  and sampled horizon curves match exactly. Zero-power binning now returns an
  empty spectrum instead of indexing an empty array.
- Synced the tested cleanup to Anvil after checking all affected source
  checksums. Anvil passed the same 41 tests and rendered 24 real-data figures
  covering A1/B1/ML at both shared radii and A1 individual comparisons. The
  A1 direct spectrum contained 6316 bins and the smoke-test LIGO SNR was
  finite (0.583457 at the test's redshifted mass 10 Msun and distance 150 Mpc).
  No production figures were overwritten, no full regeneration was launched,
  and nothing was committed or pushed.
- This is a structural cleanup, not a fresh scientific sign-off. In particular,
  ordinary raw-Psi4 time-series plots still associate raw mode samples with the
  cached uniform retarded-time grid as before. Check the size of that mismatch
  before changing their sampling or the Psi4-difference reconstruction. The
  direct detectability spectrum already consumes the matching uniform Psi4
  intermediate. Do not conflate a parity test with validation of this choice.

## Current Objective

Make the BHDisk detectability pipeline scientifically defensible and easy to
explain from numerical `Psi_4` through characteristic strain and SNR, while
retaining explicit comparison products for assumptions that cannot yet be
eliminated.

## Completed

- Preserved the complete 696-line collaborator script verbatim.
- Replaced the mixed strain/radial/tail implementation with one finite direct
  `r Psi_4` production method.
- Added a focused reusable direct-Psi4 transform in
  `helpers/gw_detectability.py`.
- Added synthetic tests for FFT normalization, angular averaging, physical
  mass scaling, and cosmology inversion.
- Made source-direction mean the sole production source average and retained
  the representative angle only as an explicit comparison diagnostic.
- Unified detector plotting and SNR on the same effective ASD convention.
- Reduced paper-facing outputs to characteristic strain, horizon, and method
  comparison figures.
- Replaced radial-extrapolation/time-tail individual diagnostics with finite
  radius, window, and angular-resolution tests.
- Added a maintained NumPy implementation of the established gauge-corrected
  retarded-time and FFI workflow, while retaining the Fortran as a regression
  reference. Its cache exposes uniformly sampled `r Psi_4` directly to this
  detectability workflow without routing the spectrum through strain.
- Regenerated all 75 configured Anvil cache products with the NumPy backend
  without failures, then verified that all six detectability cases and both
  comparison radii consume `python-uniform-rPsi4`.

## Settled Method Decisions

- The former strain-FFT path depended on the FFI cutoff. It has been replaced
  for detectability by direct finite-duration `r Psi_4/(2 pi f)^2`.
- The former representative and RMS-like production alternatives have been
  replaced by the collaborator's solid-angle mean. The representative angle
  remains only in a labeled comparison plot.
- The former radial fit and synthetic late-time tail are not used in the
  production spectrum. Finite outer-radius data are the conservative central
  result. The first usable wave-zone radius (`r=120`) is always evaluated
  against the outer valid radius (`r=170`), with one intermediate radius shown only
  in the individual validation figure.
- The direct transform takes retarded time and `r Psi_4` from the shared
  maintained preprocessing cache, removes `t_ret<0`, and only then applies the
  `1000 M_BH` transient cut. Legacy caches retain the previous fallback.
- Uniform resampling now precedes the Tukey window.
- The Wessel detector-response prescription is encoded once and reused by the
  plotted noise and SNR/horizon calculation.
- The standard `h_c=2 f |h_tilde|` and one-sided matched-filter SNR forms are
  retained, with explicit redshifted-mass and luminosity-distance scaling.

## Remaining Verification

1. Decide how the finite-radius, Tukey-window, and transient-cut systematics
   should be summarized in the paper. The first/outer-radius comparison now has
   a combined figure, while the numerical-choice brackets remain in every
   individual validation figure.
2. If tighter uncertainty is required, improve extraction-radius control or
   obtain a validated radial-extrapolation method before changing the central
   finite outer-radius result.
3. Treat any future late-time continuation as a separately labeled upper-bound
   model; do not silently merge it into this finite-signal production result.

## Publication Gate

The method and figures are reproducible and internally consistent, but the
current finite-radius/window spread is too large to describe the amplitudes or
horizons as precision predictions. Present them as finite-signal estimates
with that numerical systematic stated explicitly.
