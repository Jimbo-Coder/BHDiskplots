# GW Detectability Working Log

Last updated: 2026-09-02
Status: direct detectability is implemented and unit-tested. The shared NumPy
waveform preprocessing backend is parity-tested against the repaired Fortran
reference locally and on Anvil. The full Anvil cache and combined
detectability figures were regenerated successfully on 2026-09-02.

This file records the current objective, open decisions, and next work. Move
settled conclusions into `KNOWLEDGE.md`.

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
