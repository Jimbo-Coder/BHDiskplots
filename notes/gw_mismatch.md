# A1 versus A3: two independent mismatch calculations

This is an optional WIP diagnostic. It compares the coherent complex
**(2,2) FFI strain** of A1 and A3, using both a direct NumPy calculation and
the actual public `kuibit.gw_mismatch.mismatch_from_strains` API. Kuibit is
not a label on a second call to our manual routine. Both methods receive
the same preprocessed time arrays, complex strains, frequency band and PSD.

The default answers for the 2026-10-16 cache snapshot are:

| Weighting | Manual, optimized time and phase | Kuibit, refined grid | Difference |
| --- | ---: | ---: | ---: |
| White noise | 0.7901276721 | 0.7901304977 | 0.0000028257 |
| Cosmic Explorer, initial BH mass 50 solar masses, z=0 | 0.8279063399 | 0.8279064782 | 0.0000001383 |

On exactly matching time/phase grids, the manual and kuibit values agree
within floating-point roundoff. The small differences in this table are
finite search-grid errors, not a different definition of mismatch. These
numbers are conditional on the signal definition and interval below. They
do not yet reproduce a colleague's unspecified calculation.

As a resolution check, doubling the requested time/phase counts (513
times and 256 phase parameters in the broad search, 512 in refinement)
gives white-noise kuibit mismatch 0.7901287902, only 0.0000011182 above
the optimized manual result. The manual and kuibit values again agree
to roundoff on the shared grids.

## What signal is compared?

The read-only inputs are `gw_work/A1_psi49/rhphc.dat` and
`gw_work/A3_psi49/rhphc.dat`, with their `strain_cache.json` manifests.
Parfile radius index 8 maps to file label 9 for these runs. Columns 0, 1
and 2 contain the retarded time and the cached plus/cross components of
the (2,2) mode. We construct

\[
\tau=t_{\rm ret}/M_{{\rm BH},0},\qquad
q(\tau)=\frac{r(h_+-i h_\times)_{22}}{M_{{\rm BH},0}}.
\]

Each run uses its own configured initial BH mass (0.05 in simulation code
units for both cases). We do not align disk orbital periods or rescale one
frequency evolution to the other. Independent constant amplitude factors
cancel from the normalized overlap.

This diagnostic uses cached FFI histories, including their existing
reconstruction choices and mode-dependent integration cutoffs. It does
not validate the FFI reconstruction independently. It is separate from
the direct-Psi4 detectability calculation. In particular, an
orientation-averaged characteristic-strain **magnitude** cannot replace
these coherent strains: it has discarded the phase information needed
for mismatch. There is no synthetic time continuation here.

## Finite interval and preprocessing

The default common interval is
\(1000\leq\tau\leq8123.560977803\), the earlier of the two waveform end
times after retarded-time normalization. A3's later samples are omitted
explicitly, so unequal available simulation durations do not become a
spurious signal-shape difference.

Both modes are linearly interpolated onto the same 6324-sample uniform
grid, no finer than the coarsest input step. Its spacing is
\(\Delta\tau=1.126610940662\). We remove each mode's complex mean on this
interval, apply the same Tukey window with alpha=0.05, and append zeros
to 16384 samples. At least \(2N-1\) padded samples prevent circular
wraparound in the allowed linear-lag interval. Padding adds no physical
signal. The cropped/tapered segment is fixed during the lag search; it
is shifted as a whole, rather than re-cropped and renormalized at each lag.

The white-noise integration band is
\[
4.21137687927\times10^{-4}\leq M_{\rm BH}f
\leq0.421618486788.
\]
The low end is three cycles over the **retained**, unpadded duration;
the high end is 95 percent of the common-grid Nyquist frequency. DC and
Nyquist are excluded. Both signs of Fourier frequency are retained in
the manual calculation. The defaults are explicit diagnostic choices,
not universal optimal cuts; `--start`, `--end`, `--taper`, `--fmin` and
`--fmax` allow a different agreed comparison.

## Mismatch definition and independent manual method

For one real detector strain the standard inner product is
\[
(a,b)=4\,\mathrm{Re}\int_{f_{\min}}^{f_{\max}}
\frac{\widetilde a(f)\widetilde b^*(f)}{S_n(f)}\,df.
\]
For the complex mode we use an **ideal pair of equally weighted,
orthogonal polarization channels**: one sees plus, the other cross.
This defines an intrinsic mode comparison without choosing a particular
source sky location or interferometer antenna pattern. It is not a
claim about a physical detector network's SNR.

Define \(Q_i\) as the complex FFT of \(q_i\). With an even PSD, the
equivalent two-sided complex correlation is
\[
C(\Delta\tau)=
\frac{\sum_{k\in B} Q_{1k}Q_{3k}^*
 e^{2\pi i f_k\Delta\tau}/S_n(|f_k|)}
 {\sqrt{\left(\sum_{k\in B}|Q_{1k}|^2/S_n(|f_k|)\right)
              \left(\sum_{k\in B}|Q_{3k}|^2/S_n(|f_k|)\right)}}.
\]
Here \(B\) includes positive and negative bins in the stated band.
A constant mode rotation \(q_3\mapsto e^{i\alpha}q_3\) changes the
correlation to \(e^{-i\alpha}C\); its largest real overlap is therefore
\(|C|\), attained at \(\alpha=\arg C\). Our definition is
\[
\boxed{\mathcal M=1-\max_{|\Delta\tau|\leq T/2}|C(\Delta\tau)|}.
\]
Zero mismatch means identical normalized signals after the allowed
time/phase shifts; mismatch near one means little normalized overlap.
The white-noise result corresponds to match \(\simeq0.20987\), not a
79 percent numerical error or uncertainty in either simulation.

For justification of the two-sided expression, write
\(q=p-i c\). Summing the positive- and negative-frequency products
gives twice the real plus-plus/cross-cross inner product, together with
the quadrature that mixes plus and cross under a constant rotation.
The same factor of two appears in each norm. FFT factors of
\(\Delta\tau\), the integration measure \(\Delta f\), and the
one-sided factor four cancel between numerator and denominator. They
are not extra correction factors to insert into a normalized mismatch.
A constant PSD or detector-response multiplier also cancels. There is
no extra factor of 1/5 here and no average of normalized overlaps over
source directions.

`ManualOverlap` in `gw_mismatch.py` implements this expression using
NumPy FFTs and explicit complex Fourier sums. An inverse FFT evaluates
all sample lags in the full allowed interval, then direct sums refine
the strongest local peaks and check the boundaries. Phase maximization
is analytic. This is independent of kuibit's positive-frequency,
two-real-polarization grid calculation. Both are numerical searches;
their agreement and convergence are evidence for this comparison,
not a theorem about an arbitrary waveform's global maximum.

## Proper use of kuibit

The reference is the published **kuibit 1.6.1** implementation and its
[mismatch documentation](https://sbozzolo.github.io/kuibit/gw_mismatch.html).
We call `mismatch_from_strains` with `TimeSeries(time, q)` for each
already prepared signal. We pass `antenna_patterns=[(0,1),(1,0)]`;
kuibit orders these entries as `(F_cross,F_plus)`. Both channels use
the same noise PSD, or `noises=None` for white noise.

This explicit pair matters. A rotation preserves its summed plus/cross
norm exactly. Kuibit normalizes with the unrotated waveforms; for a
single projected channel that norm need not remain constant as its
complex input is rotated. We avoid that ambiguity by specifying the
polarization pair, rather than accepting the default antenna factors.
For a pure (2,2) comparison, a constant mode phase and a polarization
rotation are degenerate, which is the restriction stated by kuibit.
This setup should not be generalized to arbitrary mixtures of modes
without revisiting the alignment definition.

One version-specific detail is recorded explicitly: kuibit 1.6.1 builds
its parameter `p` on `[0,2*pi]` but actually rotates the signal by
`exp(1j*pi*p)`. We do not modify the library. The same-grid manual
comparison uses those actual rotations, and JSON records both the
returned parameter and the physical rotation in radians. Do not read
the returned parameter alone as our waveform phase.

The first kuibit run searches 257 times over the full interval
\([-3561.78049,+3561.78049]M_{\rm BH}\), including zero, and 128 phase
parameters. Its own best time centers a second search with 257 times
and 256 phase parameters. The refinement is not centered on the
manual answer. For each search the manual method also evaluates
exactly kuibit's grids. If their mismatches differ by more than
\(10^{-9}\), the diagnostic fails instead of publishing agreement.

The white-noise manual optimum is at \(\Delta\tau\simeq2.67782\), well
inside the search interval. Positive lag means comparing A1 with
\(q_3(\tau-\Delta\tau)\). Inspect JSON and the overlap-versus-lag plot
before interpreting any other case; a boundary maximum or an
unresolved sharp peak requires a wider/finer search.

## Detector weighting

White noise gives a scale-free waveform-shape comparison. For a detector
PSD, assign both modes the same source-frame initial BH mass and redshift:
\[
t_{\rm obs}=\tau\,(1+z)\frac{GM_{\rm BH}}{c^3},\qquad
f_{\rm obs}=\frac{f_\tau}{(1+z)GM_{\rm BH}/c^3}.
\]
The optional detector variant uses the existing `effective_detector_asd`
curve squared, not an ASD as though it were a PSD. The common band is
intersected with the configured detector band and tabulated support.
For CE at 50 solar masses and zero redshift it is 5 to 1711.9856 Hz.
Constant response multipliers cancel in the normalized result; the PSD's
frequency dependence does not. Luminosity distance only multiplies a
strain amplitude and therefore cancels at fixed redshift. Changing the
redshift still changes which part of a detector's PSD weights the signal.

Kuibit receives that same PSD sampled at the exact positive FFT bins in
both channels. The manual method evaluates it at absolute frequencies.
This avoids introducing different PSD interpolation conventions in the
two methods. This is still a polarization-pair mode mismatch with a
detector-shaped weighting, not a sky-averaged detector-network inference.

## Reproduce and inspect

This diagnostic is registered as `gw_mismatch` and excluded from default
`run_wip.py`/`run_all.py` runs. It never regenerates caches or modifies
simulation ASCII. On Anvil, use the normal Python/TeX environment:

```bash
python run_wip.py gw_mismatch
python -m wip_plots.gw_mismatch --detector ce --mass 50
```

Locally, the optional published dependency can be provided without
changing the core environment's dependency list:

```bash
uv run --with kuibit==1.6.1 python -m wip_plots.gw_mismatch --no-plot
uv run --with kuibit==1.6.1 python -m unittest tests.test_gw_mismatch
# More time and phase samples, in a separate output folder:
uv run --with kuibit==1.6.1 python -m wip_plots.gw_mismatch \
  --num-times 513 --num-phases 256 --tag white_finer --no-plot
```

`--no-plot` saves the numerical results without requiring local TeX.
`--cache-root PATH` selects an isolated cache snapshot with the usual
`A1_psi49` / `A3_psi49` layout. Outputs live under
`figures/wip/gw_mismatch/white/` or `ce/` (or the explicit `--tag`):

- `A1_A3.json`: settings, cache manifests/hashes, implementation hashes,
  kuibit version/source hash, both grid comparisons and optimized result.
- `A1_A3_inputs.npz`: the exact preprocessed complex inputs and lag curve,
  allowing a colleague to check the overlap without our cache reader.
- `A1_A3.pdf` and `A1_A3.png`: finite normalized waveforms above and
  phase-maximized overlap versus time shift below, with both grid results.

The upper-panel amplitude normalization is for display only. It is not
an additional rescaling applied to the mismatch. The two methods share
input preparation so that they test the overlap calculation, rather
than accidentally comparing different physical intervals.

Focused tests check identical/scaled/rotated modes, a known non-circular
time delay, orthogonal FFT-bin signals, PSD scaling, exchange symmetry,
common-interval padding, invalid inputs, real public-kuibit agreement
for white and colored PSDs, and explicit opt-in registration.

To compare with another calculation, agree on waveform quantity,
polarizations/modes, radius, mass/time normalization, time interval,
window/demeaning, PSD/band and allowed shifts first. Two implementations
agreeing on these inputs validates this numerical overlap calculation;
it does not make mismatch independent of those choices.
