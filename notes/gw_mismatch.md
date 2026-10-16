# A1 versus A3 mismatch, in the simulation's own code units

The complete runnable workflow is **`gw_mismatch.py`** in the repository root.
It contains the settings and file reader, preparation, the independent manual
calculation, the actual kuibit call, the analysis, and the plot. CLI overrides
are at the bottom. Run this file directly; the WIP runner calls it directly too.

The default comparison uses the stored time and strain columns directly.
**No physical BH mass, distance or redshift is assigned. Neither waveform is
divided by its BH mass.** For the current outer-radius FFI caches:

| Method | Mismatch |
| --- | ---: |
| Independent NumPy calculation, optimized time and phase | 0.7901276721 |
| kuibit 1.6.1, refined time/phase grid | 0.7901304977 |

These are the same to five decimal places: **0.79013**. On the identical
search grids the two calculations agree to roundoff; the remaining difference
from the optimized manual answer is finite-grid error. This is conditional on
the mode, interval and weighting below, not a universal mismatch for every
possible definition of an A1/A3 signal.

## Inputs and preprocessing

The inputs are `gw_work/A1_psi49/rhphc.dat` and `A3_psi49/rhphc.dat`, with
their cache manifests. Radius index 8 maps to raw Psi4 file label 9 for A cases.
Columns 0, 1 and 2 contain retarded time and the plus/cross components of the
(2,2) FFI strain mode. The complex signal is the **stored**
`r*h_plus - 1j*r*h_cross`.

The default common interval is **50 to 406.17804889015 in code time**.
The end is the earlier waveform endpoint, so A3's extra duration is excluded.
Both signals are interpolated to one uniform grid (6324 samples, spacing
0.0563305470331 code-time units), demeaned, tapered by a Tukey window with
alpha=0.05, and padded to 16384 samples. Padding does not add physical signal;
at least `2*N-1` samples prevent circular wrapping in the searched lag range.

The frequency band is **0.00842275375854 to 8.43236973575 per code-time unit**:
three cycles over the retained unpadded duration through 95% of Nyquist.
DC and Nyquist are excluded. The allowed time shift is plus/minus half the
retained duration, 178.089024445075 code-time units. The cropped/windowed
segments stay fixed during the lag search; they are shifted as whole signals.
No extra signal is extrapolated.

The FFI reconstruction is shared input preparation, not independently checked
by the two mismatch implementations. This comparison retains phase information;
an averaged characteristic-strain magnitude cannot replace the input waveform.

## Definition and the two methods

We compare both components of the complex (2,2) mode as an ideal pair of equally
weighted orthogonal polarization channels. This avoids choosing a particular
sky position or a single detector projection. It is an intrinsic mode comparison,
not a physical-network SNR calculation.

For FFTs `H1`, `H3`, an even PSD and the stated positive/negative-frequency band B,
the manual calculation evaluates

\[
C(\Delta t)=\frac{\sum_{k\in B}H_{1k}H_{3k}^*e^{2\pi i f_k\Delta t}/S_n(|f_k|)}
{\sqrt{\sum_{k\in B}|H_{1k}|^2/S_n(|f_k|)
             \sum_{k\in B}|H_{3k}|^2/S_n(|f_k|)}};
\qquad \mathcal M=1-\max_{\Delta t}|C(\Delta t)|.
\]

The absolute value analytically maximizes over a constant complex-mode phase.
The time shift moves the whole prepared segment; it does not stretch time or
change the comparison window separately at each shift.

To connect this with a 2-norm, let each band-limited, noise-weighted signal have
unit norm, \(\widehat g_i=g_i/\|g_i\|\). Expanding the squared separation gives

\[
\|\widehat g_1-e^{i\phi}\widehat g_3(t-\Delta t)\|^2
=2-2\operatorname{Re}[e^{-i\phi}C(\Delta t)].
\]

Therefore

\[
\mathcal M=\frac12\min_{\Delta t,\phi}
\|\widehat g_1-e^{i\phi}\widehat g_3(t-\Delta t)\|^2.
\]

For white noise this is the ordinary integrated squared difference after band
selection and normalization. The minimum selects the best whole-signal
alignment; it is not a minimum or maximum pointwise error. With a detector PSD
the same relation holds in the corresponding noise-weighted norm.

The manual implementation in `gw_mismatch.py` searches every sample lag by
inverse FFT and refines the strongest peaks with explicit Fourier sums.
The native-unit manual optimum is near lag **0.13389109 code-time units**.

The second implementation calls the actual installed
`kuibit.gw_mismatch.mismatch_from_strains` function with `TimeSeries` objects.
We explicitly pass `antenna_patterns=[(0,1),(1,0)]`, ordered by kuibit as
`(F_cross,F_plus)`. Summing both channels makes the norm invariant under
mode rotation. The default is `noises=None`, i.e. white noise.

Kuibit searches 257 times and 128 phase parameters across the broad interval,
then 257 times and 256 phase parameters near its own coarse maximum. The manual
code separately evaluates exactly those grids; disagreement above 1e-9 raises
an error. The kuibit refinement is not centered on the manual answer.
Kuibit 1.6.1 applies `exp(1j*pi*p)` to its phase parameter `p`; we preserve that
implementation and record the actual rotation in radians. See the
[package documentation](https://sbozzolo.github.io/kuibit/gw_mismatch.html).

## Does scaling change the answer?

A constant amplitude factor on either waveform cancels in its normalized overlap.
For **white noise**, a common time-unit conversion also cancels when the band and
allowed shifts are converted consistently. Here both initial BH masses are
0.05 in code units, so expressing both signals in `t/M_BH` and `r*h/M_BH` was a
common change of units. The new direct-code-unit calculation confirms the same
mismatch; it does not assume the BH has mass 0.05 solar masses.

A detector-shaped PSD is different. Assigning a physical mass changes where the
frequencies fall in that PSD, so the detector-weighted mismatch can change.
There is no unique detector-weighted answer until that physical scale and PSD
are specified. The optional CE comparison at 50 solar masses, z=0 gives
manual **0.8279063399**, kuibit **0.8279064782**. It requires an explicit
`--mass 50`; it is not used in the default native-unit result.

The factors of four, FFT sample spacing and common frequency-bin measure cancel
between numerator and norms. A constant PSD/antenna-response multiplier cancels
too. There is no additional factor of 1/5. A mismatch near 0.79 means low normalized
match for these signals and choices, not a 79% numerical error in a simulation.

## Run and inspect

In the normal Anvil Python environment:

```bash
python run_wip.py gw_mismatch
python gw_mismatch.py --no-plot
# Choose another pair, preserving the requested order:
python gw_mismatch.py --sims A2 A3 --radius-index 6
# Explicitly assign a physical scale only for this detector-weighted variant:
python gw_mismatch.py --detector ce --mass 50
```

Locally, provide kuibit using the usual project environment or
`uv run --with kuibit==1.6.1 python gw_mismatch.py --no-plot`.

For normal edits, change `SIM_NAMES`, `RADIUS_INDEX`, `START_TIME`, `END_TIME`,
the taper or band at the top of `gw_mismatch.py`. The `analyze` function shows
the sequence explicitly. `ManualOverlap` contains the Fourier sums;
`kuibit_check` contains the package call; `plot_comparison` contains the figure.
Keep these methods on the same prepared signals when extending the comparison.

From another computer, copy `rhphc.dat` and `strain_cache.json` for each chosen
case/radius from Anvil into `gw_work/<case>_psi4<label>/`, or point
`--cache-root` at that tree. For A/B cases, parfile index 6 means raw file label
7, hence `A1_psi47`, for example. No raw simulation files are needed here.
The diagnostic is excluded from default all-plot runs.

Outputs live in `figures/wip/gw_mismatch/white/` (or `ce/`): a PDF/PNG comparison,
`A1_A3.json` with choices, hashes and both methods' results, and
`A1_A3_inputs.npz` containing the exact prepared complex signals, `time_code`,
`lags_code` and the overlap curve. NPZ time starts at zero; the original interval
start is recorded as `selection.start` in JSON. Schema version 2 uses generic
time-shift fields in code units. Previous schema-1 outputs used M_BH units.
Display amplitude normalization in the waveform panel does not change the
mismatch calculation. `--start`, `--end`, `--fmin`, `--fmax` and `--shift-limit`
now all use **code time or inverse code time**, as appropriate.

Tests cover known shifts/rotations, identical and orthogonal signals, PSD scaling,
exchange symmetry, common-unit invariance, and agreement with the real package.
For a different pair, filenames and NPZ waveform keys follow those case names;
the default A1/A3 filenames are unchanged.
