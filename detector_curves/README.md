# Detector curves

The detectability code treats these files according to the explicit loaders
and response conversion in `gw_detectability.py`.

- `AplusDesign.txt`: frequency in Hz and strain ASD in `1/sqrt(Hz)`. This is
  the LIGO A+ design target distributed as LIGO-T1800042 and as the A+ O5
  target in LIGO-T2200043.
- `CE2_40km_strain.txt`: frequency in Hz and strain ASD in `1/sqrt(Hz)` from
  the current baseline 40 km Cosmic Explorer curve, CE-T2000017-v9,
  `cosmic_explorer_strain.txt` (27 October 2025).
- `LISA_Alloc_Sh.txt`: frequency in Hz and equivalent sky-averaged strain PSD
  in `1/Hz`, using the LISA SciRDv1 `AnalyticNoise.sensitivity()` allocation.
  The analysis takes its square root exactly once and does not apply the
  right-angle ground-detector response factor again.
- `ET10kmcolumns.txt`: unmodified public ET CoBA table, ET-0304B-22
  (28 March 2023). Column 1 is Hz; column 4 is the combined LF+HF PSD for
  one 10 km, 90-degree-equivalent interferometer. This is explicitly the
  CoBA design curve, not a claim about a latest operating detector.
  [ET source and conventions](https://apps.et-gw.eu/tds/?r=18213),
  [original table](https://apps.et-gw.eu/tds/?call_file=18213_ET10kmcolumns.txt).

The analytic DECIGO PSD is Yagi and Seto, Phys. Rev. D 83, 044011 (2011),
Eq. (5), and is kept in `gw_detectability.py` rather than a table. That PSD is
for one effective L-shaped interferometer and is not sky averaged.

The paper figures follow the convention stated in Wessel et al. (2021):

- A+, CE, and DECIGO instrument ASDs are first multiplied by `sqrt(5)` for
  the standard sky-and-polarization average of a right-angle interferometer.
- Every resulting sky-and-polarization-averaged curve is multiplied by
  `sqrt(2)` because the plotted source quantity already contains the
  `1/sqrt(2)` polarization average in `h_res`.
- The LISA file already includes the first average, so it receives only the
  second `sqrt(2)` factor.
- ET uses the square root of column 4, then `sqrt(10)/(3/2)`. The denominator
  is `sqrt(3)*sin(60 degrees)` for three independent, equal-noise 60-degree
  Michelsons relative to the supplied single 90-degree curve. Thus ET is a
  triangular-network estimate under uncorrelated-noise assumptions, not a
  single Michelson with the right-angle response applied unchanged.

Exactly these effective curves are used for both the plotted characteristic
noise and the SNR/horizon integral.
