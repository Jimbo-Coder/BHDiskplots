# Detector curves

Each table starts with `#` comments giving its source, column units, and
response convention. The numeric rows are unchanged, and `numpy.loadtxt`
ignores the comments. Strain is dimensionless: ASD has units `1/sqrt(Hz)` and
PSD has units `1/Hz`. The detectability code uses these as one-sided spectra,
as specified by the loaders and response conversion in `gw_detectability.py`.

- `AplusDesign.txt`: frequency in Hz and strain ASD in `1/sqrt(Hz)`. This is
  the [original LIGO A+ design target, LIGO-T1800042-v5](https://dcc.ligo.org/LIGO-T1800042/public),
  also distributed as an A+ O5 target in LIGO-T2200043. It is a design
  estimate, not measured noise or a later O5 projection.
- `CE2_40km_strain.txt`: frequency in Hz and strain ASD in `1/sqrt(Hz)` from
  the baseline 40 km [Cosmic Explorer curve, CE-T2000017-v9](https://dcc.cosmicexplorer.org/cgi-bin/DocDB/ShowDocument?.submit=Identifier&docid=T2000017),
  `cosmic_explorer_strain.txt` (27 October 2025). The numeric rows exactly
  match that archive. CE specifies a source 15 degrees off normal incidence;
  the later `sqrt(5/2)` factor is our simple sky-response approximation, not
  an average supplied with the CE table.
- `LISA_Alloc_Sh.txt`: frequency in Hz and equivalent sky-averaged strain PSD
  in `1/Hz`, tabulated from the LISA SciRDv1
  [`AnalyticNoise.sensitivity()` model](https://lisa.pages.in2p3.fr/LDC/_modules/ldc/lisa/noise/noise.html).
  Its calculation already includes antenna/projection averaging; it is not
  a raw TDI PSD. The original tabulation command and LDC revision were not
  recorded. The analysis takes its square root exactly once and does not
  apply the right-angle ground-detector response factor again.
- `ET10kmcolumns.txt`: public ET CoBA table, ET-0304B-22 (28 March 2023),
  with comment metadata added but numeric rows unchanged. Column 1 is Hz;
  columns 2, 3, and 4 are the ETHF, ETLF, and combined LF+HF strain PSDs
  in `1/Hz` for one 10 km L-shaped interferometer. This is a CoBA design
  curve, not a measurement or the full triangular-network sensitivity.
  [ET source and conventions](https://apps.et-gw.eu/tds/?r=18213),
  [original table](https://apps.et-gw.eu/tds/?call_file=18213_ET10kmcolumns.txt).

The analytic DECIGO one-sided strain PSD in `1/Hz` is from
[Yagi and Seto, Phys. Rev. D 83, 044011 (2011)](https://doi.org/10.1103/PhysRevD.83.044011),
Eq. (5), and is kept in `gw_detectability.py` rather than a table. It is for
one effective L-shaped interferometer and is not sky averaged.

Effective noise is defined so that `4*int h_res^2/S_eff df` equals the
detector-sky- and polarization-angle-averaged SNR^2, with Wessel Eq. (8)
`h_res^2 = (|h+|^2 + |hx|^2)/2`:

- A right-angle interferometer has `<F+^2> = <Fx^2> = 1/5` and `<F+ Fx> = 0`,
  so `<|F+ h+ + Fx hx|^2> = (2/5) h_res^2`. A+, CE, and DECIGO instrument
  ASDs are multiplied by `sqrt(5/2)`.
- ET uses the square root of column 4, then `sqrt(5/2)/(3/2)`. The denominator
  is `sqrt(3)*sin(60 degrees)` for three independent, equal-noise 60-degree
  Michelsons relative to the supplied single 90-degree curve. Thus ET is a
  triangular-network estimate under uncorrelated-noise assumptions.
- The LISA table equals twice the Robson et al. (2019) two-channel curve at
  low frequency (checked in the tests): it is one Michelson channel already
  averaged per polarization (the 20/3 normalization). Two independent
  channels halve that PSD and the `h_res` convention halves it again, so the
  effective ASD is half its square root.

One may instead put the response factor `sqrt(2/5)` on `h_res` and use the
raw instrument ASD; both choices give the same `h_c/h_n` and SNR. Equivalently,
the root-sum-square polarization amplitude `sqrt(2)*h_res` pairs with
`sqrt(5)` times the raw ASD. Wessel et al. (2021) footnote 4 describes a
`sqrt(2)` change to an already averaged sensitivity curve. Interpreting its
starting curve as `sqrt(5)` times the same raw instrument ASD would yield a
`sqrt(10)` curve; paired with Eq. (8) `h_res`, that would halve the SNR from
our direct antenna average. The published input-curve and numerical-SNR
conventions have not been independently established, so this conditional
comparison does not establish an error in Wessel's reported results. Our
results before 2026-09-28 used `sqrt(10)` with the raw PSD and `h_res`:
ground-based and DECIGO SNRs were 2x low relative to our current convention,
LISA 2*sqrt(2)x low (single channel).

Exactly these effective curves are used for both the plotted characteristic
noise and the SNR/horizon integral.
