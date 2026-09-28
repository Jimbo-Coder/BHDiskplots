# GW Methods: Equations for the Paper

Checked against the local and Anvil implementations on 2026-09-23. This is a
writing aid describing what the code does, not a claim that all systematic
errors are controlled. Equations below use LaTeX notation.

**Two separate paths:** time-series figures use fixed-frequency integration
(FFI); the measured detectability baseline Fourier-transforms Psi4 directly.
The latter does not Fourier-transform the cached FFI strain.

## 1. Units, Extraction, and Time

The simulations use $G=c=M_\odot=1$. One raw code-time unit is
$GM_\odot/c^3=4.92549095\,\mu\mathrm{s}$, not $GM_{\rm ADM}/c^3$.
Let $m_{\rm BH}$ denote the configured initial BH mass in code units
(currently 0.05), and $M_{\rm BH}$ a chosen physical source-frame BH mass.
Neither is the total ADM mass or the disk rest mass.

For the stored spin-weight $-2$ multipoles, define

$$
P_{\ell m}=r_A\Psi_{4,\ell m},\qquad
H_{\ell m}=r_A h_{\ell m},\qquad
\Psi_4=\ddot h,\qquad h=h_+-i h_\times.
$$

Here $r_A$ is the extraction sphere's recorded areal radius. These are
finite-radius wave-zone estimates, not waveforms at null infinity. Production
detectability uses extraction index 8; index 4 and intermediate spheres are
checks. No radial extrapolation or CCE is used.

The inherited Fortran prescription constructs a gauge-corrected retarded time:

$$
\begin{aligned}
\frac{dt_{\rm S}}{dt}
&=\frac{g^{tr}-\sqrt{(g^{tr})^2-g^{tt}g^{rr}}}
 {g^{tt}(1-2M_{\rm ADM}/r_A)},\\
r_*&=r_A+2M_{\rm ADM}\ln\!\left(\frac{r_A}{2M_{\rm ADM}}-1\right),\\
t_{\rm ret}&=t_{\rm S}-r_*.
\end{aligned}
$$

The metric quantities are the inverse-metric entries in the extraction file;
$t_{\rm S}$ is integrated by the trapezoidal rule, with $t_{\rm S}(t_0)=t_0$.
Modes are interpolated onto a common uniform retarded-time grid using the
reference routine's four-point polynomial stencil. Restart overlap is resolved
before analysis. Using $M_{\rm ADM}$ here does **not** make it the mass used for
the astrophysical rescaling below.

## 2. Time-Series Strain: FFI

With $t=t_{\rm ret}$, $\widetilde x(f)=\int x(t)e^{-2\pi i f t}\,dt$ and $\omega=2\pi f$,
the twice-integrated mode is computed with the regularized denominator

$$
\widetilde H_{\ell m}^{\rm FFI}(\omega)
=-\frac{\widetilde P_{\ell m}(\omega)}
 {\max(|\omega|,\omega_{0m})^2},\qquad
\omega_{0m}=\max(|m|,1)\,\Omega_{\rm orb,0}.
$$

An inverse transform gives the time series. This suppresses low-frequency
integration drift rather than deleting every frequency below the cutoff.
The $(2,2)$ cutoff is $2\Omega_{\rm orb,0}$; $m=0,\pm1$ use
$\Omega_{\rm orb,0}$. This particular cutoff rule follows our reference code;
FFI itself is described by Reisswig & Pollney [R1].

The Python backend reproduces the Fortran conjugation/storage convention:
cached columns are $r_Ah_+$ and $r_Ah_\times$, so form the natural complex mode
as $H_{\ell m}=(r_Ah_+)_{\ell m}-i(r_Ah_\times)_{\ell m}$.
The routine zero-pads outside its selected input interval; it applies **no
Tukey taper** and does not require the displayed final strain to vanish.
Do not confuse this preprocessing with the detectability window below.

## 3. A Mode, a Direction, or an Angular Average

A plot of one $(\ell,m)$ coefficient has **no observer angle assigned**.
For a chosen direction $\Omega=(\theta,\phi)$, sum complex modes coherently:

$$
H(t,\Omega)=\sum_{(\ell,m)\in\mathcal S}
 H_{\ell m}(t)\,{}_{-2}Y_{\ell m}(\Omega),\qquad
r_Ah_+=\Re H,\qquad r_Ah_\times=-\Im H.
$$

The same harmonic sum applies to $P_{\ell m}$. Do not sum mode magnitudes
first: that loses relative phase and angular interference. With normalized
harmonics, $\langle|H|^2\rangle_\Omega=(4\pi)^{-1}\sum|H_{\ell m}|^2$;
this is a time-domain squared-amplitude identity, not the mean spectral
amplitude or a waveform for a particular observer.

Current detectability uses $\mathcal S=\{(2,2),(2,1),(2,-2),(2,-1)\}$ for A1-A3.
`MODES="all"` includes all cached modes, $\ell=2,3,4$. The subset is explicitly
**not total GW emission**; excluded $m=0$ content has not been established to
be purely numerical. Negative-$m$ modes are read, not inferred from symmetry.

The comparison direction is $(\pi/2.34,0)$, following Wessel [R3]; it is not an
exact substitute for averaging a multimode signal. The separate face-on
time-series plot uses $(0,0)$, where
${}_{-2}Y_{\ell m}(0,0)=\sqrt{(2\ell+1)/(4\pi)}\,\delta_{m2}$.
Only $(2,2)$ contributes there for the default subset.

## 4. Direct-Psi4 Spectrum and Source-Orientation Average

Use dimensionless quantities

$$
\tau=t_{\rm ret}/m_{\rm BH},\qquad
q_{\ell m}=H_{\ell m}/m_{\rm BH},\qquad
p_{\ell m}=m_{\rm BH}P_{\ell m}=\frac{d^2q_{\ell m}}{d\tau^2}.
$$

Remove pre-arrival samples, then the first $1000m_{\rm BH}$ after the first
nonnegative retarded-time sample. On the retained interval of length
$\Delta\tau$, apply a symmetric Tukey window $W$ with $\alpha=0.05$
(2.5% tapered at each end), then compute

$$
\widetilde p_{\ell m}(\nu)
=\int W(\tau)p_{\ell m}(\tau)e^{-2\pi i\nu\tau}\,d\tau.
$$

The discrete FFT includes the sample-spacing factor $\Delta\tau_{\rm sample}$.
Twofold zero-padding samples the spectrum more densely; it adds no physical
duration. Retain $\nu\ge3/\Delta\tau$ and $\nu<\nu_{\rm Nyquist}$; exclude DC.
This duration-based floor is **not** the FFI orbital-frequency cutoff.
No additional detrending or amplitude-based frequency cut enters the SNR.

Reconstruct $p(\tau,\Omega)$ with the harmonic sum and define

$$
X(\nu,\Omega)=\mathcal F[W\Re p],\qquad
Y(\nu,\Omega)=\mathcal F[W\Im p],\qquad
A(\nu,\Omega)=\sqrt{\frac{|X|^2+|Y|^2}{2}}.
$$

Both positive and negative frequencies of the complex mode FFT are used to
recover these real-component transforms. The sign of the cross component
does not affect $A$. A common window makes summing modes before the FFT
equivalent to summing their complex FFTs afterward. Wessel's Eq. (8) uses this
polarization-amplitude convention [R3]. Our further source-direction average is

$$
\begin{aligned}
\mathcal A_{\rm mean}&=\frac1{4\pi}\int A\,d\Omega
\quad\text{(default)},\\
\mathcal A_{\rm RMS}&=\sqrt{\frac1{4\pi}\int A^2\,d\Omega},\\
Q(\nu)&=\frac{\mathcal A(\nu)}{(2\pi\nu)^2}.
\end{aligned}
$$

For a chosen direction, replace $\mathcal A$ by $A(\nu,\Omega)$.
The angular integral uses 24 Gauss-Legendre nodes in $\cos\theta$ and 48
periodic azimuthal samples, with normalized weights summing to one.
$Q$ is a nonnegative spectral-amplitude statistic, not a complex waveform.
Mean-amplitude averaging is not RMS averaging or an orientation-averaged SNR.

The conversion $\widetilde q=-\widetilde p/(2\pi\nu)^2$ motivates this
finite-signal estimator. Windowing and differentiating do not commute, so
$\mathcal F[W\Psi_4]/\omega^2$ need not equal $-\mathcal F[Wh]$ near the
ends or at low frequency. Window/floor sensitivity remains a systematic.

## 5. Physical Mass, Distance, and Characteristic Strain

For source-frame mass $M_{\rm BH}$ and redshift $z$, define

$$
M_z=(1+z)M_{\rm BH},\qquad T_z=GM_z/c^3,\qquad L_z=GM_z/c^2,
$$

with $q_{+,\times}=r_Ah_{+,\times}/m_{\rm BH}$ for a chosen source direction:

$$
\begin{aligned}
t_{\rm obs}&=T_z\tau,\qquad f_{\rm obs}=\nu/T_z,\\
h_{+,\times}^{\rm obs}&=\frac{L_z}{D_L}q_{+,\times},\\
|\widetilde h_{\rm eff}(f_{\rm obs})|
&=\frac{L_z}{D_L}T_z Q(\nu),\\
h_c&=2f_{\rm obs}|\widetilde h_{\rm eff}|.
\end{aligned}
$$

$h_c$ is dimensionless; $\widetilde h$ has units of seconds. There is no extra
duration or cycle-count factor. Mass rescaling preserves the dimensionless
model, including its disk/BH mass ratio; it does not independently increase
the disk mass at fixed BH mass. For the orbital-frequency figure,
$f/f_{\rm orb}=\nu/[m_{\rm BH}\Omega_{\rm orb,0}/(2\pi)]$.

The implemented flat matter-plus-Lambda cosmology is

$$
D_L(z)=(1+z)\frac{c}{H_0}\int_0^z
\frac{dz'}{\sqrt{\Omega_m(1+z')^3+1-\Omega_m}},
$$

with $H_0=67.66\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$ and $\Omega_m=0.30966$.
These are Planck18 parameter choices, **not the full Astropy Planck18 model**:
radiation/neutrino evolution is omitted. Redshift is recomputed from distance.
The illustrative $(M_{\rm BH}/M_\odot,D_L/\mathrm{Mpc})$ pairs are currently
$(50,100)$, $(10^3,500)$, and $(10^5,50)$, not fitted source parameters.

## 6. Detector Noise, SNR, and Distance Reach

For one-sided effective noise PSD $S_n$, Moore et al. [R2] give

$$
\begin{aligned}
h_n(f)&=\sqrt{fS_n(f)},\\
\rho^2&=4\int\frac{|\widetilde h_{\rm eff}(f)|^2}{S_n(f)}\,df
=\int\left(\frac{h_c}{h_n}\right)^2d\ln f.
\end{aligned}
$$

Integrate only the overlap of retained signal frequencies and the configured
detector band. This is noise-weighted integration, not convolution of the two
plotted curves; crossing a noise curve at one point does not ensure detection.
The plotted noise and SNR use identical response conventions:

| Detector input | Effective ASD used, $\sqrt{S_n}$ |
| --- | --- |
| LIGO A+, CE 40 km, analytic DECIGO instrument ASD | $\sqrt{10}$ times input ASD |
| LISA SciRDv1 already-averaged PSD | $\sqrt{2}$ times its square root |
| ET CoBA 10 km, single 90-degree-equivalent PSD | $\sqrt{10}/(3/2)$ times its square root |

The factors follow the implemented Wessel convention: right-angle response
factor $\sqrt5$, then the $\sqrt2$ adjustment discussed in footnote 4 [R3].
ET's $3/2=\sqrt3\sin60^\circ$ assumes three equal-noise, independent 60-degree
Michelsons. LISA must not receive the ground-detector response factor again.
Source-orientation averaging above is distinct from detector-sky averaging.
These are design curves, not an observed LVK network sensitivity; curve
versions and provenance are in [detector_curves/README.md](../detector_curves/README.md).

For each source-frame mass, vary $z$, recompute $D_L$ and the redshifted
spectrum, and find the outermost $\rho=8$ crossing. The plotted curve uses a
redshift scan and interpolation in $\log\rho$ (search limited to $z\le10$);
a result at that cap is a search limit, not a demonstrated horizon.
SNR evaluation uses 2048 logarithmic bins preserving integrated spectral power.
Both measured and time-on distance plots show $D_L$ on the left and $z$ on a
linked right axis, using the same $D_L(z)$ above. The redshift axis only
relabels distance; it does not change the horizon calculation.

Call this an **SNR-threshold luminosity-distance reach for the selected
amplitude statistic**, not an optimal-orientation horizon or population range.
Squaring a source-direction mean amplitude does not produce a mean SNR or
mean squared SNR. No detection-rate calculation is performed.

## 7. Optional Time Extrapolation, Not the Measured Baseline

Following the model in Wessel Eq. (10) [R3], fit only the complex $(2,2)$ mode:

$$
\begin{aligned}
M_d(\tau)&=M_{d,e}e^{-\gamma(\tau-\tau_e)},\\
q_{22}^{\rm model}&=B e^{(-\gamma+i\omega)(\tau-\tau_e)+i\phi_e}.
\end{aligned}
$$

Fit $\gamma$ from mass in linear space, $\omega,\phi_e$ from a linear fit to
unwrapped strain phase, and $B$ from the amplitude with $\gamma$ fixed. The
default mass proxy is initial outside-AH rest mass minus accumulated inward
AH rest-mass flux; it neglects other losses/sources and is **not** BH energy.
Fitting actual outside-AH rest mass is an alternative. The source scalar time
is identified approximately with retarded GW time; this is not gauge invariant.

Our fit interval is the last 40% of the retained strain, with a final-quarter
holdout check. For the time-on detectability chart, differentiate the analytic
model, $p_{22}^{\rm model}=(-\gamma+i\omega)^2q_{22}^{\rm model}$, and blend it
into measured Psi4 over the last orbit using $s(x)=10x^3-15x^4+6x^5$.
Continue until fitted remaining mass is 10% of its initial value, then taper
over three orbits. Keep the original onset taper and measured-duration floor;
all other modes retain their measured termination. These joining, fitting,
flux-proxy and acceptance choices are ours, not prescriptions from Wessel.
The tapered/joined Psi4 is not the exact second derivative of a separately
tapered/joined strain curve. A1 is currently a **conditional illustration**;
A2/A3 retain measured data only. Do not claim a validated full-lifetime signal.

## Paper Checklist and References

State the extraction sphere, mass convention, retained modes/time interval,
FFI cutoff for time-series plots, FFT window/floor for spectra, angular
statistic, detector response/curve versions, cosmology and SNR threshold.
Separate measured results from model-dependent tails. Finite-radius, window,
low-frequency and mode-selection uncertainties remain; do not claim that
all transients, memory, or nonradiative contamination have been removed.

- **[R1] Reisswig & Pollney (2011)**, *Notes on the integration of numerical
  relativity waveforms*, CQG 28, 195015, Sec. 4.2:
  [arXiv:1006.1632](https://arxiv.org/abs/1006.1632). Cite for FFI, not for our
  particular orbital-frequency cutoff or implementation details.
- **[R2] Moore, Cole & Berry (2015)**, *Gravitational-wave sensitivity curves*,
  CQG 32, 015014, Eqs. (16)-(19):
  [arXiv:1408.0740](https://arxiv.org/abs/1408.0740). Cite for $h_c$, $h_n$ and
  SNR conventions, not as validation of our preprocessing/averaging choices.
- **[R3] Wessel et al. (2021)**, *Gravitational Waves from Disks Around Spinning
  Black Holes: Simulations in Full General Relativity*, PRD 103, 043013:
  [arXiv:2011.04077](https://arxiv.org/abs/2011.04077), Secs. III.C and III.E,
  Eqs. (7), (8), (10), footnote 4, Figs. 9 and 11-13. Our direct-Psi4 spectrum,
  full directional quadrature and conditional continuation are not identical
  to their analysis. [Local PDF](reference/Wessel_2021_PRD103_043013_accepted.pdf).

Implementation pointers: [gw.py](../gw.py) (`reconstruct_strain`),
[gw_detectability.py](../gw_detectability.py) (numbered analysis sections),
[gw_strain_observer.py](../wip_plots/gw_strain_observer.py) (face-on time series).
Plot formatting is separate in
[gw_detectability_all.py](../wip_plots/gw_detectability_all.py).
