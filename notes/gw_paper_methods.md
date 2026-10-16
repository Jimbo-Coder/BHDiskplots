# GW Methods: Equations for the Paper

Checked against the local and Anvil implementations on 2026-09-28. This is a
writing aid describing what the code does, not a claim that all systematic
errors are controlled. Equations below use LaTeX notation.

**Two separate paths:** time-series figures use fixed-frequency integration
(FFI); the measured detectability baseline Fourier-transforms Psi4 directly.
The latter does not Fourier-transform the cached FFI strain.

The [short GW and PPI summary](human_gw_method.md) gives the physical argument
and the full workflow in one place. Section 10 below derives the azimuthal
selection rules and explains their limits. That interpretation was added on
2026-10-06; it does not change the analysis settings or measured results.

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
nonnegative retarded-time sample. The uniform grid ends at the last measured
Psi4 sample; zero-filled grid points beyond it are discarded. On the retained interval of length
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
| LIGO A+, CE 40 km, analytic DECIGO instrument ASD | $\sqrt{5/2}$ times input ASD |
| LISA SciRDv1 single-channel averaged PSD | $1/2$ times its square root |
| ET CoBA 10 km, single 90-degree-equivalent PSD | $\sqrt{5/2}/(3/2)$ times its square root |

The detector records $F_+h_++F_\times h_\times$. Averaging over sky position
and polarization angle, a right-angle interferometer has
$\langle F_+^2\rangle=\langle F_\times^2\rangle=1/5$ and
$\langle F_+F_\times\rangle=0$, so

$$
\langle|F_+\tilde h_++F_\times\tilde h_\times|^2\rangle
=\tfrac15\left(|\tilde h_+|^2+|\tilde h_\times|^2\right)
=\tfrac25\,|\tilde h_{\rm res}|^2 ,
$$

and $S_{\rm eff}=\tfrac52 S_n$ gives the averaged $\rho^2$ exactly. The
equivalent source-side convention is $\sqrt{2/5}\,h_{\rm res}$ with the raw
instrument noise; the ratio $h_c/h_n$ and its integral are unchanged. If the
source uses the root-sum-square polarization amplitude
$\sqrt2\,h_{\rm res}$, its noise curve must instead be $\sqrt5$ times the
raw instrument ASD.
ET's $3/2=\sqrt3\sin60^\circ$ assumes three equal-noise, independent 60-degree
Michelsons. The LISA SciRDv1 table is one channel with the per-polarization
average already applied ($20/3$ normalization, twice Robson et al. at low
frequency); two independent channels and the $h_{\rm res}$ factor each halve it.
Wessel footnote 4 [R3] describes multiplying an already sky- and
polarization-averaged sensitivity curve by $\sqrt2$ for its Eq. (8)
$h_{\rm res}$. If that starting curve means $\sqrt5$ times the same raw
right-angle instrument ASD, the result is $\sqrt{10}$ times the raw ASD.
Paired with Eq. (8) $h_{\rm res}$, it would give half the SNR of the direct
antenna average above. This is a conditional comparison of displayed-curve
definitions, not a reproduction of Wessel's SNR calculation: its exact input
curve conventions and numerical pipeline have not been established. We use
the directly averaged response for our stated input curves.

**In Moore's terms [R2].** Moore's SNR is Eq. (16),
$\rho^2=4\int|\tilde h|^2/S_n\,df$, where $\tilde h$ is the detector output
$F_+h_++F_\times h_\times$ [Eq. (48)]. The averaged response is quoted as
$F=\sqrt{1/5}$ for a right-angle interferometer. That value agrees with
either single-polarization average,
$\langle F_+^2\rangle=\langle F_\times^2\rangle=1/5$. Moore's Eq. (51), as
printed, instead integrates $F_+^2+F_\times^2$ and gives $2/5$ (also checked
by quadrature). Its formula and stated numerical value therefore differ by a
factor of two in $F^2$. Robson et al. [R5], Eqs. (4) and (7), explicitly
define the response per polarization and give $1/5$ for each; this independently
resolves the ambiguity in Moore's printed equation. Using that directly
averaged response, the SNR is

$$
\langle\rho^2\rangle=4\int\frac{|\tilde h_+|^2+|\tilde h_\times|^2}{S_n/F^2}\,df
=4\int\frac{|\tilde h_{\rm res}|^2}{S_n/(2F^2)}\,df .
$$

The $\sqrt{S_n}/F=\sqrt5\sqrt{S_n}$ curve obtained by applying Moore's
stated value to a raw instrument ASD pairs with the root-sum-square amplitude
$\sqrt{|\tilde h_+|^2+|\tilde h_\times|^2}$. For
$h_{\rm res}$ it must be divided by $\sqrt2$, giving $\sqrt{5/2}\sqrt{S_n}$.
For Eq. (8) $h_{\rm res}$, averaging the detector over sky position alone also
gives the $2/5$ response above. A $\sqrt{10}$ curve applied to the same raw
ASD would imply an effective response of $1/10$ for this amplitude, one
quarter of that direct average. The arXiv
Fig. 10 (accepted-manuscript Fig. 11) was inspected as a vector figure, but its
plotted heights alone do not identify the exact detector input files or
establish what entered the separate horizon calculation.
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

## 8. Radiated Energy, Angular Momentum, and Frequency

The `gw_radiated` diagnostic accumulates, from the same FFI modes and after
the same $1000m_{\rm BH}$ cut, over the four detectability modes only [R4]:

$$
\begin{aligned}
E_{\rm GW}(t)&=\frac1{16\pi}\int_{t_{\rm cut}}^{t}\sum_{\ell m}
 \left|\dot H_{\ell m}\right|^2dt',\\
J_{z,\rm GW}(t)&=\frac1{16\pi}\int_{t_{\rm cut}}^{t}\sum_{\ell m}
 m\,\Im\!\left[H_{\ell m}\dot H^{*}_{\ell m}\right]dt',
\end{aligned}
$$

with the sign chosen so that prograde emission gives $J_{z,\rm GW}>0$. The code
reuses the parity-tested port of the reference routine. We report
$E_{\rm GW}/M_{\rm disk,0}$ and $J_{z,\rm GW}/(M_{\rm disk,0}m_{\rm BH})$. The
instantaneous $(2,2)$ frequency is taken from the Psi4 phase, so it is
independent of the FFI cutoff:

$$
f_{22}=\frac1{2\pi}\left|\frac{d}{dt}\arg P_{22}\right|,
$$

averaged over one orbital period $P_c$ and shown as $f_{22}/f_{\rm orb}$ with
$f_{\rm orb}=\Omega_{\rm orb,0}/2\pi$. Earlier checks found that including
$m=0$ raised the FFI-derived energy by $10^4$-$10^5$. Such sensitivity warrants
investigation of integration drift and nonradiative contamination; it does
not establish that all $m=0$ emission is unphysical. These diagnostics report
the selected nonaxisymmetric modes, not the total physical energy or angular
momentum radiated by the system.

## 9. Algorithm Summary (LaTeX)

Requires `\usepackage{algorithm,algpseudocode}`. Symbols follow Secs. 1-6.
Compiled and checked 2026-09-28 in `article` and in `revtex4-2` two-column.
RevTeX 4.2 clashes with the `float` package that `algorithm` loads ("Missing
\endcsname ... \c@float@type"). There, load only `algpseudocode` and replace
`\begin{algorithm}[t]`/`\end{algorithm}` with
`\begin{figure}[t]\hrule\smallskip` / `\smallskip\hrule\end{figure}`.

```latex
\begin{algorithm}[t]
\caption{Detectability of the measured signal from direct $\Psi_4$.}
\label{alg:detectability}
\begin{algorithmic}[1]
\Require Modes $r_A\Psi_{4,\ell m}(t)$ at extraction radius $r_A$, with
  $(\ell,m)\in\{(2,\pm1),(2,\pm2)\}$; initial BH mass $m_{\rm BH}$ ($G=c=1$)
\State Build the gauge-corrected retarded time $t_{\rm ret}$ and interpolate all
  modes onto a common uniform grid (Sec.~1)
\State Keep $t_{\rm ret}\ge0$; discard the first $1000\,m_{\rm BH}$ of initial
  relaxation
\State Form $\tau=t_{\rm ret}/m_{\rm BH}$ and $p_{\ell m}=m_{\rm BH}r_A\Psi_{4,\ell m}$;
  apply a Tukey window ($\alpha=0.05$) and FFT with $2\times$ zero padding
\State Keep $3/\Delta\tau\le\nu<\nu_{\rm Nyq}$
\For{each source direction $\Omega$ (Gauss--Legendre quadrature)}
  \State $\tilde p(\nu,\Omega)\gets\sum_{\ell m}\tilde p_{\ell m}(\nu)\,{}_{-2}Y_{\ell m}(\Omega)$
  \State $A(\nu,\Omega)\gets\sqrt{(|\mathcal F[W\Re p]|^2+|\mathcal F[W\Im p]|^2)/2}$
\EndFor
\State $Q(\nu)\gets\langle A\rangle_\Omega/(2\pi\nu)^2$ \Comment{direction-mean amplitude of $h$}
\For{each source-frame mass $M_{\rm BH}$ and redshift $z$}
  \State $f\gets\nu/[M_{\rm BH}(1+z)]$, \quad
    $|\tilde h(f)|\gets Q\,M_{\rm BH}^2(1+z)^2/D_L(z)$
  \State $h_c\gets2f|\tilde h|$, \quad $h_n\gets\sqrt{fS_{\rm eff}}$ with
    $S_{\rm eff}=\tfrac52S_n$ (right-angle interferometer)
  \State $\rho^2\gets\int(h_c/h_n)^2\,d\ln f$ over the detector band
\EndFor
\State \Return the luminosity distance where $\rho=8$, for each $M_{\rm BH}$
\end{algorithmic}
\end{algorithm}
```

In prose: we Fourier-transform the windowed, retarded-time $r\Psi_4$ multipoles
directly, reconstruct both polarizations over a quadrature of source
directions, and average the polarization amplitude
$\sqrt{(|\tilde h_+|^2+|\tilde h_\times|^2)/2}$ over directions. Dividing by
$(2\pi f)^2$ gives the strain spectrum. It is rescaled to a source-frame BH
mass and luminosity distance, and compared with detector noise averaged over
sky position and polarization angle. For this amplitude, that noise is
$S_{\rm eff}=\tfrac52S_n$ for a right-angle interferometer.

## 10. PPI Interpretation and Matter to GW Mode Coupling

### Physical picture

The Papaloizou-Pringle instability (PPI) is a global nonaxisymmetric
hydrodynamic instability of a differentially rotating torus. A useful cartoon
shows disturbances on either side of corotation, where the pattern angular
speed equals the fluid angular speed. Coupling between waves carrying opposite
signs of wave energy/angular momentum, with suitable trapping at the torus
boundaries, can amplify the pattern. Low-order modes, often $m=1$ in thick
tori, can produce an orbiting overdensity. Angular-momentum redistribution
and changes to the torus can then limit growth [R3, R7]. This is a physical
interpretation to compare with the evolution, not a universal saturation time.

The Toomre diagnostics address local gravitational stability under their stated
approximations; $Q>1$ does not rule out a global hydrodynamic PPI. An orbiting
lump is a schematic explanation of a time-dependent source multipole, not a
literal radiating mass dipole. A quantitative lump/BH model must use the whole
system and a consistent center-of-mass frame.

### Density modes and radiation modes are different objects

A density mode $C_m$ labels azimuthal structure of a scalar matter field.
The plotted $C_m/C_0$ uses a density-cut diagnostic integral, with $C_0$ not
equal to the total disk mass. A radiation coefficient $h_{\ell m}$ instead
belongs to

$$
h_+-i h_\times=\sum_{\ell=2}^{\infty}\sum_{m=-\ell}^{\ell}
h_{\ell m}\,{}_{-2}Y_{\ell m}(\theta,\phi).
$$

These are **spin-weight $-2$ spherical harmonics** [R6]. The spin weight
describes transformation under rotation of the polarization basis and requires
$\ell\ge |s|=2$. Thus $h_{21}$ is an allowed quadrupolar radiation mode with
azimuthal number one; it is not dipole radiation. Neither $\ell=0$ nor
$\ell=1$ occurs in this strain expansion. Observable plus and cross strains
come from the coherent harmonic reconstruction in Sec. 3. Naming the real
and negative imaginary parts of a complex mode as modal plus/cross components
is a storage convention, not a choice of observer direction.

### Linear selection and the equatorial symmetry condition

For an axisymmetric background, consider a perturbation
$\delta\rho=\epsilon\,\hat\rho_{m_p}(r,\theta)e^{i(m_p\phi-\omega_pt)}$
plus its complex conjugate. We use $Q$ for source mass multipoles:
$Q_{ij}$ is the Cartesian quadrupole tensor and $Q_{2m}$ its spherical
components, with fixed harmonic normalization. At leading Newtonian order
the general scalar mass multipole is

$$
Q_{\ell m}=\int\rho\,r^\ell Y_{\ell m}^{*}\,d^3x,
\qquad
\delta Q_{\ell m}\propto\epsilon\int_0^{2\pi}
e^{i(m_p-m)\phi}\,d\phi
=2\pi\epsilon\,\delta_{m,m_p}\times
\text{radial and polar overlap}.
$$

The azimuthal rule is therefore $m_{\rm GW}=m_p$ at linear order, allowing
for the conjugate negative-$m$ sector and Fourier-sign conventions. It does
not fix $\ell$, guarantee a nonzero polar overlap, or predict the amplitude.
The quadrupole cartoon has $h_{2m}^{\rm mass}\propto\ddot Q_{2m}$; our actual
GW extraction uses $\Psi_4$, not this Newtonian source integral.

In particular, $Y_{21}\propto\sin\theta\cos\theta\,e^{i\phi}$ is odd under
$z\mapsto-z$. An exactly equatorially symmetric density has $Q_{21}=0$.
Consequently the statement “an $m=1$ density perturbation linearly produces
a mass-quadrupole $h_{21}$” needs this symmetry qualification. An $m=1$
radiative sector can still contain a current-quadrupole $(2,1)$ contribution
or a mass-octupole $(3,1)$ contribution. The familiar reflection-symmetric
mass/current classification is mass type for even $\ell+m$ and current type
for odd $\ell+m$ [R6]; without the symmetry both types may contribute.

### Nonlinear selection and frequency harmonics

Products in the fluid, stress, and gravitational-field equations obey

$$
e^{i(m_1\phi-\omega_1t)}e^{\pm i(m_2\phi-\omega_2t)}
=e^{i[(m_1\pm m_2)\phi-(\omega_1\pm\omega_2)t]}.
$$

For the difference term, the minus sign represents a conjugate factor;
its temporal frequency is conjugated as well if complex growth frequencies
are used. With real oscillation frequencies this gives the displayed sum
and difference rule. The signed azimuthal sectors are $m_1\pm m_2$;
$|m_1\pm m_2|$ is a shorthand when discussing positive mode numbers.

| Source product | Allowed azimuthal sectors | Interpretation |
| --- | --- | --- |
| $1\times1$ | $0,\pm2$ | An $m=1$ pattern can generate an $m=2$ source component at quadratic order. |
| $2\times2$ | $0,\pm4$ | An isolated $m=2$ pattern cannot generate $m=1$ through self-coupling. |
| $2\times1$ | $\pm1,\pm3$ | Mixed modes can feed an odd sector. |

An axisymmetric background plus only even $m$ retains symmetry under
$\phi\mapsto\phi+\pi$; odd sectors require an odd seed or some other breaking
of that symmetry. These are necessary selection rules, not measured coupling
coefficients. The mass-multipole integral is linear in the actual density:
the quadratic $1\times1$ route means that nonlinear evolution generates the
relevant source structure, not that the integral itself squares $\rho$.

For a common steadily rotating pattern, $\omega_m=m\Omega_p$, so a quadratic
$m=1$ harmonic and a direct $m=2$ component can both radiate near
$2\Omega_p$. Their frequency agreement alone cannot distinguish these routes.
A weak-perturbation scaling $h^{(2)}\sim\epsilon^2$ is a possible diagnostic
of a quadratic route, not a universal relation between our $C_m$ and $h_{\ell m}$.

### What the present comparison supports

`wip_plots/mode_appendix.py` compares density $m=1,2$ with GW $(2,1),(2,2)$
over a common interval starting no earlier than $t/P_c=10$. It folds the
complex FFT over positive/negative frequencies, uses a Tukey window with
$\alpha=0.10$, converts each GW $\Psi_4$ spectrum to characteristic-strain
shape by dividing by frequency, and normalizes **each curve to its own peak**.
These settings belong to the frequency comparison; the detectability
preprocessing remains Sec. 4 ($\alpha=0.05$).

The comparison supports statements about frequency/harmonic consistency.
It does not compare absolute matter/GW amplitudes or establish causal
nonlinear transfer. A specific attribution needs supporting phase relations,
growth/scaling evidence, source symmetry/multipoles, or explicit coupling
diagnostics. Claims about saturation by transport additionally need the
angular-momentum evolution. A modeled waveform tail cannot supply that evidence.

### Compact prose for the paper

```latex
We interpret the nonaxisymmetric disk evolution in terms of the global
Papaloizou--Pringle instability and compare its measured matter modes with
the extracted gravitational radiation. Matter modes label azimuthal
structure, whereas the radiation is expanded in spin-weight $-2$ spherical
harmonics with $\ell\ge2$. Linear response about an axisymmetric background
preserves azimuthal number; nonlinear products permit sum and difference
sectors. An $m=1$ pattern can therefore generate an $m=2$ source component
at quadratic order, while a pure even-$m$ source cannot generate odd sectors
without symmetry breaking. Polar overlap and equatorial symmetry further
restrict the radiative multipoles. Agreement of spectral peaks is consistent
with a common rotating pattern, but does not by itself establish the coupling
mechanism or its contribution to the waveform amplitude.
```

## Paper Checklist and References

State the extraction sphere, mass convention, retained modes/time interval,
FFI cutoff for time-series plots, FFT window/floor for spectra, angular
statistic, detector response ($S_{\rm eff}=\tfrac52S_n$, and how it differs from
[R3] footnote 4) and curve versions, cosmology and SNR threshold.
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
  Our detector normalization is derived from the antenna average; see Sec. 6.
- **[R4] Ruiz, Alcubierre, Nunez & Takahashi (2008)**, *Multipole expansions for
  energy and momenta carried by gravitational waves*, GRG 40, 1705:
  [arXiv:0707.4654](https://arxiv.org/abs/0707.4654). Cite for $E$, $J_z$ fluxes.
- **[R5] Robson, Cornish & Liu (2019)**, *The construction and use of LISA
  sensitivity curves*, CQG 36, 105011:
  [arXiv:1803.01944](https://arxiv.org/abs/1803.01944). The SciRDv1 table equals
  twice their two-channel curve; cite for the LISA channel/averaging factors.

- **[R6] Marsat, Faye, Blanchet & Iyer (2012)**, *Post-Newtonian Prediction
  for the (2,2) Mode of the Gravitational Wave Emitted by Compact Binaries*,
  ASP Conference Series 467, 197, Sec. 3, Eqs. (8), (11):
  [author-hosted proceedings PDF](https://apc.u-paris.fr/~beckmann/Proceedings/Marsat.pdf).
  This gives the spin-weight $-2$ decomposition and the planar-binary
  mass/current selection rule. The disk $Q_{21}=0$ statement above follows
  directly from density reflection symmetry and the parity of $Y_{21}$;
  binary dynamics are not assumed for the disk.
- **[R7] Bugli et al. (2018)**, *Papaloizou-Pringle instability suppression
  by the magnetorotational instability in relativistic accretion discs*,
  MNRAS 475, 108, Sec. 1 and Fig. 1:
  [journal article](https://academic.oup.com/mnras/article/475/1/108/4718469).
  Cite for the corotation/wave-interaction cartoon. Their magnetic suppression
  results do not establish that mechanism in our runs.

Implementation pointers: [gw.py](../gw.py) (`reconstruct_strain`),
[gw_detectability.py](../gw_detectability.py) (numbered analysis sections),
[gw_strain_observer.py](../wip_plots/gw_strain_observer.py) (face-on time series),
[gw_radiated.py](../wip_plots/gw_radiated.py) (radiated E, J and frequency).
The combined figure's formatting is in section 4 of
[gw_detectability.py](../gw_detectability.py), alongside its calculation.
[gw_detectability_all.py](../wip_plots/gw_detectability_all.py) draws additional
diagnostics using that implementation. The separate coherent-mode comparison
is entirely in [gw_mismatch.py](../gw_mismatch.py), with its definition and
inputs in [the mismatch note](gw_mismatch.md).
