# From the simulation's GW diagnostic to strain and detectability

**A reading and writing guide, checked against the local code on 2026-10-07.**

This note explains what quantity we start with, what each transformation means,
and where the factors of two, five, and solid angle enter. It describes the
current implementation, including choices that remain scientific assumptions.
It is intended to be read before turning the shorter
[paper methods](gw_paper_methods.md) into LaTeX. The
[GW and PPI summary](human_gw_method.md) supplies the physical interpretation.

There are two levels here. Sections 1–2 give the map and notation. Sections
3–13 follow the calculation and derive the normalization. Sections 14–17
cover related diagnostics, limitations, suggested paper wording, and code
pointers. If the immediate question is “why $1/5$ or $5/2$?”, read
sections 8–10 first, then return to the beginning.

The postprocessing and its input-file contract were inspected. The exact
simulation-side curvature/tetrad and sphere-quadrature implementation is not
included in this plotting checkout; section 3 distinguishes its standard
mathematical interpretation from what we verified in the reader. This note
does not claim that those upstream details, numerical-relativity convergence,
or physical distance predictions have all been independently validated.

## 1. The whole calculation in words

The simulation evolves a spacetime and a disk. In the wave zone, a diagnostic
measures the outgoing curvature scalar $\Psi_4$ and decomposes its angular
dependence into spin-weighted spherical-harmonic coefficients. The diagnostic
file therefore contains **complex curvature modes as functions of simulation
time**, together with extraction-radius and metric information. It does not
contain the strain at Earth or a detector's output.

We multiply each curvature mode by the extraction sphere's areal radius and
put it on a corrected retarded-time grid. This gives a reusable estimate of
the outgoing radiation, with the leading $1/r$ falloff removed. From there
we take two paths:

1. **Strain histories:** integrate twice using fixed-frequency integration
   (FFI), reconstruct a chosen viewing direction, and plot plus/cross strain.
2. **Measured detectability:** window the measured curvature modes, Fourier
   transform them, reconstruct the two polarizations, form a source-direction
   amplitude statistic, and divide by frequency squared. This path reads the
   curvature cache; it does not transform the FFI strain cache.

For detectability we then choose a physical initial BH mass, luminosity
distance, and corresponding redshift. These choices turn dimensionless
frequency into observed Hz and the spectral amplitude into physical strain
per Hz. We form characteristic strain $h_c$, compare it with characteristic
noise $h_n$, and integrate their squared ratio over logarithmic frequency.
Finally we can vary distance and solve for a chosen SNR threshold.

The main chain is

$$
\Psi_{4,\ell m}(t,r_A)
\longrightarrow P_{\ell m}=r_A\Psi_{4,\ell m}
\longrightarrow p_{\ell m}(\tau)=m_{\rm BH}P_{\ell m}
\longrightarrow \widetilde p_{\ell m}(\nu)
\longrightarrow \mathcal Q(\nu)
\longrightarrow |\widetilde h_{\rm stat}(f)|
\longrightarrow h_c(f)
\longrightarrow \rho_{\rm stat},\ D_{L,\rm threshold}.
$$

The label “stat” is deliberate: our default is a **mean spectral amplitude
over source viewing directions**. Its SNR-like norm is not the mean SNR of a
source population. We will show exactly where that distinction enters.

## 2. Notation: keep the masses, angles, and spectra separate

| Symbol | Meaning |
| --- | --- |
| $m_{\rm BH}$ | Initial BH mass in raw code units; `case.mlittle`, currently 0.05 |
| $m_{\rm ADM}$ | Total ADM mass in code units; `case.gw_madm` |
| $M_{\rm BH}$ | Chosen physical, source-frame initial BH mass |
| $M_z=(1+z)M_{\rm BH}$ | Redshifted BH mass scale |
| $r_A$ | Recorded areal extraction radius in code units |
| $u=t_{\rm ret}$ | Corrected retarded time in code units |
| $\Omega_s=(\theta_s,\phi_s)$ | Viewing direction in coordinates attached to the source |
| $\Omega_d=(\theta_d,\phi_d)$, $\psi$ | Sky location in the detector frame and polarization-basis angle |
| $\tau=u/m_{\rm BH}$ | Time measured in initial-BH mass units |
| $\nu$ | Fourier frequency conjugate to $\tau$; dimensionless |
| $f$ | Observed frequency in Hz |
| $H_{\ell m}=r_Ah_{\ell m}$ | Radius-scaled strain mode in code units |
| $q_{\ell m}=H_{\ell m}/m_{\rm BH}$ | Dimensionless strain mode |
| $p_{\ell m}=m_{\rm BH}r_A\Psi_{4,\ell m}$ | Dimensionless curvature mode, $q_{\ell m}''(\tau)$ |
| $\mathcal Q(\nu)$ | Nonnegative dimensionless strain-spectrum statistic; `spectrum.strain_ft` |
| $S_{\rm inst}$ | One-sided instrument-noise PSD before our response conversion |
| $S_{\rm eff}$ | Noise PSD paired with our chosen source-amplitude definition |

Here $\mathcal Q$ is a spectral statistic. It is unrelated to the source
mass-quadrupole tensor $Q_{ij}$ used later in the physical explanation.

The raw simulation uses $G=c=M_\odot=1$. A raw code-time unit is therefore

$$
t_\odot=\frac{GM_\odot}{c^3}=4.92549095\times10^{-6}\ {\rm s},
\qquad
\ell_\odot=\frac{GM_\odot}{c^2}=1476.6250385\ {\rm m}.
$$

This is the base unit convention. Astrophysically rescaling the model to a
different BH mass is a further operation: a simulated BH of code mass 0.05
must represent the chosen $M_{\rm BH}$. It is wrong to treat raw time as
already normalized by a BH of mass one, or to substitute the ADM mass without
changing the definitions consistently.

Our Fourier convention throughout is

$$
\widetilde x(f)=\int x(t)e^{-2\pi i f t}\,dt,
\qquad
\mathcal F[\ddot x]=-(2\pi f)^2\widetilde x,
$$

when boundary terms vanish or the transform is interpreted on a suitable
complete signal. A Fourier transform of a dimensionless physical strain has
units of seconds, equivalently $\mathrm{Hz}^{-1}$.

## 3. What is extracted, and what integral does it represent?

### 3.1 Curvature first, strain afterward

$\Psi_4$ is a Newman–Penrose component of the Weyl curvature evaluated
using a null tetrad. In an appropriate outgoing wave-zone convention its
radiative part is related to the strain by

$$
h=h_+-ih_\times,
\qquad
\Psi_4=\ddot h_+-i\ddot h_\times=\ddot h.
$$

This is the convention assumed by our postprocessing. The exact overall sign
and complex conjugation are convention dependent; a tetrad normalization is
also part of the upstream contract. They must be checked before combining
waveforms from different extractors. A global sign alone does not change
the amplitude norm or SNR, but inconsistent conjugation changes handedness
and mode/observer reconstruction. At finite radius this is a wave-zone
approximation, not a guarantee of a gauge-independent waveform at infinity.
[Reisswig and Pollney discuss curvature-to-strain integration and these
finite-signal issues](https://arxiv.org/html/1006.1632v3).

The physical source is the evolving spacetime of the whole BH–disk system.
Our analysis does not obtain the GR waveform by plugging a density mode
into a Newtonian quadrupole formula.

### 3.2 The spherical-harmonic projection is an integral, not an average

With harmonics normalized by

$$
\int {}_{-2}Y_{\ell m}(\Omega_s)
          {}_{-2}Y^*_{\ell' m'}(\Omega_s)\,d\Omega_s
=\delta_{\ell\ell'}\delta_{mm'},
\qquad d\Omega_s=\sin\theta_s\,d\theta_s\,d\phi_s,
$$

the standard coefficient definition is

$$
\Psi_{4,\ell m}(t,r_A)
=\int \Psi_4(t,r_A,\Omega_s)
          {}_{-2}Y^*_{\ell m}(\Omega_s)\,d\Omega_s.
$$

The inverse is a coherent sum,

$$
\Psi_4(t,r_A,\Omega_s)
=\sum_{\ell,m}\Psi_{4,\ell m}(t,r_A)
                     {}_{-2}Y_{\ell m}(\Omega_s).
$$

**There is no additional $1/(4\pi)$ in this coefficient definition.**
That factor belongs to a normalized average over directions; putting it into
an orthonormal harmonic projection would change the coefficients. Nor is
there a detector $1/5$ here. Projecting onto a radiation mode and averaging
an interferometer's response are different operations.

The coefficient formula follows by multiplying the inverse expansion by a
conjugate harmonic and integrating: orthonormality selects one coefficient.
The measure here is the angular measure of the harmonic basis. On a round
sphere, $dA=r_A^2d\Omega_s$, so a surface-area integral requires the
corresponding $r_A^{-2}$ conversion. At finite radius in a distorted metric,
coordinate-angle projection, proper-area weighting, and the choice of tetrad
must be established from the actual extractor; they are not interchangeable
by assumption. The displayed projection is the standard meaning of the
coefficients used downstream, not a new audit of the simulation diagnostic.

### 3.3 What our files and reader actually contain

`Psi4_rad.mon.N` has 47 columns:

- coordinate time;
- real/imaginary pairs for 21 modes, ordered as
  $\ell=2,3,4$, with $m=\ell,\ell-1,\ldots,-\ell$ within each $\ell$;
- areal radius and the three inverse-metric entries used for the time correction.

The radius is recorded separately. The reader treats the mode columns as
$\Psi_{4,\ell m}$, then multiplies by that radius **once**:

$$
P_{\ell m}(u)=r_A\Psi_{4,\ell m}(u).
$$

At leading radiative order $\Psi_4\propto1/r$, so $r_A\Psi_4$ is the
quantity whose leading amplitude should agree between wave-zone spheres.
Multiplying by $r_A$ does not itself extrapolate to infinity. It also does
not give the strain at a physical distance $D_L$.

For A/B cases, parfile index 8 maps to `Psi4_rad.mon.9`; the ML case uses its
configured different file offset. Use the mapping in `config.py` and the
cache manifest, not the apparent filename as a physical radius. The current
detectability baseline uses index 8, the outer common finite sphere, and
checks inner/intermediate spheres.

The baseline uses the case's original curvature cache. Separately generated
case-minus-ML difference products are comparison diagnostics; they are not
silently substituted for the measured detectability signal.

## 4. Why retarded time and restart handling come before a spectrum

A wave recorded at a larger extraction radius arrives later. Coordinate time
also depends on the simulation gauge. Comparing raw coordinate-time samples
between spheres can confuse propagation delay with source behavior.

The inherited prescription constructs a Schwarzschild-like corrected time
and subtracts a tortoise radius:

$$
\begin{aligned}
\frac{dt_{\rm S}}{dt}
&=\frac{g^{tr}-\sqrt{(g^{tr})^2-g^{tt}g^{rr}}}
        {g^{tt}(1-2m_{\rm ADM}/r_A)},\\
r_*&=r_A+2m_{\rm ADM}\ln\left(\frac{r_A}{2m_{\rm ADM}}-1\right),\\
u&=t_{\rm S}-r_*.
\end{aligned}
$$

The three $g^{ab}$ values are the recorded inverse-metric inputs; the
reference-file description identifies them as sphere averages. The Python
port integrates $t_{\rm S}$ by trapezoidal accumulation, starting with
$t_{\rm S}(t_0)=t_0$, and follows the reference's uniform coordinate-time
sampling assumption. As a check on interpretation, for a static Schwarzschild
metric this expression gives $dt_{\rm S}/dt=1$. It is a prescribed gauge
correction, not characteristic extraction or a proof of asymptotic accuracy.

Using $m_{\rm ADM}$ here approximates the exterior geometry; it does not
make $m_{\rm ADM}$ the mass used to rescale the source astrophysically.

Later restart segments replace earlier data from their restart time onward.
Repeated times within a file retain the last row. The generator multiplies
by $r_A$ and interpolates modes onto one uniform $u$ grid with the
reference's local four-point polynomial interpolation. Detectability reads
this matching time/mode cache and, after cuts, uses linear interpolation
onto its common median-spacing grid. Assigning a raw waveform to a different
cached time array would violate this construction.

Zero-filled cache rows beyond the last measured curvature sample are removed
before detectability. They are bookkeeping, not evidence for physical signal
termination or a longer simulation.

## 5. Twice integrating curvature: two paths with different purposes

### 5.1 The continuum identity

Define $H=r_Ah$. At leading radiative order we use $P=\ddot H$, with
derivatives taken in retarded time. For nonzero frequency,

$$
\widetilde H(f)=-\frac{\widetilde P(f)}{(2\pi f)^2}.
$$

The minus sign comes from differentiating $e^{2\pi ift}$ twice. It drops
out of magnitudes, but belongs in a complex waveform reconstruction.
The denominator makes very small-frequency errors dangerous. It also shows
why DC cannot be recovered by ordinary division. Curvature alone does not
determine arbitrary integration constants or a trustworthy zero-frequency
strain/memory component on a finite record.

### 5.2 FFI for time-series figures

For the strain histories, the reference-compatible algorithm replaces the
small denominator by a mode-dependent floor:

$$
\widetilde H_{\ell m}^{\rm FFI}(\omega)
=-\frac{\widetilde P_{\ell m}(\omega)}
        {\max(|\omega|,\omega_{0m})^2},
\qquad
\omega_{0m}=\max(|m|,1)\Omega_{\rm orb,0}.
$$

An inverse transform gives the time history. This regularizes low-frequency
integration drift; it does not simply delete every frequency below the floor.
For $(2,2)$, the floor is twice the configured initial orbital angular
frequency. This particular rule is inherited from our reference routine.
The general FFI motivation and prescription are described by
[Reisswig and Pollney, section 4.2](https://arxiv.org/html/1006.1632v3).

The FFI path zero-pads its chosen input interval and applies no Tukey taper.
Its output need not vanish at the last displayed time. Because the legacy
routine integrates a conjugated representation, `rhphc.dat` stores pairs
interpreted as $r_Ah_+$ and $r_Ah_\times$. Reconstruct the natural-sign
complex mode as

$$
H_{\ell m}=\text{column}_{+,\ell m}
                    -i\,\text{column}_{\times,\ell m}.
$$

The separate `rpsi4_uniform.dat` cache stores natural-sign real/imaginary
curvature modes. These two file conventions must not be mixed.

### 5.3 Direct curvature spectra for detectability

For the measured detectability baseline we Fourier-transform the retained
curvature itself and divide by $(2\pi\nu)^2$. This avoids importing the
FFI orbital-frequency floor into the detectability spectrum. It still needs
a finite-duration frequency floor and a taper. “Direct” therefore does not
mean free of low-frequency or boundary assumptions.

There is a useful exact warning. For a taper $W$,

$$
(Wq)''=Wq''+2W'q'+W''q.
$$

Consequently, with suitable vanishing boundary terms,

$$
-\frac{\mathcal F[Wq'']}{(2\pi\nu)^2}
=\mathcal F[Wq]
 +\frac{\mathcal F[2W'q'+W''q]}{(2\pi\nu)^2}.
$$

Windowing curvature and dividing by frequency squared is generally **not
exactly** the Fourier transform of the windowed strain. The extra terms are
associated with the taper, and frequency-squared division can amplify them
at low frequency. Our estimator is defined by the direct-curvature procedure;
window/floor sensitivity tests are part of judging its reliability.

## 6. The dimensionless direct-curvature spectrum, step by step

### 6.1 Remove the simulated mass scale correctly

Set

$$
\tau=\frac{u}{m_{\rm BH}},\qquad
q_{\ell m}=\frac{H_{\ell m}}{m_{\rm BH}},\qquad
p_{\ell m}=m_{\rm BH}P_{\ell m}.
$$

These definitions are linked, not arbitrary:

$$
\frac{d^2q_{\ell m}}{d\tau^2}
=m_{\rm BH}\frac{d^2H_{\ell m}}{du^2}
=m_{\rm BH}P_{\ell m}=p_{\ell m}.
$$

The reusable spectrum is therefore a transform in dimensionless time, with
dimensionless frequency $\nu$. This is why multiplying $P$ by the code
BH mass, rather than dividing it, is correct.

### 6.2 Retain the measured interval

The current settings remove $u<0$, then discard $1000m_{\rm BH}$ after
the first nonnegative sample. This is a transient-relaxation choice motivated
by comparable disk studies, not a theorem that all initial junk ends there.
The retained interval ends at measured curvature data. Its duration is

$$
\Delta\tau=\tau_{\rm last}-\tau_{\rm first}.
$$

The code shifts the retained time origin to zero. A common time translation
changes Fourier phases, not the polarization-amplitude norm.

### 6.3 Taper and transform

Apply the same symmetric Tukey window to every retained mode. The baseline
has $\alpha=0.05$, meaning 2.5% of the interval is tapered at each end.
It reduces spectral leakage caused by ending a finite record abruptly, while
also downweighting real radiation near the boundaries. No coherent-gain or
RMS-window renormalization is applied: the analyzed signal is the tapered
record as defined.

For samples spaced by $\delta\tau$, the continuous-transform approximation is

$$
\widetilde p_{\ell m}(\nu_k)
\simeq\delta\tau\sum_{j=0}^{N-1}
 W_jp_{\ell m}(\tau_j)e^{-2\pi i\nu_k\tau_j}.
$$

**The FFT must be multiplied by the sample spacing.** Dividing by the sample
count instead would produce a different normalization and an incorrect
physical spectrum when used in the subsequent SNR formula.

Twofold padding produces a denser frequency grid,
$\delta\nu_{\rm grid}=1/(N_{\rm FFT}\delta\tau)$. The physical resolving
scale is still about $1/\Delta\tau$. Padding does not add radiation,
cycles, or observing time.

The retained band is

$$
\nu\ge\frac{3}{\Delta\tau},\qquad
0<\nu<\nu_{\rm Nyquist}=\frac1{2\delta\tau}.
$$

The three-cycle floor rejects the least resolved part of the finite spectrum.
It is a provisional analysis choice. It is distinct from the FFI orbital
floor. There is no additional detrending or amplitude-based frequency cut in
the baseline SNR integration.

### 6.4 Reconstruct directions before taking amplitudes

For each source direction,

$$
p(\tau,\Omega_s)=\sum_{(\ell,m)\in\mathcal S}
 p_{\ell m}(\tau){}_{-2}Y_{\ell m}(\Omega_s).
$$

Sum the **complex coefficients** before taking a magnitude. Their relative
phases determine polarization and interference. Summing mode magnitudes first
would construct a different signal.

The baseline uses
$\mathcal S=\{(2,2),(2,1),(2,-2),(2,-1)\}$ for A1–A3. Negative-$m$
coefficients are read independently. `MODES="all"` selects the 21 cached modes
with $\ell=2,3,4$. Omitting $m=0$ defines a nonaxisymmetric quadrupole
subset; it does not establish that excluded radiation is numerical or that
the result is total emission.

Since the same window and grid are used for every mode, angular reconstruction
commutes with the FFT. The code can transform each mode once and then sum
its Fourier coefficients for many directions.

### 6.5 Why a complex mode needs both frequency signs

The real polarizations each have conjugate-symmetric transforms. A complex
mode generally does not. For the reconstructed complex field, write

$$
Z_+(\nu,\Omega_s)=\mathcal F[Wp](+\nu),\qquad
Z_-(\nu,\Omega_s)=\mathcal F[Wp](-\nu).
$$

Then

$$
\begin{aligned}
X&=\mathcal F[W\Re p](\nu)=\frac{Z_++Z_-^*}{2},\\
Y&=\mathcal F[W\Im p](\nu)=\frac{Z_+-Z_-^*}{2i}.
\end{aligned}
$$

The physical cross component carries the minus sign from $h=h_+-ih_\times$.
That sign disappears from the norm we use:

$$
A(\nu,\Omega_s)
=\sqrt{\frac{|X|^2+|Y|^2}{2}},\qquad
A^2=\frac{|Z_+|^2+|Z_-|^2}{4}.
$$

Taking only the positive-frequency magnitude of the complex field would
generally give the wrong two-polarization amplitude. There is no extra
arbitrary “double the positive bins” operation: the real-component
reconstruction and the later one-sided SNR normalization already specify
the factors.

## 7. Source-direction averaging: mean amplitude is a specific choice

For a fixed direction, define the dimensionless strain amplitude estimate

$$
Q_{\Omega_s}(\nu)=\frac{A(\nu,\Omega_s)}{(2\pi\nu)^2}.
$$

The numerator uses the same two-polarization norm as Wessel's Eq. (8), now
applied to curvature before the frequency-squared conversion.
[Wessel et al., section III.3](https://arxiv.org/html/2011.04077).

Our default then takes the normalized solid-angle **mean amplitude**:

$$
\mathcal Q_{\rm mean}(\nu)
=\frac{1}{4\pi}\int Q_{\Omega_s}(\nu)\,d\Omega_s.
$$

An available alternative is the RMS amplitude:

$$
\mathcal Q_{\rm RMS}(\nu)
=\sqrt{\frac{1}{4\pi}\int Q_{\Omega_s}(\nu)^2\,d\Omega_s}.
$$

The $1/(4\pi)$ makes these averages: a direction-independent amplitude
must remain unchanged. The code uses 24 Gauss–Legendre nodes in
$\mu_s=\cos\theta_s$, 48 periodic azimuthal samples without a duplicate
$2\pi$ endpoint, and weights summing to one.

These choices answer different questions. The mean describes the arithmetic
mean of a nonnegative spectral amplitude at each frequency. The RMS describes
mean spectral power. Neither is the coherent mean complex field, and neither
is a single observer's waveform. A frequency-by-frequency average can combine
contributions whose strongest directions differ across the band.

To see the SNR implication, let $C(f)\ge0$ include the common physical
scaling and detector-noise weighting. Then

$$
\rho_{\rm mean\ spectrum}^2
=\int C(f)\langle Q_{\Omega_s}\rangle_s^2\,df,
\qquad
\langle\rho_{\Omega_s}^2\rangle_s
=\int C(f)\langle Q_{\Omega_s}^2\rangle_s\,df.
$$

The RMS spectrum reproduces the second expression when the detector average
and band are held fixed. The default mean spectrum reproduces the first.
Jensen's inequality gives

$$
\rho_{\rm mean\ spectrum}^2\le\langle\rho_{\Omega_s}^2\rangle_s.
$$

There is no universal constant that turns our mean-spectrum SNR into the
population mean SNR or an averaged detection volume. In particular, a
population volume involves thresholding and distance cubed, not just one
amplitude average.

For comparison, harmonic orthonormality gives the time-domain identity

$$
\langle|H(u,\Omega_s)|^2\rangle_s
=\frac1{4\pi}\sum_{\ell m}|H_{\ell m}(u)|^2.
$$

It does not give $\langle|H|\rangle_s$. For our *spectral polarization*
RMS, the corresponding identity instead is

$$
\langle A(\nu,\Omega_s)^2\rangle_s
=\frac1{16\pi}\sum_{\ell m}
\left(|\widetilde p_{\ell m}(+\nu)|^2
     +|\widetilde p_{\ell m}(-\nu)|^2\right).
$$

This explicitly includes both frequency signs and the polarization convention.
Using a time-domain mode-power identity in place of the mean spectral
amplitude would change the statistic.

The optional direction $\theta_s=\pi/2.34,\phi_s=0$ follows a comparison
in Wessel. It is not an exact replacement for averaging a multimode signal.
The separate face-on strain plot uses $\theta_s=0$; there,

$$
{}_{-2}Y_{\ell m}(0,0)=\sqrt{\frac{2\ell+1}{4\pi}}\,\delta_{m2}.
$$

Thus only $(2,2)$ contributes at the north pole for the four-mode subset.
No extra factor of two for an inferred negative-$m$ partner is needed.

## 8. Where the detector $1/5$ comes from

### 8.1 A detector measures a projected strain

For an ideal right-angle Michelson in the long-wavelength approximation,
the detector strain is

$$
h_d(t)=F_+(\Omega_d,\psi)h_+(t)
       +F_\times(\Omega_d,\psi)h_\times(t).
$$

The $F$'s describe detector geometry. They do not describe disk inclination
or the harmonic projection used to extract a radiation mode. Use detector
angles $\Omega_d,\psi$ and source viewing angles $\Omega_s$ as distinct
variables even if both are casually called “sky averaging.”

With $\mu=\cos\theta_d$, write

$$
\begin{aligned}
a&=\tfrac12(1+\mu^2)\cos2\phi_d,
&b&=\mu\sin2\phi_d,\\
F_+&=a\cos2\psi-b\sin2\psi,
&F_\times&=a\sin2\psi+b\cos2\psi.
\end{aligned}
$$

These are the antenna patterns used in our independent response-quadrature
test. [Robson, Cornish and Liu, Eqs. (4)–(7)](https://arxiv.org/html/1803.01944v1)
state the per-polarization response convention explicitly.

### 8.2 Perform the average rather than memorizing the factor

Average uniformly in detector sky solid angle and polarization angle:

$$
\langle B\rangle_{d,\psi}
=\frac{1}{4\pi^2}\int_{-1}^{1}d\mu
       \int_0^{2\pi}d\phi_d\int_0^\pi d\psi\ B.
$$

First $\langle\cos^22\psi\rangle=\langle\sin^22\psi\rangle=1/2$, while
the mixed sine/cosine average vanishes. Thus either squared antenna pattern
has polarization average $(a^2+b^2)/2$. Averaging in $\phi_d$ gives

$$
\langle F_+^2\rangle_{\phi_d,\psi}
=\langle F_\times^2\rangle_{\phi_d,\psi}
=\frac{1+6\mu^2+\mu^4}{16}.
$$

Finally average $\mu$ uniformly over $[-1,1]$:

$$
\begin{aligned}
\langle F_+^2\rangle_{d,\psi}
&=\frac1{32}\int_{-1}^{1}(1+6\mu^2+\mu^4)\,d\mu\\
&=\frac1{32}\left(2+4+\frac25\right)=\frac15.
\end{aligned}
$$

The same result holds for cross, and the full averaged mixed response is zero:

$$
\boxed{\langle F_+^2\rangle=\langle F_\times^2\rangle=\frac15,
\qquad \langle F_+F_\times\rangle=0.}
$$

This is a **power** average. The corresponding single-polarization RMS
amplitude response is $1/\sqrt5$, not $1/5$. Also,
$\langle F_+^2+F_\times^2\rangle=2/5$; averaging the sum of two terms
does not give the one-polarization result.

For arbitrary complex polarization spectra, including elliptic polarization,
expand the detector power and apply these averages:

$$
\begin{aligned}
\langle|\widetilde h_d|^2\rangle_{d,\psi}
&=\langle F_+^2\rangle|\widetilde h_+|^2
 +\langle F_\times^2\rangle|\widetilde h_\times|^2\\
&\quad+2\langle F_+F_\times\rangle
          \Re(\widetilde h_+\widetilde h_\times^*)\\
&=\frac{|\widetilde h_+|^2+|\widetilde h_\times|^2}{5}.
\end{aligned}
$$

This does not require the two polarizations to have equal amplitudes or to
be statistically independent. The geometric mixed average removes the cross
term. The approximation does assume the scalar, stationary detector-response
model used here; a detailed moving-detector response is a further analysis.

## 9. Why our polarization definition gives $5/2$, not $5$

At a fixed source direction, our amplitude convention is

$$
H_{\rm pol}(f)=\sqrt{\frac{|\widetilde h_+(f)|^2
                              +|\widetilde h_\times(f)|^2}{2}}.
$$

This is a spectral norm, not a time-domain strain and not generally the
complex Fourier transform of an actual detector waveform. Wessel calls it
$\widetilde h^{\rm res}$. The factor $1/2$ defines the arithmetic mean
of the two squared polarization amplitudes. It is **not** the antenna
average and is **not** the one-sided/two-sided PSD conversion.

Substituting this definition into the previous result gives

$$
\boxed{\langle|\widetilde h_d|^2\rangle_{d,\psi}
=\frac25 H_{\rm pol}^2.}
$$

With raw instrument PSD $S_{\rm inst}$, the averaged squared SNR is

$$
\langle\rho_d^2\rangle_{d,\psi}
=4\int\frac{(2/5)H_{\rm pol}^2}{S_{\rm inst}}\,df
=4\int\frac{H_{\rm pol}^2}{S_{\rm eff}}\,df,
\qquad
\boxed{S_{\rm eff}=\frac52S_{\rm inst}.}
$$

An ASD is the square root of a PSD, so

$$
\sqrt{S_{\rm eff}}=\sqrt{\frac52}\sqrt{S_{\rm inst}}.
$$

Equivalently, leave the noise raw and replace the source amplitude by
$\sqrt{2/5}\,H_{\rm pol}$. These two placements preserve the ratio
and hence the SNR. Apply one placement consistently. Applying both counts
the same response reduction twice.

### A small example with all numbers visible

Suppose at one frequency $\widetilde h_+=\widetilde h_\times=1$ and
$S_{\rm inst}=1$, in consistent arbitrary units. Then

$$
H_{\rm pol}^2=1,\qquad
\langle|\widetilde h_d|^2\rangle=2/5,
\qquad
\frac{H_{\rm pol}^2}{S_{\rm eff}}=\frac1{5/2}=2/5.
$$

If instead the source were defined as the root-sum-square amplitude,
$H_{\rm RSS}=\sqrt{|\widetilde h_+|^2+|\widetilde h_\times|^2}=\sqrt2$,
the matching effective PSD would be $5S_{\rm inst}$, giving $2/5$ again.
The factor changes because the named source amplitude changes; the physical
detector power does not.

| Source quantity used in $4\int H^2/S\,df$ | Matching noise PSD for one right-angle detector |
| --- | --- |
| Actual $\widetilde h_d=F_+\widetilde h_++F_\times\widetilde h_\times$ | Raw $S_{\rm inst}$; no additional averaged antenna factor |
| $H_{\rm RSS}$ | $5S_{\rm inst}$ for detector-angle-averaged squared SNR |
| $H_{\rm pol}=H_{\rm RSS}/\sqrt2$, our convention | $(5/2)S_{\rm inst}$ |
| $\sqrt{2/5}\,H_{\rm pol}$ | Raw $S_{\rm inst}$ |

Replacing a fixed-direction $H_{\rm pol}$ by our mean source spectrum
retains this detector convention, but produces the mean-spectrum statistic
described in section 7. It does not magically become the full source-and-
detector population average.

### Why the literature looked inconsistent

Moore et al. quote $1/\sqrt5$ as the right-angle averaged amplitude
response, whereas their printed Eq. (51) integrates the **sum** of the two
squared antenna patterns. That displayed integral evaluates to $2/5$,
not $1/5$. The per-polarization definition in Robson et al. removes this
ambiguity. Use the derived response above rather than identifying a factor
by the phrase “sky averaged” alone.
[Moore et al., section 4.1](https://arxiv.org/html/1408.0740v2#S4.SS1).

Wessel's footnote 4 describes an additional $\sqrt2$ multiplication of
an already averaged sensitivity. If the starting ASD meant
$\sqrt5\sqrt{S_{\rm inst}}$, that operation would give
$\sqrt{10}\sqrt{S_{\rm inst}}$. Paired with our $H_{\rm pol}$, it is
twice our effective ASD and would halve the SNR for the same input data.
This is a **conditional convention comparison**, not a reproduction of
Wessel's published horizons or a claim that all their numerical results are
wrong. Establishing the exact original input curves and numerical pipeline
would be necessary for that stronger conclusion.
[Wessel et al., section III.5 and footnote 4](https://arxiv.org/html/2011.04077).

## 10. Detector-specific curves: averaging may already be inside the file

The response calculation above starts with raw right-angle instrument noise.
A supplied sensitivity table can already include an angular response, a
transfer function, or a network/channel combination. Read its definition
before multiplying it again. The implemented inputs are:

| Detector label | Input | Effective ASD paired with our source statistic |
| --- | --- | --- |
| LIGO | `AplusDesign.txt`, strain ASD | $\sqrt{5/2}$ times the input ASD |
| CE | `CE2_40km_strain.txt`, strain ASD | $\sqrt{5/2}$ times the input ASD |
| ET | `ET10kmcolumns.txt`, combined LF+HF PSD in numeric column 4 | $\sqrt{5/2}/(3/2)$ times the square root of that PSD |
| DECIGO | Analytic one-effective-interferometer PSD in `_decigo_instrument_asd` | $\sqrt{5/2}$ times its square root |
| LISA | `LISA_Alloc_Sh.txt`, already response-averaged strain-equivalent PSD | $1/2$ times its square root |

Taking the square root converts PSD to ASD once. Applying a multiplicative
ASD correction $k$ changes the PSD by $k^2$. The detector loader
interpolates the ASD in log frequency/log amplitude. The same converted
curve is used for plotting $h_n$ and evaluating SNR.

The LIGO input is the original A+ design target, not measured O5 noise. The
CE filename is historical: its numeric content is the baseline 40 km curve
documented in CE-T2000017-v9. That input uses a source 15 degrees off normal
incidence; treating it with our simple right-angle angular factor is an
approximation, not an exact normalization of that reference direction.
[LIGO A+ source](https://dcc.ligo.org/LIGO-T1800042/public),
[CE source](https://dcc.cosmicexplorer.org/cgi-bin/DocDB/ShowDocument?.submit=Identifier&docid=T2000017).

For ET, the input is the CoBA 10 km L-shaped-equivalent curve, not an entire
triangle already combined. The assumed three independent equal-noise
$60^\circ$ Michelsons give a network amplitude gain

$$
\sqrt{3}\sin60^\circ=\frac32,
$$

so the effective ASD is reduced by $3/2$ after the right-angle conversion.
This combines three detector squared SNRs, not an average of three SNRs.
It assumes the stated independent-noise model; correlated noise or another
ET configuration requires its own treatment.
[ET CoBA input documentation](https://apps.et-gw.eu/tds/?r=18213).

For LISA, the LDC `AnalyticNoise.sensitivity()` formula already contains
an antenna factor $\sqrt5$, a $60^\circ$ projection factor
$2/\sqrt3$, and an approximate transfer function. Its low-frequency
PSD normalization is $20/3$ times the displacement-noise combination
divided by arm length squared. In the retained interpretation this is a
single-channel, per-polarization-equivalent sensitivity. Two independent
effective channels reduce its PSD by two; expressing the source as
$H_{\rm RSS}/\sqrt2$ reduces the matching PSD by a further two. Hence

$$
S_{\rm eff,LISA}=\frac14S_{\rm table},
\qquad \sqrt{S_{\rm eff,LISA}}=\frac12\sqrt{S_{\rm table}}.
$$

This is the model encoded in our loader, not a rule for every LISA curve.
The unit test checks the low-frequency single-channel normalization against
twice the Robson two-channel expression. The original table-generation
command and LDC revision were not recorded. A full frequency-dependent
multi-channel response, mission-duration effects, and an independently
specified confusion foreground would require a more detailed model.
[LDC sensitivity implementation](https://lisa.pages.in2p3.fr/LDC/_modules/ldc/lisa/noise/noise.html),
[Robson et al.](https://arxiv.org/html/1803.01944v1).

The retained detector bands in observed Hz are LIGO 5–2500, ET 1–10000,
CE 5–5000, DECIGO 0.01–10, and LISA $10^{-4}$–1. These are integration
settings. They do not represent source-frame frequency cuts or measured
network performance. Complete provenance is in
[the detector-curve README](../detector_curves/README.md).

## 11. Put the source at a physical mass, distance, and redshift

### 11.1 Time, frequency, and amplitude scale differently

Choose a source-frame initial BH mass $M_{\rm BH}$, redshift $z$, and
luminosity distance $D_L$. Define

$$
M_z=(1+z)M_{\rm BH},\qquad
T_z=\frac{GM_z}{c^3},\qquad
L_z=\frac{GM_z}{c^2}.
$$

The observer-frame scaling is

$$
t_{\rm obs}=T_z\tau,\qquad
f=\frac\nu{T_z},\qquad
h_{+,\times}^{\rm obs}(t_{\rm obs})
=\frac{L_z}{D_L}\,q_{+,\times}(\tau).
$$

One way to understand the amplitude redshift is to start with source-frame
length $GM_{\rm BH}/c^2$ divided by the present-day propagation distance
$D_L/(1+z)$. This yields $GM_z/(c^2D_L)$. Time dilation independently
stretches the source time by $1+z$. In this scaling model, the redshifted
mass enters both time and amplitude; multiplying by another redshift factor
after using $M_z$ would count it twice.

Changing variables inside the Fourier integral supplies an additional time
factor:

$$
\begin{aligned}
\widetilde h^{\rm obs}(f)
&=\int\frac{L_z}{D_L}q(t_{\rm obs}/T_z)
                     e^{-2\pi ift_{\rm obs}}\,dt_{\rm obs}\\
&=\frac{L_z}{D_L}\,T_z\widetilde q(fT_z).
\end{aligned}
$$

Apply the same homogeneous scaling to the nonnegative amplitude statistic:

$$
\boxed{|\widetilde h_{\rm stat}(f)|
=\frac{L_z}{D_L}T_z\mathcal Q(\nu),\qquad \nu=fT_z.}
$$

The $T_z$ is required by the Fourier integration measure. Without it,
the units and mass dependence would be wrong. At matched dimensionless
frequencies and fixed $D_L,z$, doubling the mass halves observed frequency,
quadruples $ |\widetilde h|$, and doubles $h_c=2f|\widetilde h|$.
SNR need not scale by the same factor because the signal moves through a
frequency-dependent noise curve.

### 11.2 A concrete scale check

Ignoring redshift just for this arithmetic example, take
$M_{\rm BH}=50M_\odot$ and $D_L=100\,\mathrm{Mpc}$. Then

$$
T_z=2.462745475\times10^{-4}\ {\rm s},\qquad
\frac{L_z}{D_L}\simeq2.393\times10^{-20}.
$$

A dimensionless frequency $\nu=0.01$ maps to about $40.6\,\mathrm{Hz}$.
A projected dimensionless time-domain amplitude $q=10^{-4}$ maps to
strain about $2.393\times10^{-24}$. A spectral statistic
$\mathcal Q=1$ at that frequency maps to
$|\widetilde h_{\rm stat}|\simeq5.89\times10^{-24}\,\mathrm{s}$ and
$h_c\simeq4.79\times10^{-22}$. The time-domain and spectral examples
illustrate different quantities; they are not asserted to be one waveform.

The actual code recomputes $z$ from 100 Mpc, so these zero-redshift values
are only a transparent unit check, not the exact production example.

Rescaling preserves the dimensionless BH–disk model, including disk/BH mass
ratio, dimensionless spin, and frequency structure. It does not independently
increase disk mass at fixed BH mass. The idealized scale freedom also does
not supply new microphysics for every astrophysical mass regime.

### 11.3 Which cosmology is implemented?

The code uses a flat matter-plus-cosmological-constant approximation:

$$
D_L(z)=(1+z)\frac{c}{H_0}\int_0^z
\frac{dz'}{\sqrt{\Omega_m(1+z')^3+1-\Omega_m}},
$$

with $H_0=67.66\,\mathrm{km\,s^{-1}\,Mpc^{-1}}$ and
$\Omega_m=0.30966$. It tabulates the integral and interpolates distance
and its inverse. These parameter choices come from Planck18, but the code
does not implement the full Astropy Planck18 radiation/neutrino evolution.
Describe the actual approximation in the paper.

The fixed illustrative source pairs $(M_{\rm BH}/M_\odot,D_L/\mathrm{Mpc})$
are currently $(50,100)$, $(10^3,500)$, and $(10^5,50)$. They are
comparison scales, not fitted events or sources chosen to guarantee SNR 8.

## 12. Characteristic strain, noise, and the SNR integral

### 12.1 What one-sided noise means

For stationary noise, define a two-sided PSD $S^{(2)}(f)$ over positive
and negative frequencies. The corresponding one-sided PSD for $f>0$ is

$$
S^{(1)}(f)=2S^{(2)}(f),\qquad
\langle n(t)^2\rangle=\int_0^\infty S^{(1)}(f)\,df.
$$

This noise-convention factor of two is separate from the two-polarization
normalization. Our detector inputs are used as one-sided spectra.

The matched-filter norm of an actual real detector signal can be written
over both frequency signs using $S^{(2)}$. Reality gives equal positive
and negative contributions; substituting $S^{(1)}=2S^{(2)}$ gives

$$
\rho_d^2
=\int_{-\infty}^{\infty}
 \frac{|\widetilde h_d(f)|^2}{S^{(2)}(f)}\,df
=4\int_0^\infty\frac{|\widetilde h_d(f)|^2}{S^{(1)}(f)}\,df.
$$

That is the origin of the **4**. It is not a detector-direction factor or
a count of waveform modes. Matched filtering assumes an appropriate coherent
template and the noise model; this is an idealized sensitivity measure, not
an executed search on real detector data.

### 12.2 Why $h_c=2f|\widetilde h|$ pairs with $h_n=\sqrt{fS}$

For our source statistic and its matching effective noise define

$$
h_c(f)=2f|\widetilde h_{\rm stat}(f)|,\qquad
h_n(f)=\sqrt{fS_{\rm eff}(f)}.
$$

Then, using $d\ln f=df/f$,

$$
\begin{aligned}
\rho_{\rm stat}^2
&=4\int\frac{|\widetilde h_{\rm stat}|^2}{S_{\rm eff}}\,df\\
&=\int\frac{4f^2|\widetilde h_{\rm stat}|^2}
                 {fS_{\rm eff}}\,\frac{df}{f}\\
&=\boxed{\int\left(\frac{h_c}{h_n}\right)^2d\ln f}.
\end{aligned}
$$

These are the characteristic-strain definitions and SNR identity of
[Moore, Cole and Berry, Eqs. (16)–(21)](https://arxiv.org/html/1408.0740v2#S2.SS2).
They establish the conversion of a specified spectrum into a logarithmic
SNR integral; they do not prescribe our taper, modes, or source-direction
average.

Both $h_c$ and $h_n$ are dimensionless. ASD has units
$\mathrm{Hz}^{-1/2}$, PSD has units $\mathrm{Hz}^{-1}$, and
$\widetilde h$ has units $\mathrm{Hz}^{-1}$. Compare matching quantities:

| Signal | Noise | Units |
| --- | --- | --- |
| $h_c=2f\lvert\widetilde h\rvert$ | $h_n=\sqrt{fS_{\rm eff}}$ | Dimensionless |
| Equivalent source ASD $2\sqrt f\lvert\widetilde h\rvert$ | $\sqrt{S_{\rm eff}}$ | $\mathrm{Hz}^{-1/2}$ |

The equivalent source ASD is a plotting reparameterization of this finite
signal. It should not be confused with estimating a stationary random-process
PSD by dividing an FFT's squared magnitude by observation duration.

The pointwise ratio $h_c/h_n$ is **not the total SNR**. Instead

$$
\frac{d\rho^2}{d\ln f}=\left(\frac{h_c}{h_n}\right)^2.
$$

For example, a ratio roughly 2 across a logarithmic bandwidth of 0.1 gives
$\rho^2\simeq4\times0.1$, not $\rho=2$. A tall narrow peak can
contribute less than a lower broad signal. Nor is the SNR the unweighted
geometric area between two curves on a log–log image.

Integrate only the overlap between the retained signal band and the configured
observed detector band. The plots and SNR must use the identical
$S_{\rm eff}$. Changing only a plotted noise curve would break the link
between the picture and the reported distance reach.

### 12.3 No extra cycle-count factor belongs on this FFT

The finite FFT already contains the measured duration. To see why, consider
a real sinusoid $h(t)=h_0\cos(2\pi f_0t)$ lasting $T$, with many cycles
and nearly constant noise PSD $S_0$ across its peak. Parseval gives

$$
\int_0^\infty|\widetilde h(f)|^2df\simeq\frac{h_0^2T}{4},
\qquad \rho^2\simeq\frac{h_0^2T}{S_0}.
$$

Thus SNR grows as $\sqrt T$. The Fourier peak has height about
$h_0T/2$ and width about $1/T$; both matter in the integral. The
characteristic-strain peak from this resolved finite FFT is about
$h_0f_0T$, whereas a schematic single-point representation may use an
effective amplitude proportional to $h_0\sqrt{N_{\rm cycles}}$. Those
are different representations. Multiplying our finite-spectrum $h_c$ by
another $\sqrt{N_{\rm cycles}}$ would count coherent accumulation again.

### 12.4 One equation containing the full right-angle baseline

Combining physical scaling, the polarization convention, and the detector
average gives, for raw right-angle instrument noise,

$$
\boxed{
\rho_{\rm mean\ spectrum}^2
=\frac85\left(\frac{L_z}{D_L}\right)^2T_z
\int_{\mathcal B_\nu}
\frac{\langle A(\nu,\Omega_s)\rangle_s^2}
 {(2\pi\nu)^4S_{\rm inst}(\nu/T_z)}\,d\nu .
}
$$

Here $\mathcal B_\nu$ is the overlap of the finite-signal frequency
support and the detector band mapped into dimensionless frequency. The
$8/5$ is $4\times2/5$. The single $T_z$, rather than $T_z^2$,
remains because $df=d\nu/T_z$. This also explains the scaling in
`_snr_from_binned_power`.

For a source-direction RMS statistic, replace
$\langle A\rangle_s^2$ by $\langle A^2\rangle_s$. For ET/LISA
use their documented effective curves rather than appending the right-angle
factor again.

## 13. From an SNR to a distance reach

At each **source-frame** BH mass, the code varies redshift, computes its
luminosity distance, shifts the spectrum with $M_z=(1+z)M_{\rm BH}$,
and recomputes SNR. It looks for the outermost sampled crossing of

$$
\rho_{\rm stat}(M_{\rm BH},z)=8.
$$

The production scan uses 44 masses from $1$ to $10^7M_\odot$, a
combined low-/high-redshift scan, and a maximum search redshift of 10.
The plotted crossing is interpolated in $\log\rho$. An optional
example calculation refines a bracket by bisection. A point that reaches
the redshift cap is a search bound, not proof that the signal becomes
undetectable there. Sampling also limits the ability to find narrow or
nonmonotonic detectable intervals.

For efficiency, horizon calculations aggregate the unweighted spectral power
$\mathcal Q^2d\nu$ into 2048 logarithmic bins, then apply the detector noise
at representative frequencies. This preserves the integrated spectral power,
but evaluating the noise at bin centers is still an approximation. It is
different from smoothing a waveform or renormalizing each spectrum's peak.

At low redshift and a fixed mass, distance reach approximately follows
$\rho\propto1/D_L$. At cosmological distances, changing distance also
changes redshift, duration, and frequency placement in the detector band.
One must recompute those effects rather than multiply a reference distance
by $\rho/8$ blindly.

The clearest name for the result is:

> SNR-threshold luminosity-distance reach for the measured, selected-mode,
> mean-source-amplitude spectrum under the stated effective detector model.

“Horizon” often means the maximum reach for an optimally oriented source.
“Range” often refers to a volume-equivalent population quantity. Our default
statistic is neither of those conventional definitions. A threshold of 8 is
an assumed comparison criterion, not a false-alarm calculation or a guarantee
of detection in a real search. No event-rate inference is performed.

## 14. Related diagnostics and other easily confused factors

### 14.1 Energy and angular momentum integrate emitted radiation

The radiated-energy diagnostic is a different integral from detector SNR.
In geometrized units, using the natural-sign $H_{\ell m}$,

$$
\begin{aligned}
E_{\rm GW}(u)
&=\frac1{16\pi}\int_{u_{\rm cut}}^u
       \sum_{(\ell,m)\in\mathcal S}|\dot H_{\ell m}|^2\,du',\\
J_{z,\rm GW}(u)
&=\frac1{16\pi}\int_{u_{\rm cut}}^u
       \sum_{(\ell,m)\in\mathcal S}m\,
          \Im(H_{\ell m}\dot H_{\ell m}^*)\,du'.
\end{aligned}
$$

The sign convention makes prograde emission carry positive $J_z$. The
implemented routine handles the conjugated legacy cache accordingly.
The displayed energy and angular momentum use FFI strain/strain derivative,
the four selected modes, and accumulation after $u\ge1000m_{\rm BH}$.
This absolute-time cut differs slightly from the direct-spectrum cut's
“1000 after the first arrived sample” when arrival does not land exactly on
zero. The JSON diagnostic also records the all-cached-mode comparison.
[Ruiz et al. derive the multipolar radiation-flux expressions](https://arxiv.org/abs/0707.4654).

The $1/(16\pi)$ is a gravitational-wave flux normalization. Harmonic
orthonormality has already carried out the directional integral. These are
emitted energy/angular momentum, not averages over where a detector is
pointed; **do not multiply them by a detector $1/5$**. There is also no
luminosity-distance scaling or detector-noise weighting in these intrinsic
code-unit flux integrals.

The radiated figure normalizes by initial disk rest mass for energy and by
initial disk rest mass times initial BH mass for angular momentum. Its
$(2,2)$ frequency estimate comes from the curvature phase rate,

$$
\frac{f_{22}}{f_{\rm orb}}
=\frac{|d\arg P_{22}/du|}{\Omega_{\rm orb,0}},
$$

with unwrapped phase and a one-orbit moving average for display. Angular
frequencies appear in both numerator and denominator, so the factors of
$2\pi$ cancel. This display smoothing does not enter the direct-spectrum
SNR.

### 14.2 The quadrupole-luminosity $1/5$ is another result

For physical intuition in a slow-motion, weak-field model, define the
symmetric trace-free source quadrupole

$$
Q_{ij}=\int\rho\left(x_ix_j-\frac13\delta_{ij}r^2\right)d^3x.
$$

The leading strain is

$$
h_{ij}^{\rm TT}=\frac{2G}{c^4D}\ddot Q_{ij}^{\rm TT}.
$$

Integrating its flux over *emission directions* gives

$$
L_{\rm GW}=\frac{G}{5c^5}
 \left\langle\dddot Q_{ij}\dddot Q_{ij}\right\rangle_{\rm time}.
$$

For this definition of $Q_{ij}$, the TT projector obeys

$$
\int Q_{ij}^{\rm TT}Q_{ij}^{\rm TT}d\Omega
=\frac{8\pi}{5}Q_{ij}Q_{ij}.
$$

Substituting
$\dot h_{ij}^{\rm TT}=2G\dddot Q_{ij}^{\rm TT}/(c^4D)$ into the flux
$c^3D^2/(32\pi G)\int\dot h_{ij}^{\rm TT}\dot h_{ij}^{\rm TT}d\Omega$
gives $G/(5c^5)$. Thus this $1/5$ comes from an angular integration of
the emitted TT radiation. The time brackets denote the usual wave-period
average, not source inclination or detector response. Using a quadrupole
defined with an extra factor of three would change the corresponding
coefficient.

This is a useful interpretation of a rotating nonaxisymmetric disk, not a
step in our curvature-to-detectability pipeline. Do not apply its $1/5$
again to an extracted GR waveform. Matter $m=1$ is an azimuthal density
label, not dipole GW radiation; spin weight $-2$ requires $\ell\ge2$.
Nonlinear structure and the whole system's mass/current multipoles determine
the radiative modes. A matching matter/GW frequency is evidence of
consistency, not a calibrated matter-amplitude-to-strain conversion; see
[the existing mode-coupling derivation](gw_paper_methods.md#10-ppi-interpretation-and-matter-to-gw-mode-coupling).

### 14.3 Optional tails add a source model

The measured baseline ends with measured data. A separate trial models the
late complex $(2,2)$ strain as

$$
q_{22}^{\rm model}(\tau)
=B e^{(-\gamma+i\omega)(\tau-\tau_e)+i\phi_e}.
$$

It fits a mass-decay proxy and waveform phase/amplitude, with fit-window and
holdout checks. The current default proxy subtracts cumulative inward
horizon rest-mass flux from initial outside-horizon rest mass. This is not
a measurement of BH energy and can miss non-horizon mass losses/sources.
The optional detectability continuation differentiates the analytic model,

$$
p_{22}^{\rm model}=(-\gamma+i\omega)^2q_{22}^{\rm model},
$$

joins it into measured curvature, and applies a modeled termination. The
joining/tapering prescription is itself a further assumption. A1 is retained
as a conditional example with poor waveform-fit diagnostics visible; A2/A3
have no supported continuation in this workflow. A larger SNR from this
trial means “if this continuation model holds,” not “the simulation measured
this much radiation.”

### 14.4 Factor reference table

| Factor | Origin | Where it belongs |
| --- | --- | --- |
| $r_A$ | Remove leading radiative $1/r$ decay | Once when forming $r_A\Psi_4$ or $r_Ah$ |
| $1/(4\pi)$ | Normalize a solid-angle average | Mean/RMS over directions; absent from orthonormal mode projection |
| $1/2$ inside $H_{\rm pol}^2$ | Define mean squared polarization amplitude | Source-amplitude convention |
| $1/5$ | One-polarization detector power average | $\langle F_+^2\rangle$ or $\langle F_\times^2\rangle$ |
| $2/5$ | Two polarizations expressed through $H_{\rm pol}$ | $\langle\lvert\widetilde h_d\rvert^2\rangle/H_{\rm pol}^2$ |
| $5/2$ | Move that response into the noise | Effective PSD for our amplitude convention |
| $\sqrt{5/2}$ | Square root of the PSD conversion | Effective ASD |
| $2$ in $S^{(1)}=2S^{(2)}$ | One-sided noise convention | PSD conversion; distinct from polarization normalization |
| $4$ in SNR | Positive frequencies with a one-sided PSD | $4\int\lvert\widetilde h\rvert^2/S\,df$ |
| $2f$ | Characteristic-strain reparameterization | $h_c=2f\lvert\widetilde h\rvert$ |
| $T_z$ in $\widetilde h$ | Fourier time-coordinate Jacobian | Physical spectral scaling |
| $1/(16\pi)$ | Radiation-flux normalization in $G=c=1$ | Intrinsic mode energy/angular-momentum fluxes |
| $1/5$ in quadrupole luminosity | Emitted TT-radiation angular integral | Newtonian quadrupole formula; no extra factor on extracted modes |

## 15. What is justified, and what still needs a sensitivity statement?

Some steps are definitions or algebra: harmonic normalization, Fourier
derivatives, the sample-spacing normalization, physical-unit Jacobians,
the ideal right-angle $1/5$, and the $h_c/h_n$ SNR identity. Once the
conventions are stated, these are not tunable correction factors.

Other steps are choices that need evidence specific to the simulations:

- **Extraction:** finite radius, gauge/tetrad choices, resolution, and possible
  boundary contamination. Radius multiplication and retarded-time correction
  do not replace extrapolation or CCE.
- **Modes:** the four-mode subset answers a restricted radiation question.
  The origin of excluded $m=0$ content is not established by omitting it.
- **Finite duration:** termination, transient exclusion, taper strength, and
  the low-frequency floor can move the recovered spectral power.
- **Averaging:** mean amplitude, RMS amplitude, a fixed direction, optimal
  orientation, and a population detection volume are different observables.
- **Detector model:** curve version, reference-direction normalization,
  independent channels, frequency-dependent response, and foregrounds.
- **Astrophysical scaling:** the assumed dimensionless model, cosmology,
  coherent-template sensitivity, and any temporal continuation.

The comments and earlier review notes record sensitivity experiments for
specific data and targets: changing the transient cut from 1000 to 500–1500
BH mass units changed tested SNRs by at most about 3.2%; changing Tukey alpha
over 0.02–0.10 produced changes from about $-15\%$ to $+8\%$; raising the
duration floor from three to six cycles lowered some CE SNRs by about 35.5%.
These are historical, bounded experiments reported by the project, not
newly recomputed error bars or a guarantee for every source mass. In
particular, the low-frequency contribution should not be called converged
on that evidence alone.

For this writing pass, ten existing focused tests were rerun: detector
antenna quadrature, the characteristic-strain SNR identity, an analytic
cache-to-observed-SNR calculation, coherent multimode reconstruction,
source-direction normalization, physical mass scaling, LISA and ET curve
conventions, periodic FFI recovery, and face-on strain projection. They
support the stated algebra and implementation paths. They do not establish
NR convergence, reproduce Wessel's numerical horizons, or validate a
full-lifetime source model. Production plots and remote simulations were
not regenerated for this documentation pass.

## 16. A compact starting point for the paper

The following paragraphs are a writing scaffold. Keep the detailed
qualifications above available when deciding how strongly to phrase results.

**Extraction and strain.** We analyze the complex spin-weight $-2$
multipoles of the outgoing curvature scalar $\Psi_4$ at a finite
extraction sphere. We form $P_{\ell m}=r_A\Psi_{4,\ell m}$, using the
recorded areal radius, and interpolate onto a gauge-corrected retarded-time
grid. In our convention $h=h_+-ih_\times$ and $\Psi_4=\ddot h$ at
leading radiative order. Time-domain strain histories are reconstructed
using fixed-frequency integration with
$\omega_{0m}=\max(|m|,1)\Omega_{\rm orb,0}$. The results are finite-radius
estimates; no extrapolation to null infinity is performed.

**Measured spectrum and orientation statistic.** Detectability is calculated
from a separate direct-curvature spectrum, rather than from the FFI strain.
We use $\tau=u/m_{\rm BH}$ and $p_{\ell m}=m_{\rm BH}P_{\ell m}$,
exclude pre-arrival data and an additional $1000m_{\rm BH}$, and apply a
symmetric Tukey window with $\alpha=0.05$. We retain positive frequencies
$\nu\ge3/\Delta\tau$ below Nyquist. We coherently reconstruct the
$(2,\pm1),(2,\pm2)$ modes, form the norm
$A=[(|\mathcal F[W\Re p]|^2+|\mathcal F[W\Im p]|^2)/2]^{1/2}$, and
use $\mathcal Q(\nu)=\langle A\rangle_s/(2\pi\nu)^2$. The resulting
statistic is a source-direction mean spectral amplitude, not an orientation-
averaged squared SNR or total-mode emission.

**Physical scaling and sensitivity.** For source-frame BH mass
$M_{\rm BH}$, luminosity distance $D_L$, and redshift $z$, we use
$T_z=GM_{\rm BH}(1+z)/c^3$, $L_z=cT_z$, observed frequency
$f=\nu/T_z$, and spectral amplitude
$|\widetilde h_{\rm stat}|=(L_z/D_L)T_z\mathcal Q$. We define
$h_c=2f|\widetilde h_{\rm stat}|$ and $h_n=\sqrt{fS_{\rm eff}}$, and
evaluate $\rho_{\rm stat}^2=\int(h_c/h_n)^2d\ln f$ over the retained
signal–detector band overlap. For a raw right-angle interferometer PSD,
$S_{\rm eff}=(5/2)S_{\rm inst}$, since
$\langle F_+^2\rangle=\langle F_\times^2\rangle=1/5$ and our source norm
contains a factor $1/2$ in power. Detector-specific ET and LISA channel
conventions are treated separately. We report the luminosity distance at
$\rho_{\rm stat}=8$, recomputing the redshifted spectrum at each distance
with the stated flat matter-plus-Λ cosmology.

**Scope of reach estimates.** The distance reach refers to the measured
finite-duration signal, the selected mode set, the specified amplitude
average, and design-noise models. It is not an optimal-orientation horizon
or a volume-averaged population range. Finite-radius extraction and the
window/low-frequency choices remain sources of systematic uncertainty.
Any temporal continuation is presented separately as a conditional model.

The paper should also give the chosen extraction sphere's actual radius
or radius range from the run metadata, not just “index 8,” and identify the
detector-curve versions and cosmology. This guide intentionally does not
invent an extraction radius from a file index.

## 17. Where each statement lives in the code

| Operation | Implementation |
| --- | --- |
| Read 47-column curvature files, mode order, radius multiplication | [gw.py](../gw.py): `Psi4File`, `read_psi4_file`, `mode_columns` |
| Resolve restart overlap | [gw.py](../gw.py): `merge_psi4_restart_files` |
| Correct retarded time and interpolate | [gw.py](../gw.py): `_retarded_time`, `_four_point_interpolate`, `reconstruct_strain` |
| FFI and cache conventions | [gw.py](../gw.py): `_fixed_frequency_integrate`, `write_products` |
| Read matching direct-curvature cache | [gw_detectability.py](../gw_detectability.py): `read_rpsi4_modes` |
| Cuts, grid, Tukey window, FFT normalization and floor | [gw_detectability.py](../gw_detectability.py): `retained_psi4_modes`, `direct_psi4_spectrum` |
| Source-angle quadrature and polarization norm | [gw_detectability.py](../gw_detectability.py): `_source_directions`, `_polarization_amplitude_from_mode_ffts` |
| Detector units and response conversion | [gw_detectability.py](../gw_detectability.py): `load_detector_curves`, `effective_detector_asd` |
| Mass/redshift/distance conversion | [gw_detectability.py](../gw_detectability.py): `observer_spectrum`, `FlatLambdaCDM` |
| Characteristic strain and direct SNR norm | [gw_detectability.py](../gw_detectability.py): `characteristic_strain`, `cumulative_snr` |
| Spectral-power binning and threshold scan | [gw_detectability.py](../gw_detectability.py): `_bin_spectral_power`, `_snr_from_binned_power`, `_threshold_redshift`, `horizon_curves` |
| Combined finite-spectrum and distance figure | [gw_detectability.py](../gw_detectability.py): `analyze`, `plot_combined`, `_draw_horizon`, `_draw_characteristic_strain` |
| Additional spectrum/distance diagnostics and conditional trials | [gw_detectability.py](../gw_detectability.py), [gw_detectability_extras.py](../gw_detectability_extras.py) |
| Coherent two-signal mismatch, manual and kuibit | [gw_mismatch.py](../gw_mismatch.py): `prepare_pair`, `ManualOverlap`, `kuibit_check`, `analyze` |
| Face-on time-domain strain | [gw_waveforms.py](../wip_plots/gw_waveforms.py) |
| Selected-mode emitted energy, angular momentum, and phase frequency | [gw_radiated.py](../wip_plots/gw_radiated.py) |
| Analytic normalization checks | [test_gw_detectability.py](../tests/test_gw_detectability.py), [test_gw_ffi.py](../tests/test_gw_ffi.py) |

The highest-value cross-check when reading a formula is to identify three
things together: **which source amplitude**, **which effective noise**, and
**which quantity is averaged before squaring**. Those choices fix the factors
and the meaning of the reported reach.
