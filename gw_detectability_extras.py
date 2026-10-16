"""Optional radius/method diagnostics and conditional continuation fits.

The measured spectrum, detectability workflow and figure are in
gw_detectability.py. These routines never run for its default command.
"""
from dataclasses import dataclass
from gw import read_strain_cache, strain_cache_dir
import numpy as np
import config
import gw_detectability as core
VALIDATION_RADIUS_INDICES = (core.FIRST_RADIUS_INDEX, 7, core.OUTER_RADIUS_INDEX)
VALIDATION_TAPER_ALPHAS = (0.02, core.TAPER_ALPHA, 0.10)
VALIDATION_TRANSIENT_CUTOFFS_MBH = (500.0, core.TRANSIENT_CUTOFF_MBH, 1500.0)
VALIDATION_COARSE_ANGULAR_GRID = (12, 24)
TAIL_MASS_SOURCE = "horizon_flux"  # Or "outside_mass" for the original Wessel-like fit.
TAIL_FIT_START_FRACTION = 0.60  # Last 40% of retained measured strain.
TAIL_WINDOW_FRACTIONS = (0.40, 0.50, 0.60, 0.70, 0.80)  # Report sensitivity; never pick the best fit.
TAIL_HOLDOUT_FRACTION = 0.25  # Withhold last quarter of that interval, then refit.
TAIL_REMAINING_MASS_FRACTION = 0.10  # Wessel: 90% of initial disk mass accreted.
TAIL_FIT_LIMITS = dict(mass=0.01, amplitude=0.25, phase=0.5, holdout=0.5)
TAIL_MAX_SAMPLES = 2_000_000  # Abort, never shorten a tail silently.
TAIL_JOIN_ORBITS, TAIL_TAPER_ORBITS = 1.0, 3.0
TAIL_REVIEW_CASES = ("A1",)  # Empty tuple disables conditional examples.


def orbital_spectrum(case, spectrum):
    orbital_frequency = float(case.gw_omega_orbital) * case.mlittle / (2 * np.pi)
    if not np.isfinite(orbital_frequency) or orbital_frequency <= 0:
        raise ValueError(f"{case.name}: positive orbital frequency is required")
    return spectrum.frequency / orbital_frequency, core.characteristic_strain(spectrum.frequency, spectrum.strain_ft)


def interpolated_spectral_ratio(reference_x, reference_y, test_x, test_y):
    lower = max(float(np.min(reference_x)), float(np.min(test_x)))
    upper = min(float(np.max(reference_x)), float(np.max(test_x)))
    keep = ((reference_x >= lower) & (reference_x <= upper)
            & np.isfinite(reference_y) & (reference_y > 0))
    x = reference_x[keep]
    return x, np.interp(x, test_x, test_y) / reference_y[keep]


def _significant_ratio(label, baseline_x, baseline_y, x, y):
    x, ratio = interpolated_spectral_ratio(baseline_x, baseline_y, x, y)
    significant = (np.isfinite(ratio)
                   & (np.interp(x, baseline_x, baseline_y) >= 1e-3 * np.max(baseline_y)))
    if np.any(significant):
        change = abs(ratio[significant] - 1)
        print(f"{label}: median fractional change={np.median(change):.3g}, "
              f"95th percentile={np.percentile(change, 95):.3g}", flush=True)
    return x[significant], ratio[significant]


def validation_data(sim_name):
    """Return radius, taper and transient comparisons as arrays ready to draw."""
    case = config.all_sim_configs([sim_name])[0]
    time, modes = core.read_rpsi4_modes(case, core.OUTER_RADIUS_INDEX)
    baseline, _ = core.spectrum_for_case(case, time, modes)
    baseline_x, baseline_y = orbital_spectrum(case, baseline)
    radial, tapers, cutoffs = [], [], []
    for index in dict.fromkeys(VALIDATION_RADIUS_INDICES):
        try:
            if index == core.OUTER_RADIUS_INDEX:
                spectrum = baseline
            else:
                other_time, other_modes = core.read_rpsi4_modes(case, index)
                spectrum, _ = core.spectrum_for_case(case, other_time, other_modes)
        except (OSError, ValueError, KeyError) as exc:
            print(f"{sim_name}: radius index {index} unavailable: {exc}", flush=True)
            continue
        x, y = orbital_spectrum(case, spectrum)
        radial.append((index, x, y))
        if index != core.OUTER_RADIUS_INDEX:
            _significant_ratio(f"{sim_name} radius index {index}/outer", baseline_x, baseline_y, x, y)

    for alpha in dict.fromkeys(VALIDATION_TAPER_ALPHAS):
        spectrum = baseline if alpha == core.TAPER_ALPHA else core.spectrum_for_case(case, time, modes, taper_alpha=alpha)[0]
        x, ratio = _significant_ratio(
            f"{sim_name} Tukey alpha={alpha:g}/baseline", baseline_x, baseline_y, *orbital_spectrum(case, spectrum))
        tapers.append((alpha, x, ratio))
    for cutoff in dict.fromkeys(VALIDATION_TRANSIENT_CUTOFFS_MBH):
        spectrum = baseline if cutoff == core.TRANSIENT_CUTOFF_MBH else core.spectrum_for_case(
            case, time, modes, transient_cutoff_mbh=cutoff)[0]
        x, ratio = _significant_ratio(
            f"{sim_name} transient={cutoff:g} M_BH/baseline", baseline_x, baseline_y, *orbital_spectrum(case, spectrum))
        cutoffs.append((cutoff, x, ratio))
    coarse, _ = core.spectrum_for_case(case, time, modes,
                                 theta_nodes=VALIDATION_COARSE_ANGULAR_GRID[0],
                                 phi_nodes=VALIDATION_COARSE_ANGULAR_GRID[1])
    _significant_ratio(f"{sim_name} coarse/baseline angular grid", baseline_x, baseline_y, *orbital_spectrum(case, coarse))
    return radial, tapers, cutoffs


@dataclass
class WesselFit:
    start: float
    end: float
    gamma: float
    omega: float
    amplitude: float
    phase: float
    mass_end: float
    errors: dict

    def strain(self, time):
        return self.amplitude * np.exp((-self.gamma + 1j*self.omega)
                                      * (np.asarray(time) - self.end) + 1j*self.phase)

    def mass(self, time):
        return self.mass_end * np.exp(-self.gamma * (np.asarray(time) - self.end))


@dataclass
class WesselTrial:
    time: np.ndarray
    strain: np.ndarray
    mass_time: np.ndarray
    mass: np.ndarray
    initial_mass: float
    fit: WesselFit
    holdout_fit: WesselFit
    holdout_error: float
    issues: list[str]
    extended_time: np.ndarray | None = None
    extended_strain: np.ndarray | None = None
    finite_spectrum: core.DimensionlessSpectrum | None = None
    extended_spectrum: core.DimensionlessSpectrum | None = None
    conditional: bool = False
    density_time: np.ndarray | None = None
    density_modes: np.ndarray | None = None
    window_fits: tuple[WesselFit, ...] = ()
    mass_source: str = "outside_mass"
    orbital_period: float = 0.0


def fit_wessel_mode(time, strain, mass_time, mass, start, end):
    """Eq. (10): mass -> gamma, unwrapped phase -> omega, amplitude -> B.

    All times are in M_BH, strain is r*h_22/M_BH. Fit the mass itself in
    linear space (not log mass). Uniformly sampled waveform and scalar data
    have equal sample weights. No absolute value or positivity bound on gamma
    is imposed: a growing mass history must remain a failed decay hypothesis.
    """
    from scipy.optimize import curve_fit

    wave = (time >= start) & (time <= end)
    disk = (mass_time >= start) & (mass_time <= end)
    if wave.sum() < 16 or disk.sum() < 16 or end <= start:
        raise ValueError("Need at least 16 waveform and mass samples in the fit interval")
    h, x = strain[wave], time[wave] - end
    y, xm = mass[disk], (mass_time[disk] - end) / (end - start)
    if np.any(y <= 0) or np.any(np.abs(h) == 0):
        raise ValueError("Mass and complex-mode amplitude must be positive for this fit")
    scale = np.mean(y)
    guess = np.polyfit(xm, np.log(y / scale), 1)
    pars, _ = curve_fit(lambda u, b, g: b * np.exp(-g*u), xm, y/scale,
                        p0=(np.exp(guess[1]), -guess[0]), maxfev=10000)
    mass_end, gamma = float(pars[0]*scale), float(pars[1]/(end-start))
    phase = np.unwrap(np.angle(h))
    omega, phase_end = np.polyfit(x, phase, 1)
    envelope = np.exp(-gamma*x)
    amplitude = float(np.dot(envelope, np.abs(h)) / np.dot(envelope, envelope))
    rms = lambda a: float(np.sqrt(np.mean(np.abs(a)**2)))
    errors = dict(mass=rms(y-mass_end*np.exp(-gamma*(mass_time[disk]-end)))/rms(y),
                  amplitude=rms(np.abs(h)-amplitude*envelope)/rms(h),
                  phase=rms(phase-(omega*x+phase_end)))
    return WesselFit(float(start), float(end), gamma, float(omega), amplitude,
                     float(phase_end), mass_end, errors)


def wessel_trial(time, strain, mass_time, mass, orbital_period, *,
                 fit_start_fraction=TAIL_FIT_START_FRACTION,
                 holdout_fraction=TAIL_HOLDOUT_FRACTION,
                 allow_poor_waveform_fit=False):
    """Fit Eq. (10), retaining diagnostics even for conditional review models.

    The optional review override waives only waveform-residual limits, not
    mass-decay or sampling checks. It must not be interpreted as validation.

    Source-coordinate scalar time is compared with retarded GW source time,
    both t/M_BH with the same simulation origin. This is the usual approximate
    source-time identification, not a gauge-invariant time mapping.
    """
    time, strain = np.asarray(time, float), np.asarray(strain, complex)
    mass_time, mass = np.asarray(mass_time, float), np.asarray(mass, float)
    if not (0 <= fit_start_fraction < 1 and 0 < holdout_fraction < 1):
        raise ValueError("Fit and holdout fractions must define nonempty intervals")
    if not np.isfinite(orbital_period) or orbital_period <= 0:
        raise ValueError("Orbital period must be finite and positive")
    for t, y in ((time, strain), (mass_time, mass)):
        if t.ndim != 1 or t.size < 32 or y.shape != t.shape or not np.all(np.isfinite(y)) \
                or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
            raise ValueError("Need finite, increasing waveform and mass series")
    if mass[0] <= 0 or time[0] < mass_time[0] or time[-1] > mass_time[-1]:
        raise ValueError("Positive initial disk mass and scalar coverage of the waveform are required")
    if not np.allclose(np.diff(time), np.median(np.diff(time)), rtol=1e-5):
        raise ValueError("Use the uniformly sampled strain cache")
    start = time[0] + fit_start_fraction*np.ptp(time)
    holdout_start = time[-1] - holdout_fraction*(time[-1]-start)
    fit = fit_wessel_mode(time, strain, mass_time, mass, start, time[-1])
    held_fit = fit_wessel_mode(time, strain, mass_time, mass, start, holdout_start)
    held = time > holdout_start
    held_error = float(np.linalg.norm(strain[held]-held_fit.strain(time[held]))
                       / np.linalg.norm(strain[held]))
    issues = []
    if fit.gamma <= 0:
        issues.append("disk mass does not decay")
    elif fit.gamma*(fit.end-fit.start) <= 3*fit.errors["mass"]:
        issues.append("mass decay unresolved above fit residuals")
    if fit.gamma*orbital_period/(2*np.pi) >= 0.1:
        issues.append("decay is not slow compared with orbital motion")
    if abs(fit.omega)*(fit.end-fit.start)/(2*np.pi) < 3:
        issues.append("fewer than three fitted wave cycles")
    if not np.isfinite(fit.errors["mass"]) or fit.errors["mass"] > TAIL_FIT_LIMITS["mass"]:
        issues.append("mass residual exceeds trial limit")
    blocking = bool(issues)
    for key, error in {**fit.errors, "holdout": held_error}.items():
        if key == "mass":
            continue
        if not np.isfinite(error) or error > TAIL_FIT_LIMITS[key]:
            issues.append(f"{key} residual exceeds trial limit")
            blocking |= not np.isfinite(error)
    trial = WesselTrial(time, strain, mass_time, mass, float(mass[0]), fit,
                         held_fit, held_error, issues, orbital_period=orbital_period)
    if blocking or (issues and not allow_poor_waveform_fit):
        return trial

    # The paper does not specify a join
    # function: quintic blending over the last orbit is our explicit choice.
    stop = fit.end + np.log(fit.mass_end/(TAIL_REMAINING_MASS_FRACTION*mass[0]))/fit.gamma
    join_width, taper_width = TAIL_JOIN_ORBITS*orbital_period, TAIL_TAPER_ORBITS*orbital_period
    dt = float(np.median(np.diff(time)))
    count = int(np.ceil((stop+taper_width-time[0])/dt)) + 1
    if stop <= fit.end or fit.end-join_width < fit.start:
        issues.append("no future mass threshold or insufficient join interval")
        return trial
    if count > TAIL_MAX_SAMPLES:
        issues.append("tail exceeds sample budget; no truncated substitute")
        return trial
    extended_time = time[0] + dt*np.arange(count)
    extended = fit.strain(extended_time)
    blend = np.clip((time-(fit.end-join_width))/join_width, 0, 1)
    blend = blend**3*(10-15*blend+6*blend**2)
    extended[:time.size] = (1-blend)*strain + blend*extended[:time.size]
    taper = np.clip((extended_time-stop)/taper_width, 0, 1)
    extended *= 1-taper**3*(10-15*taper+6*taper**2)
    trial.extended_time, trial.extended_strain = extended_time, extended
    # Identical FFI input, FFT grid and measured-duration frequency floor.
    # Only the termination differs. Keep the onset taper's physical width
    # fixed; a full-length Tukey would re-window the measured signal.
    off_window = core.tukey_window(time.size, core.TAPER_ALPHA)
    on_window = np.ones(count)
    half = time.size//2
    on_window[:half] = off_window[:half]
    nfft = int(np.ceil(core.ZERO_PAD_FACTOR*count))
    floor = core.LOW_FREQUENCY_CYCLES/np.ptp(time)
    trial.finite_spectrum = _single_22_spectrum(strain*off_window, dt, nfft, floor)
    trial.extended_spectrum = _single_22_spectrum(extended*on_window, dt, nfft, floor)
    trial.conditional = bool(issues)
    return trial


def _single_22_spectrum(strain, dt, nfft, frequency_floor):
    """Same polarization/orientation statistic as production, for one mode.

    For a single mode the angular factor separates exactly, avoiding a large
    direction-by-frequency array for long tails. No extra 1/f^2: this is h.
    """
    fft = np.fft.fft(strain, n=nfft)*dt
    freq = np.fft.fftfreq(nfft, dt)
    pos = np.flatnonzero((freq > 0) & (freq >= frequency_floor))
    theta, phi, weights = core._source_directions(core.THETA_NODES, core.PHI_NODES)
    harmonics = np.abs(core._harmonic_matrix(((2, 2),), theta, phi)[:, 0])
    if core.SOURCE_AVERAGING == "mean":
        factor = weights @ harmonics
    elif core.SOURCE_AVERAGING == "rms":
        factor = np.sqrt(weights @ harmonics**2)
    else:
        raise ValueError(f"Unknown source averaging {core.SOURCE_AVERAGING}")
    amplitude = 0.5*factor*np.sqrt(np.abs(fft[pos])**2+np.abs(fft[(-pos)%nfft])**2)
    return core.DimensionlessSpectrum(freq[pos], amplitude, f"source-direction {core.SOURCE_AVERAGING}")


def disk_mass_for_tail(scalar, source):
    """Outside-AH mass or a flux-only remaining-rest-mass estimate.

    bhns.don columns 2/3 are total/inside-AH M0; column 13 is the recorded
    positive cumulative inward flux. Subtract its initial offset, not abs().
    Fitting M0(0)-F(t) to an exponential is equivalent to fitting the implied
    accumulated flux M0(0)-B exp(-gamma*t), with the initial mass fixed.
    """
    scalar = np.asarray(scalar, float)
    outside = scalar[:, 1]-scalar[:, 2]
    if source == "outside_mass":
        mass = outside
    elif source == "horizon_flux":
        mass = outside[0]-(scalar[:, 12]-scalar[0, 12])
    else:
        raise ValueError("TAIL_MASS_SOURCE must be horizon_flux or outside_mass")
    if not np.all(np.isfinite(mass)) or np.any(mass <= 0):
        raise ValueError("Temporal mass history must remain finite and positive")
    return mass


def continued_psi4_mode(time, psi4, trial):
    """Continue p=M_BH*r*Psi4 using q''=lambda^2*q, q=r*h/M_BH.

    The Wessel strain model supplies lambda=-gamma+i*omega. Analytic
    differentiation is exact for that MODEL, not validation of the data fit.
    Blend measured p into model p over the last orbit; taper p at the same
    mass threshold as the strain trial. These join/window choices are ours.
    The measured direct-Psi4 baseline never uses this routine.
    """
    time, psi4 = np.asarray(time), np.asarray(psi4)
    dt = float(np.median(np.diff(time)))
    # Join at the measured Psi4 end; allow for one sample of grid rounding.
    if trial.extended_time is None or abs(time[-1]-trial.fit.end) > 1.5*dt:
        raise ValueError("Psi4 and temporal trial endpoints differ by more than one sample")
    orbital_period = trial.orbital_period
    if orbital_period <= 0:
        raise ValueError("A positive orbital period is required for the continuation")
    stop = trial.fit.end+np.log(trial.fit.mass_end/
           (TAIL_REMAINING_MASS_FRACTION*trial.initial_mass))/trial.fit.gamma
    end = stop+TAIL_TAPER_ORBITS*orbital_period
    count = int(np.ceil((end-time[0])/dt))+1
    if count > TAIL_MAX_SAMPLES:
        raise ValueError("Psi4 tail exceeds the sample budget")
    extended_time = time[0]+dt*np.arange(count)
    lam = -trial.fit.gamma+1j*trial.fit.omega
    extended = lam**2*trial.fit.strain(extended_time)
    blend = np.clip((time-(time[-1]-TAIL_JOIN_ORBITS*orbital_period))/
                    (TAIL_JOIN_ORBITS*orbital_period), 0, 1)
    blend = blend**3*(10-15*blend+6*blend**2)
    extended[:len(time)] = (1-blend)*psi4+blend*extended[:len(time)]
    taper = np.clip((extended_time-stop)/(TAIL_TAPER_ORBITS*orbital_period), 0, 1)
    extended *= 1-taper**3*(10-15*taper+6*taper**2)
    return extended_time, extended


def temporal_trial_for_case(case):
    """Reuse the existing GW cache and restart-aware scalar reader; no raw writes."""
    from helpers.reader import DiskSim

    label = str(core.OUTER_RADIUS_INDEX + case.gw_psi4_file_index_offset)
    cached = read_strain_cache(strain_cache_dir(case, label), case.data_path/"Psi4_rad.mon")
    if cached.backend != "python" or str(cached.metadata.get("source_label")) != label \
            or not np.isclose(cached.metadata.get("madm", np.nan), case.gw_madm, rtol=1e-10, atol=0):
        raise ValueError(f"{case.name}: regenerate inconsistent strain cache")
    hp, hc = cached.hplus_hcross(2, 2)
    time = cached.time/case.mlittle
    # Undo the legacy Fortran conjugation for the natural spin -2 complex mode.
    strain = (hp-1j*hc)/case.mlittle
    sim = DiskSim(case)
    _, scalar, mass_time, _ = sim.loaddata("bhns.don")
    disk_mass = disk_mass_for_tail(scalar, TAIL_MASS_SOURCE)
    mass_time = mass_time/case.mlittle
    if mass_time[0] != 0:
        raise ValueError(f"{case.name}: initial disk mass at t=0 is unavailable")
    # Stop at the last measured Psi4 sample; later FFI rows integrate zero fill.
    measured_end = core.read_rpsi4_modes(case, core.OUTER_RADIUS_INDEX, ((2, 2),))[0][-1]/case.mlittle
    keep = ((time >= core.TRANSIENT_CUTOFF_MBH) & (time >= mass_time[0]) & (time <= mass_time[-1])
            & (time <= measured_end + 1e-9))
    trial = wessel_trial(time[keep], strain[keep], mass_time, disk_mass, case.Pc/case.mlittle,
                         allow_poor_waveform_fit=case.name in TAIL_REVIEW_CASES)
    trial.mass_source = TAIL_MASS_SOURCE
    sim.load_modes()
    valid = (sim.modes[:, 0].real > 0) & np.all(np.isfinite(sim.modes), axis=1)
    trial.density_time = sim.modes_t[valid]/case.mlittle
    trial.density_modes = np.abs(sim.modes[valid, 1:])/np.abs(sim.modes[valid, :1])
    # C0 is density-cut in the Illinois diagnostic, unlike the full M0 integral.
    # Use C1/C0 to assess structure, never substitute C0 for disk rest mass.
    trial.window_fits = tuple(
        fit_wessel_mode(trial.time, trial.strain, mass_time, disk_mass,
                        trial.time[0]+fraction*np.ptp(trial.time), trial.time[-1])
        for fraction in TAIL_WINDOW_FRACTIONS)
    return trial


def source_spectra(names=None, radius_index=core.OUTER_RADIUS_INDEX, representative=False, *,
                   cache_root=None):
    sources, directions = {}, {}
    for case in config.all_sim_configs(core.SIM_NAMES if names is None else names):
        try:
            time, modes = core.read_rpsi4_modes(case, radius_index, cache_root=cache_root)
            spectrum, info = core.spectrum_for_case(case, time, modes)
            sources[case.name] = core.SourceSpectrum(case, spectrum, info)
            if representative:
                directions[case.name] = core.spectrum_for_case(
                    case, time, modes, representative_direction=(np.pi / 2.34, 0.0)
                )[0]
            print(f"{case.name}: radius index {radius_index}, {len(info.modes)} modes, "
                  f"{info.samples} samples over {info.duration_mbh:.1f} M_BH", flush=True)
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError(f"{case.name}: {exc}") from exc
    return sources, directions
