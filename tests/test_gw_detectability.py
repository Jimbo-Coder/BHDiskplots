import unittest
import json
from pathlib import Path
import tempfile
from unittest import mock
from types import SimpleNamespace

import numpy as np
from scipy.signal.windows import tukey as scipy_tukey
import config
import gw_detectability as analysis

from gw_detectability import (
    DimensionlessSpectrum,
    DetectorCurve,
    _bin_spectral_power,
    _snr_from_binned_power,
    FlatLambdaCDM,
    characteristic_strain,
    direct_psi4_spectrum,
    observer_spectrum,
    spin_weighted_spherical_harmonic,
    tukey_window,
)


class WesselTailTests(unittest.TestCase):
    def setUp(self):
        self.time = np.arange(0., 300., .2)
        self.mass_time = np.arange(0., 320., .35)
        self.mass = .03*np.exp(-.003*self.mass_time)
        self.strain = .002*np.exp((-.003+.21j)*self.time+.6j)

    def trial(self, **overrides):
        data = dict(time=self.time, strain=self.strain, mass_time=self.mass_time,
                    mass=self.mass, orbital_period=40.)
        data.update(overrides)
        return analysis.wessel_trial(**data)

    def test_flux_mass_proxy_preserves_sign_and_initial_offset(self):
        data = np.zeros((3, 13))
        data[:, 1], data[:, 2] = [.04, .042, .044], .01
        data[:, 12] = [2., 2.001, 1.999]
        np.testing.assert_allclose(analysis.disk_mass_for_tail(data, "horizon_flux"),
                                   [.03, .029, .031])
        np.testing.assert_allclose(analysis.disk_mass_for_tail(data, "outside_mass"),
                                   [.03, .032, .034])
        data[-1, 12] = 3
        with self.assertRaisesRegex(ValueError, "positive"):
            analysis.disk_mass_for_tail(data, "horizon_flux")

    def test_analytic_strain_tail_second_derivative_is_psi4(self):
        trial = self.trial()
        lam = -.003+.21j
        p = lam**2*self.strain
        time, extended = analysis.continued_psi4_mode(self.time, p, trial)
        before_taper = time <= trial.fit.end+np.log(trial.fit.mass_end/
                       (analysis.TAIL_REMAINING_MASS_FRACTION*trial.initial_mass))/trial.fit.gamma
        np.testing.assert_allclose(extended[before_taper],
                                   lam**2*trial.fit.strain(time[before_taper]), rtol=1e-9)
        np.testing.assert_allclose(extended[:len(p)], p, rtol=1e-9)
        self.assertAlmostEqual(abs(extended[-1]), 0., places=12)

    def test_rejected_psi4_tail_keeps_baseline_exact(self):
        rejected = self.trial(mass=.03*np.exp(.003*self.mass_time))
        modes = {(2, 2): (-.003+.21j)**2*self.strain}
        options = dict(transient_cutoff_mbh=0, theta_nodes=4, phi_nodes=8)
        before, _ = direct_psi4_spectrum(self.time, modes, 1., **options)
        after, _ = direct_psi4_spectrum(self.time, modes, 1., continuation=rejected, **options)
        np.testing.assert_array_equal(before.frequency, after.frequency)
        np.testing.assert_array_equal(before.strain_ft, after.strain_ft)

    def test_psi4_join_uses_retained_endpoint_not_dropped_cache_sample(self):
        trial = self.trial()
        p = (-.003+.21j)**2*self.strain
        time, extended = analysis.continued_psi4_mode(self.time[:-1], p[:-1], trial)
        expected = (-trial.fit.gamma+1j*trial.fit.omega)**2*trial.fit.strain(self.time[-2])
        self.assertAlmostEqual(abs(extended[len(p)-2]-expected), 0., places=12)
        self.assertAlmostEqual(time[len(p)-2], self.time[-2], places=10)
        with self.assertRaisesRegex(ValueError, "endpoints"):
            analysis.continued_psi4_mode(self.time[:-3], p[:-3], trial)

    def test_psi4_tail_all_modes_reconstruct_before_averaging(self):
        trial = self.trial()
        modes = {(2, 2): (-.003+.21j)**2*self.strain,
                 (2, 1): .00002*np.exp(.15j*self.time)}
        copies = {k: v.copy() for k, v in modes.items()}
        spectrum, _ = direct_psi4_spectrum(self.time, modes, 1., continuation=trial,
                        transient_cutoff_mbh=0, theta_nodes=4, phi_nodes=8)
        time, p = analysis.continued_psi4_mode(self.time, modes[2, 2], trial)
        window = tukey_window(len(self.time))
        p[:len(self.time)//2] *= window[:len(self.time)//2]
        measured = np.pad(modes[2, 1]*window, (0, len(time)-len(self.time)))
        nfft = int(np.ceil(analysis.ZERO_PAD_FACTOR*len(time)))
        pad = nfft-len(time)
        rows = np.pad(np.stack([measured, p]), ((0, 0), (pad//2, pad-pad//2)))
        theta, phi, weights = analysis._source_directions(4, 8)
        field = analysis._harmonic_matrix(((2, 1), (2, 2)), theta, phi) @ rows
        dt = np.median(np.diff(self.time))
        f = np.fft.fftfreq(nfft, dt)
        positive = np.flatnonzero((f > 0) & (f >= analysis.LOW_FREQUENCY_CYCLES/np.ptp(self.time)))
        plus = np.fft.fft(field.real, axis=1)[:, positive]*dt
        cross = np.fft.fft(field.imag, axis=1)[:, positive]*dt
        expected = (weights @ np.sqrt((abs(plus)**2+abs(cross)**2)/2))/(2*np.pi*f[positive])**2
        np.testing.assert_allclose(spectrum.frequency, f[positive], rtol=1e-10)
        np.testing.assert_allclose(spectrum.strain_ft, expected, rtol=1e-7, atol=1e-14)
        for mode in modes:
            np.testing.assert_array_equal(modes[mode], copies[mode])

    def test_known_mass_decay_phase_and_amplitude_recovered(self):
        trial = self.trial()
        self.assertEqual(trial.issues, [])
        self.assertAlmostEqual(trial.fit.gamma, .003, places=10)
        self.assertAlmostEqual(trial.fit.omega, .21, places=10)
        self.assertLess(trial.holdout_error, 1e-10)
        np.testing.assert_allclose(trial.fit.strain(self.time), self.strain, rtol=1e-9)
        np.testing.assert_allclose(trial.extended_strain[:len(self.time)], self.strain, rtol=1e-9)
        self.assertAlmostEqual(abs(trial.extended_strain[-1]), 0., places=12)
        end_mass = trial.fit.mass(trial.extended_time[-1]-analysis.TAIL_TAPER_ORBITS*40)
        self.assertAlmostEqual(end_mass/self.mass[0], .1, delta=1e-4)
        np.testing.assert_array_equal(trial.finite_spectrum.frequency, trial.extended_spectrum.frequency)
        self.assertGreaterEqual(trial.finite_spectrum.frequency[0],
                                analysis.LOW_FREQUENCY_CYCLES/np.ptp(self.time))

    def test_growing_mass_is_not_forced_to_decay(self):
        trial = self.trial(mass=.03*np.exp(.003*self.mass_time))
        self.assertLess(trial.fit.gamma, 0)
        self.assertIn("disk mass does not decay", trial.issues)
        self.assertIsNone(trial.extended_strain)
        self.assertIsNone(trial.extended_spectrum)

    def test_bad_held_out_phase_prevents_continuation(self):
        signal = self.strain.copy()
        signal[self.time > 270] *= np.exp(2j)
        trial = self.trial(strain=signal)
        self.assertGreater(trial.holdout_error, 1)
        self.assertIn("holdout residual exceeds trial limit", trial.issues)
        self.assertIsNone(trial.extended_strain)

    def test_conditional_model_keeps_waveform_failures_visible(self):
        signal = self.strain.copy()
        signal[self.time > 270] *= np.exp(2j)
        trial = self.trial(strain=signal, allow_poor_waveform_fit=True)
        self.assertTrue(trial.conditional)
        self.assertIn("holdout residual exceeds trial limit", trial.issues)
        self.assertIsNotNone(trial.extended_spectrum)

    def test_conditional_model_cannot_override_mass_decay(self):
        trial = self.trial(mass=.03*np.exp(.003*self.mass_time), allow_poor_waveform_fit=True)
        self.assertFalse(trial.conditional)
        self.assertIn("disk mass does not decay", trial.issues)
        self.assertIsNone(trial.extended_strain)

    def test_temporal_fit_shows_only_data_and_models(self):
        import matplotlib.pyplot as plt
        from wip_plots import gw_detectability_all as plots
        trial = self.trial()
        trial.density_time = trial.time
        trial.density_modes = np.ones((trial.time.size, 2))*.01
        trial.conditional = True
        trial.issues.append("amplitude residual exceeds trial limit")
        with plt.rc_context({"text.usetex": False, "font.family": "serif"}):
            fig = plots.plot_temporal_validation(config.all_sim_configs(["A1"])[0], trial)
            try:
                fig.canvas.draw()
                self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()],
                                 ["A1 data", "Fit"])
                self.assertEqual(len(fig.axes), 3)
                self.assertEqual(len(fig.texts), 0)
                for ax in fig.axes:
                    self.assertEqual(len(ax.patches), 0)
                    self.assertEqual(len(ax.texts), 0)
                    self.assertEqual(len(ax.lines), 2)
                    np.testing.assert_allclose(ax.get_xlim(), (trial.fit.start, trial.fit.end))
                np.testing.assert_allclose(fig.axes[2].lines[0].get_ydata(),
                                           fig.axes[2].lines[1].get_ydata(), atol=1e-12)
            finally:
                plt.close(fig)

    def test_budget_does_not_silently_truncate_tail(self):
        with mock.patch.object(analysis, "TAIL_MAX_SAMPLES", 100):
            trial = self.trial()
        self.assertIn("tail exceeds sample budget; no truncated substitute", trial.issues)
        self.assertIsNone(trial.extended_strain)

    def test_transition_separates_blending_from_future_extrapolation(self):
        from dataclasses import replace
        import matplotlib.pyplot as plt
        from wip_plots import gw_detectability_all as plots
        trial = self.trial()
        before = trial.extended_strain.copy()
        case = replace(config.all_sim_configs(["A1"])[0], Pc=2.)
        with plt.rc_context({"text.usetex": False, "font.family": "serif"}):
            fig = plots.plot_temporal_transition(case, trial)
            try:
                fig.canvas.draw()
                offset_box = fig.axes[0].yaxis.get_offset_text().get_window_extent()
                for heading in fig.axes[0].texts:
                    self.assertFalse(heading.get_window_extent().overlaps(offset_box))
                self.assertEqual([text.get_text() for text in fig.axes[0].texts], ["Data end"])
                self.assertEqual(len(fig.texts), 0)
                for ax in fig.axes:
                    lines = {line.get_label(): line for line in ax.lines}
                    end = trial.time[-1]
                    join_start = end-analysis.TAIL_JOIN_ORBITS*40.
                    self.assertEqual(lines["Simulation"].get_xdata()[-1], end)
                    self.assertEqual(lines["Blend"].get_xdata()[0], join_start)
                    self.assertEqual(lines["Blend"].get_xdata()[-1], end)
                    self.assertEqual(lines["Extrapolation"].get_xdata()[0], end)
                    self.assertGreater(lines["Extrapolation"].get_xdata()[-1], end)
                    np.testing.assert_array_equal(lines["Last simulation sample"].get_xdata(), [end, end])
                    self.assertEqual(len(ax.patches), 0)
                    self.assertAlmostEqual(lines["Blend"].get_ydata()[-1],
                                           lines["Extrapolation"].get_ydata()[0])
                np.testing.assert_array_equal(trial.extended_strain, before)
            finally:
                plt.close(fig)

    def test_join_leaves_earlier_measured_samples_unchanged(self):
        signal = self.strain*(1+.04*np.sin(.025*self.time))
        trial = self.trial(strain=signal)
        self.assertEqual(trial.issues, [])
        before = self.time < trial.fit.end-40*analysis.TAIL_JOIN_ORBITS
        np.testing.assert_array_equal(trial.extended_strain[:self.time.size][before], signal[before])
        np.testing.assert_allclose(trial.extended_strain[self.time.size-1],
                                   trial.fit.strain(self.time[-1]), rtol=1e-10)

    def test_scalar_times_need_not_match_but_must_cover_waveform(self):
        with self.assertRaisesRegex(ValueError, "coverage"):
            self.trial(mass_time=self.mass_time+20)
        with self.assertRaisesRegex(ValueError, "uniformly"):
            self.trial(time=self.time+.01*np.sin(self.time))

    def test_single_mode_spectrum_matches_general_polarization_average(self):
        nfft, dt = 4096, self.time[1]-self.time[0]
        spectrum = analysis._single_22_spectrum(self.strain, dt, nfft, .01)
        fft = np.fft.fft(self.strain, n=nfft)*dt
        frequency = np.fft.fftfreq(nfft, dt)
        pos = np.flatnonzero((frequency > 0) & (frequency >= .01))
        theta, phi, weights = analysis._source_directions(24, 48)
        harmonics = analysis._harmonic_matrix(((2, 2),), theta, phi)
        expected = analysis._polarization_amplitude_from_mode_ffts(
            fft[None, pos], fft[None, (-pos)%nfft], harmonics, weights, "mean")
        np.testing.assert_allclose(spectrum.strain_ft, expected, rtol=1e-13)

    def test_accepted_trial_renders_fit_join_and_comparison(self):
        from dataclasses import replace
        from helpers import style
        from wip_plots import gw_detectability_all as plots
        trial = self.trial()
        case = replace(config.all_sim_configs(["A1"])[0], Pc=2.)
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(style, "USE_LATEX_TEXT", False), \
                mock.patch.dict(style.PAPER_PLOT_STYLE, {"text.usetex": False,
                                                        "font.family": "serif", "font.weight": "normal"}), \
                mock.patch.object(analysis, "temporal_trial_for_case", return_value=trial):
            report = plots.temporal_main(["--outdir", tmp], cases=[case])
            self.assertEqual(report["A1"]["issues"], [])
            self.assertIn("snr", report["A1"])
            self.assertTrue((Path(tmp)/"A1/gw_temporal_fit.png").is_file())
            self.assertTrue((Path(tmp)/"A1/gw_temporal_on_off.png").is_file())
            self.assertTrue((Path(tmp)/"A1/gw_temporal_transition.png").is_file())
            self.assertTrue((Path(tmp)/"gw/temporal_trial.json").is_file())
            trial.issues.append("new data rejected")
            plots.temporal_main(["--outdir", tmp], cases=[case])
            self.assertFalse((Path(tmp)/"A1/gw_temporal_on_off.png").exists())
            self.assertFalse((Path(tmp)/"A1/gw_temporal_transition.png").exists())


class DirectPsi4SpectrumTests(unittest.TestCase):
    def test_et_uses_combined_psd_and_triangular_network_response(self):
        raw = np.loadtxt(analysis.DETECTOR_CURVE_DIR/"ET10kmcolumns.txt")
        curves = analysis.load_detector_curves()
        np.testing.assert_allclose(curves["et"].asd, np.sqrt(raw[:, 3]))
        np.testing.assert_allclose(analysis.effective_detector_asd("et", raw[:, 0], curves),
                                   np.sqrt(2.5)/1.5*np.sqrt(raw[:, 3]), rtol=1e-12)

    def test_effective_noise_reproduces_sky_and_polarization_averaged_snr(self):
        # Quadrature over sky position and polarization angle of the actual
        # right-angle response F+ h+ + Fx hx, with no convention assumed.
        cos_theta, weights = np.polynomial.legendre.leggauss(16)
        phi = np.linspace(0, 2*np.pi, 32, endpoint=False)
        psi = np.linspace(0, np.pi, 16, endpoint=False)
        c, p, s = np.meshgrid(cos_theta, phi, psi, indexing="ij")
        a, b = 0.5*(1 + c**2)*np.cos(2*p), c*np.sin(2*p)
        f_plus = a*np.cos(2*s) - b*np.sin(2*s)
        f_cross = a*np.sin(2*s) + b*np.cos(2*s)
        w = weights[:, None, None]/(2*phi.size*psi.size)
        h_plus, h_cross, instrument_psd = 1.3 + 0.4j, -0.2 + 0.9j, 7.0
        averaged = np.sum(w*np.abs(f_plus*h_plus + f_cross*h_cross)**2)/instrument_psd
        h_res_squared = (abs(h_plus)**2 + abs(h_cross)**2)/2
        curves = {"ligo": analysis.DetectorCurve(np.array([1., 2.]), np.full(2, np.sqrt(instrument_psd)))}
        effective = analysis.effective_detector_asd("ligo", np.array([1.5]), curves)[0]
        self.assertAlmostEqual(h_res_squared/effective**2, averaged, places=12)

    def test_lisa_table_is_single_channel_scird_curve(self):
        curve = analysis.load_detector_curves()["lisa"]
        f = np.array([1e-4, 1e-3, 3e-3])
        L, f_star = 2.5e9, 19.09e-3
        p_oms = (1.5e-11)**2*(1 + (2e-3/f)**4)
        p_acc = (3e-15)**2*(1 + (0.4e-3/f)**2)*(1 + (f/8e-3)**4)
        robson = 10/(3*L**2)*(p_oms + 2*(1 + np.cos(f/f_star)**2)*p_acc/(2*np.pi*f)**4)*(1 + 0.6*(f/f_star)**2)
        np.testing.assert_allclose(analysis._log_interpolate_curve(curve, f)**2/robson, 2.0, rtol=2e-3)

    def test_source_scale_labels_use_latex_scientific_notation(self):
        from wip_plots import gw_detectability_all as plots
        self.assertEqual(plots._scientific_label(1e5), r"10^{5}")
        self.assertEqual(plots._scientific_label(500), r"5\times 10^{2}")
        self.assertEqual(plots._scientific_label(.025), r"2.5\times 10^{-2}")

    def test_fixed_examples_need_no_reference_case_or_horizon_scan(self):
        spectrum = DimensionlessSpectrum(np.geomspace(.001, .1, 100), np.ones(100), "mean")
        source = SimpleNamespace(spectrum=spectrum)
        with mock.patch.object(analysis, "source_spectra", return_value=({"B1": source}, {})), \
                mock.patch.object(analysis, "horizon_curves") as horizons:
            result = analysis.analyze(["B1"], compute_horizons=False)
        horizons.assert_not_called()
        self.assertIsNone(result.horizons)
        self.assertEqual([(t.detector, t.mass, t.distance) for t in result.targets],
                         list(analysis.EXAMPLE_TARGETS))

    def test_three_source_scales_share_one_axis_and_one_legend(self):
        import matplotlib.pyplot as plt
        from wip_plots import gw_detectability_all as plots
        frequency = np.geomspace(.001, 100, 40)
        source = SimpleNamespace(color="blue", linestyle="-", legend_name="A1")
        targets = [SimpleNamespace(detector=d, mass=m, distance=r,
                                   strain={"A1": (frequency, np.full(40, 1e-21))},
                                   snr={"A1": {d: 7}}) for d, m, r in analysis.EXAMPLE_TARGETS]
        result = SimpleNamespace(targets=targets, sources={"A1": source},
                                 noise={"ce": (frequency, np.full(40, 1e-23))})
        with plt.rc_context({"text.usetex": False, "font.family": "serif"}), \
                mock.patch.object(plots, "savefig") as save:
            plots._plot_characteristic_strain(result, SimpleNamespace())
            fig = save.call_args.args[0]
            try:
                self.assertEqual(len(fig.axes), 1)
                ax = fig.axes[0]
                self.assertEqual(len(ax.lines), 4)  # Three signals, one noise curve.
                self.assertEqual(len(ax.texts), 3)  # One mass/distance pair per group.
                self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()], ["A1"])
                fig.canvas.draw()
            finally:
                plt.close(fig)

    def test_curved_detector_label_is_finite_and_preserves_limits(self):
        import matplotlib.pyplot as plt
        from matplotlib.colors import to_rgba
        from wip_plots import gw_detectability_all as plots

        with plt.rc_context({"text.usetex": False, "font.family": "serif"}):
            fig, ax = plt.subplots()
            try:
                frequency = np.geomspace(.01, 10., 500)
                noise = 1e-23 * (frequency + 1 / frequency)
                ax.loglog(frequency, noise)
                limits = ax.get_xlim(), ax.get_ylim()
                plots._label_noise_curve(ax, frequency, noise, "decigo")
                patch = ax.patches[-1]
                self.assertTrue(np.all(np.isfinite(patch.get_path().vertices)))
                self.assertGreater(len(patch.get_path().vertices), 100)
                self.assertEqual(patch.get_facecolor(), to_rgba(plots.DETECTOR_COLOR))
                fig.canvas.draw()
                self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
            finally:
                plt.close(fig)

    def test_horizon_redshift_axis_tracks_distance_and_keeps_data_unchanged(self):
        import matplotlib.pyplot as plt
        from helpers.style import PAPER_PLOT_STYLE
        from wip_plots import gw_detectability_all as plots

        masses = np.geomspace(1., 1e7, 12)
        distances = np.geomspace(.05, 1000., 12)
        source = SimpleNamespace(color="blue", linestyle="-", legend_name="A1")
        result = SimpleNamespace(horizons=(masses, {"ce": {"A1": distances}}),
                                 sources={"A1": source})
        cosmology = FlatLambdaCDM()
        style = {**PAPER_PLOT_STYLE, "text.usetex": False, "font.family": "DejaVu Serif"}
        for filename, note in ((plots.OUTPUT_FILENAME_HORIZON, None),
                               (plots.OUTPUT_FILENAME_HORIZON_TIME_ON, "A1: conditional tail")):
            with self.subTest(filename=filename), plt.rc_context(style), \
                    mock.patch.object(plots, "savefig") as save:
                plots._plot_horizon(result, SimpleNamespace(), filename=filename, note=note)
                fig = save.call_args.args[0]
                try:
                    ax = fig.axes[0]
                    self.assertEqual(len(ax.child_axes), 1)
                    secondary = ax.child_axes[0]
                    self.assertEqual(secondary.get_ylabel(), r"$z$")
                    self.assertEqual(secondary.yaxis.label.get_fontsize(),
                                     ax.yaxis.label.get_fontsize())
                    np.testing.assert_array_equal(ax.lines[0].get_ydata(), distances)
                    self.assertEqual(ax.get_ylim(), plots.HORIZON_YLIM)
                    for limits in (plots.HORIZON_YLIM, (.1, 5e3)):
                        ax.set_ylim(*limits)
                        fig.canvas.draw()
                        np.testing.assert_allclose(secondary.get_ylim(),
                                                   cosmology.redshift_at_luminosity_distance(limits))
                        redshifts = np.array([1e-4, .01, .1])
                        primary_y = ax.transData.transform(np.column_stack(
                            (np.ones(3), cosmology.luminosity_distance_mpc(redshifts))))[:, 1]
                        secondary_y = secondary.transData.transform(
                            np.column_stack((np.ones(3), redshifts)))[:, 1]
                        np.testing.assert_allclose(secondary_y, primary_y, atol=1e-8)
                    self.assertFalse(any(t.tick2line.get_visible()
                                         for t in ax.yaxis.get_major_ticks()+ax.yaxis.get_minor_ticks()))
                    self.assertEqual(secondary.yaxis.get_minorticklocs().size, 0)
                    self.assertFalse(any(t.gridline.get_visible() for t in secondary.yaxis.get_major_ticks()))
                    renderer = fig.canvas.get_renderer()
                    label_box = secondary.yaxis.label.get_window_extent(renderer)
                    self.assertLessEqual(label_box.x1, fig.bbox.x1)
                    for label in secondary.get_yticklabels():
                        if label.get_visible():
                            self.assertLessEqual(label.get_window_extent(renderer).x1, label_box.x0)
                finally:
                    plt.close(fig)

    def test_fixed_source_preserves_mass_distance_and_uses_same_conversion(self):
        spectrum = DimensionlessSpectrum(np.geomspace(.001, .1, 100), np.ones(100), "mean")
        z = FlatLambdaCDM().redshift_at_luminosity_distance(100.)
        curves = analysis.load_detector_curves()
        target = analysis.observed_target({"A1": spectrum, "B1": spectrum}, 50., 100., z, curves)
        self.assertEqual((target.mass, target.distance, target.detector), (50., 100., None))
        frequency, strain_ft = observer_spectrum(spectrum, 50., 100., z)
        for case in ("A1", "B1"):
            np.testing.assert_array_equal(target.strain[case][0], frequency)
            np.testing.assert_array_equal(target.strain[case][1], characteristic_strain(frequency, strain_ft))
        self.assertEqual(target.snr["A1"], target.snr["B1"])

    def test_reference_distance_is_refined_to_snr_threshold(self):
        cosmology = FlatLambdaCDM()
        with mock.patch.object(analysis, "_snr_from_binned_power",
                               side_effect=lambda power, detector, mass, distance, curves: 16 / distance):
            z = analysis._threshold_redshift(None, "ce", 20., {}, cosmology, refine=True)
        self.assertAlmostEqual(cosmology.luminosity_distance_mpc(z), 2., places=8)
        with mock.patch.object(analysis, "_snr_from_binned_power", return_value=0.):
            self.assertTrue(np.isnan(analysis._threshold_redshift(None, "ce", 20., {}, cosmology)))
        with mock.patch.object(analysis, "_snr_from_binned_power", return_value=100.):
            with self.assertRaisesRegex(ValueError, "exceeds"):
                analysis._threshold_redshift(None, "ce", 20., {}, cosmology, refine=True)

    def test_prearrival_only_data_is_rejected(self):
        time = np.linspace(-100, -1, 100)
        with self.assertRaisesRegex(ValueError, "pre-arrival cut"):
            direct_psi4_spectrum(time, {(2, 2): np.ones(100)}, 1., transient_cutoff_mbh=0.)

    def test_cache_reader_uses_own_time_and_massless_file_offset(self):
        for name, expected_label in (("A1", "9"), ("ML", "8")):
            case = config.all_sim_configs([name])[0]
            with tempfile.TemporaryDirectory() as tmp, mock.patch("gw.GW_WORK_ROOT", Path(tmp)):
                directory = Path(tmp) / f"{name}_psi4{expected_label}"
                directory.mkdir()
                table = np.arange(8 * 43, dtype=float).reshape(8, 43)
                table[:, 0] = np.arange(8) / 8
                np.savetxt(directory / "rpsi4_uniform.dat", table)
                (directory / "strain_cache.json").write_text(json.dumps({
                    "backend": "python", "source_label": expected_label, "madm": case.gw_madm,
                }))
                time, modes = analysis.read_rpsi4_modes(case, 8)
                np.testing.assert_array_equal(time, table[:, 0])
                self.assertEqual(tuple(modes), ((2, 2), (2, 1), (2, -2), (2, -1)))
                np.testing.assert_array_equal(modes[(2, -2)], table[:, 9] + 1j * table[:, 10])
                with mock.patch.object(analysis, "MODES", "all"):
                    all_time, all_modes = analysis.read_rpsi4_modes(case, 8)
                np.testing.assert_array_equal(all_time, time)
                self.assertEqual(tuple(all_modes), tuple(analysis.CACHED_MODES))
                self.assertEqual(len(all_modes), 21)
                np.testing.assert_array_equal(all_modes[(4, -4)], table[:, 41] + 1j * table[:, 42])
                for mode in modes:
                    np.testing.assert_array_equal(all_modes[mode], modes[mode])
                with self.assertRaisesRegex(ValueError, "MODES must"):
                    analysis.read_rpsi4_modes(case, 8, modes="typo")
                # No raw Psi4, h+, flux or guessed retarded-time table was needed.
                (directory / "rpsi4_uniform.dat").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "generate_gw.py"):
                    analysis.read_rpsi4_modes(case, 8)

    def test_cache_reader_drops_trailing_zero_fill(self):
        case = config.all_sim_configs(["A1"])[0]
        with tempfile.TemporaryDirectory() as tmp, mock.patch("gw.GW_WORK_ROOT", Path(tmp)):
            directory = Path(tmp) / "A1_psi49"
            directory.mkdir()
            table = np.ones((12, 43))
            table[:, 0] = np.arange(12.0)
            table[-3:, 1:] = 0.0  # Uniform grid past the last source sample.
            np.savetxt(directory / "rpsi4_uniform.dat", table)
            (directory / "strain_cache.json").write_text(json.dumps({
                "backend": "python", "source_label": "9", "madm": case.gw_madm,
            }))
            time, modes = analysis.read_rpsi4_modes(case, 8)
            np.testing.assert_array_equal(time, np.arange(9.0))
            self.assertTrue(all(np.all(values != 0) for values in modes.values()))

    def test_analytic_mode_from_cache_through_observed_snr(self):
        # q_22 = q0 exp(2 pi i nu0 tau), so M_BH*rPsi4 = q_22''.
        # Only this mode is present in a real-format 21-mode cache. The oracle
        # uses the analytic harmonic, direct DFT sums, and a flat raw PSD.
        case = config.all_sim_configs(["A1"])[0]
        retained = 1048
        tau = np.arange(-16, 2048, dtype=float)
        nu0 = 32.0 / retained
        q0 = 0.01
        omega0 = 2.0 * np.pi * nu0
        rpsi4 = -(omega0**2 * q0 / case.mlittle) * np.exp(1j * omega0 * tau)
        table = np.zeros((tau.size, 43))
        table[:, 0] = case.mlittle * tau
        table[:, 1], table[:, 2] = rpsi4.real, rpsi4.imag

        with tempfile.TemporaryDirectory() as tmp, mock.patch("gw.GW_WORK_ROOT", Path(tmp)):
            directory = Path(tmp) / "A1_psi49"
            directory.mkdir()
            np.savetxt(directory / "rpsi4_uniform.dat", table)
            (directory / "strain_cache.json").write_text(json.dumps({
                "backend": "python", "source_label": "9", "madm": case.gw_madm,
            }))
            sources, _ = analysis.source_spectra(["A1"])

        self.assertEqual(tuple(sources), ("A1",))
        source = sources["A1"]
        self.assertEqual(source.info.samples, retained)
        self.assertEqual(source.spectrum.averaging, "source-direction mean")
        self.assertEqual(source.info.modes, ((2, -2), (2, -1), (2, 1), (2, 2)))

        # The arrival and transient cuts leave tau=1000,...,2047. The
        # symmetric padding doubles the FFT grid, without changing duration.
        nu = np.arange(1, retained) / (2.0 * retained)
        nu = nu[nu >= 3.0 / (retained - 1)]
        np.testing.assert_allclose(source.spectrum.frequency, nu, rtol=1e-12)
        relative_tau = np.arange(retained, dtype=float)
        window = scipy_tukey(retained, alpha=0.05, sym=True)
        positive = np.exp(-2j * np.pi * (nu[:, None] - nu0) * relative_tau) @ window
        negative = np.exp(-2j * np.pi * (nu[:, None] + nu0) * relative_tau) @ window
        # |_{-2}Y_22| = sqrt(5/(64 pi)) (1+cos(theta))^2; its solid-angle
        # mean is 4/3 times sqrt(5/(64 pi)). The plus/cross norm is phase
        # independent, including the finite-window negative-frequency wing.
        mean_harmonic = (4.0 / 3.0) * np.sqrt(5.0 / (64.0 * np.pi))
        expected_q_ft = (
            mean_harmonic * omega0**2 * q0
            * np.sqrt(np.abs(positive)**2 + np.abs(negative)**2)
            / (2.0 * (2.0 * np.pi * nu)**2)
        )
        np.testing.assert_allclose(source.spectrum.strain_ft, expected_q_ft,
                                   rtol=1e-8, atol=1e-12 * expected_q_ft.max())

        mass_msun, distance_mpc, redshift = 50.0, 100.0, 0.1
        raw_asd = 1.0e-23
        curves = {"ligo": DetectorCurve(np.array([5.0, 2500.0]),
                                        np.array([raw_asd, raw_asd]))}
        with mock.patch.object(analysis, "ACTIVE_DETECTORS", ("ligo",)):
            target = analysis.observed_target({"A1": source.spectrum},
                                              mass_msun, distance_mpc, redshift, curves)

        # Independent SI conversion and the <F_+^2+F_x^2> = 2/5 response
        # for the polarization norm used above.
        seconds_per_solar_mass = 4.92549095e-6
        meters_per_solar_mass = 1476.6250385
        meters_per_mpc = 3.085677581491367e22
        observer_time = mass_msun * (1.0 + redshift) * seconds_per_solar_mass
        amplitude = mass_msun * (1.0 + redshift) * meters_per_solar_mass / (
            distance_mpc * meters_per_mpc
        )
        observed_frequency = nu / observer_time
        observed_ft = expected_q_ft * amplitude * observer_time
        self.assertGreater(observed_frequency.min(), 5.0)
        self.assertLess(observed_frequency.max(), 2500.0)
        frequency, characteristic = target.strain["A1"]
        np.testing.assert_allclose(frequency, observed_frequency, rtol=1e-12)
        np.testing.assert_allclose(characteristic, 2.0 * observed_frequency * observed_ft,
                                   rtol=1e-8, atol=1e-12 * characteristic.max())
        df = np.diff(observed_frequency)
        power_integral = np.sum(
            0.5 * (observed_ft[:-1]**2 + observed_ft[1:]**2) * df
        )
        expected_snr = np.sqrt(
            (8.0 / 5.0) * power_integral / raw_asd**2
        )
        self.assertAlmostEqual(target.snr["A1"]["ligo"], expected_snr,
                               delta=1e-8 * expected_snr)

    def test_moore_hc_noise_snr_identity(self):
        f = np.geomspace(1, 100, 20000)
        h = np.exp(-f / 20) + 0j
        psd = 2 + f**2
        hc = characteristic_strain(f, h)
        hn = np.sqrt(f * psd)
        snr = analysis.cumulative_snr(f, h, psd, f[0], f[-1])[-1]
        log_integrand = (hc / hn)**2
        log_integral = np.sum((log_integrand[:-1] + log_integrand[1:]) / 2 * np.diff(np.log(f)))
        self.assertAlmostEqual(snr**2, log_integral, delta=1e-7 * log_integral)

    def test_four_mode_fft_matches_reconstruct_then_fft(self):
        # Independently follow the human proposal in the time domain first.
        tau = np.arange(256, dtype=float) / 256
        modes = {
            mode: np.exp(2j * np.pi * (8 + k) * tau) / (k + 1)
            + 0.2 * np.exp(-2j * np.pi * (3 + k) * tau)
            for k, mode in enumerate(((2, 2), (2, 1), (2, -2), (2, -1)))
        }
        cos_theta, weights = np.polynomial.legendre.leggauss(6)
        window = tukey_window(len(tau), 0.1)
        for average in ("mean", "rms"):
            spectrum, _ = direct_psi4_spectrum(
                tau, modes, 1.0, transient_cutoff_mbh=0.0, taper_alpha=0.1,
                zero_pad_factor=2.0, low_frequency_cycles=0.0,
                theta_nodes=6, phi_nodes=12, averaging=average,
            )
            expected = np.zeros_like(spectrum.frequency)
            indices = np.rint(spectrum.frequency * 2).astype(int)
            for cosine, weight in zip(cos_theta, weights):
                for phi in 2 * np.pi * np.arange(12) / 12:
                    field = sum(
                        spin_weighted_spherical_harmonic(-2, *mode, np.arccos(cosine), phi) * values
                        for mode, values in modes.items()
                    )
                    real_fft = np.fft.rfft(np.pad(field.real * window, (128, 128))) / 256
                    imag_fft = np.fft.rfft(np.pad(field.imag * window, (128, 128))) / 256
                    amplitude = np.sqrt((abs(real_fft[indices])**2 + abs(imag_fft[indices])**2) / 2)
                    amplitude /= (2 * np.pi * spectrum.frequency)**2
                    expected += (amplitude if average == "mean" else amplitude**2) * weight / 24
            if average == "rms":
                expected = np.sqrt(expected)
            np.testing.assert_allclose(spectrum.strain_ft, expected, rtol=1e-11, atol=1e-17)

    def test_detector_snr_distance_scaling_and_zero_signal(self):
        frequency = np.linspace(0.001, 0.1, 100)
        spectrum = DimensionlessSpectrum(frequency, np.ones(100), "mean")
        binned = _bin_spectral_power(spectrum, 128)
        curves = {"ligo": DetectorCurve(np.array([5., 2500.]), np.array([1e-23, 1e-23]))}
        snr = _snr_from_binned_power(binned, "ligo", 10., 100., curves)
        self.assertGreater(snr, 0)
        self.assertAlmostEqual(_snr_from_binned_power(binned, "ligo", 10., 200., curves), snr / 2)
        empty = _bin_spectral_power(DimensionlessSpectrum(frequency, np.zeros(100), "mean"), 128)
        self.assertEqual(_snr_from_binned_power(empty, "ligo", 10., 100., curves), 0.)

    def test_single_mode_recovers_double_integrated_periodic_signal(self):
        samples = 2048
        cycles = 16
        tau = np.arange(samples, dtype=float) / samples
        frequency = float(cycles)
        q = np.sin(2.0 * np.pi * frequency * tau)
        p = -(2.0 * np.pi * frequency) ** 2 * q

        spectrum, info = direct_psi4_spectrum(
            tau,
            {(2, 2): p.astype(complex)},
            1.0,
            transient_cutoff_mbh=0.0,
            taper_alpha=0.0,
            zero_pad_factor=1.0,
            low_frequency_cycles=0.0,
            representative_direction=(np.pi / 2.34, 0.0),
        )

        peak = int(np.argmin(np.abs(spectrum.frequency - frequency)))
        harmonic = spin_weighted_spherical_harmonic(-2, 2, 2, np.pi / 2.34, 0.0)
        expected_q_ft = 0.5 * (tau[-1] - tau[0] + info.dt_mbh)
        expected = abs(harmonic) * expected_q_ft / np.sqrt(2.0)
        self.assertAlmostEqual(spectrum.frequency[peak], frequency, places=12)
        self.assertAlmostEqual(spectrum.strain_ft[peak], expected, delta=2.0e-3 * expected)

    def test_source_direction_quadrature_is_normalized(self):
        samples = 1024
        cycles = 8
        tau = np.arange(samples, dtype=float) / samples
        q = np.sin(2.0 * np.pi * cycles * tau)
        p = -(2.0 * np.pi * cycles) ** 2 * q
        spectrum, _ = direct_psi4_spectrum(
            tau,
            {(2, 2): p.astype(complex)},
            1.0,
            transient_cutoff_mbh=0.0,
            taper_alpha=0.0,
            zero_pad_factor=1.0,
            low_frequency_cycles=0.0,
            theta_nodes=20,
            phi_nodes=32,
            averaging="mean",
        )
        peak = int(np.argmin(np.abs(spectrum.frequency - cycles)))

        cos_theta, weights = np.polynomial.legendre.leggauss(80)
        theta = np.arccos(cos_theta)
        mean_abs_harmonic = 0.5 * np.sum(
            weights
            * np.array(
                [abs(spin_weighted_spherical_harmonic(-2, 2, 2, th, 0.0)) for th in theta]
            )
        )
        expected = mean_abs_harmonic * 0.5 / np.sqrt(2.0)
        self.assertAlmostEqual(spectrum.strain_ft[peak], expected, delta=2.0e-3 * expected)

    def test_observer_scaling_preserves_characteristic_strain_mass_law(self):
        samples = 1024
        cycles = 8
        tau = np.arange(samples, dtype=float) / samples
        q = np.sin(2.0 * np.pi * cycles * tau)
        p = -(2.0 * np.pi * cycles) ** 2 * q
        spectrum, _ = direct_psi4_spectrum(
            tau,
            {(2, 2): p.astype(complex)},
            1.0,
            transient_cutoff_mbh=0.0,
            taper_alpha=0.0,
            zero_pad_factor=1.0,
            low_frequency_cycles=0.0,
            representative_direction=(np.pi / 2.34, 0.0),
        )
        f1, h1 = observer_spectrum(spectrum, 10.0, 100.0, redshift=0.0)
        f2, h2 = observer_spectrum(spectrum, 20.0, 100.0, redshift=0.0)
        np.testing.assert_allclose(f2, 0.5 * f1)
        np.testing.assert_allclose(characteristic_strain(f2, h2), 2.0 * characteristic_strain(f1, h1))

    def test_flat_cosmology_round_trip(self):
        cosmology = FlatLambdaCDM()
        redshift = np.array([0.01, 0.5, 2.0, 6.0])
        distance = cosmology.luminosity_distance_mpc(redshift)
        np.testing.assert_allclose(cosmology.redshift_at_luminosity_distance(distance), redshift, rtol=2e-6)



if __name__ == "__main__":
    unittest.main()
