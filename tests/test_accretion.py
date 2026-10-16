"""Accretion sign, mass budget, and dimensional rescaling contracts."""
from dataclasses import replace
import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import matplotlib.pyplot as plt

from config import DISK_SIMS
from helpers.time_units import MILLISECONDS_PER_SOLAR_MASS
from wip_plots.accretion import accretion_history, late_mass_budget, plot_mass_budget


class AccretionTests(unittest.TestCase):
    def test_late_budget_rebases_every_quantity_on_common_interval(self):
        import gw_detectability as analysis
        history = dict(time_pc=np.array([0., 1., 2., 3.]),
                       bh_time_pc=np.array([.1, 1., 2., 2.5]),
                       mass_loss=np.array([0., 2., 4., 6.]),
                       recorded_accreted_mass=np.array([1., 4., 7., 10.]),
                       bh_mass_gain=np.array([.1, 1., 2., 2.5]))
        sim = SimpleNamespace(config=self.config)
        trial = SimpleNamespace(fit=SimpleNamespace(start=20., end=80.))
        with mock.patch.object(analysis, "temporal_trial_for_case", return_value=trial):
            late = late_mass_budget(sim, history)
        np.testing.assert_allclose(late["time_pc"], [.5, 1., 2.])
        np.testing.assert_allclose(late["mass_loss"], [0., 1., 3.])
        np.testing.assert_allclose(late["recorded_accreted_mass"], [0., 1.5, 4.5])
        np.testing.assert_allclose(late["bh_mass_gain"], [0., .5, 1.5])
        self.assertEqual(history["recorded_accreted_mass"][0], 1.)
        trial.fit.end = 120.
        with mock.patch.object(analysis, "temporal_trial_for_case", return_value=trial), \
                self.assertRaisesRegex(ValueError, "cover"):
            late_mass_budget(sim, history)

    def setUp(self):
        self.config = replace(DISK_SIMS[0], Pc=2, mlittle=0.05, disk_to_bh_mass_ratio=1)
        self.data = np.zeros((4, 14))
        self.data[:, 0] = [0, 1, 3, 5]
        self.data[:, 1] = 0.05 - 0.001 * self.data[:, 0]
        self.data[:, 11] = -0.001
        self.data[:, 12] = 0.001 * self.data[:, 0]

    def test_flux_matches_mass_loss_on_nonuniform_time_grid(self):
        h = accretion_history(self.data, self.config)
        np.testing.assert_allclose(h["time_pc"], [0, 0.5, 1.5, 2.5])
        np.testing.assert_allclose(h["mass_loss"], [0, .001, .003, .005])
        np.testing.assert_allclose(h["accreted_mass"], h["mass_loss"])
        np.testing.assert_allclose(h["recorded_accreted_mass"], h["accreted_mass"])

    def test_budget_plot_uses_raw_mass_not_independent_normalization(self):
        h = accretion_history(self.data, self.config)
        h["bh_time_pc"] = h["time_pc"]
        h["bh_mass_gain"] = .7*h["recorded_accreted_mass"]
        sim = SimpleNamespace(config=replace(self.config, name="B1"))
        fig = plot_mass_budget([sim], {"B1": h}, late=True)
        try:
            ax = fig.axes[3]
            np.testing.assert_allclose(ax.lines[0].get_ydata(), h["mass_loss"])
            np.testing.assert_allclose(ax.lines[1].get_ydata(), self.data[:, 12])
            np.testing.assert_allclose(ax.lines[2].get_ydata(), .7*self.data[:, 12])
            self.assertEqual(len(fig.axes), 6)
            self.assertIn(r"M_\odot", fig._supylabel.get_text())
        finally:
            plt.close(fig)

    def test_outside_mass_subtracts_inside_ah(self):
        self.data[:, 1] = 0.05
        self.data[:, 2] = 0.001 * self.data[:, 0]
        h = accretion_history(self.data, self.config)
        np.testing.assert_allclose(h["mass_loss"], [0, .001, .003, .005])

    def test_outward_flux_remains_negative(self):
        self.data[:, 11] *= -1
        h = accretion_history(self.data, self.config)
        self.assertTrue(np.all(h["rate_per_second"] < 0))
        np.testing.assert_allclose(h["accreted_mass"], [0, -.001, -.003, -.005])

    def test_physical_rate_uses_bh_not_adm_mass(self):
        h = accretion_history(self.data, self.config, source_bh_mass=10)
        seconds = 10 / 0.05 * MILLISECONDS_PER_SOLAR_MASS / 1000
        np.testing.assert_allclose(h["rate_per_second"], 0.001 / 0.05 / seconds)
        larger = accretion_history(self.data, self.config, source_bh_mass=100)
        np.testing.assert_allclose(larger["rate_per_second"], h["rate_per_second"] / 10)
        other_adm = accretion_history(self.data, replace(self.config, gw_madm=1e5))
        np.testing.assert_allclose(other_adm["rate_per_second"], h["rate_per_second"])

    def test_budget_and_recorded_integral_start_at_first_sample(self):
        self.data[:, 12] += 0.5
        h = accretion_history(self.data[1:], self.config)
        self.assertEqual(h["mass_loss"][0], 0)
        self.assertEqual(h["recorded_accreted_mass"][0], 0)
        self.assertEqual(h["accreted_mass"][0], 0)

    def test_missing_ah_pattern_is_a_plot_gap_not_replaced_data(self):
        self.data[1, 11] = 0
        self.data[1, 12] = self.data[0, 12]
        h = accretion_history(self.data, self.config)
        np.testing.assert_array_equal(h["unavailable_flux"], [False, True, False, False])
        self.assertEqual(h["rate_per_second"][1], 0)
        np.testing.assert_allclose(h["accreted_mass"], [0, .0005, .0015, .0035])

    def test_true_zero_with_regular_integral_is_not_marked_missing(self):
        self.data[1, 11] = 0
        self.data[:, 12] = [0, 0.0005, 0.0015, 0.0035]
        h = accretion_history(self.data, self.config)
        self.assertFalse(np.any(h["unavailable_flux"]))

    def test_multiple_missing_ah_samples_are_gaps_too(self):
        self.data[1:3, 11] = 0
        self.data[1:3, 12] = 0
        h = accretion_history(self.data, self.config)
        np.testing.assert_array_equal(h["unavailable_flux"], [False, True, True, False])

    def test_no_horizon_gap_inferred_without_matching_integral(self):
        self.data[1, 11] = 0
        self.data[1, 12] = 0
        self.data[2, 12] = 0.004
        h = accretion_history(self.data, self.config)
        self.assertFalse(np.any(h["unavailable_flux"]))

    def test_reject_invalid_data_instead_of_bridging_gaps(self):
        for col, row, value in ((0, 1, 0), (11, 1, np.nan), (1, 1, -1)):
            data = self.data.copy()
            data[row, col] = value
            with self.assertRaises(ValueError):
                accretion_history(data, self.config)
        with self.assertRaises(ValueError):
            accretion_history(self.data, self.config, source_bh_mass=0)


if __name__ == "__main__":
    unittest.main()
