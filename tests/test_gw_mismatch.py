"""Numerical contracts for the optional coherent (2,2) mismatch diagnostic."""
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from gw_mismatch import ManualOverlap, compare, prepare_pair
import run_wip


class TestManualMismatch(unittest.TestCase):
    def setUp(self):
        self.dt = .5
        self.t = np.arange(1024)*self.dt
        self.h = np.exp(-((self.t-180)/35)**2)*np.exp(-2j*np.pi*(.04*self.t+.0001*self.t**2))

    def metric(self, other, psd=None):
        return ManualOverlap(self.h, other, self.dt, .01, .5, psd)

    def test_identical_scaled_rotated(self):
        result, _ = self.metric(7*self.h*np.exp(.8j)).optimum(40.)
        self.assertAlmostEqual(result["mismatch"], 0., places=12)
        self.assertAlmostEqual(result["time_shift_MBH"], 0., places=5)
        self.assertAlmostEqual(result["rotation_rad"], -.8, places=5)

    def test_known_delay_recovered_without_wrap(self):
        shifted = np.r_[np.zeros(20), self.h[:-20]]*np.exp(.6j)
        result, _ = self.metric(shifted).optimum(40.)
        self.assertAlmostEqual(result["mismatch"], 0., places=10)
        self.assertAlmostEqual(result["time_shift_MBH"], -10., places=4)
        self.assertAlmostEqual(result["rotation_rad"], -.6, places=4)

    def test_orthogonal_frequency_bins(self):
        t = np.arange(1024)
        a = np.exp(-2j*np.pi*10*t/1024)
        b = np.exp(-2j*np.pi*30*t/1024)
        metric = ManualOverlap(a, b, 1., .001, .4)
        result, _ = metric.optimum(100.)
        self.assertAlmostEqual(result["mismatch"], 1., places=12)

    def test_psd_scale_invariance_and_symmetry(self):
        other = self.h*np.exp(-.001j*(self.t-180)**2)
        noise = lambda f: 1+50*f*f
        a, _ = self.metric(other, noise).optimum(40.)
        b, _ = self.metric(other, lambda f: 9*noise(f)).optimum(40.)
        c, _ = ManualOverlap(other, self.h, self.dt, .01, .5, noise).optimum(40.)
        self.assertAlmostEqual(a["mismatch"], b["mismatch"], places=12)
        self.assertAlmostEqual(a["mismatch"], c["mismatch"], places=10)

    def test_common_interval_and_padding(self):
        time, a, b, meta = prepare_pair(self.t, self.h, self.t[:800], self.h[:800], start=40.)
        self.assertEqual(meta["end_MBH"], self.t[799])
        self.assertGreaterEqual(len(time), 2*meta["retained_samples"]-1)
        np.testing.assert_allclose(a, b)
        self.assertTrue(np.all(a[meta["retained_samples"]:] == 0))

    def test_bad_inputs_fail(self):
        with self.assertRaises(ValueError):
            self.metric(np.zeros_like(self.h))
        with self.assertRaises(ValueError):
            self.metric(self.h, lambda f: -np.ones_like(f))
        with self.assertRaises(ValueError):
            prepare_pair(self.t, self.h, self.t, self.h, start=self.t[-1])


@unittest.skipUnless(importlib.util.find_spec("kuibit"), "optional kuibit not installed")
class TestKuibitAgreement(unittest.TestCase):
    setUp = TestManualMismatch.setUp
    def test_public_api_white_and_colored_noise(self):
        other = self.h*np.exp(.0004j*(self.t-180)**2+.7j)
        for psd in (None, lambda f: 1+100*f*f):
            with self.subTest(weighted=psd is not None):
                result, _ = compare(self.t, self.h, other, .01, .5, 25., psd, ntime=33, nphase=32)
                for run in result["grid_checks"]:
                    self.assertLess(run["absolute_difference"], 1e-12)
                self.assertLess(result["kuibit_minus_manual_optimized"], .001)

    def test_identical_public_api(self):
        result, _ = compare(self.t, self.h, self.h, .01, .5, 20., ntime=17, nphase=16)
        self.assertAlmostEqual(result["grid_checks"][-1]["kuibit"]["mismatch"], 0., places=12)


class TestOptionalRegistration(unittest.TestCase):
    def test_default_excludes_mismatch(self):
        with patch.object(run_wip, "import_module") as imported:
            run_wip.main([])
        self.assertNotIn("wip_plots.gw_mismatch", [c.args[0] for c in imported.call_args_list])

    def test_explicit_runs_mismatch(self):
        with patch.object(run_wip, "import_module") as imported:
            run_wip.main(["gw_mismatch"])
        imported.assert_called_once_with("wip_plots.gw_mismatch")
        imported.return_value.main.assert_called_once_with([])


if __name__ == "__main__":
    unittest.main()
