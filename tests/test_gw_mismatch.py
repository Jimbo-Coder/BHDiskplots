"""Numerical contracts for the optional coherent (2,2) mismatch diagnostic."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from gw_mismatch import ManualOverlap, compare, prepare_pair
import gw_mismatch
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
        self.assertAlmostEqual(result["time_shift"], 0., places=5)
        self.assertAlmostEqual(result["rotation_rad"], -.8, places=5)

    def test_known_delay_recovered_without_wrap(self):
        shifted = np.r_[np.zeros(20), self.h[:-20]]*np.exp(.6j)
        result, _ = self.metric(shifted).optimum(40.)
        self.assertAlmostEqual(result["mismatch"], 0., places=10)
        self.assertAlmostEqual(result["time_shift"], -10., places=4)
        self.assertAlmostEqual(result["rotation_rad"], -.6, places=4)

    def test_orthogonal_frequency_bins(self):
        t = np.arange(1024)
        a = np.exp(-2j*np.pi*10*t/1024)
        b = np.exp(-2j*np.pi*30*t/1024)
        metric = ManualOverlap(a, b, 1., .001, .4)
        result, _ = metric.optimum(100.)
        self.assertAlmostEqual(result["mismatch"], 1., places=12)

    def test_common_time_and_amplitude_units_do_not_change_white_mismatch(self):
        other = self.h*np.exp(-.001j*(self.t-180)**2)
        original, _ = self.metric(other).optimum(40.)
        code_scale = .05
        native = ManualOverlap(code_scale*self.h, code_scale*other,
                               code_scale*self.dt, .01/code_scale, .5/code_scale)
        result, _ = native.optimum(40.*code_scale)
        self.assertAlmostEqual(original["mismatch"], result["mismatch"], places=12)
        self.assertAlmostEqual(result["time_shift"], code_scale*original["time_shift"], places=6)

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
        self.assertEqual(meta["end"], self.t[799])
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
        self.assertNotIn("gw_mismatch", [c.args[0] for c in imported.call_args_list])

    def test_explicit_runs_mismatch(self):
        with patch.object(run_wip, "import_module") as imported:
            run_wip.main(["gw_mismatch"])
        imported.assert_called_once_with("gw_mismatch")
        imported.return_value.main.assert_called_once_with([])


class TestSelectedPair(unittest.TestCase):
    def test_requested_order_and_output_names_are_preserved(self):
        """Config order must not silently swap the user's first/second signal."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            time = np.arange(256) * .5
            for name, amplitude in (("A1", 1.), ("B3", 3.)):
                folder = root / f"{name}_psi47"
                folder.mkdir()
                strain = amplitude * np.exp(-.3j * time)
                np.savetxt(folder / "rhphc.dat",
                           np.column_stack((time, strain.real, -strain.imag)))
                (folder / "strain_cache.json").write_text('{}')
            answer = {
                "manual_optimized": {"mismatch": .1, "near_search_boundary": False},
                "grid_checks": [{"kuibit": {"mismatch": .1}}],
                "kuibit_minus_manual_optimized": 0.,
            }
            with patch.object(gw_mismatch, "compare", return_value=(answer, (time, time))) as calculate, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = gw_mismatch.main([
                    "--sims", "B3", "A1", "--radius-index", "6",
                    "--cache-dir", str(root), "--outdir", str(root), "--no-plot"])
            self.assertEqual([row["case"] for row in result["inputs"]], ["B3", "A1"])
            first, second = calculate.call_args.args[1:3]
            np.testing.assert_allclose(first, 3 * second, atol=1e-14)
            output = root / "wip/gw_mismatch/white/B3_A1"
            stored = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(stored["manual_optimized"]["mismatch"], .1)
            with np.load(output.with_name('B3_A1_inputs.npz')) as arrays:
                np.testing.assert_array_equal(arrays['B3'], first)
                np.testing.assert_array_equal(arrays['A1'], second)

    def test_unknown_or_missing_simulation_fails_clearly(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "Unknown simulation.*Z1"):
                gw_mismatch.read_strain_pair(["Z1", "A1"], tmp, 6)
            with self.assertRaisesRegex(FileNotFoundError, "A1_psi47.*rhphc.dat"):
                gw_mismatch.read_strain_pair(["A1", "A3"], tmp, 6)


if __name__ == "__main__":
    unittest.main()
