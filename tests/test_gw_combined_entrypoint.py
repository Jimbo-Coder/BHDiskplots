"""The single-figure command selects the requested cache/radius and fails clearly."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import config
import gw_detectability as analysis
import gw_detectability as command


class CombinedEntryPointTests(unittest.TestCase):
    def test_cache_override_selects_requested_file_without_raw_data(self):
        case = config.all_sim_configs(["A1"])[0]
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"A1_psi47"
            folder.mkdir()
            table = np.ones((16, 43))
            table[:, 0] = np.arange(16.)
            table[:, 1] = np.arange(16.)+5
            np.savetxt(folder/"rpsi4_uniform.dat", table)
            (folder/"strain_cache.json").write_text(json.dumps({
                "backend": "python", "source_label": "7", "madm": case.gw_madm}))
            time, modes = analysis.read_rpsi4_modes(case, 6, cache_root=tmp)
            np.testing.assert_array_equal(time, table[:, 0])
            np.testing.assert_array_equal(modes[2, 2], table[:, 1]+1j*table[:, 2])
            with self.assertRaises(FileNotFoundError):
                analysis.read_rpsi4_modes(case, 7, cache_root=tmp)

    def test_strict_mode_stops_instead_of_omitting_missing_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "A1.*Missing"):
                analysis.analyze(["A1"], radius_index=6, cache_root=tmp)

    def test_radius_and_cache_reach_analysis_without_extra_diagnostics(self):
        spectrum = analysis.DimensionlessSpectrum(np.geomspace(.001, .1, 100), np.ones(100), "mean")
        with mock.patch.object(analysis, "read_rpsi4_modes", return_value=(np.arange(10.), {})) as read, \
                mock.patch.object(analysis, "direct_psi4_spectrum", return_value=(spectrum, SimpleNamespace(samples=10,duration_mbh=9.))), \
                mock.patch.object(analysis, "horizon_curves", return_value="reach") as reaches, \
                mock.patch.object(analysis, "EXAMPLE_TARGETS", (("ce", 50., 100.),)):
            result = analysis.analyze(["A1"], radius_index=6, cache_root="copied_cache")
        self.assertEqual(read.call_count, 1)
        self.assertEqual(read.call_args.args[0].name, "A1")
        self.assertEqual(read.call_args.args[1], 6)
        self.assertEqual(read.call_args.kwargs["cache_root"], "copied_cache")
        reaches.assert_called_once()
        self.assertEqual(result.horizons, "reach")
        self.assertEqual(result.inner, {})

    def test_missing_cache_is_reported_before_tex_or_analysis(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(command, "apply_paper_style") as style, \
                mock.patch.object(command, "analyze") as calculate, \
                contextlib.redirect_stderr(io.StringIO()) as error, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as stopped:
                command.main(["--cache-dir", tmp, "--radius-index", "6"])
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("A1_psi47/rpsi4_uniform.dat", error.getvalue())
        self.assertIn("not included in GitHub", error.getvalue())
        calculate.assert_not_called()
        style.assert_not_called()

    def test_direct_command_plots_only_selected_radius(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in command.SIM_NAMES:
                folder = Path(tmp)/f"{name}_psi47"
                folder.mkdir()
                for filename in ("rpsi4_uniform.dat", "strain_cache.json"):
                    (folder/filename).touch()
            result = SimpleNamespace(sources={})
            with mock.patch.object(command, "analyze", return_value=result) as calculate, \
                    mock.patch.object(command, "apply_paper_style"), \
                    mock.patch.object(command, "plot_combined") as draw, \
                    contextlib.redirect_stdout(io.StringIO()):
                command.main(["--cache-dir", tmp, "--radius-index", "6", "--no-save"])
            calculate.assert_called_once_with(list(command.SIM_NAMES), radius_index=6,
                                              cache_root=Path(tmp),
                                              compare_radius=False, compare_direction=False,
                                              compute_horizons=True, include_reference=False)
            self.assertEqual(draw.call_args.kwargs["filename"], "gw/detectability/radius_6.png")
            self.assertEqual(draw.call_count, 1)
            self.assertEqual(analysis.OUTER_RADIUS_INDEX, 8)


if __name__ == "__main__":
    unittest.main()
