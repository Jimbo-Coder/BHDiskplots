from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock
from dataclasses import replace

import numpy as np

from config import FORTRAN_GW_ROOT
from gw import (
    N_PSI4_COLUMNS,
    Psi4File,
    generate_python_strain,
    generate_fortran_strain,
    mode_order,
    GWRun,
    read_strain_cache,
    read_difference,
    subtract_waveforms,
)
from gw import (
    _fixed_frequency_integrate,
    _four_point_interpolate,
    _padded_fft_size,
    reconstruct_strain,
)


class PythonFFITests(unittest.TestCase):
    def test_generate_then_read_case_and_difference_without_touching_sources(self):
        import generate_gw
        from config import all_sim_configs

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            runs, sources = {}, {}
            for name, label, factor in (("A1", "5", 1.), ("ML", "4", .1)):
                directory = root / name
                directory.mkdir()
                source = directory / f"Psi4_rad.mon.{label}"
                rows = self.synthetic_psi4().data.copy()
                rows[:, 1:-4] *= factor
                np.savetxt(source, rows)
                sources[source] = source.read_bytes()
                cfg = replace(all_sim_configs([name])[0], data_roots=(directory,),
                              gw_omega_orbital=.1, gw_madm=1.)
                runs[name] = GWRun(cfg)
            with (
                mock.patch("gw.GW_WORK_ROOT", root / "cache"),
                mock.patch.object(generate_gw, "GW_REGENERATE_EXISTING", True),
                mock.patch.object(generate_gw, "GW_STRAIN_BACKEND", "python"),
            ):
                for run in runs.values():
                    self.assertEqual(generate_gw.generate_simulation(run), 1)
                self.assertEqual(generate_gw.generate_difference_radius(runs, 4), 1)
                result = read_difference(runs["A1"].at_radius(4), runs["ML"].at_radius(4))
                self.assertGreater(result.time.size, 4)
                self.assertTrue(np.isfinite(result.rhphc).all())
                with (
                    mock.patch.object(generate_gw, "GW_REGENERATE_EXISTING", False),
                    mock.patch.object(generate_gw, "generate_strain", side_effect=AssertionError("must reuse")),
                ):
                    self.assertEqual(generate_gw.generate_simulation(runs["A1"]), 1)
            for source, original in sources.items():
                self.assertEqual(source.read_bytes(), original)

    def test_radius_switch_reuses_files_and_selects_correct_cached_result(self):
        from config import all_sim_configs

        source = self.synthetic_psi4()
        files = {"5": replace(source, label="5"), "9": replace(source, label="9")}
        inner, outer = mock.Mock(), mock.Mock()
        with (
            mock.patch("gw.load_psi4", return_value=(files, {4: 120., 8: 170.})) as load,
            mock.patch("gw.read_strain_cache", side_effect=[inner, outer]) as read,
            mock.patch("gw.generate_strain", side_effect=AssertionError("plotting must not integrate")),
        ):
            run = GWRun(all_sim_configs(["A1"])[0])
            first = run.at_radius(4)
            second = run.at_radius(8)
            self.assertIs(first.strain_result, inner)
            self.assertIs(second.strain_result, outer)
            self.assertIs(run.at_radius(4), first)
            self.assertEqual(first.psi4_radius, 120.)
            self.assertEqual(first.parfile_index, 4)
            self.assertEqual(second.parfile_index, 8)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(read.call_count, 2)
        with (
            mock.patch("gw.load_psi4", return_value=({"8": replace(source, label="8")}, {7: 170.})),
            mock.patch("gw.read_strain_cache", return_value=outer),
        ):
            ml = GWRun(all_sim_configs(["ML"])[0]).at_radius(8)
        self.assertEqual(second.psi4.label, "9")
        self.assertEqual(ml.psi4.label, "8")

    def test_missing_cache_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp) / "missing"
            with self.assertRaises(FileNotFoundError):
                read_strain_cache(workdir, Path("source"))
            self.assertFalse(workdir.exists())

    def test_complex_difference_only_uses_shared_time_interval(self):
        time = np.arange(6.)
        result = subtract_waveforms(time, (2 + 3j) * time,
                                    [1.5, 3.5], [1.5 + 1.5j, 3.5 + 3.5j])
        np.testing.assert_array_equal(result[0], [2., 3.])
        np.testing.assert_allclose(result[1], [2 + 4j, 3 + 6j])
        self.assertIsNone(subtract_waveforms([0, 1], [0, 1], [3, 4], [3, 4]))

    @staticmethod
    def synthetic_psi4(sample_count=33):
        dt = 0.125
        time = dt * np.arange(sample_count)
        rows = np.zeros((sample_count, N_PSI4_COLUMNS), dtype=float)
        rows[:, 0] = time
        rows[:, -4] = 100.0 + 0.05 * np.sin(0.3 * time)
        lapse_factor = 1.0 - 2.0 / rows[:, -4]
        rows[:, -3] = -1.0
        rows[:, -2] = 0.0
        rows[:, -1] = lapse_factor**2
        envelope = np.sin(np.pi * np.arange(sample_count) / (sample_count - 1)) ** 2
        for mode_index in range(21):
            harmonic = 2 + mode_index % 5
            omega = 2.0 * np.pi * harmonic / (sample_count * dt)
            internal = -omega**2 * envelope * np.exp(1j * (omega * time + 0.03 * mode_index))
            raw = np.conj(internal) / rows[:, -4]
            rows[:, 1 + 2 * mode_index] = raw.real
            rows[:, 2 + 2 * mode_index] = raw.imag
        return Psi4File(Path("synthetic/Psi4_rad.mon.1"), "1", rows, 0)

    def test_padding_matches_legacy_fortran_policy(self):
        self.assertEqual(_padded_fft_size(64), 256)
        self.assertEqual(_padded_fft_size(65), 512)
        self.assertEqual(_padded_fft_size(127), 512)
        self.assertEqual(_padded_fft_size(128), 512)

    def test_four_point_interpolation_is_exact_for_cubic_data(self):
        source_time = np.linspace(-2.0, 3.0, 12)
        target_time = np.linspace(-1.9, 2.9, 41)
        first = source_time**3 - 2.0 * source_time + 1.0
        second = 0.5 * source_time**2 + 1j * (source_time**3 + 2.0)
        source = np.column_stack((first, second))
        result = _four_point_interpolate(source_time, source, target_time)
        expected = np.column_stack(
            (
                target_time**3 - 2.0 * target_time + 1.0,
                0.5 * target_time**2 + 1j * (target_time**3 + 2.0),
            )
        )
        np.testing.assert_allclose(result, expected, rtol=2.0e-13, atol=2.0e-13)

    def test_ffi_recovers_periodic_complex_strain_and_derivative(self):
        sample_count = 256
        dt = 0.125
        harmonic = 7
        omega = 2.0 * np.pi * harmonic / (sample_count * dt)
        time = dt * np.arange(sample_count)
        strain = np.exp(1j * omega * time)
        rpsi4 = (-omega**2 * strain)[:, None]

        recovered, recovered_dot = _fixed_frequency_integrate(
            rpsi4,
            dt,
            ((2, 2),),
            omega_orbital=0.1,
        )

        np.testing.assert_allclose(recovered[:, 0], strain, rtol=2.0e-13, atol=2.0e-13)
        np.testing.assert_allclose(recovered_dot[:, 0], 1j * omega * strain, rtol=2.0e-13, atol=2.0e-13)

    def test_python_backend_writes_reusable_legacy_cache(self):
        sample_count = 32
        dt = 0.25
        time = dt * np.arange(sample_count)
        rows = np.zeros((sample_count, N_PSI4_COLUMNS), dtype=float)
        rows[:, 0] = time
        rows[:, -4] = 100.0
        lapse_factor = 1.0 - 2.0 / rows[:, -4]
        rows[:, -3] = -1.0
        rows[:, -2] = 0.0
        rows[:, -1] = lapse_factor**2
        omega = 2.0 * np.pi * 2.0 / (sample_count * dt)
        internal_rpsi4 = -omega**2 * np.exp(1j * omega * time)
        raw_psi4 = np.conj(internal_rpsi4) / rows[:, -4]
        rows[:, 1] = raw_psi4.real
        rows[:, 2] = raw_psi4.imag
        psi4 = Psi4File(Path("source/Psi4_rad.mon.1"), "1", rows, 0)

        products = reconstruct_strain(rows, tuple(mode_order()), 0.1, 1.0)
        self.assertEqual(products.rhphc.shape, (sample_count, N_PSI4_COLUMNS - 4))
        self.assertEqual(products.rpsi4_uniform.shape, products.rhphc.shape)
        self.assertTrue(np.all(np.isfinite(products.rhphc)))

        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            result = generate_python_strain(
                psi4,
                workdir,
                omega_orbital=0.1,
                madm=1.0,
            )
            for filename in (
                "rhphc.dat",
                "rhphcdot.dat",
                "omega22.dat",
                "ejv_GW.dat",
                "times.dat",
                "rpsi4_uniform.dat",
                "strain_cache.json",
            ):
                self.assertTrue((workdir / filename).is_file(), filename)
            manifest = json.loads((workdir / "strain_cache.json").read_text())
            self.assertEqual(manifest["backend"], "python")
            self.assertEqual(manifest["source_rows"], sample_count)
            self.assertEqual(result.backend, "python")
            uniform_time, uniform_modes = result.rpsi4_modes(((2, 2), (2, 1)))
            np.testing.assert_allclose(uniform_time, result.time)
            np.testing.assert_allclose(uniform_modes[(2, 2)], result.rpsi4(2, 2))
            self.assertEqual(uniform_modes[(2, 1)].shape, result.time.shape)

            cached = read_strain_cache(workdir, psi4.path)
            np.testing.assert_allclose(cached.rhphc, result.rhphc)

    @unittest.skipUnless(shutil.which("gfortran"), "gfortran is not installed")
    def test_python_backend_matches_bounds_checked_fortran_reference(self):
        source = FORTRAN_GW_ROOT / "ccc_ffi_hplus_hcross_ejkick.f90"
        psi4 = self.synthetic_psi4()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            executable = root / "rhphc_reference"
            subprocess.run(
                [
                    shutil.which("gfortran"),
                    "-O0",
                    "-fcheck=all",
                    "-fbacktrace",
                    "-o",
                    str(executable),
                    str(source),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            python_result = generate_python_strain(
                psi4,
                root / "python",
                omega_orbital=0.1,
                madm=1.0,
            )
            fortran_result = generate_fortran_strain(
                psi4,
                root / "fortran",
                omega_orbital=0.1,
                madm=1.0,
                psi4_hlm_dir=root,
                executable=executable.name,
            )
            for name in ("rhphc", "rhphcdot", "omega22", "ejv_gw"):
                rtol = 5.0e-11 if name == "ejv_gw" else 2.0e-12
                atol = 2.0e-8 if name == "ejv_gw" else 2.0e-10
                np.testing.assert_allclose(
                    getattr(python_result, name),
                    getattr(fortran_result, name),
                    rtol=rtol,
                    atol=atol,
                    equal_nan=True,
                    err_msg=name,
                )


class ObserverStrainTests(unittest.TestCase):
    @staticmethod
    def waveform():
        from config import all_sim_configs
        from gw import StrainResult, mode_columns

        case = all_sim_configs(["A1"])[0]
        data = np.zeros((5, N_PSI4_COLUMNS))
        data[:, 0] = case.mlittle * np.arange(5.)
        re, im = mode_columns(2, 2)
        data[:, re] = case.mlittle * np.array([1., 2., 4., 2., 1.])
        data[:, im] = -0.5 * data[:, re]
        for mode in ((2, 1), (2, -1), (2, -2), (2, 0)):
            re, im = mode_columns(*mode)
            data[:, re:im + 1] = 1.e6  # Must not contribute at the north pole.
        return case, StrainResult(Path("cache"), Path("input"), data, None, None, None, "", "")

    def test_face_on_projection_and_physical_units(self):
        from gw_detectability import METERS_PER_M_SUN, METERS_PER_MPC, SECONDS_PER_M_SUN
        from wip_plots.gw_strain_observer import observer_waveform

        case, strain = self.waveform()
        time, hp, hc, peak = observer_waveform(case, strain, source_mass=50., distance=100.,
                                               min_tret_mbh=0., redshift=0.)
        amplitude = np.sqrt(5 / (4 * np.pi)) * 50 * METERS_PER_M_SUN / (100 * METERS_PER_MPC)
        np.testing.assert_allclose(hp, amplitude * np.array([1., 2., 4., 2., 1.]), atol=0)
        np.testing.assert_allclose(hc, -0.5 * hp, atol=0)
        np.testing.assert_allclose(time, (np.arange(5.) - 2) * 50 * SECONDS_PER_M_SUN, atol=1e-18)
        self.assertEqual(peak, 2.)

    def test_mass_distance_redshift_and_code_mass_scaling(self):
        from wip_plots.gw_strain_observer import observer_waveform

        case, strain = self.waveform()
        time, hp, _, _ = observer_waveform(case, strain, min_tret_mbh=0.)
        for kwargs, time_factor, amplitude_factor in (
            ({"source_mass": 100.}, 2., 2.), ({"distance": 200.}, 1., 0.5),
            ({"redshift": 1.}, 2., 2.),
        ):
            other = observer_waveform(case, strain, min_tret_mbh=0., **kwargs)
            np.testing.assert_allclose(other[0], time * time_factor, atol=1e-18)
            np.testing.assert_allclose(other[1], hp * amplitude_factor, atol=0)
        scaled = replace(strain, rhphc=2 * strain.rhphc)
        other = observer_waveform(replace(case, mlittle=2 * case.mlittle), scaled, min_tret_mbh=0.)
        np.testing.assert_allclose(other[0], time, atol=1e-18)
        np.testing.assert_allclose(other[1], hp, atol=0)

    def test_time_selection_does_not_extrapolate_or_force_endpoints_to_zero(self):
        from wip_plots.gw_strain_observer import observer_waveform

        case, strain = self.waveform()
        time, hp, _, peak = observer_waveform(case, strain, min_tret_mbh=1.5, align_peak=False)
        self.assertEqual(time.size, 3)
        self.assertGreater(time[0], 0.)
        self.assertGreater(hp[-1], 0.)
        self.assertEqual(peak, 2.)
        with self.assertRaises(ValueError):
            observer_waveform(case, strain, min_tret_mbh=10.)
        with self.assertRaises(ValueError):
            observer_waveform(case, strain, distance=0.)

    def test_all_mode_selection_uses_only_m_two_at_pole(self):
        from gw import mode_columns
        from wip_plots.gw_strain_observer import observer_waveform

        case, strain = self.waveform()
        subset = observer_waveform(case, strain, min_tret_mbh=0.)
        all_modes = observer_waveform(case, strain, modes="all", min_tret_mbh=0.)
        np.testing.assert_array_equal(all_modes[1], subset[1])
        re, _ = mode_columns(3, 2)
        strain.rhphc[:, re] = case.mlittle
        all_modes = observer_waveform(case, strain, modes="all", min_tret_mbh=0.)
        ratio = (all_modes[1] - subset[1]) / subset[1][0]
        np.testing.assert_allclose(ratio, np.sqrt(7 / 5))


if __name__ == "__main__":
    unittest.main()
