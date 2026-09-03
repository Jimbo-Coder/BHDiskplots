from pathlib import Path
import json
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from config import FORTRAN_GW_ROOT
from gw_psi4 import (
    N_PSI4_COLUMNS,
    Psi4File,
    convert_to_strain_with_python,
    convert_to_strain_with_rhphc,
    mode_order,
)
from helpers.gw_ffi import (
    _fixed_frequency_integrate,
    _four_point_interpolate,
    _padded_fft_size,
    reconstruct_strain,
)


class PythonFFITests(unittest.TestCase):
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
            result = convert_to_strain_with_python(
                psi4,
                workdir,
                omega_orbital=0.1,
                madm=1.0,
                reuse_existing=False,
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

            cached = convert_to_strain_with_python(
                psi4,
                workdir,
                omega_orbital=0.1,
                madm=1.0,
                reuse_existing=True,
                generate_if_missing=False,
            )
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
            python_result = convert_to_strain_with_python(
                psi4,
                root / "python",
                omega_orbital=0.1,
                madm=1.0,
                reuse_existing=False,
            )
            fortran_result = convert_to_strain_with_rhphc(
                psi4,
                root / "fortran",
                omega_orbital=0.1,
                madm=1.0,
                psi4_hlm_dir=root,
                executable=executable.name,
                reuse_existing=False,
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


if __name__ == "__main__":
    unittest.main()
