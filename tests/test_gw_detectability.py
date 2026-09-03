import unittest

import numpy as np

from helpers.gw_detectability import (
    FlatLambdaCDM,
    characteristic_strain,
    direct_psi4_spectrum,
    observer_spectrum,
    retarded_time_from_cache,
    spin_weighted_spherical_harmonic,
)


class DirectPsi4SpectrumTests(unittest.TestCase):
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

    def test_retarded_time_cache_prefix_is_extended_without_changing_cache(self):
        coordinate = np.linspace(0.0, 20.0, 201)
        cached = 0.997 * coordinate[:151] - 5.0
        mapped, method = retarded_time_from_cache(coordinate, cached)
        np.testing.assert_array_equal(mapped[: cached.size], cached)
        np.testing.assert_allclose(mapped, 0.997 * coordinate - 5.0)
        self.assertIn("linear-extension", method)


if __name__ == "__main__":
    unittest.main()
