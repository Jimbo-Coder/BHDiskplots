import unittest

import numpy as np

from wip_plots.mode_appendix import measure_growth, normalized_spectrum


class ModeAppendixTests(unittest.TestCase):
    def test_exponential_rise_rate_is_measured_in_orbital_periods(self):
        time_pc = np.linspace(0, 12, 1201)
        amplitude = np.exp(0.4*time_pc)
        result = measure_growth(time_pc, amplitude)

        self.assertAlmostEqual(result.rate, 0.4, delta=0.01)
        self.assertGreater(result.r2, 0.99)
        self.assertLess(result.rise_start, result.rise_end)
        self.assertTrue(result.peak_at_end)

    def test_already_large_at_start_has_no_measured_rise(self):
        time_pc = np.linspace(0, 12, 1201)
        result = measure_growth(time_pc, np.ones_like(time_pc))
        self.assertIsNone(result.rate)

    def test_complex_spectrum_uses_absolute_frequency(self):
        time_pc = np.linspace(0, 20, 2001)
        for sign in (-1, 1):
            with self.subTest(sign=sign):
                signal = np.exp(2j*np.pi*sign*1.25*time_pc)
                frequency, amplitude = normalized_spectrum(time_pc, signal, 3, 20)
                self.assertAlmostEqual(frequency[np.argmax(amplitude)], 1.25, delta=0.03)
                self.assertAlmostEqual(np.max(amplitude), 1)

    def test_psi4_conversion_weights_characteristic_strain_by_inverse_frequency(self):
        time_pc = np.linspace(0, 20, 2001)
        psi4 = np.exp(2j*np.pi*time_pc) + np.exp(4j*np.pi*time_pc)
        frequency, amplitude = normalized_spectrum(
            time_pc, psi4, 0, 20, psi4_to_hc=True)
        first = amplitude[np.argmin(np.abs(frequency - 1))]
        second = amplitude[np.argmin(np.abs(frequency - 2))]
        self.assertAlmostEqual(first/second, 2, delta=0.1)


if __name__ == "__main__":
    unittest.main()
