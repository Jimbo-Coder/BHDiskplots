from types import SimpleNamespace
import unittest

import numpy as np

import config
from gw import MODES, mode_columns
from wip_plots import gw_radiated


def monochromatic_sim(amplitude=0.3, omega=0.7, zero_rows=5):
    case = config.all_sim_configs(["A1"])[0]
    time = np.arange(0.0, 400.0, 0.01)
    columns = 1 + 2 * len(MODES)
    rhphc, rhphcdot, rpsi4 = (np.zeros((time.size, columns)) for _ in range(3))
    for table in (rhphc, rhphcdot, rpsi4):
        table[:, 0] = time
    re, im = mode_columns(2, 2)
    h = amplitude * np.exp(1j * omega * time)
    for table, values in ((rhphc, h), (rhphcdot, 1j * omega * h), (rpsi4, -omega**2 * h)):
        table[:, re], table[:, im] = values.real, values.imag
    rpsi4[-zero_rows:, 1:] = 0.0  # Zero fill past the last source sample.
    result = SimpleNamespace(time=time, rhphc=rhphc, rhphcdot=rhphcdot, rpsi4_uniform=rpsi4,
                             rpsi4=lambda ell, emm: rpsi4[:, re] + 1j * rpsi4[:, im])
    return SimpleNamespace(config=case, strain_result=result), time


class RadiatedGWTests(unittest.TestCase):
    def test_energy_and_angular_momentum_of_a_monochromatic_22_wave(self):
        amplitude, omega = 0.3, 0.7
        sim, time = monochromatic_sim(amplitude, omega)
        t, energy, angular = gw_radiated.radiated(sim, ((2, 2), (2, 1)))
        start = gw_radiated.TRANSIENT_CUTOFF_MBH * sim.config.mlittle
        self.assertGreaterEqual(t[0], start)
        self.assertLess(t[-1], time[-5])  # Zero-fill rows are excluded.
        duration = t[-1] - (t[0] - np.median(np.diff(t)))
        self.assertAlmostEqual(energy[-1], amplitude**2 * omega**2 * duration / (16*np.pi), delta=1e-3*energy[-1])
        np.testing.assert_allclose(np.abs(angular[-1] / energy[-1]), 2 / omega, rtol=1e-6)

    def test_22_frequency_is_psi4_phase_rate_over_orbital_frequency(self):
        omega = 0.7
        sim, _ = monochromatic_sim(omega=omega)
        _, ratio = gw_radiated.frequency_22(sim)
        np.testing.assert_allclose(ratio, omega / sim.config.gw_omega_orbital, rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
