"""Small numerical contracts for the initial Newtonian Toomre diagnostic."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

from wip_plots.toomre import (
    column_density, epicyclic_squared, extract_cocal_profiles,
    meidt_parameter, plot_case_comparison, romeo_thickness_factor,
)


class ToomreTests(unittest.TestCase):
    def test_raw_cocal_export_order_and_coordinates(self):
        radius = np.array([0.1, 0.2, 0.4, 0.8])
        theta = np.linspace(0, np.pi, 5)
        phi = np.linspace(0, 2 * np.pi, 5)
        q = np.sin(theta[:, None])**2 * radius
        omega = np.broadcast_to(radius**-1.5, q.shape)
        with TemporaryDirectory() as directory:
            source = Path(directory)
            header = f"{len(radius)-1} {len(theta)-1} {len(phi)-1}"
            np.savetxt(source / "rnsgrids_3D.las", np.concatenate((radius, theta, phi)),
                       header=header, comments="")
            fields = np.tile(np.column_stack((q.ravel(), omega.ravel())), (len(phi), 1))
            np.savetxt(source / "rnsflu_3D.las", fields, header=header, comments="")
            np.savetxt(source / "omeg_xp70.txt", np.column_stack((radius, omega[2])))
            output = source / "profiles"
            extract_cocal_profiles(source, output, 70)
            equator = np.loadtxt(output / "emdg_xp70.txt")
            np.testing.assert_allclose(equator, np.column_stack((radius, q[2])))
            meridians = np.loadtxt(output / "emdg_xz70.txt").reshape(2, 5, 4, 3)
            np.testing.assert_allclose(meridians[0, :, :, 0], np.sin(theta[:, None])*radius)
            np.testing.assert_allclose(meridians[1, :, :, 0], -meridians[0, :, :, 0])
            np.testing.assert_allclose(meridians[0, :, :, 1], np.cos(theta[:, None])*radius)
            np.testing.assert_allclose(meridians[0, :, :, 2], q)
            np.testing.assert_allclose(meridians[1, :, :, 2], q)
            np.savetxt(source / "omeg_xp70.txt", np.column_stack((radius, omega[2]*2)))
            with self.assertRaisesRegex(ValueError, "does not match"):
                extract_cocal_profiles(source, output, 70)

    def test_keplerian_epicycle(self):
        r = np.geomspace(0.4, 8, 300)
        omega = np.sqrt(0.05 / r**3)
        np.testing.assert_allclose(epicyclic_squared(r, omega), omega**2, rtol=1e-10)

    def test_solid_body_epicycle(self):
        r = np.linspace(1, 5, 1000)
        omega = np.full_like(r, 0.3)
        np.testing.assert_allclose(epicyclic_squared(r, omega), 4 * omega**2, rtol=2e-4)

    def test_negative_epicycle_is_not_made_positive(self):
        r = np.linspace(1, 5, 1000)
        self.assertTrue(np.all(epicyclic_squared(r, r**-3) < 0))

    def test_romeo_thickness_factor(self):
        for anisotropy, expected in ((0, 1), (0.25, 1.0375), (0.5, 1.15),
                                     (0.75, 1.325), (1, 1.5)):
            with self.subTest(anisotropy=anisotropy):
                self.assertAlmostEqual(romeo_thickness_factor(anisotropy), expected)

    def test_romeo_anisotropy_domain(self):
        for anisotropy in (-0.01, 1.01, np.nan, np.inf):
            with self.subTest(anisotropy=anisotropy), self.assertRaises(ValueError):
                romeo_thickness_factor(anisotropy)

    def test_meidt_midplane_density_and_epicycle(self):
        rho_mid = np.array([1., 2., 1.])
        kappa2 = 4 * np.pi * np.array([1., 1., 2.])
        np.testing.assert_allclose(meidt_parameter(kappa2, rho_mid), [1., 0.5, 2.])

    def test_meidt_invalid_samples_are_not_plotted(self):
        kappa2 = np.array([-1., 0., np.nan, np.inf, 1., 1., 1., 1.])
        rho_mid = np.array([1., 1., 1., 1., -1., 0., np.nan, np.inf])
        self.assertTrue(np.isnan(meidt_parameter(kappa2, rho_mid)).all())

    def test_case_comparison_keeps_three_methods_on_each_case(self):
        sims = [SimpleNamespace(config=SimpleNamespace(gw_madm=2), legend_name=name)
                for name in ("A2", "A3", "B3", "ML")]
        profiles = [dict(r=np.array([2., 4.]), Q_N=np.array([2., 1.]),
                         Q_R=np.array([3., 1.5]), Q_M=np.array([0.5, 0.25]))
                    for _ in sims]
        fig = plot_case_comparison(sims, profiles)
        try:
            visible = [ax for ax in fig.axes if ax.get_visible()]
            self.assertEqual(len(visible), len(sims))
            for ax, sim, profile in zip(visible, sims, profiles):
                self.assertEqual(ax.texts[0].get_text(), sim.legend_name)
                self.assertEqual(ax.get_yscale(), "log")
                for line, field in zip(ax.lines[:3], ("Q_N", "Q_R", "Q_M")):
                    np.testing.assert_array_equal(line.get_xdata(), profile["r"] / 2)
                    np.testing.assert_array_equal(line.get_ydata(), profile[field])
            self.assertEqual([text.get_text() for text in fig.legends[0].get_texts()],
                             ["Toomre", "Romeo", "Meidt"])
        finally:
            plt.close(fig)

    def test_gaussian_vertical_column(self):
        radius = np.linspace(0.01, 20, 1200)
        theta = np.linspace(0, np.pi, 601)
        R = np.sin(theta[:, None]) * radius
        z = np.cos(theta[:, None]) * radius
        density = np.exp(-R**2 / 8 - z**2 / 0.5)
        density[density < 1e-15] = 0
        selected = np.linspace(0.4, 3, 40)
        sigma = column_density(radius, theta, density, selected)
        expected = np.exp(-selected**2 / 8) * np.sqrt(0.5 * np.pi)
        np.testing.assert_allclose(sigma, expected, rtol=5e-4)

    def test_vertical_samples_include_midplane(self):
        with self.assertRaisesRegex(ValueError, "odd"):
            column_density(np.arange(1, 4), np.linspace(0, np.pi, 3),
                           np.ones((3, 3)), np.array([1.0]), samples=100)

    def test_missing_outer_coverage_fails(self):
        with self.assertRaisesRegex(ValueError, "boundary"):
            column_density(np.arange(1, 4), np.linspace(0, np.pi, 3),
                           np.ones((3, 3)), np.array([1.0]))


if __name__ == "__main__":
    unittest.main()
