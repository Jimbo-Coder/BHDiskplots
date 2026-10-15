from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np

from wip_plots import disk_compactness_comparison as comparison


class CompactnessTests(unittest.TestCase):
    def test_all_models_are_retained_and_selected_cases_match(self):
        sequences = comparison.load_sequences()
        self.assertEqual([len(sequences[family]) for family in ("A", "B")], [17, 43])
        expected = {name: (radius/comparison.BH_MASS_CODE, mass/comparison.BH_MASS_CODE)
                    for name,radius,mass in comparison.OUR_DISKS}
        for family, selected in comparison.SEQUENCE_CASES.items():
            models = sequences[family]
            self.assertEqual([model.number for model in models], list(range(1,len(models)+1)))
            self.assertTrue(np.any(np.diff([model.outer_radius for model in models]) < 0))
            for number,name in selected.items():
                np.testing.assert_allclose(models[number-1].coordinates, expected[name], rtol=1e-12)

    def test_parser_uses_rest_mass_and_each_models_own_bh_mass(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bhtphyseq.dat"
            path.write_text("== Sequence number == 2\nBlack hole mass (M)= 2D-1\n"
                            "Outer point (width)= 4.0 0 0\nRest, proper mass = .3 99\n")
            model, = comparison.read_cocal_sequence(path)
            self.assertEqual(model.number,2)
            np.testing.assert_allclose(model.coordinates,(20.,1.5))
            path.write_text(path.read_text().replace("Rest, proper mass", "Missing field"))
            with self.assertRaisesRegex(ValueError,"missing Rest"):
                comparison.read_cocal_sequence(path)

    def test_sequence_overview_shows_full_ranges_at_one_column_width(self):
        with plt.rc_context({"text.usetex":False}):
            fig = comparison.plot(True,include_sequences=True,sequence_overview=True)
            try:
                main, overview = fig.axes
                self.assertAlmostEqual(fig.get_figwidth(),3.75)
                self.assertGreater(overview.get_ylim()[1],4.2649)
                self.assertLess(overview.get_xlim()[0],11.45)
                self.assertEqual([len(line.get_xdata()) for line in overview.lines],[17,43])
                self.assertEqual(main.get_ylim(),comparison.COMPARISON_YLIM)
                fig.canvas.draw()
            finally:
                plt.close(fig)

    def test_sequences_can_be_disabled(self):
        with plt.rc_context({"text.usetex":False}):
            fig = comparison.plot(True,include_sequences=False)
            try:
                self.assertEqual(len(fig.axes),1)
                np.testing.assert_allclose(fig.get_size_inches(),(3.75,3.))
            finally:
                plt.close(fig)
