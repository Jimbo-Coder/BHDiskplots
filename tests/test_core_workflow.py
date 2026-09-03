from pathlib import Path
import inspect
from importlib import import_module
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock

import numpy as np

from config import (
    CACHE_ROOT,
    FORTRAN_GW_ROOT,
    GW_COMPARISON_PARFILE_INDICES,
    GW_FIRST_WAVEZONE_PARFILE_INDEX,
    GW_OUTERMOST_PARFILE_INDEX,
    GW_WORK_ROOT,
    INITIAL_DATA_ROOT,
    PLOTS_DIR,
    REPOSITORY_ROOT,
    all_sim_configs,
)
from plot_settings import GW_TIME_SCALE
from gw_psi4 import N_PSI4_COLUMNS, Psi4File, convert_to_strain_with_rhphc, read_psi4_file
from helpers.reader_gw import (
    filter_psi4_by_expected_radius,
    subtract_psi4_on_retarded_time,
)
from helpers import reader_2d
from helpers import style
from helpers.gw_units import gw_time_values, gw_time_xlabel
from helpers.time_units import ORBITAL_PERIOD_LATEX, time_xlabel
from helpers.time_series import merge_restart_time_series


def psi4_rows(times):
    rows = np.zeros((len(times), N_PSI4_COLUMNS), dtype=float)
    rows[:, 0] = times
    rows[:, -4] = 100.0
    return rows


def carpet_2d_iteration(iteration, time_value, ref_levels=(0,)):
    lines = []
    for component, ref_level in enumerate(ref_levels):
        lines.extend(
            [
                f"# iteration {iteration}\n",
                f"# refinement level {ref_level} multigrid level 0 map 0 "
                f"component {component} time level 0\n",
                "# column format: it tl rl c ml ix iy iz time x y z rho\n",
                f"{iteration} 0 {ref_level} {component} 0 0 0 0 "
                f"{time_value} 0.0 0.0 0.0 1.0\n",
            ]
        )
    return "".join(lines)


class RestartAndCacheTests(unittest.TestCase):
    def test_savefig_creates_nested_output_directory(self):
        from helpers.plot_common import savefig

        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                outdir=Path(tmp),
                no_save=False,
                show=False,
            )
            fig = mock.Mock()
            savefig(fig, args, Path("gw") / "example.png")

            expected = Path(tmp) / "gw" / "example.png"
            self.assertTrue(expected.parent.is_dir())
            fig.savefig.assert_called_once_with(expected)

    def test_individual_figures_are_saved_under_case_directory(self):
        from helpers.plot_common import save_individual_fig

        args = SimpleNamespace(outdir=Path("figures"))
        sim = SimpleNamespace(config=SimpleNamespace(name="A1"))
        with mock.patch("helpers.plot_common.savefig") as save:
            save_individual_fig(mock.Mock(), args, sim, "example.png")
        self.assertEqual(save.call_args.args[2], Path("A1") / "example.png")

    def test_rho2d_cleanup_only_removes_superseded_matching_tag(self):
        from paper_plots.rho2d_individual import _remove_superseded_snapshot

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            case_dir = root / "B3"
            case_dir.mkdir()
            old_final = case_dir / "rho2d_B3_xy_iter10_idx-1.png"
            new_final = case_dir / "rho2d_B3_xy_iter20_idx-1.png"
            initial = case_dir / "rho2d_B3_xy_iter0_idx0.png"
            unrelated = case_dir / "rhphc_B3_l2m2.png"
            for path in (old_final, new_final, initial, unrelated):
                path.touch()

            args = SimpleNamespace(outdir=root, no_save=False)
            _remove_superseded_snapshot(
                args,
                "B3",
                "xy",
                "idx-1",
                new_final.name,
            )

            self.assertFalse(old_final.exists())
            self.assertTrue(new_final.exists())
            self.assertTrue(initial.exists())
            self.assertTrue(unrelated.exists())

    def test_nonpaper_outputs_are_routed_out_of_figure_root(self):
        from helpers.gw_difference import GW_DIFFERENCE_OUTPUT_SUBDIR
        from wip_plots import (
            disp_all,
            gw_detectability_all,
            gw_psi4_all,
            gw_strain_all,
        )

        self.assertTrue(disp_all.OUTPUT_FILENAME.startswith("wip/"))
        self.assertTrue(gw_psi4_all.OUTPUT_TEMPLATE.startswith("gw/"))
        self.assertTrue(gw_strain_all.OUTPUT_TEMPLATE.startswith("gw/"))
        self.assertTrue(
            gw_detectability_all.OUTPUT_FILENAME_CHARACTERISTIC_STRAIN.startswith("gw/")
        )
        self.assertEqual(GW_DIFFERENCE_OUTPUT_SUBDIR, "gw/difference")

    def test_paper_filenames_remain_latex_compatible(self):
        expected = {
            "paper_plots.constraints_all": "constraints.png",
            "paper_plots.initial_data_all": "initial_rho_ell_xp.png",
            "paper_plots.modes_all": "modes.png",
            "paper_plots.phase_all": "phase.png",
            "paper_plots.rhomax_all": "rhomax.png",
            "paper_plots.spin_all": "J_Xi.png",
        }
        for module_name, filename in expected.items():
            with self.subTest(module=module_name):
                module = import_module(module_name)
                self.assertEqual(module.OUTPUT_FILENAME, filename)

    def test_dimensionless_spin_paper_limits_start_at_point_six(self):
        from paper_plots.spin_all import SPIN_YLIM as standalone_limits
        from paper_plots.triple_m_all import SPIN_YLIM as combined_limits

        self.assertEqual(standalone_limits, (0.6, 1.0))
        self.assertEqual(combined_limits, standalone_limits)

    def test_gw_radius_policy_uses_first_wavezone_and_outermost(self):
        self.assertEqual(GW_FIRST_WAVEZONE_PARFILE_INDEX, 4)
        self.assertEqual(GW_OUTERMOST_PARFILE_INDEX, 8)
        self.assertEqual(GW_COMPARISON_PARFILE_INDICES, (4, 8))

    def test_combined_time_domain_gw_plots_generate_both_radii(self):
        module_names = (
            "wip_plots.gw_psi4_all",
            "wip_plots.gw_strain_all",
            "wip_plots.gw_strain_polarization_panel",
        )
        for module_name in module_names:
            with self.subTest(module=module_name):
                module = import_module(module_name)
                with (
                    mock.patch.object(module, "setup"),
                    mock.patch.object(module, "load_sims", return_value=[]) as load,
                    mock.patch.object(module, "plot", return_value=mock.Mock()),
                    mock.patch.object(module, "savefig") as save,
                ):
                    module.main([])
                self.assertEqual(
                    [call.kwargs["psi4_parfile_index"] for call in load.call_args_list],
                    list(GW_COMPARISON_PARFILE_INDICES),
                )
                self.assertEqual(save.call_count, 8)

    def test_known_hamiltonian_burst_is_broken_not_interpolated(self):
        from paper_plots.constraints_all import _hamiltonian_for_plot

        sim = mock.Mock()
        sim.config.name = "B3"
        sim.ham_t = np.array([267.49, 267.50, 267.70, 267.84, 267.85])
        sim.ham_r = np.arange(5.0)
        plotted = _hamiltonian_for_plot(sim)
        np.testing.assert_array_equal(
            np.isnan(plotted),
            [False, True, True, True, False],
        )
        self.assertEqual(plotted[0], sim.ham_r[0])
        self.assertEqual(plotted[-1], sim.ham_r[-1])

    def test_tex_module_is_loaded_only_when_required_font_is_missing(self):
        with (
            mock.patch.object(
                style,
                "_tex_file_available",
                side_effect=(False, True),
            ),
            mock.patch.object(style, "_load_module_environment") as load,
        ):
            style.ensure_tex_path()
        load.assert_called_once_with(style.TEXLIVE_MODULE)

    def test_orbital_period_labels_use_lowercase_c(self):
        self.assertEqual(ORBITAL_PERIOD_LATEX, r"P_c")
        self.assertEqual(time_xlabel(normalize_by_pc=True), r"$t/P_c$")
        self.assertEqual(
            gw_time_xlabel("P_c"),
            r"$t_{\mathrm{ret}}/P_c$",
        )

    def test_gw_time_scale_supports_all_four_configured_choices(self):
        self.assertEqual(GW_TIME_SCALE, "M_BH")
        sim = SimpleNamespace(
            config=SimpleNamespace(
                name="A1",
                mlittle=2.0,
                gw_madm=4.0,
                Pc=5.0,
            )
        )
        values = np.array([10.0, 20.0])
        np.testing.assert_allclose(gw_time_values(values, sim, "M_BH"), [5.0, 10.0])
        np.testing.assert_allclose(gw_time_values(values, sim, "M_ADM"), [2.5, 5.0])
        np.testing.assert_allclose(gw_time_values(values, sim, "P_c"), [2.0, 4.0])
        np.testing.assert_allclose(gw_time_values(values, sim, "code"), values)
        self.assertEqual(gw_time_xlabel("M_BH"), r"$t_{\mathrm{ret}}/M_{\mathrm{BH}}$")
        self.assertEqual(gw_time_xlabel("M_ADM"), r"$t_{\mathrm{ret}}/M$")
        self.assertEqual(gw_time_xlabel("code"), r"$t_{\mathrm{ret}}\;(\mathrm{code})$")

    def test_top_level_plot_runners_accept_empty_selection(self):
        import run_paper
        import run_wip

        with mock.patch.dict(run_paper.PAPER_PLOTS, clear=True):
            run_paper.main([])
        with mock.patch.dict(run_wip.WIP_PLOTS, clear=True):
            run_wip.main([])

    def test_registered_plots_accept_explicit_argv(self):
        from run_paper import PAPER_PLOTS
        from run_wip import WIP_PLOTS

        modules = [
            *(f"paper_plots.{name}" for name in PAPER_PLOTS.values()),
            *(f"wip_plots.{name}" for name in WIP_PLOTS.values()),
        ]
        for module_name in modules:
            with self.subTest(module=module_name):
                main = import_module(module_name).main
                self.assertIn("argv", inspect.signature(main).parameters)

    def test_2d_bounds_cache_invalidates_when_source_grows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rho_b.xy.asc"
            path.write_text("initial\n")
            reader_2d._cached_2d_file_bounds.cache_clear()
            with (
                mock.patch.object(
                    reader_2d,
                    "first_2d_iteration_time_info",
                    return_value=(10, 1.0, 100),
                ) as first,
                mock.patch.object(
                    reader_2d,
                    "last_2d_iteration_time_info",
                    return_value=(20, 2.0, 200),
                ) as last,
            ):
                reader_2d._2d_file_bounds(path, [0, 1])
                reader_2d._2d_file_bounds(path, [0, 1])
                self.assertEqual(first.call_count, 1)
                self.assertEqual(last.call_count, 1)

                path.write_text("source grew\n")
                reader_2d._2d_file_bounds(path, [0, 1])
                self.assertEqual(first.call_count, 2)
                self.assertEqual(last.call_count, 2)

    def test_2d_timeline_cache_reuses_unchanged_source_and_extends_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "shared-cache"
            path = root / "rho_b.xy.asc"
            path.write_text(
                carpet_2d_iteration(0, 0.0)
                + carpet_2d_iteration(10, 1.0)
            )

            original_scan = reader_2d._scan_valid_2d_iteration_time_infos
            with mock.patch.object(reader_2d, "TWO_D_INDEX_CACHE_ROOT", cache_root):
                reader_2d._cached_valid_2d_iteration_time_infos.cache_clear()
                first = reader_2d.valid_2d_iteration_time_infos(path, ref_level=[0])
                self.assertEqual([info[:2] for info in first], [(0, 0.0), (10, 1.0)])
                self.assertEqual(len(list(cache_root.glob("*.json"))), 1)

                reader_2d._cached_valid_2d_iteration_time_infos.cache_clear()
                with mock.patch.object(
                    reader_2d,
                    "_scan_valid_2d_iteration_time_infos",
                    side_effect=AssertionError("unchanged source was rescanned"),
                ):
                    second = reader_2d.valid_2d_iteration_time_infos(path, ref_level=[0])
                self.assertEqual(second, first)

                with path.open("a") as source:
                    source.write(carpet_2d_iteration(20, 2.0))
                reader_2d._cached_valid_2d_iteration_time_infos.cache_clear()
                with mock.patch.object(
                    reader_2d,
                    "_scan_valid_2d_iteration_time_infos",
                    wraps=original_scan,
                ) as scan:
                    extended = reader_2d.valid_2d_iteration_time_infos(path, ref_level=[0])

                self.assertEqual(
                    [info[:2] for info in extended],
                    [(0, 0.0), (10, 1.0), (20, 2.0)],
                )
                self.assertEqual(scan.call_count, 1)
                self.assertEqual(scan.call_args.kwargs["start_byte"], first[-1][2])

    def test_fast_2d_header_scan_matches_linewise_validity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rho_b.xy.asc"
            complete = carpet_2d_iteration(0, 0.0, ref_levels=(0, 1))
            complete = complete.replace(
                "# column format: it tl rl c ml ix iy iz time x y z rho\n",
                "# column format: it tl rl c ml ix iy iz time x y z rho\n"
                "# auxiliary header retained before numeric data\n",
            )
            path.write_text(
                complete
                + carpet_2d_iteration(10, 1.0, ref_levels=(0,))
                + carpet_2d_iteration(20, 2.0, ref_levels=(0, 1))
            )

            linewise = list(
                reader_2d._scan_valid_2d_iteration_time_infos_linewise(
                    path,
                    ref_level=[0, 1],
                )
            )
            grep = reader_2d.shutil.which("grep")
            self.assertIsNotNone(grep)
            with mock.patch.object(reader_2d.shutil, "which", return_value=grep):
                external = list(
                    reader_2d._scan_valid_2d_iteration_time_infos_external(
                        path,
                        ref_level=[0, 1],
                    )
                )
            self.assertEqual(external, linewise)
            self.assertEqual([info[:2] for info in external], [(0, 0.0), (20, 2.0)])

    def test_2d_timeline_cache_preserves_later_restart_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "shared-cache"
            path = root / "rho_b.xy.asc"
            path.write_text(
                carpet_2d_iteration(0, 0.0)
                + carpet_2d_iteration(10, 1.0)
                + carpet_2d_iteration(20, 2.0)
            )

            with mock.patch.object(reader_2d, "TWO_D_INDEX_CACHE_ROOT", cache_root):
                reader_2d._cached_valid_2d_iteration_time_infos.cache_clear()
                reader_2d.valid_2d_iteration_time_infos(path, ref_level=[0])
                with path.open("a") as source:
                    source.write(
                        carpet_2d_iteration(15, 1.5)
                        + carpet_2d_iteration(25, 2.5)
                    )
                reader_2d._cached_valid_2d_iteration_time_infos.cache_clear()
                infos = reader_2d.valid_2d_iteration_time_infos(path, ref_level=[0])

            self.assertEqual(
                [info[:2] for info in infos],
                [(0, 0.0), (10, 1.0), (15, 1.5), (25, 2.5)],
            )

    def test_2d_timeline_cache_rechecks_incomplete_appended_iteration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cache_root = root / "shared-cache"
            path = root / "rho_b.xy.asc"
            path.write_text(
                carpet_2d_iteration(0, 0.0, ref_levels=(0, 1))
                + carpet_2d_iteration(10, 1.0, ref_levels=(0,))
            )

            with mock.patch.object(reader_2d, "TWO_D_INDEX_CACHE_ROOT", cache_root):
                reader_2d._cached_valid_2d_iteration_time_infos.cache_clear()
                initial = reader_2d.valid_2d_iteration_time_infos(
                    path,
                    ref_level=[0, 1],
                )
                self.assertEqual([info[:2] for info in initial], [(0, 0.0)])

                with path.open("a") as source:
                    source.write(
                        carpet_2d_iteration(10, 1.0, ref_levels=(1,))
                        + carpet_2d_iteration(20, 2.0, ref_levels=(0, 1))
                    )
                reader_2d._cached_valid_2d_iteration_time_infos.cache_clear()
                completed = reader_2d.valid_2d_iteration_time_infos(
                    path,
                    ref_level=[0, 1],
                )

            self.assertEqual(
                [info[:2] for info in completed],
                [(0, 0.0), (10, 1.0), (20, 2.0)],
            )

    def test_scalar_restart_replaces_old_overlap(self):
        old = np.array([[0.0, 10.0], [1.0, 11.0], [2.0, 12.0], [3.0, 13.0]])
        restart = np.array([[2.5, 25.0], [3.5, 35.0], [4.5, 45.0]])
        merged = merge_restart_time_series((old, restart))
        np.testing.assert_allclose(merged[:, 0], [0.0, 1.0, 2.0, 2.5, 3.5, 4.5])
        np.testing.assert_allclose(merged[:, 1], [10.0, 11.0, 12.0, 25.0, 35.0, 45.0])

    def test_psi4_duplicate_time_keeps_last_row(self):
        rows = psi4_rows([0.0, 1.0, 1.0, 2.0])
        rows[:, 1] = [10.0, 11.0, 99.0, 12.0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Psi4_rad.mon.1"
            np.savetxt(path, rows)
            result = read_psi4_file(path, label="1")
        np.testing.assert_allclose(result.time, [0.0, 1.0, 2.0])
        np.testing.assert_allclose(result.data[:, 1], [10.0, 99.0, 12.0])
        self.assertEqual(result.repeated_times, 1)

    def test_plot_cache_read_cannot_fall_through_to_fortran(self):
        psi4 = Psi4File(
            path=Path("source/Psi4_rad.mon.1"),
            label="1",
            data=psi4_rows([0.0, 1.0]),
            repeated_times=0,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "Run generate_gw.py"):
                convert_to_strain_with_rhphc(
                    psi4,
                    workdir=Path(tmp),
                    omega_orbital=0.1,
                    madm=1.0,
                    reuse_existing=True,
                    generate_if_missing=False,
                )

    def test_massless_restart_is_ordered_oldest_to_newest(self):
        massless = all_sim_configs(["ml"])[0]
        self.assertEqual(len(massless.data_roots), 2)
        self.assertIn("jamiescalars", str(massless.data_roots[0]))
        self.assertEqual(massless.data_roots[1].name, "massless")
        self.assertNotIn("scratch", str(PLOTS_DIR))
        self.assertNotIn("scratch", str(GW_WORK_ROOT))

    def test_repository_owned_paths_are_checkout_relative(self):
        self.assertEqual(PLOTS_DIR, REPOSITORY_ROOT / "figures")
        self.assertEqual(CACHE_ROOT, REPOSITORY_ROOT / "cache")
        self.assertEqual(GW_WORK_ROOT, REPOSITORY_ROOT / "gw_work")
        self.assertEqual(FORTRAN_GW_ROOT, REPOSITORY_ROOT / "psi4_hlm_ref")
        self.assertEqual(
            INITIAL_DATA_ROOT,
            REPOSITORY_ROOT / "data" / "initial_profiles",
        )

    def test_mislabeled_psi4_radius_is_rejected(self):
        good_rows = psi4_rows([0.0, 1.0])
        good_rows[:, -4] = 171.0
        bad_rows = psi4_rows([0.0, 1.0])
        bad_rows[:, -4] = 12.8
        files = {
            "9": Psi4File(Path("Psi4_rad.mon.9"), "9", good_rows, 0),
            "10": Psi4File(Path("Psi4_rad.mon.*"), "10", bad_rows, 0),
        }
        valid = filter_psi4_by_expected_radius(files, {8: 170.0, 9: 180.0})
        self.assertEqual(tuple(valid), ("9",))


class Psi4DifferenceTests(unittest.TestCase):
    def test_subtraction_aligns_on_retarded_time_and_preserves_case_metadata(self):
        case_rows = psi4_rows([10.0, 11.0, 12.0, 13.0])
        reference_rows = psi4_rows([20.0, 21.0, 22.0, 23.0])
        case_rows[:, 1:43] = np.arange(4, dtype=float)[:, None] + 10.0
        reference_rows[:, 1:43] = np.arange(4, dtype=float)[:, None] + 3.0
        case = Psi4File(Path("case"), "5", case_rows, 0)
        reference = Psi4File(Path("reference"), "5", reference_rows, 0)

        difference = subtract_psi4_on_retarded_time(
            case,
            [0.0, 1.0, 2.0, 3.0],
            reference,
            [0.0, 1.0, 2.0, 3.0],
            label="5_minus_ML",
        )

        np.testing.assert_allclose(difference.data[:, 0], case_rows[:, 0])
        np.testing.assert_allclose(difference.data[:, 1:43], 7.0)
        np.testing.assert_allclose(difference.data[:, -4:], case_rows[:, -4:])


if __name__ == "__main__":
    unittest.main()
