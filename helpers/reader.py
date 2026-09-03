"""1D diagnostic and simulation data loading for BHDisk plots."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from config import DiskSimConfig, GW_WORK_ROOT, all_sim_configs
from .reader_2d import TwoDFilePosition, read_2d_file
from .reader_gw import (
    convert_psi4_to_strain,
    load_psi4,
    psi4_file_label_for_index,
    select_psi4_file,
    selected_psi4_mode,
    subtract_psi4_on_retarded_time,
)
from .style import apply_styles
from .time_series import merge_restart_time_series


class DiskSim:
    def __init__(self, config: DiskSimConfig):
        self.config = config
        self.data_paths_1d = tuple(Path(path) for path in config.data_roots)
        self.data_path_1d = self.data_paths_1d[0]
        self.data_path_initial_data = Path(config.initial_data_path) if config.initial_data_path is not None else None
        self.data_path_2d = self.data_path_1d / "beta100"
        self.data_paths_2d = tuple(path / "beta100" for path in self.data_paths_1d)
        self.supplemental_2d_paths = tuple(Path(path) for path in config.supplemental_2d_paths)
        self.loaded_data_paths = {}
        self.loaded = set()
        self.legend_name = None
        self.linestyle = "-"
        self.markerstyle = "o"
        self.color = "k"

    @property
    def args(self):
        # Compatibility for code adapted from the notebook.
        return self.config

    def loaddata(self, fname, coloi=None, tcol=0):
        filepaths = [root / fname for root in self.data_paths_1d]
        existing = [path for path in filepaths if path.exists()]
        if not existing:
            raise FileNotFoundError(f"None of the configured sources contain {fname}: {filepaths}")
        segments = [np.loadtxt(path) for path in existing]
        data_clean = merge_restart_time_series(segments, tcol=tcol)
        t = data_clean[:, tcol]
        scalar = data_clean[:, coloi] if coloi is not None else None
        self.loaded_data_paths[fname] = tuple(existing)
        return existing[0], data_clean, t, scalar

    def load_constraints(self):
        if "constraints" in self.loaded:
            return True
        self.hampath, self.hamdata, self.ham_t, self.ham_r = self.loaddata("bhns-ham.con", coloi=5)
        self.mompath, self.momdata, self.mom_t, _ = self.loaddata("bhns-mom.con", None)
        mom_coloi = 5
        self.mom_Ni = self.momdata[:, mom_coloi:mom_coloi + 3]
        self.mom_Nd = self.momdata[:, mom_coloi + 3]
        self.mom_r = np.sum(self.mom_Ni, axis=1) / self.mom_Nd
        self.loaded.add("constraints")
        return True

    def load_modes(self):
        if "modes" in self.loaded:
            return True
        self.modepath, self.modedata, self.modes_t, _ = self.loaddata("bhns-dens_mode.con", None)
        self.modes_t = self.modedata[:, 0]
        modes = 3
        real_index = np.concatenate(([1], np.arange(2, 2 * modes, 2)))
        imag_index = np.arange(3, 2 * modes + 1, 2)
        self.modes_re = self.modedata[:, real_index]
        self.modes_im = np.zeros_like(self.modes_re)
        self.modes_im[:, 1:] = self.modedata[:, imag_index]
        self.modes = self.modes_re + 1j * self.modes_im
        self.loaded.add("modes")
        return True

    def load_rhomax(self):
        if "rhomax" in self.loaded:
            return True
        self.rhomaxpath, self.rhomaxdata, self.rhomax_t, self.rhomax = self.loaddata("bhns.mon", coloi=8)
        self.loaded.add("rhomax")
        return True

    def load_M0MADM(self):
        if "M0MADM" in self.loaded:
            return True
        restmass_coloi = 2
        admmass_coloi = 10
        m0dot_bh_coloi = 12
        self.M0MADMpath, M0MADMdata, self.M0MADM_t, _ = self.loaddata("bhns.don", coloi=None)
        self.restmass = M0MADMdata[:, restmass_coloi - 1]
        self.admmass = M0MADMdata[:, admmass_coloi - 1]
        self.M0dot_BH_t = self.M0MADM_t
        self.M0dot_BH = M0MADMdata[:, m0dot_bh_coloi - 1]
        self.loaded.add("M0MADM")
        return True

    def load_J(self):
        if "J" in self.loaded:
            return True
        self.Jpath, self.Jdata, self.J_t, _ = self.loaddata("bhns_BHspin.mon", None)
        self.J = np.sqrt(np.sum(np.square(self.Jdata[:, 1:4]), axis=1))
        self.loaded.add("J")
        return True

    def load_Rs(self):
        if "Rs" in self.loaded:
            return True
        self.Rspath, self.Rsdata, self.Rs_t, self.Rs = self.loaddata("beta100/BH_diagnostics.ah1.gp", coloi=27, tcol=1)
        self.loaded.add("Rs")
        return True

    def load_spin_parameter(self):
        self.load_J()
        self.load_Rs()
        self.Rs_interp = np.interp(self.J_t, self.Rs_t, self.Rs)
        self.Mbh = (1 / (2 * self.Rs_interp)) * np.sqrt(self.Rs_interp**4 + 4 * self.J**2)
        self.loaded.add("spin_parameter")
        return True

    def load_psi4(self, psi4_parfile_index=None, psi4_mode=None):
        if "psi4" in self.loaded:
            return True
        self.psi4_files, self.psi4_extraction_radii = load_psi4(self.data_paths_1d)
        self.psi4_mode = psi4_mode if psi4_mode is not None else self.config.psi4_mode
        self.psi4_parfile_index = int(psi4_parfile_index if psi4_parfile_index is not None else self.config.psi4_parfile_index)
        self.psi4_label, self.psi4 = select_psi4_file(
            self.psi4_files,
            self.psi4_parfile_index,
            file_label=self.gw_extraction_file_label(self.psi4_parfile_index),
        )
        if self.psi4 is None:
            print(f"{self.config.name}: no Psi4_rad.mon.N files found")
            return False
        self.psi4_radius = self.psi4_extraction_radii.get(self.gw_extraction_radius_index(self.psi4_parfile_index))
        self.psi4_t = self.psi4.time
        ell, emm = self.psi4_mode
        self.rpsi4_lm = selected_psi4_mode(self.psi4, ell=ell, emm=emm, multiply_by_r=True)
        self.loaded.add("psi4")
        return True

    def gw_parfile_indices(self, gw_parfile_indices=None):
        indices = gw_parfile_indices if gw_parfile_indices is not None else self.config.gw_parfile_indices
        return [int(index) for index in indices]

    def gw_extraction_plot_label(self, parfile_index):
        radius = self.psi4_extraction_radii.get(self.gw_extraction_radius_index(parfile_index))
        if radius is None and hasattr(self, "psi4_files"):
            psi4_file = self.psi4_files.get(self.gw_extraction_file_label(parfile_index))
            if psi4_file is not None:
                radius = float(np.nanmedian(psi4_file.r_areal))
        if radius is None:
            return rf"$i_{{par}}={int(parfile_index)}$"
        if radius == 0:
            radius_label = "0"
        else:
            exponent = int(np.floor(np.log10(abs(radius))))
            mantissa = radius / (10.0**exponent)
            if exponent != 0 and np.isclose(mantissa, 1.0):
                radius_label = rf"10^{{{exponent}}}"
            elif exponent != 0 and (abs(radius) >= 1.0e4 or abs(radius) < 1.0e-2):
                radius_label = rf"{mantissa:.3g}\times10^{{{exponent}}}"
            else:
                radius_label = f"{radius:g}"
        return rf"$r={radius_label}$"

    def gw_extraction_file_label(self, parfile_index):
        label_index = int(parfile_index) + int(getattr(self.config, "gw_psi4_file_index_offset", 1))
        return str(label_index)

    def gw_extraction_radius_index(self, parfile_index):
        return int(self.gw_extraction_file_label(parfile_index)) - 1

    def gw_workdir(self, file_label):
        return GW_WORK_ROOT / f"{self.config.name}_psi4{file_label}"

    def gw_difference_workdir(self, reference_name):
        return GW_WORK_ROOT / f"{self.config.name}_psi4{self.psi4_label}_minus_{reference_name}"

    def load_strain(
        self,
        regenerate_gw=False,
        psi4_parfile_index=None,
        psi4_mode=None,
        generate_if_missing=False,
    ):
        if "strain" in self.loaded and not regenerate_gw:
            return True
        if self.config.gw_omega_orbital is None or self.config.gw_madm is None:
            print(f"{self.config.name}: missing gw_omega_orbital/gw_madm; skipping strain")
            return False
        if not self.load_psi4(psi4_parfile_index=psi4_parfile_index, psi4_mode=psi4_mode):
            return False
        workdir = self.gw_workdir(self.psi4_label)
        self.strain_result = convert_psi4_to_strain(
            self.psi4,
            workdir=workdir,
            omega_orbital=self.config.gw_omega_orbital,
            madm=self.config.gw_madm,
            regenerate=regenerate_gw,
            generate_if_missing=generate_if_missing,
        )
        self.rh_t = self.strain_result.time
        ell, emm = self.psi4_mode
        self.rh_plus_lm, self.rh_cross_lm = self.strain_result.hplus_hcross(ell=ell, emm=emm)
        self.loaded.add("strain")
        return True

    def strain_from_psi4_difference(
        self,
        reference,
        regenerate_gw=False,
        generate_if_missing=False,
        backend="python",
    ):
        """Reconstruct strain after subtracting a reference Psi4 waveform."""
        if "strain" not in self.loaded or "strain" not in reference.loaded:
            raise ValueError("Both simulations must have strain loaded to define retarded time")
        difference = subtract_psi4_on_retarded_time(
            self.psi4,
            self.rh_t,
            reference.psi4,
            reference.rh_t,
            label=f"{self.psi4_label}_minus_{reference.config.name}",
        )
        return convert_psi4_to_strain(
            difference,
            workdir=self.gw_difference_workdir(reference.config.name),
            omega_orbital=self.config.gw_omega_orbital,
            madm=self.config.gw_madm,
            regenerate=regenerate_gw,
            generate_if_missing=generate_if_missing,
            backend=backend,
        )

    def load_strain_radii(
        self,
        regenerate_gw=False,
        gw_parfile_indices=None,
        psi4_mode=None,
        generate_if_missing=False,
    ):
        if "strain_radii" in self.loaded and not regenerate_gw:
            return True
        if self.config.gw_omega_orbital is None or self.config.gw_madm is None:
            print(f"{self.config.name}: missing gw_omega_orbital/gw_madm; skipping radii strain")
            return False
        indices = self.gw_parfile_indices(gw_parfile_indices)
        if not self.load_psi4(psi4_parfile_index=indices[-1] if indices else None, psi4_mode=psi4_mode):
            return False
        ell, emm = self.psi4_mode
        self.strain_radii = {}
        self.rh_t_radii = {}
        self.rh_plus_lm_radii = {}
        self.rh_cross_lm_radii = {}
        for parfile_index in indices:
            extraction_index = parfile_index + 1
            radius_label = str(extraction_index)
            file_label = self.gw_extraction_file_label(parfile_index)
            if file_label not in self.psi4_files:
                print(f"{self.config.name}: parfile index {parfile_index} maps to Psi4 label {file_label}, but that file was not loaded; skipping strain")
                continue
            workdir = self.gw_workdir(file_label)
            result = convert_psi4_to_strain(
                self.psi4_files[file_label],
                workdir=workdir,
                omega_orbital=self.config.gw_omega_orbital,
                madm=self.config.gw_madm,
                regenerate=regenerate_gw,
                generate_if_missing=generate_if_missing,
            )
            rh_plus_lm, rh_cross_lm = result.hplus_hcross(ell=ell, emm=emm)
            self.strain_radii[radius_label] = result
            self.rh_t_radii[radius_label] = result.time
            self.rh_plus_lm_radii[radius_label] = rh_plus_lm
            self.rh_cross_lm_radii[radius_label] = rh_cross_lm
        if not self.strain_radii:
            return False
        self.loaded.add("strain_radii")
        return True

    def load_initial_data(self):
        if "initial_data" in self.loaded:
            return True
        if self.data_path_initial_data is None:
            print(f"{self.config.name}: no initial data path configured")
            return False
        def xp_suffix(path):
            match = re.search(r"_xp(\d+)\.txt$", path.name)
            return int(match.group(1)) if match else None
        emdg_by_suffix = {xp_suffix(path): path for path in self.data_path_initial_data.glob("emdg_xp*.txt") if xp_suffix(path) is not None}
        ell_by_suffix = {xp_suffix(path): path for path in self.data_path_initial_data.glob("ell_xp*.txt") if xp_suffix(path) is not None}
        common_suffixes = sorted(set(emdg_by_suffix) & set(ell_by_suffix))
        if not common_suffixes:
            print(f"{self.config.name}: missing matching emdg_xp*.txt/ell_xp*.txt pair in {self.data_path_initial_data}")
            return False
        suffix = common_suffixes[-1]
        self.emdg_path = emdg_by_suffix[suffix]
        self.ell_path = ell_by_suffix[suffix]
        self.emdg_data = np.loadtxt(self.emdg_path)
        self.ell_data = np.loadtxt(self.ell_path)
        self.emdg_x = self.emdg_data[:, 0] # -> divide
        self.emdg = self.emdg_data[:, 1]
        self.ell_x = self.ell_data[:, 0]
        self.ell = self.ell_data[:, 1]
        self.rho_initial = np.full_like(self.emdg, np.nan, dtype=float)
        rho_mask = np.isfinite(self.emdg) & (self.emdg > 0)
        self.rho_initial[rho_mask] = np.power(self.emdg[rho_mask] / self.config.kappa, 1.0 / (self.config.gamma - 1.0))
        self.initial_data_xp_suffix = suffix
        self.loaded.add("initial_data")
        return True

    def load_rho2d(
        self,
        variable="rho_b",
        plane="xy",
        iteration=-1,
        ref_level=None,
        start_byte=None,
        required_ref_levels=None,
        region=None,
        selection_grid_shape=None,
    ):
        if ref_level is None:
            ref_level = list(range(8, 13))
        filepath = self.data_path_2d / f"{variable}.{plane}.asc"
        if isinstance(start_byte, TwoDFilePosition):
            filepath = start_byte.filepath
            start_byte = start_byte.start_byte
        self.rho2d = read_2d_file(
            filepath,
            variable=variable,
            plane=plane,
            iteration=iteration,
            ref_level=ref_level,
            components="all",
            start_byte=start_byte,
            required_ref_levels=required_ref_levels,
            region=region,
            selection_grid_shape=selection_grid_shape,
        )
        self.rho2d_source_path = Path(filepath)
        self.loaded.add("rho2d")
        return True

    def rho2d_source_paths(self, variable="rho_b", plane="xy"):
        filename = f"{variable}.{plane}.asc"
        authoritative = tuple(root / filename for root in self.data_paths_2d)
        supplemental = tuple(root / filename for root in self.supplemental_2d_paths)
        return authoritative, supplemental


def load_sims(
    diagnostics=(),
    names=None,
    skip_missing=True,
    psi4_parfile_index=None,
    psi4_mode=None,
    gw_parfile_indices=None,
):
    sims = apply_styles([DiskSim(config) for config in all_sim_configs(names)])
    loaded = []
    for sim in sims:
        try:
            ok = True
            for diagnostic in diagnostics:
                if diagnostic == "constraints":
                    ok = sim.load_constraints() and ok
                elif diagnostic == "modes":
                    ok = sim.load_modes() and ok
                elif diagnostic == "rhomax":
                    ok = sim.load_rhomax() and ok
                elif diagnostic == "spin_parameter":
                    ok = sim.load_spin_parameter() and ok
                elif diagnostic == "J_Rs":
                    ok = sim.load_J() and sim.load_Rs() and ok
                elif diagnostic == "Rs":
                    ok = sim.load_Rs() and ok
                elif diagnostic == "M0MADM":
                    ok = sim.load_M0MADM() and ok
                elif diagnostic == "tripleM":
                    ok = sim.load_rhomax() and sim.load_spin_parameter() and sim.load_M0MADM() and ok
                elif diagnostic == "psi4":
                    ok = sim.load_psi4(psi4_parfile_index=psi4_parfile_index, psi4_mode=psi4_mode) and ok
                elif diagnostic == "strain":
                    ok = sim.load_strain(
                        psi4_parfile_index=psi4_parfile_index,
                        psi4_mode=psi4_mode,
                    ) and ok
                elif diagnostic == "strain_radii":
                    ok = sim.load_strain_radii(
                        gw_parfile_indices=gw_parfile_indices,
                        psi4_mode=psi4_mode,
                    ) and ok
                elif diagnostic == "initial_data":
                    ok = sim.load_initial_data() and ok
                elif diagnostic == "rho2d":
                    ok = sim.load_rho2d() and ok
                else:
                    raise ValueError(f"Unknown diagnostic {diagnostic}")
            if ok:
                loaded.append(sim)
        except (OSError, ValueError, IndexError) as exc:
            if not skip_missing:
                raise
            print(f"{sim.config.name}: skipping; {exc}")
    return loaded
