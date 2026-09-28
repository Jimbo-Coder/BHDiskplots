"""1D diagnostic and simulation data loading for BHDisk plots."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from config import DiskSimConfig, all_sim_configs
from .reader_2d import TwoDFilePosition, read_2d_file
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
        ncols = self.modedata.shape[1]
        if ncols < 6 or ncols % 2:
            raise ValueError(f"{self.modepath}: expected time, C0, and complex mode pairs")
        modes = ncols // 2
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
