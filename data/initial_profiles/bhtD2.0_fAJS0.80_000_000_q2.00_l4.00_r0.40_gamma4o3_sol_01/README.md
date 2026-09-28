# Massless Initial Profiles

Read-only source (retrieved 2026-09-12):
`/anvil/projects/x-mca99s008/Initial_data/BHT/bhtD2.0_fAJS0.80_000_000_q2.00_l4.00_r0.40_gamma4o3_sol_01/`.

`omeg_xp70.txt` and `peos_parameter_output.dat` are unchanged source copies.
`emdg_xp70.txt` and `emdg_xz70.txt` are derived from raw COCAL fluid/grid data,
not evolution data. No interpolation, mass rescaling, or NaN filling is done
during this export. The plot reader applies its documented atmosphere checks.

Reproduce locally after copying the raw files and `omeg_xp70.txt` to SOURCE:

```python
from config import all_sim_configs
from wip_plots.toomre import extract_cocal_profiles
extract_cocal_profiles(SOURCE, all_sim_configs(["ML"])[0].initial_data_path, 70)
```

The routine follows `IO_input_matter_BHT_export.f90` and
`IO_output_solution_3D_BHT.f90`: columns q=P/rho_0 and Omega, radius fastest,
then theta, then phi. Exports phi=0/pi meridians and theta=pi/2 positive-x ray.
The raw grid has 401 radial, 97 theta, 49 phi points. The full equatorial Omega
profile matches the existing suffix-70 export to its printed precision.
q_max=0.00340363620916 and Omega_peak=0.441137056342 match sequence 1 in
`bhtphyseq.dat`; its disk rest mass is 9.20354e-9 in solar-mass code units.
The existing ML config (K=1.45708, Gamma=4/3, M=0.05) is unchanged.

SHA256 of original inputs:

| File | SHA256 |
| --- | --- |
| rnsflu_3D.las | 5e7c0efbb183d2fb599bab6cc5648c0a49cdc6fbf8bb7d5cd44a1316b358fe69 |
| rnsgrids_3D.las | a5f4f1509a7b56e043d75a57e0cd6cab7adc041b81eb0f36157fe302f943475b |
| omeg_xp70.txt | 3374249aebcaa6bbfce76993ee97df99ae73b420011583f53c224d1ad682472d |

Raw files remain outside the repository; normal plotting only needs these
small exports. No `ell_xp` file is fabricated or needed for Toomre Q.
