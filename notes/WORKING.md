# BHDiskplots Work Log

Last updated: 2026-09-28

## Current Work

- Sep28 full non-movie rerun at commit 485e76a (corrected detector response):
  login00, PID 1124179, 12:29-12:38 EDT, run_logs/nonmovie_20260928_122954.log
  (.pid/.info/.exit/_tests.log). Preflight tests passed; GW cache processed=75,
  failures=0; paper + WIP + individual all --extra; 143 figures saved, exit 0.
  Detectability/gw_radiated numbers equal the earlier Sep28 regeneration.
  The local uncommitted MareNostrum data_roots change was NOT deployed.

- Sep28 audit and corrections:
  - Detector response CORRECTED. With Eq. (8) h_res, a right-angle
    interferometer's sky- and polarization-angle-averaged response is
    <|F+h+ + Fx hx|^2> = (2/5) h_res^2, so S_eff = (5/2) S_n (ASD x sqrt(5/2)).
    Wessel footnote 4's sqrt(10) multiplies where its own stated aim (a
    sky-position-only curve) requires dividing, and halves every SNR. The
    SciRDv1 LISA table is the single-channel 20/3 curve (exactly 2x Robson+2019
    at low f); now two channels and h_res give 0.5 x sqrt(PSD). A quadrature
    test of the actual F+/Fx response and a Robson comparison pin both.
    Expected: ground/DECIGO SNRs 2x, LISA 2.83x the earlier values; horizons
    grow similarly at low z. All earlier SNR/horizon numbers in these notes
    use the superseded convention.
  - rpsi4_uniform.dat rows past the last source sample are exact zero fill
    (2-6 rows per cache). Detectability and the temporal trial now stop at
    the last measured sample (local A1-A3 CE SNR change <= 0.21%).
  - New `run_wip.py gw_radiated`: cumulative E_GW/M_disk and J_GW for the
    detectability modes (legacy parity-tested flux code, after the 1000 M_BH
    cut) plus the smoothed (2,2) Psi4 frequency over f_orb. JSON summary also
    reports all-mode E/J, to expose any m=0 dominance.
  - README condensed to an index; unused imports removed; stale
    .gitattributes path fixed; Anvil's newer 2D index taken locally.
  - Verified on Anvil: 112 tests pass (run_logs/audit_20260928_tests.log);
    `run_wip.py detectability temporal_trial gw_radiated` exit 0
    (run_logs/audit_20260928_122239.log; pre-change JSON reports kept in
    run_logs/pre_audit_20260928/). Every SNR changed by x2.000 (LISA x2.828)
    apart from negligible band-edge values; fit parameters moved <0.6% from
    the zero-fill trim; no pass/fail or conditional verdict changed. Example
    SNRs now: 50 Msun/100 Mpc CE 1.4/4.9/3.5 (A1/A2/A3); 1e3 Msun/500 Mpc
    DECIGO 13/43/36; 1e5 Msun/50 Mpc LISA 18/60/50. Figures inspected
    (figures/review/audit_20260928/).
  - gw_radiated results (outer radius, after 1000 M_BH): l=2, m!=0
    E_GW/M_disk = 1.8e-8, 1.7e-8, 4.4e-9 (A1-A3), 5.7e-9, 9.2e-10, 2.5e-10
    (B1-B3). Including all 21 modes raises E by 10^4-10^5 (e.g. A1 3.2e-4):
    the m=0 FFI content dominates the "radiated" energy, further evidence it
    is non-radiative drift/junk rather than disk emission. Late (2,2)
    frequency is 1.4-3.3 f_orb, i.e. not locked to 2 f_orb.

- Sep23 luminosity-distance figures: added a linked right-hand redshift axis
  using the same FlatLambdaCDM distance mapping as the analysis, with sparse
  ticks and no duplicate distance ticks or extra gridlines on the right.
  Both measured and conditional-tail horizon plots use the shared plotter.
  All 103 tests pass locally and on Anvil, including tick-position agreement,
  changing distance limits, unchanged data and label-spacing checks.
  Regenerated detectability successfully (exit 0), retrieved and inspected
  the actual LaTeX figures; the complete off/on SNR and tail-fit report is
  exactly unchanged. Log: run_logs/detectability_redshift_20260923_133304.log.
  Added notes/gw_paper_methods.md earlier today and documented the linked
  axis there. Sources, notes and new figures are synchronized; no GW cache,
  raw simulation data or scientific-analysis settings changed.

- Sep22 ground-based detectability example: refreshed the relevant Anvil
  files and verified they matched locally. Changed only the 50 Msun example
  distance from 10 to 100 Mpc; both measured and conditional-tail charts
  share it and recompute cosmological redshift. DECIGO/LISA examples and
  waveform, window, cutoff, averaging and detector settings are unchanged.
  The cache-first full run below is now confirmed complete: 75 cache
  products, zero failures, then paper + WIP + individual all --extra.
  All 102 local tests passed. Regenerated combined detectability products
  successfully on Anvil (exit 0); inspected both off/on strain figures.
  Log: run_logs/detectability_100mpc_20260922_233340.log. Retrieved the
  figures/report locally. Fit reports and DECIGO/LISA example SNRs are
  exactly unchanged; only the ground-based example distance/redshift changed.

- Sep22 cache-first correction: full non-movie runs now default to
  RUN_GW_CACHE=True. Existing GW_REGENERATE_EXISTING=True rebuilds all
  configured A/B/ML radii and disk-minus-ML products from current source data.
  No separate per-plot regeneration. The runner already aborts on cache
  failure; regression tests now check that and the default stage ordering.
  Updated README to match. The prior Sep22 run completed all plots but used
  the September 12 GW cache. Both local and Anvil preflight suites passed
  102 tests. Launched the cache-first run with condactivate under nohup:
  login03.anvil.rcac.purdue.edu, PID 3364090, started 20:11:23 EDT.
  Log: run_logs/nonmovie_20260922_201114.log, with matching .pid/.info files
  and _tests.log. Verified detached (PPID 1), regenerating A1 cache products
  with the Python backend. Paper + WIP + individual all --extra follow only
  after successful cache generation. Later verified: 75 cache products,
  zero failures, and the final full non-movie completion message.
  No movies, raw-data changes, commits or pushes.

- Sep22 requested full non-movie rerun: refreshed the Anvil source tree and
  verified all 75 active source/reference files already matched locally; no
  source overwrite or plot-setting change was needed. Both local and Anvil
  preflight suites passed all 100 tests. Launched with condactivate and
  single-thread BLAS under nohup, verified detached (PPID 1):
  login01.anvil.rcac.purdue.edu, PID 1459497, started 17:27:52 EDT.
  Log: run_logs/nonmovie_20260922_172743.log, with matching .pid/.info files
  and _tests.log. Runs paper + WIP + individual all --extra; preserves
  RUN_GW_CACHE=False, so existing GW caches are reused. No movies, raw-data
  changes, figure deletion, commits or pushes. At launch it entered paper
  plots; subsequently confirmed the final completion message. Moved the inspected, completed
  Sep14 root nohup0.out into run_logs/nonmovie_20260914_legacy.log. The two
  local-only movie scripts remain untracked and were not deployed.

- Sep21 follow-up: budget figures now use raw code-unit mass changes, labeled
  Delta M [M_sun], not percentages. The prior normalization used ONE initial
  outside-AH mass for all three curves in a case, never separate curve scaling.
  Extended the late comparison to B1-B3 using each case's own existing GW fit
  interval. B cases have positive inward rest-mass flux but decreasing estimated
  BH mass. Area-derived mass grows while M_BH-M_irr falls by slightly more;
  cancellation also makes A1's net gain small. This does not establish whether
  the drift is physical or numerical. No time extrapolation in these budgets.
  ET horizon curves are dashed to distinguish them from solid CE curves; no
  sensitivity or spectrum values changed. Refreshed the affected Anvil sources
  and B strain caches read-only before local edits; all 100 local tests pass.
  Verification logs: run_logs/raw_budget_20260921_tests.log and
  run_logs/raw_budget_20260921_render.log. Scientific values in KNOWLEDGE.md.

- Sep21 local todo.txt: refreshed Anvil sources first; all 73 downloaded files
  matched locally, with no remote-only edits. Copied only the needed scalar
  diagnostics and A1-A3 outer GW caches read-only into /tmp for local work.
  Added public ET CoBA 10 km triangular-network noise, A-only detectability,
  flux-based remaining-mass fits and full/late-interval BH mass comparisons.
  Restored the requested combined time-on strain AND horizon charts using
  direct Psi4 throughout: analytic q'' for the Wessel model, same measured
  onset/frequency floor, unchanged other modes, explicit conditional A1 only.
  Real A1-A3 off arrays equal the pre-edit baseline exactly. A2/A3 on arrays
  reuse that baseline, since waveform fits still fail. Fit/holdout limits and
  conditional-review list were NOT relaxed. All 99 local tests pass, including
  analytic differentiation, ET response, angular reconstruction and overlap
  checks. All 99 tests also pass on Anvil. Synced only finished changes after
  verifying remote source hashes, then regenerated detectability, temporal
  diagnostics, accretion and standalone restmass with condactivate. Retrieved
  and inspected the actual Anvil LaTeX figures; all local/Anvil off/on SNRs
  match (relative tolerance 1e-7). Logs: run_logs/notes_20260921_tests.log and
  run_logs/notes_20260921_render.log. Fresh local review images are under
  figures/review/notes_20260921/ with integer suffixes. Writing prompts remain
  in todo.txt; interpretation guidance/results are in KNOWLEDGE.md. No raw
  simulation files, paper-plot calculations, GW caches, commits or pushes changed.

- Temporal comparison correction, Sep16: the user identified that the new
  combined time-off spectrum differed from the established direct-Psi4 chart.
  The new pair used cached FFI strain, which was an unrequested baseline change.
  Removed that pair and restored the analysis, renderer and GW tests exactly
  to the pre-change Anvil source snapshot. Existing direct-Psi4 production and
  individual FFI tail diagnostics are retained. A combined time-on chart is
  still pending agreement on a baseline-preserving method, not completed.
  Stopped the new comparison run before its combined pair was saved on Anvil.
  Restored standalone restmass.py and default WIP registration are retained.
  Removed the unnecessary reference archive; curated references remain.
  Corrected sources synced after checking the prior deployed hashes. All 92
  tests pass locally and on Anvil; restmass and the original detectability
  figures regenerated successfully. Retrieved and visually checked the actual
  Anvil baseline and restmass PNGs (local review suffix 20260916_2).
  Logs: run_logs/baseline_correction_tests_20260916.log and
  run_logs/baseline_correction_render_20260916.log. No new tail method, mass
  rescaling, windows, cutoffs or fit thresholds were introduced in the fix.

- Source synchronization audit, Sep16: pulled Anvil into a local staging
  directory before comparing checksums. Configuration, readers, paper plots,
  ordinary GW analysis/cache and the recent Toomre/accretion scripts matched.
  Remote lacked the local temporal implementation, renderer/entry point,
  tests, current notes/README and saved Wessel PDF. Its two extra user notes
  were copied locally unchanged before review. Existing production
  functions/classes match structurally in both GW files (39 analysis, 11 plot).
  Deployed all missing active files with baseline hash checks: all 128 source,
  reference and input files match. All 90 tests passed on Anvil. The Anvil
  temporal trial completed: A1 has fit/transition/on-off figures, while the
  other five cases have fit diagnostics only because the models were rejected.
  Logs: run_logs/source_sync_20260916_tests.log and temporal_trial_20260916.log.
  This entry supersedes older LOCAL ONLY statements as files are synchronized;
  old entries describe what had happened at their recorded time, not policy.
  At this earlier sync, the combined three-rescaling extrapolation figure
  had not yet been implemented; see the newer entry above.

- Shibata styling and cleanup: replaced script-specific font/LaTeX/DPI
  overrides with the shared setup, axis formatting, legend and save helpers.
  Data, classification boundaries and output names are unchanged; both local
  previews inspected, 91 local tests pass. Retained the detailed human workflow
  and carried over the unfinished temporal/radial requests before removing
  the short plot note and empty restore reminder. Collected original collaborator
  scripts from the obsolete singular BHDiskplot tree into
  notes/reference/collaborator_scripts_ref.tar.gz. Its two old tar files are
  exact subsets of the collected directories, and current notes already retain
  the GW reference and knowledge. No active code or symlinks depend on that tree.
  Both figures regenerated on Anvil and actual LaTeX PNGs inspected; the bold
  local Mathtext preview was a font fallback, not the final Anvil styling.
  91 tests pass on Anvil. Removed the two redundant remote notes and the obsolete
  singular tree after verifying preserved reference-file hashes. No scratch
  changes. All 127 active source/reference/input files match locally/remotely.
  Logs: run_logs/cleanup_20260916_tests.log and shibata_20260916.log.
  Fresh review images: figures/review/shibata_20260916_2/*_anvil_2.png.
  The optional historical archive was subsequently removed at user request;
  it was never a runtime dependency and curated references were already present.

- Added `wip/toomre_cases.png`: one six-panel A/B figure with original,
  Romeo and Meidt curves overlaid for each case. Method colors/line styles
  have one shared legend; no new individual files or entry points. Original
  calculations and figures unchanged. All 90 local tests and 12 focused
  Anvil tests pass; finished script/tests synced after remote hash checks.
  Anvil render retrieved and inspected as
  figures/review/toomre_20260916_1/toomre_cases_anvil_20260916_1.png.
  Re-read the staged diagnostic writer and flux/volume routines to explain
  mass-budget provenance: blue is decrease in column 2 minus column 3;
  orange integrates negative column 12, not a derivative or fitted decay.
  Both use the first outside-AH scalar mass as their percentage denominator.
  The flux routine accounts for horizon motion and shape change; no changes
  to accretion code/data were made. No full workflow restart was needed.

- Launched the requested nonmovie run on Anvil, 2026-09-16 12:59:03 EDT:
  login04.anvil.rcac.purdue.edu, PID 3529606,
  `run_logs/nonmovie_20260916_125903.log` and matching .pid. Verified detached
  (PPID 1) at 1m11s, progressing through paper panels. Uses condactivate and
  unchanged config: paper + WIP + individual all --extra, RUN_GW_CACHE=False,
  no movies or experimental temporal trial. Confirmed finished during the
  Toomre comparison sync: final B3 validation saved and log ends with
  "Complete non-movie workflow finished." Previous Sep15 run finished.
  Refreshed the relevant remote runner/config/helpers before syncing only
  wip_plots/accretion.py and the remote runner's new accretion entry. Escaped
  the budget ylabel percent for Anvil LaTeX; ten focused tests still pass.
  The local temporal opt-in runner changes remain local, not exposed against
  the older remote GW implementation. No scratch changes, commit or push.

- Added requested Romeo/Falstad and Meidt WIP comparisons in the existing
  toomre.py, generated by the same `run_wip.py toomre` command. Two-panel
  figures retain Q_N above each alternative. T=1.5 assumes isotropic thermal
  support; Q_M=kappa^2/(4 pi rho_mid) is the separate local 3D diagnostic.
  Do not combine their corrections or claim validated GR stability results.
  Original six-case numerical arrays and rendered toomre.png are exactly
  unchanged. All 89 local tests pass; local Mathtext renders inspected.
  Baseline and previews: figures/review/toomre_20260916_1/. Synced only the
  finished toomre.py and its tests after checking remote baseline hashes;
  all 11 focused tests passed on Anvil, all three figures regenerated with
  condactivate, and the retrieved LaTeX comparison figures were inspected.
  Log: run_logs/toomre_20260916_comparisons.log. Preview files have suffix
  _anvil_20260916_1.png. README/knowledge/work log updates remain local.
  No GW code, scratch inputs, commits or pushes were modified/performed.

- Local accretion investigation: added `run_wip.py accretion`, also in normal
  WIP runs. One plot file owns a linear Wessel-style rate figure (A/B separated,
  source BH mass 10 Msun) and measured-mass-loss versus integrated-AH-flux panels.
  Initial-data time zero is retained; no automatic PPI-saturation alignment.
  The Sep15 six-case scalar inputs reproduce the simulation's int_M0dot within
  0.22% of final accreted mass, but do NOT close the outside-AH mass budget.
  A1: 6.793% accreted versus 0.843% measured loss. Restart overlap is not the
  explanation. Some zero-flux samples precisely follow the source's missing-AH
  branch; show these as gaps without changing raw data or the numerical integral.
  Details and the six-case table are in KNOWLEDGE.md. Local 85 tests pass;
  both six-case figures rendered/inspected using Mathtext (not Anvil LaTeX).
  Final previews: figures/review/accretion_20260916_1/wip/*_20260916_3.png.
  Paper M0dot, readers, GW pipeline, and simulation data unchanged. No SSH,
  Anvil sync, commit, or push. The physical/diagnostic cause remains unresolved.

- Launched the requested Anvil nonmovie workflow on 2026-09-15 at 22:44:55 EDT,
  using the current remote checkout and its condactivate environment. Host
  login06.anvil.rcac.purdue.edu, detached PID 2736314; repo-relative log
  run_logs/nonmovie_20260915_224455.log and matching .pid file. Verified after
  SSH disconnect (PPID 1): prior paper figures saved, currently entering panels.
  Scope: paper + combined WIP + all individual extras; RUN_GW_CACHE=False.
  No movies, no experimental temporal trial, and no local files synced.
  A nonfatal ML initial-profile warning reports a missing matching emdg/ell
  x-slice pair; the initial-data figure was still saved. Completion confirmed
  on Sep16 by the final Complete non-movie workflow finished log marker.

- Simplified the temporal figures for human review, local only. Removed all
  pass/fail headings, fit scores, held-out shading, fit-start markers and long
  captions. The fit figure now shows only disk mass, strain amplitude and
  unwrapped phase versus their fits over the fitted interval; a common phase
  constant is removed for display. Transition keeps data/blend/extrapolation
  curves and one labeled data-end boundary. On/off is now spectrum-only.
  The measured curve stays solid blue through the blend. Numerical scores,
  warnings and conditional status remain unchanged in the JSON report.
  All six reports exactly match the preceding run; 75 tests pass. Rendered
  and visually inspected all three A1 figures in
  figures/review/temporal_20260915_6/A1/; fresh display aliases end _12.
  No scientific changes, Anvil sync, commit or push.

- Added a separate temporal transition view after the fit-only diagnostic was
  mistaken for an extrapolation plot. `gw_temporal_transition.png` shows the
  real strain and amplitude around the join: measured signal, last-orbit blend,
  and future tail have distinct colors. Dotted blue preserves the original
  measured signal during blending; the dashed line marks the last data sample.
  A1 remains illustrative only, with failed fit checks explicitly labeled.
  All six numerical reports are exactly unchanged. Local 75 tests pass,
  including region boundaries and stale-output removal. Fresh display:
  figures/review/temporal_20260915_5/A1/gw_temporal_transition_20260915_10.png.
  No scientific edits, Anvil sync, production changes, commit or push.

- Temporal diagnostic annotation cleanup, local only: descriptive panel headings,
  shared Simulation / Full late-time fit / Earlier-fit model legend, contrasting
  earlier-fit curve, and one consistent held-out band across all validation
  panels. Captions explain the band, fit-start line, and complex-strain test
  error. Comparison marks the smooth-join interval and simulation endpoint;
  spectra say Measured only / With model tail and show the source scale.
  All six numerical JSON reports match the preceding run exactly. Regenerated
  figures: figures/review/temporal_20260915_4/; fresh A1 display aliases end _8.
  Local 74 tests pass. No scientific settings, Anvil files or production plots
  changed.

- Wessel physical-assumption follow-up, LOCAL ONLY: refreshed the six density-mode
  histories and reread the Illinois diagnostic writer/integrands from the
  collaborator project. Full M0 is a rho_star volume integral; C0 uses a density
  cutoff, so C0 cannot replace total disk mass. Added matter-mode context and
  fit-window results to the existing temporal diagnostics. A1 is an explicit
  conditional review case, retaining waveform failures; other cases remain
  unextended. CE SNR at 50 Msun/100 Mpc: 0.204 off, 0.725 on, but 0.645-1.296
  across tested windows. The default total duration is 163 times measured.
  Source structure and phase are not stationary enough to call this a prediction.
  Details/provenance: KNOWLEDGE.md; evidence:
  run_logs/wessel_physics_assessment_20260915.json and
  figures/review/temporal_20260915_3/. Production remains finite and unchanged.
  No Anvil writes, sync, commit or push. Local 73-test suite passes.

- Saved the unchanged Wessel accepted-manuscript PDF in notes/reference/.
  Source URL, checksum, and printed/PDF page mapping are in KNOWLEDGE.md.
  Read the local copy for subsequent method questions; no Anvil sync or push.

- Local Wessel temporal-continuation trial: refreshed the nine relevant source
  files, six outer strain caches and six bhns.don diagnostics from Anvil. All
  refreshed source files matched before editing; remote inputs were read only.
  Analysis remains in gw_detectability.py and rendering in the existing plot
  file. `run_wip.py temporal_trial` is opt-in, excluded from default/full runs.
  Fits mass decay, complex (2,2) phase and amplitude; checks a withheld final
  interval. Only accepted trials can make a smooth joined/tapered tail and
  like-for-like FFI off/on spectra. No negative-m mode is invented.
  All six cases failed the original strict criteria; 30 fits over five window choices
  also fail. This does not establish bad simulation data or an extraction bug.
  Local diagnostics: figures/review/temporal_20260915_2/{A1,...,B3}/.
  Detailed criteria, caveats and evidence are in KNOWLEDGE.md. Direct-Psi4
  spectra are exactly unchanged for all six cases versus the refreshed source.
  Local 71-test suite passes, including accepted synthetic off/on rendering;
  rejected real-data diagnostic figures were rendered and inspected.
  Changes are LOCAL ONLY, with no commit, push, Anvil writes or production run.

- Added Shibata Fig. 4-style face-on strain at 50 Msun/100 Mpc, one A/B figure
  each and one row per case. Existing FFI cache, explicit polar projection,
  physical units, peak alignment, and source knobs in one plot file. Registered
  as `strain_observer` in WIP/full runs. Fresh outer-radius caches copied down
  from Anvil; all refreshed source files matched local before editing.
  Local and Anvil 63-test suites pass; both three-row production outputs
  rendered and inspected. Log: `run_logs/strain_observer_20260914.log`.
  Finished scripts and notes synced; no cache rebuild or full-workflow restart.
  Scientific scope and normalization are recorded in KNOWLEDGE.md; ordinary
  FFI and direct-Psi4 detectability calculations have not been changed.

- Detectability presentation cleanup: three fixed mass/distance rescalings now
  share one h_c(f) axis; detector noise is drawn once and the A/B legend retained.
  Fixed examples are 50 Msun/10 Mpc, 1000 Msun/500 Mpc, 100000 Msun/50 Mpc;
  no reference-case optimization or retuning. Mass/distance labels use LaTeX
  scientific notation, without repeated redshifts or SNR annotations.
  Luminosity-distance horizons share one axis with detector group labels and
  the SNR threshold. Display ranges omit tiny tails only visually, not from SNR.
  Local 59-test suite passes; all six spectra and every horizon array are
  exactly unchanged against the pre-edit code. Evidence:
  `run_logs/detectability_layout_20260914.json`. Local renders inspected.
  Anvil: 59 tests pass and the three affected production figures regenerated;
  log `run_logs/detectability_layout_20260914.log`. No full-workflow restart.
  The Anvil human note also requests time-extrapolation comparisons and possibly
  radial extrapolation; those remain pending and are NOT part of this cleanup.
  Its restoration reminder refers to the intact `notes/human_gw_method.md`.

- Added initial `toomre` WIP plot using existing COCAL meridional density and
  equatorial rotation profiles for all six disks plus ML; no new HDF5 reader or raw
  simulation edits. Standard paper style, input/plot knobs at file top.
  Local seven-case render and 56 tests pass, including raw COCAL export ordering,
  rotation consistency, analytic Gaussian column
  integration and Keplerian/solid-body/negative epicyclic-frequency tests.
  Integration/derivative sensitivity and input conventions are recorded in
  KNOWLEDGE.md. This is explicitly a Newtonian diagnostic for GR initial data;
  physical interpretation for thick disks remains a collaboration decision.

- Detector annotations now bend with their noise curves in screen coordinates,
  using serif glyphs and the same color as the corresponding lines. One small
  rendering helper in `gw_detectability_all.py` handles both figure layouts;
  no analysis changes or added dependencies. Positions, normal insets and size
  remain top-of-file presentation settings.
  Verified both production layouts after regeneration on Anvil. All 49 tests
  pass locally and on Anvil, including finite glyph geometry and unchanged
  axis limits. Log: `run_logs/detectability_curved_labels_20260911.log`.
- Replaced the bottom detector legend with compact labels beneath the noise
  curves in both characteristic-strain figures. Shared label anchors/offsets
  are presentation knobs in the plotting file. The A/B case legend remains;
  reclaimed the unused bottom margin. Local real-cache renders inspected and
  48 tests pass; analysis, source scales, detector curves and SNRs unchanged.
  Production TeX render checked on Anvil; increased DECIGO's inset to clear
  its descending curve. Log: `run_logs/detectability_inline_labels_20260911.log`.
- Added a separate fixed 50 Msun / 100 Mpc six-case detectability figure,
  `figures/gw/gw_detectability_shibata.png`, to the normal detectability run.
  Shared observer conversion and rendering keep it consistent with the
  threshold examples. These are Shibata's reference source scales, not a fit
  to GW190521, not its inferred distance, and not a reproduction of Shibata's
  full waveform. Our existing modes, orientation average and detector models
  remain unchanged. The same cosmology supplies redshift at 100 Mpc.
  Local headless and Anvil verification: 48 tests pass on each. Production
  detectability figures regenerated; log `run_logs/detectability_shibata_20260911.log`.
  At the fixed scale, CE SNR ranges 0.312-2.46; all six cases remain below 8.
  No commit or push.
- Replaced borrowed Wessel example distances with our A1 SNR=8 examples.
  Choose the sampled maximum-reach mass per detector, round to two significant
  digits, then refine the outermost threshold crossing in redshift. All cases
  use the same pair; matching letters and stars connect the two main figures.
  CE: 62 Msun / 9.2799 Mpc; DECIGO: 3800 Msun / 1062.7620 Mpc;
  LISA: 340000 Msun / 103.2403 Mpc. Four-mode physics settings are unchanged.
  Local checks: 47 tests pass; three reference SNRs agree with 8 to 1e-7
  relative tolerance; both main figures rendered and inspected.
  Evidence: `run_logs/detectability_threshold_examples_20260911.json`.
  Anvil verification: 47 tests pass and the combined detectability figures
  regenerated with production TeX. Log:
  `run_logs/detectability_threshold_20260911.log`. No commit or push.
- Made the top-of-file detectability choices a numbered method specification,
  with reasons and measured limitations beside the defaults. Moved detector
  tables/bands and cosmology parameters out of buried implementation sections.
  Ran 120 local real-cache spectra (20 choices per disk), measuring SNR and
  cumulative bands for four detector/source pairs. Results and decisions are
  summarized in KNOWLEDGE.md; machine-readable evidence is in
  `run_logs/detectability_choices_20260911.json`. Keep the existing defaults,
  but identify subset completeness and finite-radius/low-frequency sensitivity
  as unresolved, not as established cleaning of the waveform.
  Fixed one guard: entirely pre-arrival data now fail instead of being treated
  as arrived. Local and Anvil verification: 46 tests pass on each; synced
  source hashes match. Ordinary GW processing and
  all default figure data remain unchanged; no figure regeneration needed.
- Mode option: `MODES` now accepts the default four-mode tuple or `"all"`
  (all 21 cached ell=2..4 modes, not the earlier 12-mode ell<=3 selection).
  No separate pipeline or CLI. Fewer modes do not establish cleaner physics.
  Wessel Sec. III.3 uses ell<=3; Sec. III.5 motivates the initial 1000 M_BH
  exclusion by hydrodynamic relaxation. Our alpha and time-cut comparisons
  test sensitivity, but the three-cycles/duration frequency floor remains a
  heuristic. We have not shown that all physical signal power lies in a
  validated band; low-frequency leakage is amplified by the 1/f^2 conversion.
  No new cutoff or automatic power-percentile truncation was introduced.
  Selector verification: 45 local tests pass; both 4- and 21-mode spectra
  produce finite results for all six cases using the existing outer cache.
  Default figures are unchanged, so no figure regeneration is needed.
- Keep one canonical BHDiskplots checkout per machine and gather notes here.
  Retired 57 local sync/review/script roots, initially with a verified recovery
  archive. Removed that redundant archive at the user's follow-up request;
  selected useful references remain in `notes/reference/`.
- Mac and Anvil started at `25c8b3d4868e10cd0688e68a446923e5932063ff`.
  Source checksum comparison found no code differences. Anvil had two new
  `human*` notes and newer 2D index contents; those were pulled locally.
  The previously local-only `wip_plots/disk_compactness_comparison.py` is now
  registered as `run_wip.py shibata` and synced to Anvil, with its reference
  and comparison PNG/PDF outputs under `figures/wip/` on both machines.
- [Current human GW proposal](human_gw_method.md) is preserved verbatim.
- Implemented the small human plot note: initial-data coordinates now divide
  by the case's ADM mass instead of BH mass and use `r/M`; the reference-curve
  anchors use the same ADM mass and their labels use `r`. The positive x ray
  is the coordinate radial profile, not an areal-radius reconstruction.
  The completed `human_notes.md` was removed after rendering verification.
- Detectability now has two editing locations: `gw_detectability.py` for the
  scientific settings and seven-step analysis, and
  `wip_plots/gw_detectability_all.py` for combined/individual figure rendering.
  Removed the old helper module and separate individual plotting script; no
  compatibility wrappers. Shared simulation metadata remains in `config.py`.
- Adopted the human note's four modes. Retained angular mean, alpha=0.05,
  transient/frequency cuts, cosmology and detector conventions; these are
  explicit review choices rather than claims that Moore prescribes them.
  Standard FFI time-domain analysis and plots are untouched.
- Local verification: 45 tests pass. All 18 case/radius spectra match the
  preceding calculation with identical four-mode inputs (rtol=2e-13).
  Ten real-data figures render using the downloaded caches and local mathtext;
  the Mac lacks the production TeX installation. Six-case combined analysis
  and four figures took about five seconds locally with BLAS threads limited.
  Four-mode spectra are substantially weaker than the historical 12-mode
  spectra; those historical SNR values must not be reused for current plots.
- Anvil verification completed: 45 tests pass; all four combined and six
  individual detectability figures regenerated with production TeX. Inspected
  characteristic strain, horizon and A1 validation; corrected the frequency
  label/detector legend overlap and kept legends clear of the plotted spectra.
  Source checksums match across hosts, including unchanged `gw.py` and the
  verbatim human note. Figures and the run log were pulled into the canonical
  local checkout. Log: `run_logs/detectability_refactor_20260911.log`.
  No full workflow regeneration, commit or push was performed.

## Human GW Proposal Versus Current Code

The comparison below is source-checked on 2026-09-11, not a claim that all
detector response factors or astrophysical assumptions have been approved.

- Steps 1-3: already implemented by `direct_psi4_spectrum` in
  `gw_detectability.py`. It transforms complex modes once, then sums
  spin-weight -2 harmonics, preserving positive and negative frequencies to
  reconstruct the real/imaginary polarization transforms. FFT linearity makes
  this equivalent to reconstructing each direction before its FFT when all
  modes use the same samples/window. The new test compares those two routes.
- Its amplitude is `sqrt((|FFT(Re Psi4)|^2 + |FFT(Im Psi4)|^2)/2)`.
  The human note's expression omits magnitudes/squares; the implementation
  uses the explicit amplitude above, not a literal complex sum. Angular mean is the configured
  default; RMS is already supported. Neither is the detector-sky average.
- Mode selection now follows the proposal: (2,2), (2,1), (2,-2), (2,-1).
  Change `MODES` at the top of `gw_detectability.py` without another reader
  or workflow implementation.
- Window selection differs: current alpha is 0.05; the note tentatively says
  0.1. Current pre-arrival, 1000 M_BH transient and 3-cycles/duration frequency
  cuts are additional assumptions absent from the new note. Review them on
  the data, rather than treating a paper on strain conventions as validating
  those simulation-specific values.
- Steps 4-5: characteristic strain and noise conversions exist already.
  [Moore et al., equations 16-21](https://arxiv.org/html/1408.0740v2#S2.SS2)
  give `h_c=2f|h_tilde|`, `h_n=sqrt(f S_n)` and the corresponding SNR integral.
  These equations do not settle which source-orientation statistic to use or
  whether each imported detector curve already includes response averaging.
- Steps 6-7: source-mass/redshift/distance scaling and SNR-threshold horizons
  exist. Current physical detector comparisons use Hz; dimensionless
  radius/method comparisons use `f/f_orbit`. A detector curve plotted against
  `f/f_orbit` also needs the chosen source mass/redshift and orbital frequency;
  do not reuse one dimensionless detector axis across differently scaled cases.

The implementation retains shared `rPsi4` preparation and keeps one analysis
file and one plotting file. Characteristic strain and SNR horizons are the
primary outputs; radius/direction/parameter checks are optional figures in the
same plotting file. No replacement GW reader or second transform pipeline.

## Historical Work (Through 2026-09-09)

The following records describe earlier work at the time it was performed.
Statements about not having pushed or regenerated apply to those checkpoints;
the consolidation was subsequently committed as `25c8b3d`.

This file records the current objective, open decisions, and next work. Move
settled conclusions into `KNOWLEDGE.md`.

## Code Cleanup

- Second pass: `GWRun.at_radius(index)` returns a stable radius-specific
  `Waveform`, not a mutable current selection. All modes remain in that result.
  Removed unused per-simulation mode/radius-list settings and reader accessors.
- Cache reads and generation are separate functions; only `generate_gw.py`
  chooses rebuild versus reuse. The numerical FFI functions are unchanged.
  Radius-to-LaTeX formatting lives with display helpers. Waveform plots select
  explicit named series functions rather than inferring behavior from names.
- Second-pass local verification: 42 tests, including generation/readback of
  case and difference caches without source edits. The same 24 synthetic
  figures match in all 344 recorded data/layout arrays and label text.
- Second pass synced after source checksum checks: Anvil also passed all 42
  tests and rendered 24 real-data figures; A1's direct spectrum again had 6316
  bins. No production figure regeneration, commit, or push was performed.

- `gw.py` now owns extraction reading, restart merging, the unchanged FFI
  equations, and cache access. Scalar/2D readers no longer own GW conversion.
- Nine ordinary waveform plotting scripts are consolidated in
  `wip_plots/gw_waveforms.py`. Combined jobs load each radius once for all modes
  and plot types; individual jobs reuse loaded extraction files and strain.
- Shared run, unit, and detectability settings are in `config.py`. Detector
  response and SNR calculations moved out of the plot into
  `helpers/gw_detectability.py`; no response/window/cutoff convention changed.
- Local verification: 41 tests; 24 old/new synthetic figures agree in plotted
  data, axis bounds, panel positions, and labels. Detector curves, binned power,
  and sampled horizon curves match exactly. Zero-power binning now returns an
  empty spectrum instead of indexing an empty array.
- Synced the tested cleanup to Anvil after checking all affected source
  checksums. Anvil passed the same 41 tests and rendered 24 real-data figures
  covering A1/B1/ML at both shared radii and A1 individual comparisons. The
  A1 direct spectrum contained 6316 bins and the smoke-test LIGO SNR was
  finite (0.583457 at the test's redshifted mass 10 Msun and distance 150 Mpc).
  No production figures were overwritten, no full regeneration was launched,
  and nothing was committed or pushed.
- This is a structural cleanup, not a fresh scientific sign-off. In particular,
  ordinary raw-Psi4 time-series plots still associate raw mode samples with the
  cached uniform retarded-time grid as before. Check the size of that mismatch
  before changing their sampling or the Psi4-difference reconstruction. The
  direct detectability spectrum already consumes the matching uniform Psi4
  intermediate. Do not conflate a parity test with validation of this choice.

## Current Objective

Make the BHDisk detectability pipeline scientifically defensible and easy to
explain from numerical `Psi_4` through characteristic strain and SNR, while
retaining explicit comparison products for assumptions that cannot yet be
eliminated.

## Completed

- Preserved the complete 696-line collaborator script verbatim.
- Replaced the mixed strain/radial/tail implementation with one finite direct
  `r Psi_4` production method.
- Added a focused reusable direct-Psi4 transform in
  `helpers/gw_detectability.py`.
- Added synthetic tests for FFT normalization, angular averaging, physical
  mass scaling, and cosmology inversion.
- Made source-direction mean the sole production source average and retained
  the representative angle only as an explicit comparison diagnostic.
- Unified detector plotting and SNR on the same effective ASD convention.
- Reduced paper-facing outputs to characteristic strain, horizon, and method
  comparison figures.
- Replaced radial-extrapolation/time-tail individual diagnostics with finite
  radius, window, and angular-resolution tests.
- Added a maintained NumPy implementation of the established gauge-corrected
  retarded-time and FFI workflow, while retaining the Fortran as a regression
  reference. Its cache exposes uniformly sampled `r Psi_4` directly to this
  detectability workflow without routing the spectrum through strain.
- Regenerated all 75 configured Anvil cache products with the NumPy backend
  without failures, then verified that all six detectability cases and both
  comparison radii consume `python-uniform-rPsi4`.

## Settled Method Decisions

- The former strain-FFT path depended on the FFI cutoff. It has been replaced
  for detectability by direct finite-duration `r Psi_4/(2 pi f)^2`.
- The former representative and RMS-like production alternatives have been
  replaced by the collaborator's solid-angle mean. The representative angle
  remains only in a labeled comparison plot.
- The former radial fit and synthetic late-time tail are not used in the
  production spectrum. Finite outer-radius data are the conservative central
  result. The first usable wave-zone radius (`r=120`) is always evaluated
  against the outer valid radius (`r=170`), with one intermediate radius shown only
  in the individual validation figure.
- The direct transform takes retarded time and `r Psi_4` from the shared
  maintained preprocessing cache, removes `t_ret<0`, and only then applies the
  `1000 M_BH` transient cut. Legacy caches retain the previous fallback.
- Uniform resampling now precedes the Tukey window.
- The Wessel detector-response prescription is encoded once and reused by the
  plotted noise and SNR/horizon calculation.
- The standard `h_c=2 f |h_tilde|` and one-sided matched-filter SNR forms are
  retained, with explicit redshifted-mass and luminosity-distance scaling.

## Remaining Verification

1. Decide how the finite-radius, Tukey-window, and transient-cut systematics
   should be summarized in the paper. The first/outer-radius comparison now has
   a combined figure, while the numerical-choice brackets remain in every
   individual validation figure.
2. If tighter uncertainty is required, improve extraction-radius control or
   obtain a validated radial-extrapolation method before changing the central
   finite outer-radius result.
3. Treat any future late-time continuation as a separately labeled upper-bound
   model; do not silently merge it into this finite-signal production result.

## Publication Gate

The method and figures are reproducible and internally consistent, but the
current finite-radius/window spread is too large to describe the amplitudes or
horizons as precision predictions. Present them as finite-signal estimates
with that numerical systematic stated explicitly.
