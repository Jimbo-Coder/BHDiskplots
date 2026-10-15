# Initial profiles

These files are the matching `emdg_xp*.txt` and `ell_xp*.txt` inputs used by
`paper_plots/initial_data_all.py`. They were copied verbatim from the six
corresponding BHT initial-data directories on Anvil. Keeping this small input
set in the repository makes the paper figure reproducible without another
machine-specific data root.

The bundled `bhtphyseq.dat` tables also make the compactness comparison
reproducible. The three A directories contain identical copies of the full
17-model A sequence; the three B directories contain the full 43-model B
sequence. The comparison reads one copy per family (A1 and B1), matches evolved
models by sequence number, and retains all unevolved models for the optional
sequence overlay. See the repository README's compactness instructions.
