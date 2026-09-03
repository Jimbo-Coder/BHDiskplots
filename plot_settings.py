"""Shared display-unit choices for BHDisk figures.

Plot-specific limits, layouts, and filenames stay at the top of each plot
script. Only choices that must remain identical across several figures belong
here.
"""

# Non-GW time axes.
TIME_NORMALIZE_BY_PC = True
TIME_CODE_UNIT_MASS_MSUN = 1.0

# Time-domain GW axes and amplitudes.
GW_NORMALIZE_BY_M = True
GW_TIME_SCALE = "M_BH"  # "M_BH", "M_ADM", "P_c", or "code"
