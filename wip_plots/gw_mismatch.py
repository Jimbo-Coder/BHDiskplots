#!/usr/bin/env python3
"""Compatibility entry point; read or edit the complete root gw_mismatch.py."""
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gw_mismatch import main

if __name__ == "__main__":
    main()
