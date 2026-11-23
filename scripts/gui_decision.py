#!/usr/bin/env python3
"""Run from the repository root after installing requirements/gui_decision.txt."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from improving.gui_decision.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
