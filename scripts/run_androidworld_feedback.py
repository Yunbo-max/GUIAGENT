#!/usr/bin/env python3
"""Second GUI benchmark: native AndroidWorld, learned feedback spectrum."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from improving.androidworld_feedback import main

if __name__ == "__main__":
    main()
