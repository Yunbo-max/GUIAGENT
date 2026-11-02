#!/usr/bin/env python3
"""Qwen3-VL MobileGym feedback-only experiment entrypoint."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from improving.mobilegym_feedback import main

if __name__ == "__main__":
    main()
