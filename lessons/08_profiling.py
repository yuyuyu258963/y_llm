"""Chapter 08 delegates to the reproducible cache-versus-recompute benchmark."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parents[1] / "benchmarks" / "decode.py"), run_name="__main__")
