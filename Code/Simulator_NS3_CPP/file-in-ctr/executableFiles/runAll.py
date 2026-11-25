#!/usr/bin/env python3
"""Run all released experiments; use --help for options."""
from pathlib import Path
import subprocess
import sys

launcher = Path(__file__).resolve().parents[2] / "scripts" / "turbo.py"
raise SystemExit(subprocess.call([sys.executable, str(launcher), "run", "all"] + sys.argv[1:]))
