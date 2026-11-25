#!/usr/bin/env python3
"""Run this experiment through the portable release launcher; use --help for options."""
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve()
launcher = here.parents[3] / "scripts" / "turbo.py"
raise SystemExit(subprocess.call([sys.executable, str(launcher), "run", here.parent.name] + sys.argv[1:]))
