"""Output locations. Results go to <repo>/tests/results/<prefix>/<prefix>_<date>/.

Override the base folder with the PDPTW_RESULTS_DIR environment variable
(useful on CLAIX, e.g. to write to $WORK or a sciebo-synced folder).
"""
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def results_dir(prefix):
    base = Path(os.environ.get("PDPTW_RESULTS_DIR", REPO_ROOT / "tests" / "results"))
    out = base / prefix / f"{prefix}_{datetime.now().strftime('%Y-%m-%d')}"
    out.mkdir(parents=True, exist_ok=True)
    return str(out)
