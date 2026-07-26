"""Shared filesystem defaults for the integrated SSCI evaluation package."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_DATA_DIR = REPO_ROOT / "real_data"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "output"
