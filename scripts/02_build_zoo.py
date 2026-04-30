"""
Phase B1: render the algorithm zoo grid from the cached thumbnails.

Usage: python scripts/02_build_zoo.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.zoo import render_zoo  # noqa: E402


def main() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = config.ZOO_DIR / f"algorithm_zoo_{timestamp}.png"
    render_zoo(out)


if __name__ == "__main__":
    main()
