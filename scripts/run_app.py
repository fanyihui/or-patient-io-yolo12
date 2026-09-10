#!/usr/bin/env python3
"""启动 OR Gate 手术室入室监测 Web 应用。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.webapp import main  # noqa: E402


if __name__ == "__main__":
    main()
