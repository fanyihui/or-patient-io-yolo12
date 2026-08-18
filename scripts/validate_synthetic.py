#!/usr/bin/env python3
"""端到端验证：推床合成视频应检测到 1 次入室 + 1 次出室。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.pipeline import ORIOPipeline, load_config  # noqa: E402


def main() -> int:
    source = ROOT / "data/samples/or_door_synthetic.mp4"
    if not source.exists():
        print("请先运行: python scripts/generate_synthetic_video.py")
        return 2

    cfg = load_config(ROOT / "configs/default.yaml")
    cfg["model"]["device"] = "cpu"
    cfg.setdefault("target", {})["mode"] = "bed_patient"
    out = ROOT / "outputs/validate"
    pipe = ORIOPipeline(cfg, project_root=ROOT)
    summary = pipe.process_video(source, out)

    ok = summary["enters"] >= 1 and summary["exits"] >= 1
    print(json.dumps({"ok": ok, "enters": summary["enters"], "exits": summary["exits"]}, ensure_ascii=False))
    if not ok:
        return 1
    print("验证通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
