#!/usr/bin/env python3
"""运行手术室入/出室检测流水线。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.pipeline import ORIOPipeline, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="OR bed+patient enter/exit (YOLO12+ByteTrack)")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs/run"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--target-mode",
        type=str,
        default=None,
        choices=["bed_patient", "stretcher", "person"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.device is not None:
        cfg["model"]["device"] = args.device
    if args.target_mode is not None:
        cfg.setdefault("target", {})["mode"] = args.target_mode

    pipe = ORIOPipeline(cfg, project_root=ROOT)
    summary = pipe.process_video(args.source, args.output, max_frames=args.max_frames)

    summary_path = Path(args.output) / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 运行结果 ===")
    print(f"target: {summary.get('target_mode')}")
    print(f"enter: {summary['enters']}  exit: {summary['exits']}")
    print(f"events: {summary['events_path']}")
    print(f"video:  {summary['video_path']}")


if __name__ == "__main__":
    main()
