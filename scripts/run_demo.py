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


def resolve_device(requested: str | None) -> str | int:
    """auto / None → GPU 0（若可用）否则 cpu；显式值原样返回。"""
    value = (requested or "auto").strip().lower()
    if value not in ("auto", ""):
        # "0" / "0,1" 保持字符串也可被 ultralytics 接受；单卡数字更直观
        if value.isdigit():
            return int(value)
        return value
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            print(f"[device] CUDA available → using GPU 0 ({name})")
            return 0
        print("[device] CUDA not available → using CPU")
        return "cpu"
    except Exception as e:  # noqa: BLE001
        print(f"[device] torch/CUDA check failed ({e}) → using CPU")
        return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="OR bed+patient enter/exit (YOLO-World + ByteTrack)")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--output", type=str, default=str(ROOT / "outputs/run"))
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="auto / cpu / 0 / 0,1；默认读配置，配置为 auto 时自动选 GPU",
    )
    parser.add_argument(
        "--target-mode",
        type=str,
        default=None,
        choices=["bed_patient", "stretcher", "person"],
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    requested = args.device if args.device is not None else str(cfg.get("model", {}).get("device", "auto"))
    cfg.setdefault("model", {})["device"] = resolve_device(requested)
    if args.target_mode is not None:
        cfg.setdefault("target", {})["mode"] = args.target_mode

    print(f"[device] model.device = {cfg['model']['device']!r}")
    pipe = ORIOPipeline(cfg, project_root=ROOT)
    summary = pipe.process_video(args.source, args.output, max_frames=args.max_frames)

    summary_path = Path(args.output) / "summary.json"
    summary["device"] = cfg["model"]["device"]
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== 运行结果 ===")
    print(f"device: {cfg['model']['device']}")
    print(f"target: {summary.get('target_mode')}")
    print(f"enter: {summary['enters']}  exit: {summary['exits']}")
    print(f"events: {summary['events_path']}")
    print(f"video:  {summary['video_path']}")


if __name__ == "__main__":
    main()
