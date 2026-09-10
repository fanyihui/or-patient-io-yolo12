#!/usr/bin/env python3
"""基于流媒体的实时入/出室识别与事件记录。

支持:
  - RTSP / RTSPS:  rtsp://user:pass@ip:554/stream
  - HTTP(S) MJPEG/HLS: http://ip/video.mjpg  /  https://.../index.m3u8
  - RTMP: rtmp://...
  - 本地摄像头: 0  /  1  /  /dev/video0

事件即时写入:
  - events.jsonl  (追加)
  - events.csv    (追加)
  - events.json   (汇总快照)
标注视频可选写出 annotated*.mp4（可按时长切段）。
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.pipeline import ORIOPipeline, load_config  # noqa: E402
from or_io.stream import is_stream_source, source_label  # noqa: E402


def resolve_device(requested: str | None) -> str | int:
    value = (requested or "auto").strip().lower()
    if value not in ("auto", ""):
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
    parser = argparse.ArgumentParser(
        description="OR bed+patient realtime stream enter/exit recorder"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="rtsp://... / http(s)://... / rtmp://... / 摄像头索引(0) / 视频文件",
    )
    parser.add_argument("--config", type=str, default=str(ROOT / "configs/default.yaml"))
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录；默认 outputs/live_<timestamp>_<source_label>",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="调试用：最多处理帧数")
    parser.add_argument("--device", type=str, default=None, help="auto / cpu / 0")
    parser.add_argument("--target-fps", type=float, default=None, help="处理帧率上限，降负载保实时")
    parser.add_argument("--no-video", action="store_true", help="不写标注视频，只记事件")
    parser.add_argument("--no-reconnect", action="store_true", help="断流后不重连")
    parser.add_argument("--preview", action="store_true", help="弹窗预览（需 GUI；无头环境无效）")
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

    stream_cfg = cfg.setdefault("stream", {})
    if args.target_fps is not None:
        stream_cfg["target_fps"] = float(args.target_fps)
    if args.no_reconnect:
        stream_cfg["reconnect"] = False

    out_cfg = cfg.setdefault("output", {})
    out_cfg.setdefault("save_events_json", True)
    out_cfg.setdefault("save_events_jsonl", True)
    out_cfg.setdefault("save_events_csv", True)
    if args.no_video:
        out_cfg["save_video"] = False

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = source_label(args.source)
    output = Path(args.output) if args.output else (ROOT / "outputs" / f"live_{stamp}_{label}")

    live = is_stream_source(args.source)
    print(f"[device] model.device = {cfg['model']['device']!r}")
    print(f"[source] {args.source!r} live={live}")
    print(f"[output] {output}")

    pipe = ORIOPipeline(cfg, project_root=ROOT)
    if live:
        summary = pipe.process_stream(
            args.source,
            output,
            max_frames=args.max_frames,
            preview=bool(args.preview),
        )
    else:
        print("[warn] source 看起来像文件；仍按离线处理。实时流请用 rtsp/http/摄像头索引。")
        summary = pipe.process_video(args.source, output, max_frames=args.max_frames)

    summary["device"] = cfg["model"]["device"]
    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 实时运行结果 ===")
    print(f"device: {cfg['model']['device']}")
    print(f"live:   {summary.get('live')}")
    print(f"frames: {summary.get('frames')}  proc_fps: {summary.get('fps_process')}")
    print(f"enter:  {summary['enters']}  exit: {summary['exits']}")
    print(f"events: {summary['events_path']}")
    print(f"jsonl:  {output / 'events.jsonl'}")
    print(f"csv:    {output / 'events.csv'}")
    print(f"video:  {summary.get('video_path')}")
    print(f"summary:{summary_path}")


if __name__ == "__main__":
    main()
