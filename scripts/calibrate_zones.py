#!/usr/bin/env python3
"""标定门口：支持门线（2 点）或双 ROI 多边形。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--out-config", default=None)
    parser.add_argument("--mode", choices=["roi", "line"], default="roi")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = cv2.imread(args.source)
    if frame is None:
        raise SystemExit(f"无法读取: {args.source}")

    h, w = frame.shape[:2]
    outside: list[tuple[int, int]] = []
    inside: list[tuple[int, int]] = []
    line_pts: list[tuple[int, int]] = []
    editing = "outside"

    def on_mouse(event, x, y, flags, param):  # noqa: ARG001
        nonlocal editing
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if args.mode == "line":
            if len(line_pts) < 2:
                line_pts.append((x, y))
            return
        if editing == "outside":
            outside.append((x, y))
        else:
            inside.append((x, y))

    win = "calibrate_zone"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    print("ROI: 点 OUT → n → 点 IN → n → s 保存；q 退出")

    while True:
        vis = frame.copy()
        if args.mode == "line":
            for p in line_pts:
                cv2.circle(vis, p, 5, (0, 255, 255), -1)
            if len(line_pts) == 2:
                cv2.line(vis, line_pts[0], line_pts[1], (0, 220, 255), 2)
        else:
            if len(outside) >= 1:
                cv2.polylines(vis, [np.array(outside, dtype=np.int32)], False, (40, 120, 255), 2)
            if len(inside) >= 1:
                cv2.polylines(vis, [np.array(inside, dtype=np.int32)], False, (40, 200, 80), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            outside.clear()
            inside.clear()
            line_pts.clear()
            editing = "outside"
        if key == ord("n") and args.mode == "roi" and editing == "outside" and len(outside) >= 3:
            editing = "inside"
        if key == ord("s"):
            with open(args.config, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            cfg.setdefault("zone", {})
            if args.mode == "line" and len(line_pts) == 2:
                cfg["zone"]["mode"] = "line"
                cfg["door_line"] = {
                    "p1": [round(line_pts[0][0] / w, 4), round(line_pts[0][1] / h, 4)],
                    "p2": [round(line_pts[1][0] / w, 4), round(line_pts[1][1] / h, 4)],
                    "outside_side": cfg.get("door_line", {}).get("outside_side", "left"),
                }
            elif len(outside) >= 3 and len(inside) >= 3:
                cfg["zone"]["mode"] = "roi"
                cfg["zone"]["rois"] = {
                    "outside": {"polygon": [[round(x / w, 4), round(y / h, 4)] for x, y in outside]},
                    "inside": {"polygon": [[round(x / w, 4), round(y / h, 4)] for x, y in inside]},
                }
            else:
                print("需要足够的点")
                continue
            out = args.out_config or args.config
            with open(out, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
            print(f"已保存到 {out}")
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
