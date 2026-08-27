#!/usr/bin/env python3
"""探测视频/图片上 YOLO 实际检出哪些类别（排查床/病人漏检）。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.model_loader import load_detector  # noqa: E402
from or_io.pipeline import load_config, resolve_device  # noqa: E402
from or_io.stretcher_filter import TargetFilter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe bed/patient detections")
    parser.add_argument("--source", required=True, help="视频或图片路径")
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--max-frames", type=int, default=60)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save", type=str, default="", help="可选：保存叠加图到该路径")
    args = parser.parse_args()

    cfg = load_config(args.config)
    requested = args.device if args.device is not None else str(cfg.get("model", {}).get("device", "auto"))
    cfg.setdefault("model", {})["device"] = resolve_device(requested)
    det = load_detector(cfg["model"])
    filt = TargetFilter.from_config(cfg)
    filt.bind_class_ids(det.bed_class_ids, det.person_class_ids, ())
    if det.backend == "world":
        filt.extra_bed_like_ids = ()

    src = Path(args.source)
    is_image = src.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    frames = []
    if is_image:
        img = cv2.imread(str(src))
        if img is None:
            raise SystemExit(f"无法读取图片: {src}")
        frames = [img]
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise SystemExit(f"无法打开视频: {src}")
        i = 0
        while i < args.max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            i += 1
        cap.release()

    class_counter: Counter[str] = Counter()
    role_counter: Counter[str] = Counter()
    examples = []
    last = None
    for fi, frame in enumerate(frames):
        h, w = frame.shape[:2]
        kwargs = dict(
            source=frame,
            conf=float(cfg["model"].get("conf", 0.12)),
            iou=float(cfg["model"].get("iou", 0.5)),
            imgsz=int(cfg["model"].get("imgsz", 640)),
            device=cfg["model"].get("device", "cpu"),
            verbose=False,
        )
        if det.detect_classes is not None:
            kwargs["classes"] = det.detect_classes
        r0 = det.model.predict(**kwargs)[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        for box, conf, cls_id in zip(xyxy, confs, clss):
            name = det.names.get(int(cls_id), str(cls_id))
            role = filt.classify_role(int(cls_id), box.tolist(), w, h)
            class_counter[name] += 1
            role_counter[role] += 1
            if len(examples) < 40:
                examples.append(
                    {
                        "frame": fi,
                        "class": name,
                        "role": role,
                        "conf": round(float(conf), 3),
                        "xyxy": [round(float(v), 1) for v in box.tolist()],
                    }
                )
            color = {
                "bed": (30, 180, 255),
                "patient_head": (40, 220, 160),
                "lying_patient": (40, 200, 80),
                "person": (160, 160, 160),
            }.get(role, (120, 120, 120))
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{role}/{name} {conf:.2f}",
                (x1, max(20, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )
        last = frame

    summary = {
        "backend": det.backend,
        "frames": len(frames),
        "class_counts": dict(class_counter),
        "role_counts": dict(role_counter),
        "bed_class_ids": list(filt.bed_class_ids),
        "person_class_ids": list(filt.person_class_ids),
        "examples": examples,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.save and last is not None:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), last)
        print(f"saved: {out}")


if __name__ == "__main__":
    main()
