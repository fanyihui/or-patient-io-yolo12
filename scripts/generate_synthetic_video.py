#!/usr/bin/env python3
"""生成「推床入/出室」合成验证视频。"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import urllib.request


SAMPLE_URLS = [
    "https://ultralytics.com/images/bus.jpg",
    "https://raw.githubusercontent.com/ultralytics/yolov5/master/data/images/bus.jpg",
]


def download_sample(dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and dst.stat().st_size > 10_000:
        return dst
    last_err = None
    for url in SAMPLE_URLS:
        try:
            print(f"下载样例图: {url}")
            urllib.request.urlretrieve(url, dst)
            if dst.exists() and dst.stat().st_size > 10_000:
                return dst
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"无法下载样例图: {last_err}")


def extract_largest_person(image: np.ndarray) -> np.ndarray:
    try:
        from ultralytics import YOLO

        model = YOLO("yolo12n.pt")
        res = model.predict(image, classes=[0], conf=0.25, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
            x1, y1, x2, y2 = xyxy[int(np.argmax(areas))].astype(int)
            h, w = image.shape[:2]
            pad = 6
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
            crop = image[y1:y2, x1:x2].copy()
            if crop.size > 0:
                return crop
    except Exception as e:  # noqa: BLE001
        print(f"人物裁剪回退: {e}")
    h, w = image.shape[:2]
    return image[h // 4 : 3 * h // 4, w // 3 : 2 * w // 3].copy()


def make_stretcher_sprite(person: np.ndarray, out_w: int = 506, out_h: int = 200) -> np.ndarray:
    if person.shape[0] >= person.shape[1]:
        body = cv2.rotate(person, cv2.ROTATE_90_CLOCKWISE)
    else:
        body = person.copy()
    return cv2.resize(body, (out_w, out_h))


def paste(frame: np.ndarray, patch: np.ndarray, x: int, y: int) -> None:
    fh, fw = frame.shape[:2]
    ph, pw = patch.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(fw, x + pw), min(fh, y + ph)
    if x1 >= x2 or y1 >= y2:
        return
    frame[y1:y2, x1:x2] = patch[y1 - y : y2 - y, x1 - x : x2 - x]


def build_video(
    out_path: Path,
    stretcher: np.ndarray,
    width: int = 960,
    height: int = 540,
    fps: int = 20,
) -> dict:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    y_bed = (height - stretcher.shape[0]) // 2
    x_outside = 60
    x_inside = 650
    enter_frames = 55
    stay_frames = 20
    exit_frames = 55

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    total = enter_frames + stay_frames + exit_frames

    def bed_x(i: int) -> int:
        if i < enter_frames:
            t = i / max(enter_frames - 1, 1)
            return int(x_outside + t * (x_inside - x_outside))
        if i < enter_frames + stay_frames:
            return x_inside
        t = (i - enter_frames - stay_frames) / max(exit_frames - 1, 1)
        return int(x_inside + t * (x_outside - x_inside))

    for i in range(total):
        frame = np.full((height, width, 3), 210, dtype=np.uint8)
        cv2.rectangle(frame, (width // 2 - 4, 40), (width // 2 + 4, height - 40), (160, 160, 170), -1)
        cv2.putText(frame, "OUT", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 90), 2)
        cv2.putText(frame, "OR", (width - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 80, 90), 2)
        paste(frame, stretcher, bed_x(i), y_bed)
        writer.write(frame)
    writer.release()

    door_x = width * 0.5
    half = stretcher.shape[1] / 2
    enter_frac = (door_x - (x_outside + half)) / max(x_inside - x_outside, 1)
    enter_frac = float(np.clip(enter_frac, 0, 1))
    return {
        "enter_frame_approx": int(round(enter_frac * (enter_frames - 1))),
        "exit_frame_approx": int(round(enter_frames + stay_frames + enter_frac * (exit_frames - 1))),
        "fps": fps,
        "path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="生成推床入出室合成验证视频")
    parser.add_argument("--out", type=str, default="data/samples/or_door_synthetic.mp4")
    parser.add_argument("--sample-image", type=str, default="data/samples/bus.jpg")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sample = download_sample(root / args.sample_image)
    img = cv2.imread(str(sample))
    if img is None:
        raise RuntimeError(f"读取失败: {sample}")
    person = extract_largest_person(img)
    stretcher = make_stretcher_sprite(person)
    expected = build_video(root / args.out, stretcher)
    print("合成推床视频已生成:")
    for k, v in expected.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
