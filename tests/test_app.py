"""ROI 存储与 Web API 单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.roi_store import apply_roi_to_config, build_zone_payload, load_roi, save_roi_yaml
from or_io.webapp import create_app


def test_roi_roundtrip(tmp_path: Path | None = None):
    out = Path(tmp_path) if tmp_path else (ROOT / "outputs" / "_test_roi")
    out.mkdir(parents=True, exist_ok=True)
    payload = build_zone_payload(
        [[0.1, 0.1], [0.4, 0.1], [0.4, 0.4], [0.1, 0.4]],
        [[0.45, 0.2], [0.8, 0.2], [0.8, 0.55], [0.45, 0.55]],
        960,
        540,
        already_normalized=True,
        meta={"site": "demo"},
    )
    path = save_roi_yaml(out / "demo_door_roi.yaml", payload, base_config=None)
    loaded = load_roi(path)
    assert loaded["zone"]["mode"] == "roi"
    assert len(loaded["zone"]["rois"]["outside"]["polygon"]) == 4
    assert loaded["zone"]["rois"]["inside"]["polygon"][0][0] == 0.45

    base = {"model": {"backend": "fusion"}, "zone": {"mode": "line"}}
    merged = apply_roi_to_config(base, payload)
    assert merged["zone"]["mode"] == "roi"
    assert base["zone"]["mode"] == "line"  # deepcopy


def _make_video(path: Path, frames: int = 30) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w, h = 320, 240
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (30, 50, 40)
        cv2.rectangle(img, (20 + i, 80), (120 + i, 160), (200, 180, 80), -1)
        writer.write(img)
    writer.release()
    return path


def test_web_connect_snapshot_roi():
    from fastapi.testclient import TestClient

    video = _make_video(ROOT / "outputs" / "_test_app" / "clip.mp4")
    app = create_app(ROOT, ROOT / "configs" / "default.yaml")
    client = TestClient(app)

    r = client.get("/api/health")
    assert r.status_code == 200

    r = client.post("/api/connect", json={"source": str(video)})
    assert r.status_code == 200, r.text
    assert r.json()["connected"] is True

    r = client.get("/api/frame.jpg?which=live")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/jpeg")
    assert len(r.content) > 100

    r = client.post("/api/snapshot")
    assert r.status_code == 200
    assert r.json()["has_snapshot"] is True

    r = client.post(
        "/api/roi",
        json={
            "outside": [[0.05, 0.1], [0.35, 0.1], [0.35, 0.5], [0.05, 0.5]],
            "inside": [[0.4, 0.15], [0.75, 0.15], [0.75, 0.55], [0.4, 0.55]],
            "site_name": "unit_test_or",
            "already_normalized": True,
            "save": True,
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["has_roi"] is True
    assert Path(body["roi_path"]).exists()

    roi_file = ROOT / "configs" / "sites" / "unit_test_or.roi.yaml"
    assert roi_file.exists()
    loaded = load_roi(roi_file)
    assert loaded["zone"]["rois"]["outside"]["polygon"][0][0] == 0.05

    r = client.get("/api/roi")
    assert r.status_code == 200
    assert r.json()["roi"]["zone"]["mode"] == "roi"

    r = client.get("/")
    assert r.status_code == 200
    assert "OR Gate" in r.text

    r = client.post("/api/disconnect")
    assert r.status_code == 200


if __name__ == "__main__":
    test_roi_roundtrip()
    test_web_connect_snapshot_roi()
    print("app tests passed")
