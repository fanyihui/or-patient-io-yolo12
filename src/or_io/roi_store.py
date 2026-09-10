"""手术室门口 ROI 读写（归一化多边形坐标）。"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import yaml


Point = Tuple[float, float]
Polygon = List[Point]


def normalize_polygon(
    points: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
    *,
    already_normalized: bool = False,
) -> Polygon:
    """像素或归一化点 → 归一化 [0,1] 多边形。"""
    out: Polygon = []
    for p in points:
        if len(p) < 2:
            continue
        x, y = float(p[0]), float(p[1])
        if already_normalized:
            nx, ny = x, y
        else:
            nx, ny = x / max(frame_w, 1), y / max(frame_h, 1)
        out.append((round(min(max(nx, 0.0), 1.0), 6), round(min(max(ny, 0.0), 1.0), 6)))
    if len(out) < 3:
        raise ValueError("ROI 多边形至少需要 3 个点")
    return out


def polygon_to_pixel(poly: Sequence[Sequence[float]], frame_w: int, frame_h: int) -> List[Tuple[int, int]]:
    return [(int(float(x) * frame_w), int(float(y) * frame_h)) for x, y in poly]


def build_zone_payload(
    outside: Sequence[Sequence[float]],
    inside: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
    *,
    already_normalized: bool = True,
    meta: Dict[str, Any] | None = None,
) -> dict:
    out_poly = normalize_polygon(outside, frame_w, frame_h, already_normalized=already_normalized)
    in_poly = normalize_polygon(inside, frame_w, frame_h, already_normalized=already_normalized)
    payload = {
        "version": 1,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "frame_size": {"width": int(frame_w), "height": int(frame_h)},
        "zone": {
            "mode": "roi",
            "rois": {
                "outside": {"polygon": [list(p) for p in out_poly], "label": "门外/走廊"},
                "inside": {"polygon": [list(p) for p in in_poly], "label": "手术室内"},
            },
        },
        "meta": meta or {},
    }
    return payload


def save_roi_yaml(path: str | Path, payload: dict, base_config: dict | None = None) -> Path:
    """保存 ROI；若给 base_config 则合并进完整检测配置。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if base_config is None:
        data = payload
    else:
        data = deepcopy(base_config)
        data["zone"] = deepcopy(payload["zone"])
        data.setdefault("roi_meta", {})
        data["roi_meta"] = {
            "saved_at": payload.get("saved_at"),
            "frame_size": payload.get("frame_size"),
            **(payload.get("meta") or {}),
        }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)
    return path


def load_roi(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "zone" in data and "rois" in (data.get("zone") or {}):
        return {
            "version": data.get("version", 1),
            "saved_at": (data.get("roi_meta") or {}).get("saved_at") or data.get("saved_at"),
            "frame_size": (data.get("roi_meta") or {}).get("frame_size") or data.get("frame_size"),
            "zone": data["zone"],
            "meta": data.get("roi_meta") or data.get("meta") or {},
        }
    raise ValueError(f"无法从 {path} 解析 zone.rois")


def apply_roi_to_config(cfg: dict, roi_payload: dict) -> dict:
    out = deepcopy(cfg)
    out["zone"] = deepcopy(roi_payload["zone"])
    return out
