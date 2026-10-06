"""固定手术床 ROI 与「推床 → 手术床」转移事件。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .events import IOEvent
from .stretcher_filter import BedPatientPair, Detection


def _center(xyxy: Sequence[float]) -> Tuple[float, float]:
    return (float(xyxy[0] + xyxy[2]) * 0.5, float(xyxy[1] + xyxy[3]) * 0.5)


def _area(xyxy: Sequence[float]) -> float:
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(_area(a) + _area(b) - inter, 1e-6)


def poly_from_normalized(
    polygon: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    pts = np.array([[float(x) * frame_w, float(y) * frame_h] for x, y in polygon], dtype=np.float32)
    if pts.shape[0] < 3:
        raise ValueError("手术床 ROI 至少 3 个点")
    return pts


def point_in_poly(poly: np.ndarray, x: float, y: float) -> bool:
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0


def box_overlap_poly_ratio(xyxy: Sequence[float], poly: np.ndarray) -> float:
    """检测框与多边形的近似重叠比（用框中心 + IoU 粗略兜底）。"""
    cx, cy = _center(xyxy)
    if point_in_poly(poly, cx, cy):
        return 1.0
    # 采样框内若干点
    x1, y1, x2, y2 = map(float, xyxy)
    hits = 0
    total = 0
    for fx in (0.25, 0.5, 0.75):
        for fy in (0.25, 0.5, 0.75):
            total += 1
            px = x1 + (x2 - x1) * fx
            py = y1 + (y2 - y1) * fy
            if point_in_poly(poly, px, py):
                hits += 1
    return hits / max(total, 1)


@dataclass
class TransferConfig:
    enabled: bool = True
    # 患者中心在手术床 ROI 内连续确认帧数
    confirm_frames: int = 8
    # 手术床连续空床帧数后重置，允许再次转移
    empty_reset_frames: int = 45
    # 要求近期有推床（床+患者）靠近手术床
    require_stretcher_nearby: bool = True
    stretcher_near_dist_ratio: float = 0.38
    stretcher_memory_frames: int = 120
    # 患者框与手术床 ROI 重叠比例门槛
    patient_overlap_min: float = 0.35
    debounce_frames: int = 90
    # 与固定手术床 ROI 重叠很大的「床」检测不当作推床
    fixed_bed_iou_min: float = 0.35
    fixed_bed_center_in_roi: bool = True


@dataclass
class TransferMonitor:
    """检测：患者从推床转移到固定手术床。"""

    cfg: TransferConfig = field(default_factory=TransferConfig)
    or_bed_poly: Optional[np.ndarray] = None  # 像素坐标
    source: str = ""

    _occupy_hits: int = 0
    _empty_hits: int = 0
    _occupied: bool = False
    _last_transfer_frame: int = -10**9
    _last_stretcher_near_frame: int = -10**9
    _last_patient_tid: int = -1

    def set_or_bed_poly(self, poly: Optional[np.ndarray]) -> None:
        self.or_bed_poly = poly

    def is_fixed_or_bed_box(self, xyxy: Sequence[float]) -> bool:
        """床框是否落在固定手术床 ROI（不当作可移动推床）。"""
        if self.or_bed_poly is None:
            return False
        cx, cy = _center(xyxy)
        if self.cfg.fixed_bed_center_in_roi and point_in_poly(self.or_bed_poly, cx, cy):
            return True
        # 用轴对齐近似：多边形外接矩形与床框 IoU
        x = self.or_bed_poly[:, 0]
        y = self.or_bed_poly[:, 1]
        bed_xyxy = (float(x.min()), float(y.min()), float(x.max()), float(y.max()))
        return _iou(xyxy, bed_xyxy) >= self.cfg.fixed_bed_iou_min

    def _patient_on_or_bed(
        self,
        detections: Sequence[Detection],
        roles: Dict[int, str],
    ) -> Optional[Detection]:
        if self.or_bed_poly is None:
            return None
        best: Optional[Detection] = None
        best_score = 0.0
        for d in detections:
            role = roles.get(d.track_id, "other")
            if role not in ("patient_head", "lying_patient"):
                continue
            overlap = box_overlap_poly_ratio(d.xyxy, self.or_bed_poly)
            if overlap < self.cfg.patient_overlap_min:
                continue
            cx, cy = _center(d.xyxy)
            if not point_in_poly(self.or_bed_poly, cx, cy) and overlap < 0.6:
                continue
            score = overlap * float(d.conf)
            if score > best_score:
                best_score = score
                best = d
        return best

    def _note_stretcher_nearby(
        self,
        frame_idx: int,
        frame_w: int,
        frame_h: int,
        pairs: Sequence[BedPatientPair],
        detections: Sequence[Detection],
        roles: Dict[int, str],
    ) -> None:
        if self.or_bed_poly is None:
            return
        ox = float(np.mean(self.or_bed_poly[:, 0]))
        oy = float(np.mean(self.or_bed_poly[:, 1]))
        thr = self.cfg.stretcher_near_dist_ratio * float(np.hypot(frame_w, frame_h))

        # 确认的推床+患者配对（排除固定手术床本身）
        for pair in pairs:
            if self.is_fixed_or_bed_box(pair.bed_xyxy):
                continue
            cx, cy = pair.centroid
            if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 <= thr:
                self._last_stretcher_near_frame = frame_idx
                return

        # 兜底：可移动床框靠近手术床
        for d in detections:
            if roles.get(d.track_id) != "bed":
                continue
            if self.is_fixed_or_bed_box(d.xyxy):
                continue
            cx, cy = _center(d.xyxy)
            if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 <= thr:
                self._last_stretcher_near_frame = frame_idx
                return

    def update(
        self,
        *,
        frame_idx: int,
        fps: float,
        frame_w: int,
        frame_h: int,
        detections: Sequence[Detection],
        roles: Dict[int, str],
        pairs: Sequence[BedPatientPair],
    ) -> Optional[IOEvent]:
        if not self.cfg.enabled or self.or_bed_poly is None:
            return None

        self._note_stretcher_nearby(frame_idx, frame_w, frame_h, pairs, detections, roles)
        patient = self._patient_on_or_bed(detections, roles)

        if patient is None:
            self._occupy_hits = 0
            self._empty_hits += 1
            if self._empty_hits >= self.cfg.empty_reset_frames:
                self._occupied = False
            return None

        self._empty_hits = 0
        self._occupy_hits += 1
        self._last_patient_tid = int(patient.track_id)

        if self._occupied:
            return None
        if self._occupy_hits < self.cfg.confirm_frames:
            return None
        if frame_idx - self._last_transfer_frame < self.cfg.debounce_frames:
            return None

        if self.cfg.require_stretcher_nearby:
            age = frame_idx - self._last_stretcher_near_frame
            if age < 0 or age > self.cfg.stretcher_memory_frames:
                # 尚未满足「近期有推床靠近」——保持累计，但不触发
                return None

        self._occupied = True
        self._last_transfer_frame = frame_idx
        cx, cy = _center(patient.xyxy)
        evt = IOEvent(
            event="transfer_to_or_bed",  # type: ignore[arg-type]
            track_id=int(patient.track_id),
            frame_idx=frame_idx,
            video_time_sec=round(frame_idx / max(fps, 1e-6), 3),
            wall_time_iso=datetime.now(timezone.utc).isoformat(),
            centroid=(cx, cy),
            bbox_xyxy=tuple(float(v) for v in patient.xyxy),  # type: ignore[arg-type]
            confidence=float(patient.conf),
            source=self.source,
        )
        return evt


def transfer_config_from_dict(d: dict | None) -> TransferConfig:
    d = d or {}
    return TransferConfig(
        enabled=bool(d.get("enabled", True)),
        confirm_frames=int(d.get("confirm_frames", 8)),
        empty_reset_frames=int(d.get("empty_reset_frames", 45)),
        require_stretcher_nearby=bool(d.get("require_stretcher_nearby", True)),
        stretcher_near_dist_ratio=float(d.get("stretcher_near_dist_ratio", 0.38)),
        stretcher_memory_frames=int(d.get("stretcher_memory_frames", 120)),
        patient_overlap_min=float(d.get("patient_overlap_min", 0.35)),
        debounce_frames=int(d.get("debounce_frames", 90)),
        fixed_bed_iou_min=float(d.get("fixed_bed_iou_min", 0.35)),
        fixed_bed_center_in_roi=bool(d.get("fixed_bed_center_in_roi", True)),
    )
