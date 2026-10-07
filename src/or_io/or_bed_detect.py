"""固定手术床自动识别：开放词汇手术台优先 + 位置稳定锁定。"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from .stretcher_filter import Detection
from .transfer import _area, _center, _iou


DEFAULT_OR_TABLE_NAME_KEYS = (
    "operating table",
    "surgical table",
    "or table",
    "operating bed",
    "surgical bed",
    "operations table",
)


def resolve_or_table_class_ids(
    names: Dict[int, str],
    or_table_prompts: Sequence[str] | None = None,
) -> Tuple[int, ...]:
    keys = {_norm(x) for x in (or_table_prompts or DEFAULT_OR_TABLE_NAME_KEYS)}
    ids = tuple(i for i, n in names.items() if _norm(n) in keys)
    return ids


def _norm(s: str) -> str:
    return str(s).strip().lower()


def box_to_poly(xyxy: Sequence[float], expand: float = 0.06) -> np.ndarray:
    """检测框 → 略放大的四边形（像素）。"""
    x1, y1, x2, y2 = map(float, xyxy)
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    dx, dy = w * expand, h * expand
    return np.array(
        [
            [x1 - dx, y1 - dy],
            [x2 + dx, y1 - dy],
            [x2 + dx, y2 + dy],
            [x1 - dx, y2 + dy],
        ],
        dtype=np.float32,
    )


def poly_to_normalized(poly: np.ndarray, frame_w: int, frame_h: int) -> List[List[float]]:
    return [
        [round(float(x) / max(frame_w, 1), 6), round(float(y) / max(frame_h, 1), 6)]
        for x, y in poly.tolist()
    ]


@dataclass
class AutoOrBedConfig:
    enabled: bool = True
    # 锁定前至少观察的帧数
    lock_frames: int = 25
    # 中心位移相对画面对角线的上限（越小越“固定”）
    max_center_motion_ratio: float = 0.025
    # 候选最小面积比
    min_area_ratio: float = 0.04
    min_aspect_wh: float = 1.05
    # 框略放大成 ROI
    poly_expand: float = 0.08
    # 开放词汇手术台类加权
    or_table_score_bonus: float = 2.5
    # 锁定后若连续丢失过久可重新搜索
    lost_unlock_frames: int = 180
    # 历史窗口
    history: int = 40


@dataclass
class _CandHist:
    centers: Deque[Tuple[float, float]] = field(default_factory=lambda: deque(maxlen=40))
    boxes: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=40))
    confs: Deque[float] = field(default_factory=lambda: deque(maxlen=40))
    hits: int = 0
    is_or_table: bool = False
    last_frame: int = -1


@dataclass
class AutoOrBedDetector:
    """无手动 ROI 时，自动锁定固定手术床。"""

    cfg: AutoOrBedConfig = field(default_factory=AutoOrBedConfig)
    or_table_class_ids: Tuple[int, ...] = ()
    bed_class_ids: Set[int] = field(default_factory=set)

    locked_poly: Optional[np.ndarray] = None
    locked_box: Optional[Tuple[float, float, float, float]] = None
    locked_track_id: Optional[int] = None
    locked_from: str = ""  # or_table_class | stationary_bed
    status: str = "idle"  # idle | searching | locked

    _hist: Dict[int, _CandHist] = field(default_factory=dict)
    _miss: int = 0

    def reset(self) -> None:
        self.locked_poly = None
        self.locked_box = None
        self.locked_track_id = None
        self.locked_from = ""
        self.status = "idle"
        self._hist.clear()
        self._miss = 0

    def _is_candidate(self, d: Detection, frame_w: int, frame_h: int) -> Tuple[bool, bool]:
        """返回 (是否候选, 是否手术台类)。"""
        cid = int(d.class_id)
        is_table = cid in self.or_table_class_ids
        is_bed = cid in self.bed_class_ids or is_table
        if not is_bed:
            return False, False
        area = _area(d.xyxy) / max(frame_w * frame_h, 1)
        if area < self.cfg.min_area_ratio:
            return False, False
        w = float(d.xyxy[2] - d.xyxy[0])
        h = float(d.xyxy[3] - d.xyxy[1])
        if w / max(h, 1e-6) < self.cfg.min_aspect_wh:
            return False, False
        return True, is_table

    def _motion_ratio(self, hist: _CandHist, frame_diag: float) -> float:
        if len(hist.centers) < 2:
            return 0.0
        xs = [c[0] for c in hist.centers]
        ys = [c[1] for c in hist.centers]
        span = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))
        return span / max(frame_diag, 1e-6)

    def _score(self, hist: _CandHist, frame_w: int, frame_h: int) -> float:
        if hist.hits < 3 or not hist.boxes:
            return -1.0
        frame_diag = float(np.hypot(frame_w, frame_h))
        motion = self._motion_ratio(hist, frame_diag)
        if motion > self.cfg.max_center_motion_ratio * 2.5:
            return -1.0
        area = _area(hist.boxes[-1]) / max(frame_w * frame_h, 1)
        conf = float(np.mean(hist.confs)) if hist.confs else 0.0
        stability = max(0.0, 1.0 - motion / max(self.cfg.max_center_motion_ratio, 1e-6))
        score = stability * 3.0 + area * 8.0 + conf + min(hist.hits, 40) * 0.05
        if hist.is_or_table:
            score += self.cfg.or_table_score_bonus
        return score

    def update(
        self,
        *,
        frame_idx: int,
        frame_w: int,
        frame_h: int,
        detections: Sequence[Detection],
        roles: Optional[Dict[int, str]] = None,
    ) -> Optional[np.ndarray]:
        """
        更新自动识别。返回当前手术床多边形（像素），未锁定则为 None。
        若已锁定则持续返回 locked_poly（可微调）。
        """
        if not self.cfg.enabled:
            return self.locked_poly

        roles = roles or {}
        # 已锁定：用重叠检测刷新位置，或计丢失
        if self.locked_poly is not None and self.locked_box is not None:
            matched = None
            best_iou = 0.0
            for d in detections:
                ok, is_table = self._is_candidate(d, frame_w, frame_h)
                if not ok and roles.get(d.track_id) not in ("bed", "or_bed"):
                    continue
                iou = _iou(d.xyxy, self.locked_box)
                if iou > best_iou:
                    best_iou = iou
                    matched = d
            if matched is not None and best_iou >= 0.25:
                self._miss = 0
                # 轻微平滑更新
                a = 0.25
                ob = self.locked_box
                nb = matched.xyxy
                smoothed = tuple(
                    (1 - a) * ob[i] + a * float(nb[i]) for i in range(4)
                )  # type: ignore[misc]
                self.locked_box = smoothed  # type: ignore[assignment]
                self.locked_poly = box_to_poly(smoothed, self.cfg.poly_expand)
                self.locked_track_id = int(matched.track_id)
                self.status = "locked"
                return self.locked_poly
            self._miss += 1
            if self._miss >= self.cfg.lost_unlock_frames:
                print("[or_bed] auto lock lost → re-search")
                self.reset()
                self.status = "searching"
            else:
                self.status = "locked"
                return self.locked_poly

        # 搜索 / 累积候选
        self.status = "searching"
        seen: Set[int] = set()
        for d in detections:
            ok, is_table = self._is_candidate(d, frame_w, frame_h)
            role = roles.get(d.track_id)
            if not ok and role not in ("bed", "or_bed"):
                continue
            if role == "equipment_cart":
                continue
            tid = int(d.track_id)
            seen.add(tid)
            hist = self._hist.get(tid)
            if hist is None:
                hist = _CandHist(centers=deque(maxlen=self.cfg.history), boxes=deque(maxlen=self.cfg.history), confs=deque(maxlen=self.cfg.history))
                self._hist[tid] = hist
            hist.centers.append(_center(d.xyxy))
            hist.boxes.append(tuple(float(v) for v in d.xyxy))  # type: ignore[arg-type]
            hist.confs.append(float(d.conf))
            hist.hits += 1
            hist.is_or_table = hist.is_or_table or is_table
            hist.last_frame = frame_idx

        # 清理过旧
        stale = [tid for tid, h in self._hist.items() if frame_idx - h.last_frame > self.cfg.history * 2]
        for tid in stale:
            self._hist.pop(tid, None)

        # 尝试锁定
        best_tid = None
        best_score = 0.0
        for tid, hist in self._hist.items():
            if hist.hits < self.cfg.lock_frames:
                continue
            motion = self._motion_ratio(hist, float(np.hypot(frame_w, frame_h)))
            if motion > self.cfg.max_center_motion_ratio:
                # 手术台类允许稍松一点
                if not (hist.is_or_table and motion <= self.cfg.max_center_motion_ratio * 1.8):
                    continue
            sc = self._score(hist, frame_w, frame_h)
            if sc > best_score:
                best_score = sc
                best_tid = tid

        if best_tid is None:
            return None

        hist = self._hist[best_tid]
        # 用最近若干框的中位数更稳
        boxes = list(hist.boxes)[-min(10, len(hist.boxes)) :]
        arr = np.array(boxes, dtype=np.float32)
        med = tuple(float(x) for x in np.median(arr, axis=0))
        self.locked_box = med  # type: ignore[assignment]
        self.locked_poly = box_to_poly(med, self.cfg.poly_expand)
        self.locked_track_id = best_tid
        self.locked_from = "or_table_class" if hist.is_or_table else "stationary_bed"
        self.status = "locked"
        self._miss = 0
        print(
            f"[or_bed] auto-locked track={best_tid} via={self.locked_from} "
            f"hits={hist.hits} score={best_score:.2f}"
        )
        return self.locked_poly

    def locked_norm_polygon(self, frame_w: int, frame_h: int) -> Optional[List[List[float]]]:
        if self.locked_poly is None:
            return None
        return poly_to_normalized(self.locked_poly, frame_w, frame_h)


def auto_or_bed_config_from_dict(d: dict | None) -> AutoOrBedConfig:
    d = d or {}
    return AutoOrBedConfig(
        enabled=bool(d.get("auto_detect", d.get("enabled", True))),
        lock_frames=int(d.get("lock_frames", 25)),
        max_center_motion_ratio=float(d.get("max_center_motion_ratio", 0.025)),
        min_area_ratio=float(d.get("min_area_ratio", 0.04)),
        min_aspect_wh=float(d.get("min_aspect_wh", 1.05)),
        poly_expand=float(d.get("poly_expand", 0.08)),
        or_table_score_bonus=float(d.get("or_table_score_bonus", 2.5)),
        lost_unlock_frames=int(d.get("lost_unlock_frames", 180)),
        history=int(d.get("history", 40)),
    )
