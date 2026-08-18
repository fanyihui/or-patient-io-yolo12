"""门口区域判定：虚拟线 / 双 ROI。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence, Tuple

import cv2
import numpy as np

Side = Literal["outside", "inside", "on_line"]
EventName = Literal["enter", "exit"]


class ZoneClassifier(Protocol):
    def classify(self, x: float, y: float) -> Side: ...


def side_transition(
    last_stable: Side | None,
    current: Side,
) -> tuple[EventName | None, Side | None]:
    if current == "on_line":
        return None, last_stable
    if last_stable is None:
        return None, current
    if last_stable == "outside" and current == "inside":
        return "enter", current
    if last_stable == "inside" and current == "outside":
        return "exit", current
    return None, current


@dataclass(frozen=True)
class DoorLine:
    x1: float
    y1: float
    x2: float
    y2: float
    outside_side: Literal["left", "right", "top", "bottom"] = "left"

    @classmethod
    def from_normalized(
        cls,
        p1: Sequence[float],
        p2: Sequence[float],
        frame_w: int,
        frame_h: int,
        outside_side: Literal["left", "right", "top", "bottom"] = "left",
    ) -> "DoorLine":
        return cls(
            x1=float(p1[0]) * frame_w,
            y1=float(p1[1]) * frame_h,
            x2=float(p2[0]) * frame_w,
            y2=float(p2[1]) * frame_h,
            outside_side=outside_side,
        )

    def endpoints(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2))

    def _signed_side(self, x: float, y: float) -> float:
        return (self.x2 - self.x1) * (y - self.y1) - (self.y2 - self.y1) * (x - self.x1)

    def classify(self, x: float, y: float, on_eps: float = 4.0) -> Side:
        s = self._signed_side(x, y)
        dx, dy = self.x2 - self.x1, self.y2 - self.y1
        length = max(np.hypot(dx, dy), 1e-6)
        dist = abs(s) / length
        if dist <= on_eps:
            return "on_line"

        left_is_positive = s > 0
        if self.outside_side == "left":
            return "outside" if left_is_positive else "inside"
        if self.outside_side == "right":
            return "inside" if left_is_positive else "outside"
        if self.outside_side == "top":
            mid_y = (self.y1 + self.y2) / 2.0
            return "outside" if y < mid_y else "inside"
        if self.outside_side == "bottom":
            mid_y = (self.y1 + self.y2) / 2.0
            return "outside" if y > mid_y else "inside"
        return "inside"

    @staticmethod
    def side_transition(
        last_stable: Side | None,
        current: Side,
    ) -> tuple[EventName | None, Side | None]:
        return side_transition(last_stable, current)


def _norm_poly_to_px(
    polygon: Sequence[Sequence[float]],
    frame_w: int,
    frame_h: int,
) -> np.ndarray:
    pts = np.array([[float(x) * frame_w, float(y) * frame_h] for x, y in polygon], dtype=np.float32)
    if pts.shape[0] < 3:
        raise ValueError("ROI polygon 至少需要 3 个点")
    return pts


@dataclass(frozen=True)
class DualROIZones:
    outside_poly: np.ndarray
    inside_poly: np.ndarray

    @classmethod
    def from_normalized(
        cls,
        outside: Sequence[Sequence[float]],
        inside: Sequence[Sequence[float]],
        frame_w: int,
        frame_h: int,
    ) -> "DualROIZones":
        return cls(
            outside_poly=_norm_poly_to_px(outside, frame_w, frame_h),
            inside_poly=_norm_poly_to_px(inside, frame_w, frame_h),
        )

    @staticmethod
    def _contains(poly: np.ndarray, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

    def classify(self, x: float, y: float) -> Side:
        in_out = self._contains(self.outside_poly, x, y)
        in_in = self._contains(self.inside_poly, x, y)
        if in_out and not in_in:
            return "outside"
        if in_in and not in_out:
            return "inside"
        return "on_line"

    @staticmethod
    def side_transition(
        last_stable: Side | None,
        current: Side,
    ) -> tuple[EventName | None, Side | None]:
        return side_transition(last_stable, current)


def build_zone(cfg: dict, frame_w: int, frame_h: int) -> DoorLine | DualROIZones:
    zone_cfg = cfg.get("zone") or {}
    mode = str(zone_cfg.get("mode", "line")).lower()

    if mode == "roi":
        rois = zone_cfg.get("rois") or {}
        outside = rois.get("outside", {}).get("polygon")
        inside = rois.get("inside", {}).get("polygon")
        if not outside or not inside:
            raise ValueError("zone.mode=roi 时需要配置 zone.rois.outside/inside.polygon")
        return DualROIZones.from_normalized(outside, inside, frame_w, frame_h)

    dl = zone_cfg.get("door_line") or cfg.get("door_line")
    if not dl:
        raise ValueError("缺少 door_line / zone.door_line 配置")
    return DoorLine.from_normalized(
        dl["p1"],
        dl["p2"],
        frame_w,
        frame_h,
        dl.get("outside_side", "left"),
    )
