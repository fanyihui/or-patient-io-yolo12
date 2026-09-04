"""可视化叠加。"""

from __future__ import annotations

from typing import Iterable, Tuple, Union

import cv2
import numpy as np

from .events import IOEvent
from .zones import DoorLine, DualROIZones

Zone = Union[DoorLine, DualROIZones]


def draw_zone(frame: np.ndarray, zone: Zone) -> None:
    overlay = frame.copy()
    if isinstance(zone, DoorLine):
        (x1, y1), (x2, y2) = zone.endpoints()
        cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 255), 2, cv2.LINE_AA)
        mx, my = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.putText(frame, "DOOR", (mx + 8, my), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2, cv2.LINE_AA)
        return

    out_pts = zone.outside_poly.astype(np.int32).reshape(-1, 1, 2)
    in_pts = zone.inside_poly.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(overlay, [out_pts], (80, 160, 255))
    cv2.fillPoly(overlay, [in_pts], (80, 220, 120))
    cv2.addWeighted(overlay, 0.28, frame, 0.72, 0, frame)
    cv2.polylines(frame, [out_pts], True, (40, 120, 255), 2, cv2.LINE_AA)
    cv2.polylines(frame, [in_pts], True, (40, 200, 80), 2, cv2.LINE_AA)

    def _label(poly: np.ndarray, text: str, color: tuple[int, int, int]) -> None:
        cx = int(np.mean(poly[:, 0]))
        cy = int(np.mean(poly[:, 1]))
        cv2.putText(frame, text, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    _label(zone.outside_poly, "OUT ROI", (40, 120, 255))
    _label(zone.inside_poly, "IN ROI", (40, 200, 80))


def draw_door_line(frame: np.ndarray, door: DoorLine) -> None:
    draw_zone(frame, door)


def draw_track(
    frame: np.ndarray,
    track_id: int,
    xyxy: Tuple[float, float, float, float],
    conf: float,
    side: str,
    is_target: bool,
    role: str = "person",
    class_name: str = "person",
) -> None:
    x1, y1, x2, y2 = map(int, xyxy)
    role_color = {
        "bed": (30, 180, 255),
        "equipment_cart": (180, 180, 40),
        "lying_patient": (40, 200, 80),
        "patient_head": (40, 220, 160),
        "person": (160, 160, 160),
        "other": (140, 140, 140),
    }
    color = role_color.get(role, (160, 160, 160))
    if not is_target and role in ("person", "other", "equipment_cart"):
        color = (160, 160, 160) if role != "equipment_cart" else (180, 180, 40)
    thickness = 3 if is_target else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    prefix = {
        "bed": "BED",
        "equipment_cart": "CART",
        "lying_patient": "LYING",
        "patient_head": "HEAD",
        "person": "STAFF",
    }.get(role, class_name)
    if is_target:
        prefix = "TARGET-" + prefix
    label = f"{prefix} ID{track_id} {conf:.2f} {side}"
    cv2.putText(
        frame,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_pair(frame: np.ndarray, pair, side: str, confirmed: bool) -> None:
    color = (0, 200, 255) if confirmed else (0, 160, 200)
    ux1, uy1, ux2, uy2 = map(int, pair.union_xyxy)
    cv2.rectangle(frame, (ux1, uy1), (ux2, uy2), color, 3)
    bx1, by1, bx2, by2 = map(int, pair.bed_xyxy)
    px1, py1, px2, py2 = map(int, pair.patient_xyxy)
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), (30, 180, 255), 2)
    cv2.rectangle(frame, (px1, py1), (px2, py2), (40, 200, 80), 2)
    bcx, bcy = int((bx1 + bx2) / 2), int((by1 + by2) / 2)
    pcx, pcy = int((px1 + px2) / 2), int((py1 + py2) / 2)
    cv2.line(frame, (bcx, bcy), (pcx, pcy), color, 2, cv2.LINE_AA)
    tag = "BED+HEAD" if confirmed else "pairing..."
    if getattr(pair, "evidence", "") == "lying_full":
        tag = "BED+LYING" if confirmed else "pairing..."
    elif getattr(pair, "evidence", "") == "merged":
        tag = "MERGED" if confirmed else "pairing..."
    cv2.putText(
        frame,
        f"{tag} bed={pair.bed_track_id} pat={pair.patient_track_id} {side}",
        (ux1, max(22, uy1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_event_banner(frame: np.ndarray, events: Iterable[IOEvent], hold: int = 45) -> None:
    y = 28
    for evt in events:
        color = (50, 220, 50) if evt.event == "enter" else (40, 40, 230)
        text = (
            f"{evt.event.upper()}  track={evt.track_id}  "
            f"t={evt.video_time_sec:.2f}s  frame={evt.frame_idx}"
        )
        cv2.putText(frame, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 28


def draw_hud(
    frame: np.ndarray,
    frame_idx: int,
    fps: float,
    n_enter: int,
    n_exit: int,
    target_mode: str = "bed_patient",
) -> None:
    h, w = frame.shape[:2]
    panel = (
        f"target={target_mode}  frame={frame_idx}  "
        f"time={frame_idx / max(fps, 1e-6):.1f}s  enter={n_enter}  exit={n_exit}"
    )
    cv2.rectangle(frame, (0, h - 36), (w, h), (20, 20, 20), -1)
    cv2.putText(
        frame,
        panel,
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
