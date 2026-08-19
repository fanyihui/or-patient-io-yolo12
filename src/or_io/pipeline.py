"""YOLO12 + ByteTrack 入/出室检测流水线（病床 + 平躺患者）。"""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from .events import EventManager, IOEvent
from .stretcher_filter import BedPatientPair, Detection, TargetFilter
from .visualize import draw_event_banner, draw_hud, draw_pair, draw_track, draw_zone
from .zones import DoorLine, DualROIZones, Side, build_zone, side_transition

Zone = Union[DoorLine, DualROIZones]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_detect_classes(cfg: dict) -> List[int]:
    model_classes = cfg.get("model", {}).get("classes")
    if model_classes is not None:
        return list(model_classes)
    target = cfg.get("target") or {}
    mode = str(target.get("mode", "bed_patient"))
    if mode in ("bed_patient", "stretcher"):
        st = target.get("stretcher") or target.get("bed_patient") or {}
        ids = set(st.get("bed_class_ids") or [59])
        ids.update(st.get("extra_bed_like_ids") or [57])
        ids.add(0)
        return sorted(ids)
    return [0]


def resolve_device(requested) -> str | int:
    """把配置里的 device 解析成 ultralytics 可接受的值。auto → GPU0 或 cpu。"""
    if requested is None:
        value = "auto"
    else:
        value = str(requested).strip().lower()
    if value not in ("auto", ""):
        return int(value) if value.isdigit() else value
    try:
        import torch

        if torch.cuda.is_available():
            return 0
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


class ORIOPipeline:
    def __init__(self, config: dict, project_root: Optional[Path] = None):
        self.cfg = config
        self.root = project_root or Path.cwd()
        self.model = YOLO(config["model"]["weights"])
        # 统一解析 device，避免 yaml 里写 auto/cpu 时行为不一致
        self.cfg.setdefault("model", {})["device"] = resolve_device(config.get("model", {}).get("device"))
        tracker_rel = config["tracker"]["config"]
        self.tracker_cfg = str((self.root / tracker_rel).resolve())
        if not Path(self.tracker_cfg).exists():
            self.tracker_cfg = "bytetrack.yaml"

        self.target_filter = TargetFilter.from_config(config)
        self.detect_classes = resolve_detect_classes(config)
        ev = config.get("events", {})
        self.event_manager = EventManager(
            debounce_frames=int(ev.get("debounce_frames", 15)),
            min_crossing_disp=float(ev.get("min_crossing_disp", 0.02)),
        )
        self._prev_centroid: Dict[int, Tuple[float, float]] = {}
        self._last_stable_side: Dict[int, Side] = {}
        self._recent_events: Deque[Tuple[int, IOEvent]] = deque(maxlen=32)
        self.zone: Optional[Zone] = None

    def _ensure_zone(self, frame_w: int, frame_h: int) -> Zone:
        if self.zone is None:
            self.zone = build_zone(self.cfg, frame_w, frame_h)
        return self.zone

    def _handle_pair_event(
        self,
        pair: BedPatientPair,
        confirmed: bool,
        frame_idx: int,
        fps: float,
        frame_diag: float,
        zone: Zone,
    ) -> None:
        tid = pair.event_track_id
        cx, cy = pair.centroid
        side_now = zone.classify(cx, cy)
        prev = self._prev_centroid.get(tid)
        crossing, new_side = side_transition(self._last_stable_side.get(tid), side_now)
        if new_side is not None:
            self._last_stable_side[tid] = new_side
        if confirmed:
            evt = self.event_manager.consider(
                track_id=tid,
                crossing=crossing,
                frame_idx=frame_idx,
                fps=fps,
                centroid=(cx, cy),
                bbox_xyxy=pair.union_xyxy,
                confidence=pair.conf,
                frame_diag=frame_diag,
                prev_centroid=prev,
            )
            if evt is not None:
                self._recent_events.append((frame_idx, evt))
                print(
                    f"[{evt.event.upper()}] bed={pair.bed_track_id}+patient={pair.patient_track_id} "
                    f"video_t={evt.video_time_sec:.2f}s frame={evt.frame_idx}"
                )
        self._prev_centroid[tid] = (cx, cy)

    def process_video(
        self,
        source: str | Path,
        output_dir: str | Path,
        max_frames: Optional[int] = None,
    ) -> dict:
        source = str(source)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_diag = float(np.hypot(width, height))
        zone = self._ensure_zone(width, height)
        zone_mode = str((self.cfg.get("zone") or {}).get("mode", "line"))
        target_mode = self.target_filter.mode

        out_cfg = self.cfg.get("output", {})
        writer = None
        out_video = output_dir / "annotated.mp4"
        if out_cfg.get("save_video", True):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_video), fourcc, fps, (width, height))

        model_cfg = self.cfg["model"]
        names = self.model.names
        frame_idx = 0
        t0 = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            results = self.model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_cfg,
                conf=float(model_cfg.get("conf", 0.35)),
                iou=float(model_cfg.get("iou", 0.5)),
                imgsz=int(model_cfg.get("imgsz", 640)),
                device=model_cfg.get("device", "cpu"),
                classes=self.detect_classes,
                verbose=False,
            )
            r0 = results[0]
            boxes = r0.boxes

            if out_cfg.get("draw_door_line", True) or out_cfg.get("draw_zone", True):
                draw_zone(frame, zone)

            dets: List[Detection] = []
            if boxes is not None and boxes.id is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for box, tid, conf, cls_id in zip(xyxy, ids, confs, clss):
                    dets.append(
                        Detection(
                            track_id=int(tid),
                            class_id=int(cls_id),
                            conf=float(conf),
                            xyxy=tuple(float(v) for v in box.tolist()),  # type: ignore[arg-type]
                        )
                    )

            paired_ids: set[int] = set()
            if target_mode == "bed_patient":
                pairs = self.target_filter.associate(dets, width, height, frame_idx)
                for pair in pairs:
                    confirmed = self.target_filter.confirm_pair(pair)
                    paired_ids.add(pair.bed_track_id)
                    paired_ids.add(pair.patient_track_id)
                    if out_cfg.get("draw_tracks", True):
                        draw_pair(frame, pair, zone.classify(*pair.centroid), confirmed)
                    self._handle_pair_event(pair, confirmed, frame_idx, fps, frame_diag, zone)
                self.target_filter.mark_unseen_beds([p.event_track_id for p in pairs])

                if out_cfg.get("draw_tracks", True):
                    for d in dets:
                        if d.track_id in paired_ids:
                            continue
                        role = self.target_filter.classify_role(d.class_id, d.xyxy, width, height)
                        cx = (d.xyxy[0] + d.xyxy[2]) * 0.5
                        cy = (d.xyxy[1] + d.xyxy[3]) * 0.5
                        side = zone.classify(cx, cy)
                        cls_name = names.get(d.class_id, str(d.class_id)) if isinstance(names, dict) else str(d.class_id)
                        draw_track(
                            frame, d.track_id, d.xyxy, d.conf, side, False, role=role, class_name=str(cls_name)
                        )
            else:
                person_boxes = [
                    list(d.xyxy)
                    for d in dets
                    if self.target_filter.classify_role(d.class_id, d.xyxy, width, height) == "person"
                ]
                for d in dets:
                    role = self.target_filter.classify_role(d.class_id, d.xyxy, width, height)
                    is_target = self.target_filter.accept_track(
                        d.track_id, d.class_id, d.xyxy, width, height, person_boxes=person_boxes
                    )
                    cx = (d.xyxy[0] + d.xyxy[2]) * 0.5
                    cy = (d.xyxy[1] + d.xyxy[3]) * 0.5
                    side_now = zone.classify(cx, cy)
                    cls_name = names.get(d.class_id, str(d.class_id)) if isinstance(names, dict) else str(d.class_id)
                    if out_cfg.get("draw_tracks", True):
                        draw_track(
                            frame, d.track_id, d.xyxy, d.conf, side_now, is_target, role=role, class_name=str(cls_name)
                        )
                    prev = self._prev_centroid.get(d.track_id)
                    if is_target or role in ("bed", "lying_patient", "patient_head"):
                        crossing, new_side = side_transition(
                            self._last_stable_side.get(d.track_id), side_now
                        )
                        if new_side is not None:
                            self._last_stable_side[d.track_id] = new_side
                        if is_target:
                            evt = self.event_manager.consider(
                                track_id=d.track_id,
                                crossing=crossing,
                                frame_idx=frame_idx,
                                fps=fps,
                                centroid=(cx, cy),
                                bbox_xyxy=d.xyxy,
                                confidence=d.conf,
                                frame_diag=frame_diag,
                                prev_centroid=prev,
                            )
                            if evt is not None:
                                self._recent_events.append((frame_idx, evt))
                                print(
                                    f"[{evt.event.upper()}] track={evt.track_id} "
                                    f"video_t={evt.video_time_sec:.2f}s frame={evt.frame_idx}"
                                )
                    self._prev_centroid[d.track_id] = (cx, cy)

            hold = 45
            visible = [e for fidx, e in self._recent_events if frame_idx - fidx <= hold]
            draw_event_banner(frame, visible)
            n_enter = sum(1 for e in self.event_manager.events if e.event == "enter")
            n_exit = sum(1 for e in self.event_manager.events if e.event == "exit")
            draw_hud(frame, frame_idx, fps, n_enter, n_exit, target_mode=target_mode)

            if writer is not None:
                writer.write(frame)
            frame_idx += 1

        cap.release()
        if writer is not None:
            writer.release()

        events_path = output_dir / "events.json"
        if out_cfg.get("save_events_json", True):
            self.event_manager.save_json(events_path)

        elapsed = time.time() - t0
        return {
            "frames": frame_idx,
            "fps_process": round(frame_idx / max(elapsed, 1e-6), 2),
            "video_fps": fps,
            "zone_mode": zone_mode,
            "target_mode": target_mode,
            "detect_classes": self.detect_classes,
            "enters": sum(1 for e in self.event_manager.events if e.event == "enter"),
            "exits": sum(1 for e in self.event_manager.events if e.event == "exit"),
            "events_path": str(events_path),
            "video_path": str(out_video) if writer is not None else None,
            "events": [e.to_dict() for e in self.event_manager.events],
        }
