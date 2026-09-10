"""YOLO（COCO / YOLO-World）+ ByteTrack 入/出室检测流水线（病床 + 平躺患者）。"""

from __future__ import annotations

import signal
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

from .events import EventManager, IOEvent, LiveEventRecorder
from .model_loader import load_detector
from .stream import StreamConfig, is_stream_source, iter_frames, normalize_source, source_label
from .stretcher_filter import PSEUDO_BED_CLASS_ID, BedPatientPair, Detection, TargetFilter
from .video_writer import AnnotatedVideoWriter
from .visualize import draw_event_banner, draw_hud, draw_pair, draw_track, draw_zone
from .zones import DoorLine, DualROIZones, Side, build_zone, side_transition

Zone = Union[DoorLine, DualROIZones]


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_detect_classes(cfg: dict) -> List[int] | None:
    """仅 COCO 路径需要；开放词汇由 set_classes 限定，返回 None。"""
    backend = str(cfg.get("model", {}).get("backend") or "world").lower()
    if backend in ("world", "yolo-world", "open_vocab", "open-vocab"):
        return None
    model_classes = cfg.get("model", {}).get("classes")
    if model_classes is not None:
        return list(model_classes)
    target = cfg.get("target") or {}
    mode = str(target.get("mode", "bed_patient"))
    if mode in ("bed_patient", "stretcher"):
        st = target.get("stretcher") or target.get("bed_patient") or {}
        ids = set(st.get("bed_class_ids") or [59])
        ids.update(st.get("extra_bed_like_ids") or [56, 57, 60])
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
        self.cfg.setdefault("model", {})["device"] = resolve_device(config.get("model", {}).get("device"))

        self.detector = load_detector(self.cfg["model"])
        self.model = self.detector.model
        self.class_names = self.detector.names

        tracker_rel = config["tracker"]["config"]
        self.tracker_cfg = str((self.root / tracker_rel).resolve())
        if not Path(self.tracker_cfg).exists():
            self.tracker_cfg = "bytetrack.yaml"

        self.target_filter = TargetFilter.from_config(config)
        self.target_filter.bind_class_ids(
            bed_class_ids=self.detector.bed_class_ids,
            person_class_ids=self.detector.person_class_ids,
            extra_bed_like_ids=(),
            equipment_class_ids=getattr(self.detector, "equipment_class_ids", ()),
        )
        self.target_filter.extra_bed_like_ids = ()
        self.target_filter.bed_class_ids = tuple(
            sorted(set(self.target_filter.bed_class_ids) | {PSEUDO_BED_CLASS_ID})
        )

        self.detect_classes = self.detector.detect_classes
        self.class_names[PSEUDO_BED_CLASS_ID] = "pseudo_bed"
        ev = config.get("events", {})
        self.event_manager = EventManager(
            debounce_frames=int(ev.get("debounce_frames", 15)),
            min_crossing_disp=float(ev.get("min_crossing_disp", 0.02)),
        )
        self._prev_centroid: Dict[int, Tuple[float, float]] = {}
        self._last_stable_side: Dict[int, Side] = {}
        self._recent_events: Deque[Tuple[int, IOEvent]] = deque(maxlen=32)
        self.zone: Optional[Zone] = None
        self._tmp_id = 10_000_000
        self._sec_id = 8_000_000
        self._live_recorder: Optional[LiveEventRecorder] = None
        self._stop_requested = False
        self._on_event_hook: Optional[Callable[[IOEvent], None]] = None

    def request_stop(self) -> None:
        self._stop_requested = True

    def _stream_config(self) -> StreamConfig:
        s = self.cfg.get("stream") or {}
        return StreamConfig(
            reconnect=bool(s.get("reconnect", True)),
            reconnect_delay_sec=float(s.get("reconnect_delay_sec", 2.0)),
            max_reconnect_attempts=int(s.get("max_reconnect_attempts", 0)),
            buffer_size=int(s.get("buffer_size", 1)),
            open_timeout_ms=int(s.get("open_timeout_ms", 8000)),
            read_timeout_ms=int(s.get("read_timeout_ms", 5000)),
            drop_pending=bool(s.get("drop_pending", True)),
            max_pending_grabs=int(s.get("max_pending_grabs", 8)),
            target_fps=float(s.get("target_fps", 0.0)),
            ffmpeg_options=str(
                s.get(
                    "ffmpeg_options",
                    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay",
                )
            ),
        )

    def _ensure_zone(self, frame_w: int, frame_h: int) -> Zone:
        if self.zone is None:
            self.zone = build_zone(self.cfg, frame_w, frame_h)
        return self.zone

    def _name_of(self, class_id: int) -> str:
        if int(class_id) == PSEUDO_BED_CLASS_ID:
            return "pseudo_bed"
        return str(self.class_names.get(int(class_id), class_id))

    @staticmethod
    def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    def _secondary_detections(self, frame, model_cfg: dict, frame_w: int, frame_h: int) -> List[Detection]:
        """COCO 辅检：补推床/家具漏检；用量化中心生成较稳的临时 track id。"""
        sec = self.detector.secondary_model
        if sec is None:
            return []
        conf = float(model_cfg.get("secondary_conf") or min(0.08, float(model_cfg.get("conf", 0.08))))
        kwargs = dict(
            source=frame,
            conf=conf,
            iou=float(model_cfg.get("iou", 0.5)),
            imgsz=int(model_cfg.get("secondary_imgsz") or model_cfg.get("imgsz", 640)),
            device=model_cfg.get("device", "cpu"),
            verbose=False,
        )
        if self.detector.secondary_detect_classes is not None:
            kwargs["classes"] = self.detector.secondary_detect_classes
        r0 = sec.predict(**kwargs)[0]
        boxes = r0.boxes
        if boxes is None or len(boxes) == 0:
            return []
        offset = int(self.detector.secondary_id_offset)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        allow = set(self.detector.secondary_bed_class_ids) | set(self.detector.secondary_person_class_ids)
        out: List[Detection] = []
        for box, conf_v, cls_id in zip(xyxy, confs, clss):
            cid = int(cls_id)
            if cid not in allow:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            # 量化中心 → 跨帧较稳的伪 track id
            tid = 800_000 + int(cx / 48) * 1000 + int(cy / 48) + cid * 17
            out.append(
                Detection(
                    track_id=int(tid),
                    class_id=offset + cid,
                    conf=float(conf_v),
                    xyxy=(x1, y1, x2, y2),
                )
            )
        return out

    def _merge_detections(self, primary: List[Detection], secondary: List[Detection]) -> List[Detection]:
        if not secondary:
            return primary
        merged = list(primary)
        for s in secondary:
            if any(self._iou(s.xyxy, p.xyxy) >= 0.45 for p in primary):
                continue
            merged.append(s)
        return merged

    def _boxes_to_detections(self, boxes) -> List[Detection]:
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        if boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
        else:
            ids = np.arange(self._tmp_id, self._tmp_id + len(xyxy), dtype=int)
            self._tmp_id += len(xyxy)
        dets: List[Detection] = []
        for box, tid, conf, cls_id in zip(xyxy, ids, confs, clss):
            dets.append(
                Detection(
                    track_id=int(tid),
                    class_id=int(cls_id),
                    conf=float(conf),
                    xyxy=tuple(float(v) for v in box.tolist()),  # type: ignore[arg-type]
                )
            )
        return dets

    def _emit_event(self, evt: IOEvent) -> None:
        self._recent_events.append((evt.frame_idx, evt))
        if self._live_recorder is not None:
            self._live_recorder.append(evt, self.event_manager.events)
        if self._on_event_hook is not None:
            self._on_event_hook(evt)

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
                self._emit_event(evt)
                print(
                    f"[{evt.event.upper()}] bed={pair.bed_track_id}+patient={pair.patient_track_id} "
                    f"wall={evt.wall_time_iso} video_t={evt.video_time_sec:.2f}s frame={evt.frame_idx}"
                )
        self._prev_centroid[tid] = (cx, cy)

    def process_frame(
        self,
        frame,
        frame_idx: int,
        fps: float,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        role_hist: Optional[Dict[str, int]] = None,
    ):
        """处理单帧：检测/跟踪/配对/画框；返回标注后的 frame。"""
        if width is None or height is None:
            height, width = frame.shape[:2]
        frame_diag = float(np.hypot(width, height))
        zone = self._ensure_zone(width, height)
        target_mode = self.target_filter.mode
        out_cfg = self.cfg.get("output", {})
        model_cfg = self.cfg["model"]
        hide_standing = bool(out_cfg.get("hide_standing_staff", True))
        hide_equipment = bool(out_cfg.get("hide_equipment_carts", True))
        draw_roles = out_cfg.get("draw_roles")
        if draw_roles is not None:
            draw_roles = [str(x) for x in draw_roles]
        if role_hist is None:
            role_hist = {}

        track_kwargs = dict(
            source=frame,
            persist=True,
            tracker=self.tracker_cfg,
            conf=float(model_cfg.get("conf", 0.15)),
            iou=float(model_cfg.get("iou", 0.5)),
            imgsz=int(model_cfg.get("imgsz", 640)),
            device=model_cfg.get("device", "cpu"),
            verbose=False,
        )
        if self.detect_classes is not None:
            track_kwargs["classes"] = self.detect_classes
        results = self.model.track(**track_kwargs)
        r0 = results[0]
        boxes = r0.boxes

        if out_cfg.get("draw_door_line", True) or out_cfg.get("draw_zone", True):
            draw_zone(frame, zone)

        dets = self._boxes_to_detections(boxes)
        if self.detector.backend == "fusion":
            sec = self._secondary_detections(frame, model_cfg, width, height)
            dets = self._merge_detections(dets, sec)

        standing_boxes = self.target_filter.collect_standing_boxes(dets, width, height)
        if self.target_filter.allow_pseudo_bed:
            roles_pre = self.target_filter.classify_roles(dets, width, height)
            existing = [d for d in dets if roles_pre.get(d.track_id) == "bed"]
            dets = list(dets) + self.target_filter.synthesize_pseudo_beds(
                dets, width, height, existing_beds=existing
            )
            standing_boxes = self.target_filter.collect_standing_boxes(dets, width, height)

        roles = self.target_filter.classify_roles(dets, width, height)
        for d in dets:
            role = roles.get(d.track_id, "other")
            role_hist[role] = role_hist.get(role, 0) + 1

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
                    if not self.target_filter.should_draw_detection(
                        d.class_id,
                        d.xyxy,
                        width,
                        height,
                        hide_standing_staff=hide_standing,
                        hide_equipment_carts=hide_equipment,
                        draw_roles=draw_roles,
                        standing_boxes=standing_boxes,
                    ):
                        continue
                    role = roles.get(d.track_id, "other")
                    cx = (d.xyxy[0] + d.xyxy[2]) * 0.5
                    cy = (d.xyxy[1] + d.xyxy[3]) * 0.5
                    side = zone.classify(cx, cy)
                    draw_track(
                        frame,
                        d.track_id,
                        d.xyxy,
                        d.conf,
                        side,
                        False,
                        role=role,
                        class_name=self._name_of(d.class_id),
                    )
        else:
            person_boxes = [
                list(d.xyxy)
                for d in dets
                if roles.get(d.track_id) == "person"
            ]
            for d in dets:
                role = roles.get(d.track_id, "other")
                is_target = self.target_filter.accept_track(
                    d.track_id, d.class_id, d.xyxy, width, height, person_boxes=person_boxes
                )
                if hide_standing and self.target_filter.is_upright_person(
                    d.class_id, d.xyxy, width, height, standing_boxes=standing_boxes
                ):
                    is_target = False
                cx = (d.xyxy[0] + d.xyxy[2]) * 0.5
                cy = (d.xyxy[1] + d.xyxy[3]) * 0.5
                side_now = zone.classify(cx, cy)
                if out_cfg.get("draw_tracks", True) and self.target_filter.should_draw_detection(
                    d.class_id,
                    d.xyxy,
                    width,
                    height,
                    hide_standing_staff=hide_standing,
                    hide_equipment_carts=hide_equipment,
                    draw_roles=draw_roles,
                    standing_boxes=standing_boxes,
                ):
                    draw_track(
                        frame,
                        d.track_id,
                        d.xyxy,
                        d.conf,
                        side_now,
                        is_target,
                        role=role,
                        class_name=self._name_of(d.class_id),
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
                            self._emit_event(evt)
                            print(
                                f"[{evt.event.upper()}] track={evt.track_id} "
                                f"wall={evt.wall_time_iso} video_t={evt.video_time_sec:.2f}s "
                                f"frame={evt.frame_idx}"
                            )
                self._prev_centroid[d.track_id] = (cx, cy)

        hold = 45
        visible = [e for fidx, e in self._recent_events if frame_idx - fidx <= hold]
        draw_event_banner(frame, visible)
        n_enter = sum(1 for e in self.event_manager.events if e.event == "enter")
        n_exit = sum(1 for e in self.event_manager.events if e.event == "exit")
        draw_hud(frame, frame_idx, fps, n_enter, n_exit, target_mode=target_mode)
        return frame

    def _build_summary(
        self,
        *,
        frame_idx: int,
        elapsed: float,
        fps: float,
        zone_mode: str,
        role_hist: dict,
        events_path: Path,
        video_path: Optional[str],
        source: str,
        live: bool,
    ) -> dict:
        return {
            "frames": frame_idx,
            "fps_process": round(frame_idx / max(elapsed, 1e-6), 2),
            "video_fps": fps,
            "zone_mode": zone_mode,
            "target_mode": self.target_filter.mode,
            "backend": self.detector.backend,
            "detect_classes": self.detect_classes,
            "bed_class_ids": list(self.target_filter.bed_class_ids),
            "person_class_ids": list(self.target_filter.person_class_ids),
            "role_counts": role_hist,
            "enters": sum(1 for e in self.event_manager.events if e.event == "enter"),
            "exits": sum(1 for e in self.event_manager.events if e.event == "exit"),
            "events_path": str(events_path),
            "video_path": video_path,
            "source": source,
            "live": live,
            "started_at": getattr(self, "_run_started_at", None),
            "finished_at": datetime.now(timezone.utc).isoformat(),
            "events": [e.to_dict() for e in self.event_manager.events],
        }

    def process_video(
        self,
        source: str | Path,
        output_dir: str | Path,
        max_frames: Optional[int] = None,
    ) -> dict:
        """离线文件或一次性源；若 source 是流地址则走实时模式。"""
        source_s = str(source)
        if is_stream_source(source_s):
            return self.process_stream(source_s, output_dir, max_frames=max_frames)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.event_manager.source = source_s
        self._run_started_at = datetime.now(timezone.utc).isoformat()

        out_cfg = self.cfg.get("output", {})
        recorder = None
        if out_cfg.get("save_events_jsonl", False) or out_cfg.get("save_events_csv", False):
            recorder = LiveEventRecorder(
                output_dir,
                save_jsonl=bool(out_cfg.get("save_events_jsonl", False)),
                save_csv=bool(out_cfg.get("save_events_csv", False)),
                save_json_snapshot=bool(out_cfg.get("save_events_json", True)),
                snapshot_every_events=int(out_cfg.get("events_flush_every", 1)),
                source=source_s,
            )
            self._live_recorder = recorder

        stream_cfg = self._stream_config()
        # 文件模式：不丢帧、不限 target_fps
        file_cfg = StreamConfig(
            reconnect=False,
            drop_pending=False,
            target_fps=0.0,
            buffer_size=stream_cfg.buffer_size,
        )

        role_hist = {
            "bed": 0,
            "equipment_cart": 0,
            "patient_head": 0,
            "lying_patient": 0,
            "person": 0,
            "other": 0,
        }
        writer: Optional[AnnotatedVideoWriter] = None
        fps = 25.0
        width = height = 0
        zone_mode = str((self.cfg.get("zone") or {}).get("mode", "line"))
        t0 = time.time()
        frames_done = 0
        last_video_path = None

        try:
            for frame_idx, frame, fps_hint in iter_frames(source_s, file_cfg, live=False):
                if max_frames is not None and frame_idx >= max_frames:
                    break
                fps = fps_hint or fps
                height, width = frame.shape[:2]
                if writer is None and out_cfg.get("save_video", True):
                    writer = AnnotatedVideoWriter(
                        output_dir,
                        fps,
                        (width, height),
                        enabled=True,
                        segment_minutes=0.0,
                    )
                self._ensure_zone(width, height)
                annotated = self.process_frame(
                    frame, frame_idx, fps, width=width, height=height, role_hist=role_hist
                )
                if writer is not None:
                    last_video_path = writer.write(annotated, frame_idx)
                frames_done = frame_idx + 1
        finally:
            if writer is not None:
                writer.release()
            events_path = output_dir / "events.json"
            if recorder is not None:
                recorder.close(self.event_manager.events)
                self._live_recorder = None
            elif out_cfg.get("save_events_json", True):
                self.event_manager.save_json(events_path)

        elapsed = time.time() - t0
        print(f"[detect-stats] frames={frames_done} roles={role_hist}")
        return self._build_summary(
            frame_idx=frames_done,
            elapsed=elapsed,
            fps=fps,
            zone_mode=zone_mode,
            role_hist=role_hist,
            events_path=output_dir / "events.json",
            video_path=str(last_video_path) if last_video_path else None,
            source=source_s,
            live=False,
        )

    def process_stream(
        self,
        source: str | int,
        output_dir: str | Path,
        max_frames: Optional[int] = None,
        *,
        preview: bool = False,
        on_event: Optional[Callable[[IOEvent], None]] = None,
    ) -> dict:
        """
        实时流识别与记录。
        支持: rtsp://, rtmp://, http(s)://, 摄像头索引(0/1), /dev/video0
        事件即时写入 events.jsonl / events.csv，并刷新 events.json。
        Ctrl+C 优雅退出并落盘。
        """
        src = normalize_source(source)
        source_s = str(src)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._stop_requested = False
        self.event_manager.source = source_s
        self._run_started_at = datetime.now(timezone.utc).isoformat()

        out_cfg = self.cfg.get("output", {})
        stream_cfg = self._stream_config()
        zone_mode = str((self.cfg.get("zone") or {}).get("mode", "line"))

        recorder = LiveEventRecorder(
            output_dir,
            save_jsonl=bool(out_cfg.get("save_events_jsonl", True)),
            save_csv=bool(out_cfg.get("save_events_csv", True)),
            save_json_snapshot=bool(out_cfg.get("save_events_json", True)),
            snapshot_every_events=int(out_cfg.get("events_flush_every", 1)),
            source=source_s,
        )
        self._live_recorder = recorder
        self._on_event_hook = on_event

        prev_sigint = signal.getsignal(signal.SIGINT)
        prev_sigterm = signal.getsignal(signal.SIGTERM)

        def _handle_stop(signum, frame):  # noqa: ANN001
            print(f"\n[stream] signal {signum} → stopping…")
            self.request_stop()

        signal.signal(signal.SIGINT, _handle_stop)
        signal.signal(signal.SIGTERM, _handle_stop)

        role_hist = {
            "bed": 0,
            "equipment_cart": 0,
            "patient_head": 0,
            "lying_patient": 0,
            "person": 0,
            "other": 0,
        }
        writer: Optional[AnnotatedVideoWriter] = None
        fps = float(stream_cfg.target_fps) if stream_cfg.target_fps > 0 else 25.0
        width = height = 0
        t0 = time.time()
        last_status = t0
        frames_done = 0
        last_video_path = None
        status_every = float((self.cfg.get("stream") or {}).get("status_every_sec", 10.0))

        print(
            f"[stream] start source={source_s!r} label={source_label(src)} "
            f"target_fps={stream_cfg.target_fps or 'all'} drop_pending={stream_cfg.drop_pending}"
        )
        print(f"[stream] output={output_dir}")

        try:
            for frame_idx, frame, fps_hint in iter_frames(src, stream_cfg, live=True):
                if self._stop_requested:
                    break
                if max_frames is not None and frame_idx >= max_frames:
                    break

                if stream_cfg.target_fps > 0:
                    fps = stream_cfg.target_fps
                elif fps_hint:
                    fps = fps_hint

                height, width = frame.shape[:2]
                if writer is None and out_cfg.get("save_video", True):
                    writer = AnnotatedVideoWriter(
                        output_dir,
                        fps,
                        (width, height),
                        enabled=True,
                        segment_minutes=float(out_cfg.get("video_segment_minutes", 0) or 0),
                    )
                self._ensure_zone(width, height)
                annotated = self.process_frame(
                    frame, frame_idx, fps, width=width, height=height, role_hist=role_hist
                )
                if writer is not None:
                    last_video_path = writer.write(annotated, frame_idx)

                if preview:
                    try:
                        cv2.imshow("OR-IO live", annotated)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (27, ord("q")):
                            print("[stream] preview quit")
                            break
                    except Exception as e:  # noqa: BLE001
                        print(f"[stream] preview unavailable ({e}); continue headless")
                        preview = False

                frames_done = frame_idx + 1
                now = time.time()
                if status_every > 0 and now - last_status >= status_every:
                    elapsed = now - t0
                    n_enter = sum(1 for e in self.event_manager.events if e.event == "enter")
                    n_exit = sum(1 for e in self.event_manager.events if e.event == "exit")
                    print(
                        f"[stream] frames={frames_done} "
                        f"proc_fps={(frames_done / max(elapsed, 1e-6)):.2f} "
                        f"enter={n_enter} exit={n_exit}"
                    )
                    last_status = now
        finally:
            signal.signal(signal.SIGINT, prev_sigint)
            signal.signal(signal.SIGTERM, prev_sigterm)
            self._on_event_hook = None
            if writer is not None:
                writer.release()
            if preview:
                try:
                    cv2.destroyAllWindows()
                except Exception:  # noqa: BLE001
                    pass
            recorder.close(self.event_manager.events)
            self._live_recorder = None

        elapsed = time.time() - t0
        print(f"[detect-stats] frames={frames_done} roles={role_hist}")
        return self._build_summary(
            frame_idx=frames_done,
            elapsed=elapsed,
            fps=fps,
            zone_mode=zone_mode,
            role_hist=role_hist,
            events_path=output_dir / "events.json",
            video_path=str(last_video_path) if last_video_path else None,
            source=source_s,
            live=True,
        )
