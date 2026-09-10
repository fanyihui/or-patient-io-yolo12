"""应用会话：一路视频预览、截图、ROI、实时监测。"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np
import yaml

from .events import IOEvent
from .pipeline import ORIOPipeline, load_config, resolve_device
from .roi_store import apply_roi_to_config, build_zone_payload, load_roi, save_roi_yaml
from .stream import StreamConfig, is_stream_source, normalize_source, open_capture, grab_latest_frame


class AppSession:
    def __init__(self, project_root: Path, default_config: str | Path):
        self.root = Path(project_root)
        self.default_config_path = Path(default_config)
        self.base_cfg = load_config(self.default_config_path)

        self.source: Optional[str] = None
        self.connected = False
        self.frame_w = 0
        self.frame_h = 0
        self._latest: Optional[np.ndarray] = None
        self._snapshot: Optional[np.ndarray] = None
        self._lock = threading.RLock()

        self._cap: Optional[cv2.VideoCapture] = None
        self._grab_thread: Optional[threading.Thread] = None
        self._grab_stop = threading.Event()

        self.roi_payload: Optional[dict] = None
        self.roi_path: Optional[Path] = None

        self.monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self.pipeline: Optional[ORIOPipeline] = None
        self.events: List[dict] = []
        self.monitor_stats: Dict[str, Any] = {
            "frames": 0,
            "enters": 0,
            "exits": 0,
            "started_at": None,
            "last_frame_at": None,
        }
        self.output_dir: Optional[Path] = None
        self._annotated: Optional[np.ndarray] = None
        self.status_message = "未连接"
        self._on_event: Optional[Callable[[dict], None]] = None

    def set_event_hook(self, fn: Optional[Callable[[dict], None]]) -> None:
        self._on_event = fn

    def connect(self, source: str, *, reconnect: bool = True) -> dict:
        self.disconnect()
        src = normalize_source(source)
        cfg = StreamConfig(
            reconnect=reconnect and is_stream_source(src),
            drop_pending=is_stream_source(src),
            buffer_size=1,
            target_fps=0.0,
        )
        cap = open_capture(src, cfg)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {source}")
        ok, frame = cap.read()
        if not ok or frame is None:
            # 再试一次
            ok, frame = grab_latest_frame(cap, 3)
        if not ok or frame is None:
            cap.release()
            raise RuntimeError(f"已打开但读不到帧: {source}")

        h, w = frame.shape[:2]
        with self._lock:
            self._cap = cap
            self.source = str(src)
            self.connected = True
            self.frame_w, self.frame_h = w, h
            self._latest = frame.copy()
            self.status_message = f"已连接 {self.source}"
        self._grab_stop.clear()
        self._grab_thread = threading.Thread(target=self._grab_loop, args=(cfg,), daemon=True)
        self._grab_thread.start()
        return {"source": self.source, "width": w, "height": h, "live": is_stream_source(src)}

    def _grab_loop(self, cfg: StreamConfig) -> None:
        while not self._grab_stop.is_set():
            with self._lock:
                cap = self._cap
            if cap is None:
                break
            if cfg.drop_pending and is_stream_source(self.source or ""):
                ok, frame = grab_latest_frame(cap, cfg.max_pending_grabs)
            else:
                ok, frame = cap.read()
                if not ok and self.source and not is_stream_source(self.source):
                    # 文件循环
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
            if ok and frame is not None:
                with self._lock:
                    self._latest = frame
                    self.frame_h, self.frame_w = frame.shape[:2]
            else:
                time.sleep(0.05)
            time.sleep(0.01)

    def disconnect(self) -> None:
        self.stop_monitor()
        self._grab_stop.set()
        if self._grab_thread and self._grab_thread.is_alive():
            self._grab_thread.join(timeout=2.0)
        self._grab_thread = None
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
            self.connected = False
            self.status_message = "未连接"

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def snapshot_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._snapshot is None else self._snapshot.copy()

    def annotated_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._annotated is not None:
                return self._annotated.copy()
            return None if self._latest is None else self._latest.copy()

    def take_snapshot(self, save_dir: Optional[Path] = None) -> dict:
        frame = self.latest_frame()
        if frame is None:
            raise RuntimeError("无可用画面，请先连接视频")
        with self._lock:
            self._snapshot = frame.copy()
            h, w = frame.shape[:2]
        path = None
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = save_dir / f"snapshot_{stamp}.jpg"
            cv2.imwrite(str(path), frame)
        self.status_message = "已截图，可标注门口 ROI"
        return {"width": w, "height": h, "path": str(path) if path else None}

    def encode_jpeg(self, which: str = "live", quality: int = 80) -> bytes:
        if which == "snapshot":
            frame = self.snapshot_frame()
        elif which == "monitor":
            frame = self.annotated_frame()
        else:
            frame = self.latest_frame()
        if frame is None:
            raise RuntimeError("无画面")
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("JPEG 编码失败")
        return buf.tobytes()

    def set_roi(
        self,
        outside: list,
        inside: list,
        *,
        already_normalized: bool = True,
        site_name: str = "default",
        save: bool = True,
    ) -> dict:
        frame = self.snapshot_frame()
        if frame is None:
            frame = self.latest_frame()
        if frame is None:
            raise RuntimeError("无画面尺寸，请先连接并截图")
        h, w = frame.shape[:2]
        payload = build_zone_payload(
            outside,
            inside,
            w,
            h,
            already_normalized=already_normalized,
            meta={"source": self.source, "site": site_name},
        )
        self.roi_payload = payload
        if save:
            import json

            out_dir = self.root / "configs" / "sites"
            path = out_dir / f"{site_name}.yaml"
            save_roi_yaml(path, payload, base_config=self.base_cfg)
            save_roi_yaml(out_dir / f"{site_name}.roi.yaml", payload, base_config=None)
            (out_dir / f"{site_name}.roi.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            self.roi_path = path
            self.status_message = f"ROI 已保存: {path.name}"
        else:
            self.status_message = "ROI 已更新（未落盘）"
        return payload

    def load_roi_file(self, path: str | Path) -> dict:
        payload = load_roi(path)
        self.roi_payload = payload
        self.roi_path = Path(path)
        self.status_message = f"已加载 ROI: {Path(path).name}"
        return payload

    def get_roi(self) -> Optional[dict]:
        return deepcopy(self.roi_payload) if self.roi_payload else None

    def start_monitor(self, *, device: str | int | None = None, output_name: Optional[str] = None) -> dict:
        if not self.connected:
            raise RuntimeError("请先连接视频")
        if not self.roi_payload:
            raise RuntimeError("请先标注并保存手术室门 ROI")
        if self.monitoring:
            return {"ok": True, "message": "监测已在运行"}

        cfg = apply_roi_to_config(self.base_cfg, self.roi_payload)
        cfg.setdefault("model", {})["device"] = resolve_device(
            device if device is not None else cfg.get("model", {}).get("device", "auto")
        )
        # 监测时降低处理负载
        stream = cfg.setdefault("stream", {})
        stream.setdefault("target_fps", 8)
        stream.setdefault("drop_pending", True)
        out_cfg = cfg.setdefault("output", {})
        out_cfg["save_video"] = bool(out_cfg.get("save_video", True))
        out_cfg["save_events_jsonl"] = True
        out_cfg["save_events_csv"] = True

        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = output_name or f"app_{stamp}"
        self.output_dir = self.root / "outputs" / name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 固化本次使用的配置
        with open(self.output_dir / "run_config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
        if self.roi_payload:
            save_roi_yaml(self.output_dir / "door_roi.yaml", self.roi_payload, base_config=None)

        self.status_message = "正在加载检测模型…"
        pipe = ORIOPipeline(cfg, project_root=self.root)
        pipe.event_manager.source = self.source or ""
        self.pipeline = pipe
        self.events = []
        self.monitor_stats = {
            "frames": 0,
            "enters": 0,
            "exits": 0,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "last_frame_at": None,
            "device": cfg["model"]["device"],
            "output_dir": str(self.output_dir),
        }

        from .events import LiveEventRecorder

        recorder = LiveEventRecorder(
            self.output_dir,
            save_jsonl=True,
            save_csv=True,
            save_json_snapshot=True,
            source=self.source or "",
        )
        pipe._live_recorder = recorder

        def _hook(evt: IOEvent) -> None:
            row = evt.to_dict()
            row["local_time"] = datetime.now().astimezone().isoformat(timespec="seconds")
            self.events.append(row)
            self.monitor_stats["enters"] = sum(1 for e in self.events if e["event"] == "enter")
            self.monitor_stats["exits"] = sum(1 for e in self.events if e["event"] == "exit")
            if self._on_event:
                self._on_event(row)

        pipe._on_event_hook = _hook

        self._monitor_stop.clear()
        self.monitoring = True
        self.status_message = "实时监测中"
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        return {"ok": True, "output_dir": str(self.output_dir), "device": cfg["model"]["device"]}

    def _monitor_loop(self) -> None:
        assert self.pipeline is not None
        pipe = self.pipeline
        fps = float((pipe.cfg.get("stream") or {}).get("target_fps") or 8) or 8.0
        min_interval = 1.0 / fps
        frame_idx = 0
        last_t = 0.0
        role_hist: Dict[str, int] = {}
        try:
            while not self._monitor_stop.is_set():
                now = time.time()
                if now - last_t < min_interval:
                    time.sleep(0.005)
                    continue
                frame = self.latest_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue
                last_t = now
                h, w = frame.shape[:2]
                annotated = pipe.process_frame(
                    frame, frame_idx, fps, width=w, height=h, role_hist=role_hist
                )
                with self._lock:
                    self._annotated = annotated
                frame_idx += 1
                self.monitor_stats["frames"] = frame_idx
                self.monitor_stats["last_frame_at"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:  # noqa: BLE001
            self.status_message = f"监测异常: {e}"
        finally:
            self.monitoring = False
            if pipe._live_recorder is not None:
                pipe._live_recorder.close(pipe.event_manager.events)
                pipe._live_recorder = None
            if self.output_dir is not None:
                pipe.event_manager.save_json(self.output_dir / "events.json")
            if not self.status_message.startswith("监测异常"):
                self.status_message = "监测已停止"

    def stop_monitor(self) -> None:
        self._monitor_stop.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        self._monitor_thread = None
        self.monitoring = False
        if self.pipeline is not None:
            self.pipeline._on_event_hook = None

    def state(self) -> dict:
        return {
            "connected": self.connected,
            "source": self.source,
            "width": self.frame_w,
            "height": self.frame_h,
            "has_snapshot": self._snapshot is not None,
            "has_roi": self.roi_payload is not None,
            "roi_path": str(self.roi_path) if self.roi_path else None,
            "monitoring": self.monitoring,
            "status": self.status_message,
            "stats": deepcopy(self.monitor_stats),
            "event_count": len(self.events),
            "enters": sum(1 for e in self.events if e.get("event") == "enter"),
            "exits": sum(1 for e in self.events if e.get("event") == "exit"),
        }
