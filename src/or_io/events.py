"""入/出室事件记录（批量 JSON + 实时 JSONL/CSV）。"""

from __future__ import annotations

import csv
import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, TextIO, Tuple


EventType = Literal["enter", "exit"]


@dataclass
class IOEvent:
    event: EventType
    track_id: int
    frame_idx: int
    video_time_sec: float
    wall_time_iso: str
    centroid: Tuple[float, float]
    bbox_xyxy: Tuple[float, float, float, float]
    confidence: float
    source: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid"] = list(self.centroid)
        d["bbox_xyxy"] = list(self.bbox_xyxy)
        if not d.get("source"):
            d.pop("source", None)
        return d


@dataclass
class EventManager:
    debounce_frames: int = 15
    min_crossing_disp: float = 0.02
    events: List[IOEvent] = field(default_factory=list)
    _last_event_frame: Dict[int, int] = field(default_factory=dict, repr=False)
    source: str = ""

    def consider(
        self,
        track_id: int,
        crossing: Optional[EventType],
        frame_idx: int,
        fps: float,
        centroid: Tuple[float, float],
        bbox_xyxy: Tuple[float, float, float, float],
        confidence: float,
        frame_diag: float,
        prev_centroid: Optional[Tuple[float, float]] = None,
        wall_time_iso: Optional[str] = None,
    ) -> Optional[IOEvent]:
        if crossing is None:
            return None

        last = self._last_event_frame.get(track_id, -10**9)
        if frame_idx - last < self.debounce_frames:
            return None

        if prev_centroid is not None and frame_diag > 0 and self.min_crossing_disp > 0:
            dx = centroid[0] - prev_centroid[0]
            dy = centroid[1] - prev_centroid[1]
            disp = (dx * dx + dy * dy) ** 0.5 / frame_diag
            if disp < self.min_crossing_disp:
                return None

        evt = IOEvent(
            event=crossing,
            track_id=track_id,
            frame_idx=frame_idx,
            video_time_sec=round(frame_idx / max(fps, 1e-6), 3),
            wall_time_iso=wall_time_iso or datetime.now(timezone.utc).isoformat(),
            centroid=(float(centroid[0]), float(centroid[1])),
            bbox_xyxy=tuple(float(v) for v in bbox_xyxy),  # type: ignore[arg-type]
            confidence=float(confidence),
            source=self.source,
        )
        self.events.append(evt)
        self._last_event_frame[track_id] = frame_idx
        return evt

    def save_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "count": len(self.events),
            "enters": sum(1 for e in self.events if e.event == "enter"),
            "exits": sum(1 for e in self.events if e.event == "exit"),
            "source": self.source or None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "events": [e.to_dict() for e in self.events],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class LiveEventRecorder:
    """实时落盘：JSONL 追加 + CSV 追加 + 周期性刷新汇总 JSON。"""

    CSV_FIELDS = (
        "wall_time_iso",
        "event",
        "track_id",
        "frame_idx",
        "video_time_sec",
        "confidence",
        "cx",
        "cy",
        "x1",
        "y1",
        "x2",
        "y2",
        "source",
    )

    def __init__(
        self,
        output_dir: str | Path,
        *,
        save_jsonl: bool = True,
        save_csv: bool = True,
        save_json_snapshot: bool = True,
        snapshot_every_events: int = 1,
        source: str = "",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_jsonl = save_jsonl
        self.save_csv = save_csv
        self.save_json_snapshot = save_json_snapshot
        self.snapshot_every_events = max(1, int(snapshot_every_events))
        self.source = source
        self.events_path = self.output_dir / "events.json"
        self.jsonl_path = self.output_dir / "events.jsonl"
        self.csv_path = self.output_dir / "events.csv"
        self._lock = threading.Lock()
        self._jsonl_fp: Optional[TextIO] = None
        self._csv_fp: Optional[TextIO] = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._written = 0
        self._open_files()

    def _open_files(self) -> None:
        if self.save_jsonl:
            self._jsonl_fp = open(self.jsonl_path, "a", encoding="utf-8")
        if self.save_csv:
            new_file = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
            self._csv_fp = open(self.csv_path, "a", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_fp, fieldnames=self.CSV_FIELDS)
            if new_file:
                self._csv_writer.writeheader()
                self._csv_fp.flush()

    def append(self, evt: IOEvent, all_events: Optional[List[IOEvent]] = None) -> None:
        row = evt.to_dict()
        with self._lock:
            if self._jsonl_fp is not None:
                self._jsonl_fp.write(json.dumps(row, ensure_ascii=False) + "\n")
                self._jsonl_fp.flush()
            if self._csv_writer is not None and self._csv_fp is not None:
                x1, y1, x2, y2 = evt.bbox_xyxy
                self._csv_writer.writerow(
                    {
                        "wall_time_iso": evt.wall_time_iso,
                        "event": evt.event,
                        "track_id": evt.track_id,
                        "frame_idx": evt.frame_idx,
                        "video_time_sec": evt.video_time_sec,
                        "confidence": round(evt.confidence, 4),
                        "cx": round(evt.centroid[0], 2),
                        "cy": round(evt.centroid[1], 2),
                        "x1": round(x1, 2),
                        "y1": round(y1, 2),
                        "x2": round(x2, 2),
                        "y2": round(y2, 2),
                        "source": evt.source or self.source,
                    }
                )
                self._csv_fp.flush()
            self._written += 1
            if self.save_json_snapshot and all_events is not None:
                if self._written % self.snapshot_every_events == 0:
                    self._write_snapshot(all_events)

    def _write_snapshot(self, events: List[IOEvent]) -> None:
        payload = {
            "count": len(events),
            "enters": sum(1 for e in events if e.event == "enter"),
            "exits": sum(1 for e in events if e.event == "exit"),
            "source": self.source or None,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "events": [e.to_dict() for e in events],
        }
        tmp = self.events_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.events_path)

    def flush_snapshot(self, events: List[IOEvent]) -> None:
        with self._lock:
            if self.save_json_snapshot:
                self._write_snapshot(events)
            if self._jsonl_fp is not None:
                self._jsonl_fp.flush()
            if self._csv_fp is not None:
                self._csv_fp.flush()

    def close(self, events: Optional[List[IOEvent]] = None) -> None:
        with self._lock:
            if events is not None and self.save_json_snapshot:
                self._write_snapshot(events)
            if self._jsonl_fp is not None:
                self._jsonl_fp.close()
                self._jsonl_fp = None
            if self._csv_fp is not None:
                self._csv_fp.close()
                self._csv_fp = None
            self._csv_writer = None

    def __enter__(self) -> "LiveEventRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
