"""入/出室事件记录。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple


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

    def to_dict(self) -> dict:
        d = asdict(self)
        d["centroid"] = list(self.centroid)
        d["bbox_xyxy"] = list(self.bbox_xyxy)
        return d


@dataclass
class EventManager:
    debounce_frames: int = 15
    min_crossing_disp: float = 0.02
    events: List[IOEvent] = field(default_factory=list)
    _last_event_frame: Dict[int, int] = field(default_factory=dict, repr=False)

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
            wall_time_iso=datetime.now(timezone.utc).isoformat(),
            centroid=(float(centroid[0]), float(centroid[1])),
            bbox_xyxy=tuple(float(v) for v in bbox_xyxy),  # type: ignore[arg-type]
            confidence=float(confidence),
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
            "events": [e.to_dict() for e in self.events],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
