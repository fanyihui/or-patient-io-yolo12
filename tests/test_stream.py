"""流媒体源与实时事件落盘单元测试。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.events import IOEvent, LiveEventRecorder
from or_io.stream import is_stream_source, normalize_source, source_label
from or_io.video_writer import AnnotatedVideoWriter
import numpy as np


def test_is_stream_source():
    assert is_stream_source("rtsp://192.168.1.10:554/stream1")
    assert is_stream_source("rtsps://cam/secret")
    assert is_stream_source("http://host/video.mjpg")
    assert is_stream_source("https://cdn/live/index.m3u8")
    assert is_stream_source("rtmp://live/app/stream")
    assert is_stream_source(0)
    assert is_stream_source("0")
    assert is_stream_source("/dev/video0")
    assert not is_stream_source("data/samples/or_door_synthetic.mp4")
    assert not is_stream_source("/tmp/video.mp4")


def test_normalize_and_label():
    assert normalize_source("1") == 1
    assert normalize_source(0) == 0
    assert isinstance(normalize_source("rtsp://a/b"), str)
    assert "rtsp" in source_label("rtsp://192.168.0.5/live")
    assert source_label(0) == "cam0"


def test_live_event_recorder(tmp_path: Path | None = None):
    out = Path(tmp_path) if tmp_path is not None else (ROOT / "outputs" / "_test_live_events")
    if tmp_path is None:
        out = ROOT / "outputs" / "_test_live_events"
        if out.exists():
            for p in out.glob("*"):
                p.unlink()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    rec = LiveEventRecorder(
        out,
        save_jsonl=True,
        save_csv=True,
        save_json_snapshot=True,
        snapshot_every_events=1,
        source="rtsp://test",
    )
    events = []
    evt = IOEvent(
        event="enter",
        track_id=7,
        frame_idx=12,
        video_time_sec=1.2,
        wall_time_iso="2026-09-10T06:00:00+00:00",
        centroid=(100.0, 200.0),
        bbox_xyxy=(10.0, 20.0, 110.0, 220.0),
        confidence=0.88,
        source="rtsp://test",
    )
    events.append(evt)
    rec.append(evt, events)
    evt2 = IOEvent(
        event="exit",
        track_id=7,
        frame_idx=40,
        video_time_sec=4.0,
        wall_time_iso="2026-09-10T06:00:05+00:00",
        centroid=(300.0, 210.0),
        bbox_xyxy=(200.0, 20.0, 400.0, 220.0),
        confidence=0.81,
        source="rtsp://test",
    )
    events.append(evt2)
    rec.append(evt2, events)
    rec.close(events)

    lines = (out / "events.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["event"] == "enter"
    assert json.loads(lines[1])["event"] == "exit"

    csv_text = (out / "events.csv").read_text(encoding="utf-8")
    assert "enter" in csv_text and "exit" in csv_text

    snap = json.loads((out / "events.json").read_text(encoding="utf-8"))
    assert snap["count"] == 2
    assert snap["enters"] == 1
    assert snap["exits"] == 1


def test_annotated_writer(tmp_path: Path | None = None):
    out = Path(tmp_path) if tmp_path is not None else (ROOT / "outputs" / "_test_writer")
    out.mkdir(parents=True, exist_ok=True)
    for p in out.glob("*.mp4"):
        p.unlink()
    w = AnnotatedVideoWriter(out, fps=5.0, frame_size=(64, 48), enabled=True, segment_minutes=0)
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    path = w.write(frame, 0)
    w.release()
    assert path is not None and path.exists()
    assert path.stat().st_size > 0


if __name__ == "__main__":
    test_is_stream_source()
    test_normalize_and_label()
    test_live_event_recorder()
    test_annotated_writer()
    print("stream tests passed")
