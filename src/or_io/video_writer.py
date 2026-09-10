"""分段/连续标注视频写出。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import cv2


class AnnotatedVideoWriter:
    """写出 annotated.mp4；segment_minutes>0 时按时间切段。"""

    def __init__(
        self,
        output_dir: str | Path,
        fps: float,
        frame_size: Tuple[int, int],
        *,
        enabled: bool = True,
        segment_minutes: float = 0.0,
        prefix: str = "annotated",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.fps = max(float(fps), 1.0)
        self.frame_size = frame_size
        self.enabled = enabled
        self.segment_minutes = float(segment_minutes or 0.0)
        self.prefix = prefix
        self.writer: Optional[cv2.VideoWriter] = None
        self.current_path: Optional[Path] = None
        self._seg_start_frame = 0
        self._frames_in_seg = 0
        self._frame_budget = (
            int(self.segment_minutes * 60 * self.fps) if self.segment_minutes > 0 else 0
        )
        if self.enabled:
            self._open_new(0)

    def _open_new(self, frame_idx: int) -> None:
        self.release_writer_only()
        if self.segment_minutes > 0:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            name = f"{self.prefix}_{stamp}_f{frame_idx}.mp4"
        else:
            name = f"{self.prefix}.mp4"
        self.current_path = self.output_dir / name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(
            str(self.current_path), fourcc, self.fps, self.frame_size
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"无法创建视频写出: {self.current_path}")
        self._seg_start_frame = frame_idx
        self._frames_in_seg = 0

    def write(self, frame, frame_idx: int = 0) -> Optional[Path]:
        if not self.enabled:
            return None
        if self.writer is None:
            self._open_new(frame_idx)
        assert self.writer is not None
        if self._frame_budget > 0 and self._frames_in_seg >= self._frame_budget:
            self._open_new(frame_idx)
        h, w = frame.shape[:2]
        fw, fh = self.frame_size
        if (w, h) != (fw, fh):
            frame = cv2.resize(frame, (fw, fh))
        self.writer.write(frame)
        self._frames_in_seg += 1
        return self.current_path

    def release_writer_only(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

    def release(self) -> Optional[Path]:
        path = self.current_path
        self.release_writer_only()
        return path
