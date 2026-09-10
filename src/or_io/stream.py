"""流媒体源打开、重连与实时读帧。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np


_STREAM_SCHEMES = ("rtsp", "rtsps", "rtmp", "http", "https", "udp", "tcp")
_STREAM_EXT = (".m3u8", ".mjpg", ".mjpeg")


def is_stream_source(source: str | int) -> bool:
    """判断是否为实时流（RTSP/HTTP/摄像头等），而非本地文件。"""
    if isinstance(source, int):
        return True
    s = str(source).strip()
    if not s:
        return False
    if s.isdigit():
        return True
    lower = s.lower()
    if "://" in lower:
        scheme = urlparse(lower).scheme
        return scheme in _STREAM_SCHEMES
    if lower.startswith("/dev/video"):
        return True
    return any(lower.endswith(ext) for ext in _STREAM_EXT)


def normalize_source(source: str | int) -> str | int:
    """摄像头索引转 int；其余保持字符串。"""
    if isinstance(source, int):
        return source
    s = str(source).strip()
    if s.isdigit():
        return int(s)
    return s


@dataclass
class StreamConfig:
    reconnect: bool = True
    reconnect_delay_sec: float = 2.0
    max_reconnect_attempts: int = 0  # 0 = 无限
    buffer_size: int = 1
    open_timeout_ms: int = 8000
    read_timeout_ms: int = 5000
    # 处理跟不上时：跳过堆积帧，保实时
    drop_pending: bool = True
    max_pending_grabs: int = 8
    target_fps: float = 0.0  # <=0 表示尽量每帧都处理
    ffmpeg_options: str = "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay"


def open_capture(source: str | int, cfg: Optional[StreamConfig] = None) -> cv2.VideoCapture:
    """打开文件或流；对流设置小缓冲与超时。"""
    cfg = cfg or StreamConfig()
    src = normalize_source(source)
    stream = is_stream_source(src)

    if stream and isinstance(src, str) and src.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        # OpenCV/FFmpeg 低延迟选项（不同构建支持程度不一，失败则忽略）
        try:
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:  # noqa: BLE001
            pass
        if cfg.ffmpeg_options:
            try:
                os_environ_set = __import__("os").environ
                os_environ_set["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = cfg.ffmpeg_options
            except Exception:  # noqa: BLE001
                pass

    cap = cv2.VideoCapture(src)
    if stream:
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(cfg.buffer_size)))
        except Exception:  # noqa: BLE001
            pass
        # 部分后端支持超时（毫秒）
        for prop, value in (
            (getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None), cfg.open_timeout_ms),
            (getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None), cfg.read_timeout_ms),
        ):
            if prop is None:
                continue
            try:
                cap.set(prop, float(value))
            except Exception:  # noqa: BLE001
                pass
    return cap


def grab_latest_frame(cap: cv2.VideoCapture, max_grabs: int = 8) -> Tuple[bool, Optional[np.ndarray]]:
    """丢掉缓冲中的旧帧，取最新一帧（降低 RTSP 延迟）。"""
    ok = False
    frame = None
    for _ in range(max(1, max_grabs)):
        grabbed = cap.grab()
        if not grabbed:
            break
        ok = True
    if ok:
        ok, frame = cap.retrieve()
    return ok, frame


def iter_frames(
    source: str | int,
    cfg: Optional[StreamConfig] = None,
    *,
    live: Optional[bool] = None,
) -> Iterator[Tuple[int, np.ndarray, float]]:
    """
    产出 (frame_idx, frame, fps_hint)。
    实时流断线时按配置自动重连；文件读完正常结束。
    """
    cfg = cfg or StreamConfig()
    src = normalize_source(source)
    live_mode = is_stream_source(src) if live is None else bool(live)

    attempts = 0
    frame_idx = 0
    last_process_t = 0.0
    min_interval = (1.0 / cfg.target_fps) if cfg.target_fps and cfg.target_fps > 0 else 0.0

    while True:
        cap = open_capture(src, cfg)
        if not cap.isOpened():
            attempts += 1
            if not live_mode or not cfg.reconnect:
                raise RuntimeError(f"无法打开源: {src}")
            if cfg.max_reconnect_attempts and attempts > cfg.max_reconnect_attempts:
                raise RuntimeError(f"重连次数耗尽: {src}")
            print(f"[stream] open failed, retry in {cfg.reconnect_delay_sec:.1f}s (attempt={attempts})")
            time.sleep(cfg.reconnect_delay_sec)
            continue

        fps_hint = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 25.0
        attempts = 0
        print(f"[stream] opened source={src!r} live={live_mode} fps_hint={fps_hint:.2f}")

        while True:
            if live_mode and cfg.drop_pending:
                ok, frame = grab_latest_frame(cap, cfg.max_pending_grabs)
            else:
                ok, frame = cap.read()

            if not ok or frame is None:
                cap.release()
                if not live_mode:
                    return
                if not cfg.reconnect:
                    return
                attempts += 1
                if cfg.max_reconnect_attempts and attempts > cfg.max_reconnect_attempts:
                    raise RuntimeError(f"读帧失败且重连耗尽: {src}")
                print(f"[stream] read failed, reconnect in {cfg.reconnect_delay_sec:.1f}s")
                time.sleep(cfg.reconnect_delay_sec)
                break

            now = time.time()
            if min_interval > 0 and (now - last_process_t) < min_interval:
                continue
            last_process_t = now

            yield frame_idx, frame, fps_hint
            frame_idx += 1


def source_label(source: str | int) -> str:
    """用于目录名/日志的短标签。"""
    src = normalize_source(source)
    if isinstance(src, int):
        return f"cam{src}"
    s = str(src)
    if "://" in s:
        p = urlparse(s)
        host = p.hostname or "stream"
        path = re.sub(r"[^a-zA-Z0-9._-]+", "_", (p.path or "").strip("/"))[:40]
        return f"{p.scheme}_{host}" + (f"_{path}" if path else "")
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)[:60]
