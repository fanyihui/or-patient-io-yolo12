"""手术室入室监测 Web 应用（FastAPI）。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from or_io.app_session import AppSession  # noqa: E402


class ConnectBody(BaseModel):
    source: str = Field(..., description="rtsp://... / http://... / 0 / 视频文件路径")


class RoiBody(BaseModel):
    outside: List[List[float]]
    inside: List[List[float]]
    site_name: str = "or_door"
    already_normalized: bool = True
    save: bool = True


class MonitorBody(BaseModel):
    device: Optional[str] = None
    output_name: Optional[str] = None


def create_app(project_root: Path | None = None, config_path: Path | None = None) -> FastAPI:
    root = Path(project_root or ROOT)
    cfg_path = Path(config_path or (root / "configs" / "default.yaml"))
    session = AppSession(root, cfg_path)
    static_dir = Path(__file__).resolve().parent / "static"

    app = FastAPI(title="OR Patient Enter Monitor", version="1.0.0")
    app.state.session = session
    app.state.root = root

    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        index_path = static_dir / "index.html"
        if not index_path.exists():
            raise HTTPException(500, "缺少 static/index.html")
        return HTMLResponse(index_path.read_text(encoding="utf-8"))

    @app.get("/api/health")
    def health() -> dict:
        return {"ok": True, "service": "or-patient-io-app"}

    @app.get("/api/state")
    def state() -> dict:
        return session.state()

    @app.post("/api/connect")
    def connect(body: ConnectBody) -> dict:
        try:
            info = session.connect(body.source.strip())
            return {"ok": True, **info, **session.state()}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e

    @app.post("/api/disconnect")
    def disconnect() -> dict:
        session.disconnect()
        return {"ok": True, **session.state()}

    @app.post("/api/snapshot")
    def snapshot() -> dict:
        try:
            snap_dir = root / "outputs" / "snapshots"
            info = session.take_snapshot(snap_dir)
            return {"ok": True, **info, **session.state()}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e

    @app.get("/api/frame.jpg")
    def frame_jpg(which: str = "live", q: int = 80) -> Response:
        try:
            data = session.encode_jpeg(which=which, quality=q)
            return Response(content=data, media_type="image/jpeg")
        except Exception as e:  # noqa: BLE001
            raise HTTPException(404, str(e)) from e

    @app.get("/api/preview.mjpg")
    def preview_mjpg(which: str = "live", fps: float = 8.0) -> StreamingResponse:
        interval = 1.0 / max(float(fps), 1.0)

        def gen():
            while True:
                try:
                    # 监测中优先看标注画面
                    w = which
                    if which == "auto":
                        w = "monitor" if session.monitoring else "live"
                    jpg = session.encode_jpeg(which=w, quality=70)
                except Exception:  # noqa: BLE001
                    time.sleep(interval)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                time.sleep(interval)

        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")

    @app.post("/api/roi")
    def save_roi(body: RoiBody) -> dict:
        try:
            payload = session.set_roi(
                body.outside,
                body.inside,
                already_normalized=body.already_normalized,
                site_name=body.site_name.strip() or "or_door",
                save=body.save,
            )
            return {"ok": True, "roi": payload, **session.state()}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e

    @app.get("/api/roi")
    def get_roi() -> dict:
        roi = session.get_roi()
        if not roi:
            raise HTTPException(404, "尚未标注 ROI")
        return {"ok": True, "roi": roi}

    @app.post("/api/roi/load")
    def load_roi(path: str) -> dict:
        try:
            payload = session.load_roi_file(path)
            return {"ok": True, "roi": payload, **session.state()}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e

    @app.post("/api/monitor/start")
    def monitor_start(body: MonitorBody | None = None) -> dict:
        body = body or MonitorBody()
        try:
            info = session.start_monitor(device=body.device, output_name=body.output_name)
            return {**info, **session.state()}
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, str(e)) from e

    @app.post("/api/monitor/stop")
    def monitor_stop() -> dict:
        session.stop_monitor()
        return {"ok": True, **session.state()}

    @app.get("/api/events")
    def events(limit: int = 200) -> dict:
        items = session.events[-max(1, limit) :]
        return {
            "ok": True,
            "count": len(session.events),
            "enters": sum(1 for e in session.events if e.get("event") == "enter"),
            "exits": sum(1 for e in session.events if e.get("event") == "exit"),
            "events": list(reversed(items)),
        }

    return app


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="OR patient enter monitor web app")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(ROOT, Path(args.config))
    print(f"[app] open http://127.0.0.1:{args.port}/")
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
