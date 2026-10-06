"""固定手术床 ROI 与推床→手术床转移单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.stretcher_filter import BedPatientPair, Detection
from or_io.transfer import TransferConfig, TransferMonitor, poly_from_normalized


def test_poly_and_fixed_bed():
    poly = poly_from_normalized([[0.4, 0.4], [0.7, 0.4], [0.7, 0.7], [0.4, 0.7]], 1000, 1000)
    mon = TransferMonitor(cfg=TransferConfig(confirm_frames=3, require_stretcher_nearby=True))
    mon.set_or_bed_poly(poly)
    assert mon.is_fixed_or_bed_box((450, 450, 650, 650))
    assert not mon.is_fixed_or_bed_box((50, 50, 150, 120))


def test_transfer_event_after_stretcher_nearby():
    poly = poly_from_normalized([[0.35, 0.35], [0.75, 0.35], [0.75, 0.75], [0.35, 0.75]], 1000, 1000)
    mon = TransferMonitor(
        cfg=TransferConfig(
            confirm_frames=3,
            require_stretcher_nearby=True,
            stretcher_memory_frames=50,
            debounce_frames=5,
            patient_overlap_min=0.3,
        )
    )
    mon.set_or_bed_poly(poly)

    # 推床靠近手术床
    stretcher = Detection(1, 59, 0.9, (200, 400, 380, 560))
    head_on_stretcher = Detection(2, 0, 0.8, (220, 420, 280, 480))
    pair = BedPatientPair(
        bed_track_id=1,
        patient_track_id=2,
        bed_xyxy=stretcher.xyxy,
        patient_xyxy=head_on_stretcher.xyxy,
        bed_conf=0.9,
        patient_conf=0.8,
        score=1.0,
        evidence="head_on_bed",
    )
    roles = {1: "bed", 2: "patient_head"}
    # 先记推床靠近
    for f in range(5):
        mon.update(
            frame_idx=f,
            fps=10,
            frame_w=1000,
            frame_h=1000,
            detections=[stretcher, head_on_stretcher],
            roles=roles,
            pairs=[pair],
        )

    # 患者转移到手术床 ROI 内
    head_on_or = Detection(3, 0, 0.85, (480, 480, 560, 560))
    roles2 = {3: "patient_head", 1: "bed"}
    evt = None
    for f in range(10, 20):
        evt = mon.update(
            frame_idx=f,
            fps=10,
            frame_w=1000,
            frame_h=1000,
            detections=[stretcher, head_on_or],
            roles=roles2,
            pairs=[pair],
        )
        if evt is not None:
            break
    assert evt is not None
    assert evt.event == "transfer_to_or_bed"
    assert evt.track_id == 3


def test_no_transfer_without_stretcher_when_required():
    poly = poly_from_normalized([[0.35, 0.35], [0.75, 0.35], [0.75, 0.75], [0.35, 0.75]], 1000, 1000)
    mon = TransferMonitor(
        cfg=TransferConfig(confirm_frames=3, require_stretcher_nearby=True, stretcher_memory_frames=5)
    )
    mon.set_or_bed_poly(poly)
    head = Detection(9, 0, 0.9, (480, 480, 560, 560))
    evt = None
    for f in range(0, 12):
        evt = mon.update(
            frame_idx=f,
            fps=10,
            frame_w=1000,
            frame_h=1000,
            detections=[head],
            roles={9: "patient_head"},
            pairs=[],
        )
    assert evt is None


def test_or_bed_payload_roundtrip(tmp_path=None):
    from or_io.roi_store import build_or_bed_payload, build_zone_payload, load_roi, save_roi_yaml

    out = Path(tmp_path) if tmp_path else (ROOT / "outputs" / "_test_or_bed")
    out.mkdir(parents=True, exist_ok=True)
    payload = build_zone_payload(
        [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
        [[0.1, 0.35], [0.3, 0.35], [0.3, 0.55], [0.1, 0.55]],
        960,
        540,
        or_bed=[[0.4, 0.4], [0.7, 0.4], [0.7, 0.7], [0.4, 0.7]],
    )
    assert "or_bed" in payload
    path = save_roi_yaml(out / "site.roi.yaml", payload, base_config=None)
    loaded = load_roi(path)
    assert loaded["or_bed"]["fixed"] is True
    assert len(loaded["or_bed"]["polygon"]) == 4


if __name__ == "__main__":
    test_poly_and_fixed_bed()
    test_transfer_event_after_stretcher_nearby()
    test_no_transfer_without_stretcher_when_required()
    test_or_bed_payload_roundtrip()
    print("transfer tests passed")
