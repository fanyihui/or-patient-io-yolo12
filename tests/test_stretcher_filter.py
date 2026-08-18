"""病床 + 平躺患者 关联单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.stretcher_filter import Detection, TargetFilter


def test_associate_bed_and_lying():
    f = TargetFilter(mode="bed_patient", min_hits=1, allow_merged_detection=False)
    dets = [
        Detection(1, 59, 0.8, (100, 200, 500, 360)),
        Detection(2, 0, 0.7, (150, 220, 450, 320)),
        Detection(3, 0, 0.9, (50, 80, 110, 300)),
    ]
    pairs = f.associate(dets, 960, 540)
    assert len(pairs) == 1
    assert pairs[0].bed_track_id == 1
    assert pairs[0].patient_track_id == 2


def test_empty_bed_not_paired():
    f = TargetFilter(mode="bed_patient", min_hits=1, allow_merged_detection=False)
    assert f.associate([Detection(1, 59, 0.9, (100, 200, 500, 360))], 960, 540) == []


def test_merged_box():
    f = TargetFilter(
        mode="bed_patient", min_hits=1, allow_merged_detection=True,
        merged_min_aspect_wh=1.5, merged_min_area_ratio=0.05,
    )
    pairs = f.associate([Detection(5, 0, 0.6, (80, 200, 600, 360))], 960, 540)
    assert len(pairs) == 1
    assert pairs[0].bed_track_id == pairs[0].patient_track_id == 5


if __name__ == "__main__":
    test_associate_bed_and_lying()
    test_empty_bed_not_paired()
    test_merged_box()
    print("stretcher tests passed")
