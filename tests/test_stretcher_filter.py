"""病床 + 盖被只露头 关联单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.stretcher_filter import Detection, TargetFilter


def test_bed_with_head_only():
    """带栏杆病床 + 床上小头框 → 配对成功。"""
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=False,
        patient_appearance="covered_head",
    )
    dets = [
        Detection(1, 59, 0.85, (100, 200, 620, 380)),  # bed with rails
        Detection(2, 0, 0.70, (140, 230, 210, 300)),  # head only on bed
        Detection(3, 0, 0.90, (40, 80, 110, 360)),  # standing staff
    ]
    pairs = f.associate(dets, 960, 540)
    assert len(pairs) == 1
    assert pairs[0].bed_track_id == 1
    assert pairs[0].patient_track_id == 2
    assert pairs[0].evidence == "head_on_bed"


def test_empty_bed_not_paired():
    f = TargetFilter(mode="bed_patient", min_hits=1, allow_merged_detection=False)
    assert f.associate([Detection(1, 59, 0.9, (100, 200, 500, 360))], 960, 540) == []


def test_staff_beside_bed_not_patient():
    """床旁高大医护不应被当成患者。"""
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=False,
        patient_appearance="covered_head",
    )
    dets = [
        Detection(1, 59, 0.85, (200, 220, 700, 400)),
        Detection(3, 0, 0.92, (720, 100, 820, 420)),  # tall staff outside bed
    ]
    assert f.associate(dets, 960, 540) == []


def test_merged_box():
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=True,
        merged_min_aspect_wh=1.5,
        merged_min_area_ratio=0.05,
    )
    pairs = f.associate([Detection(5, 0, 0.6, (80, 200, 600, 360))], 960, 540)
    assert len(pairs) == 1
    assert pairs[0].evidence == "merged"


def test_roles():
    f = TargetFilter(mode="bed_patient")
    assert f.classify_role(59, (100, 200, 600, 360), 960, 540) == "bed"
    assert f.classify_role(0, (140, 230, 200, 290), 960, 540) == "patient_head"
    assert f.classify_role(0, (40, 80, 110, 400), 960, 540) == "person"


if __name__ == "__main__":
    test_bed_with_head_only()
    test_empty_bed_not_paired()
    test_staff_beside_bed_not_patient()
    test_merged_box()
    test_roles()
    print("stretcher tests passed")
