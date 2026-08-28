"""病床 + 盖被只露头 关联单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.stretcher_filter import PSEUDO_BED_CLASS_ID, Detection, TargetFilter


def test_bed_with_head_only():
    """带栏杆病床 + 床上小头框 → 配对成功。"""
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=False,
        patient_appearance="covered_head",
        allow_pseudo_bed=False,
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
    f = TargetFilter(mode="bed_patient", min_hits=1, allow_merged_detection=False, allow_pseudo_bed=False)
    assert f.associate([Detection(1, 59, 0.9, (100, 200, 500, 360))], 960, 540) == []


def test_staff_beside_bed_not_patient():
    """床旁高大医护不应被当成患者。"""
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=False,
        patient_appearance="covered_head",
        allow_pseudo_bed=False,
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
        allow_pseudo_bed=False,
        merged_min_aspect_wh=1.5,
        merged_min_area_ratio=0.05,
    )
    pairs = f.associate([Detection(5, 0, 0.6, (80, 200, 600, 360))], 960, 540)
    assert len(pairs) == 1
    assert pairs[0].evidence == "merged"


def test_roles():
    f = TargetFilter(mode="bed_patient", allow_pseudo_bed=False)
    assert f.classify_role(59, (100, 200, 600, 360), 960, 540) == "bed"
    assert f.classify_role(0, (140, 230, 200, 290), 960, 540) == "patient_head"
    assert f.classify_role(0, (40, 80, 110, 400), 960, 540) == "person"


def test_pseudo_bed_from_head_when_stretcher_missing():
    """真实推床类漏检时：盖被头 → 伪床 → 配对。"""
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=False,
        allow_pseudo_bed=True,
        patient_appearance="covered_head",
    )
    f.bind_class_ids(bed_class_ids=(59,), person_class_ids=(0,), extra_bed_like_ids=())
    dets = [
        Detection(2, 0, 0.72, (400, 250, 460, 310)),  # covered head only
        Detection(3, 0, 0.90, (300, 120, 360, 380)),  # standing staff nearby
    ]
    pairs = f.associate(dets, 960, 540)
    assert len(pairs) == 1
    assert pairs[0].patient_track_id == 2
    assert pairs[0].bed_track_id >= 900_000
    assert f.classify_role(PSEUDO_BED_CLASS_ID, pairs[0].bed_xyxy, 960, 540) == "bed"


def test_open_vocab_class_binding():
    """开放词汇：hospital bed / human head 映射到自定义 class id。"""
    f = TargetFilter(mode="bed_patient", min_hits=1, allow_merged_detection=False, allow_pseudo_bed=False)
    f.bind_class_ids(bed_class_ids=(2,), person_class_ids=(0, 1), extra_bed_like_ids=())
    dets = [
        Detection(10, 2, 0.8, (100, 200, 620, 380)),
        Detection(11, 1, 0.7, (140, 230, 210, 300)),
        Detection(12, 0, 0.9, (40, 80, 110, 360)),
    ]
    assert f.classify_role(2, dets[0].xyxy, 960, 540) == "bed"
    assert f.classify_role(1, dets[1].xyxy, 960, 540) == "patient_head"
    pairs = f.associate(dets, 960, 540)
    assert len(pairs) == 1
    assert pairs[0].bed_track_id == 10
    assert pairs[0].patient_track_id == 11


def test_prompt_id_resolve():
    from or_io.model_loader import resolve_prompt_class_ids

    bed_ids, person_ids, names = resolve_prompt_class_ids(
        ["person", "human head", "hospital bed", "stretcher"],
        ["hospital bed", "stretcher"],
        ["person", "human head"],
    )
    assert bed_ids == (2, 3)
    assert person_ids == (0, 1)
    assert names[2] == "hospital bed"


def test_default_prompt_sets():
    from or_io.model_loader import (
        DEFAULT_BED_PROMPTS,
        DEFAULT_PERSON_PROMPTS,
        DEFAULT_WORLD_PROMPTS,
        resolve_prompt_class_ids,
    )

    bed_ids, person_ids, _ = resolve_prompt_class_ids(
        DEFAULT_WORLD_PROMPTS, DEFAULT_BED_PROMPTS, DEFAULT_PERSON_PROMPTS
    )
    assert person_ids == (0, 1, 2, 3)
    assert 4 in bed_ids  # hospital bed
    assert len(bed_ids) >= 4


if __name__ == "__main__":
    test_bed_with_head_only()
    test_empty_bed_not_paired()
    test_staff_beside_bed_not_patient()
    test_merged_box()
    test_roles()
    test_pseudo_bed_from_head_when_stretcher_missing()
    test_open_vocab_class_binding()
    test_prompt_id_resolve()
    test_default_prompt_sets()
    print("stretcher tests passed")
