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


def test_upright_head_not_patient_head():
    """直立医护上半身的头/脸不能算作躺着的患者头。"""
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=False,
        allow_pseudo_bed=True,
        reject_upright_heads=True,
    )
    # staff body tall; head box sits in upper torso/head region
    staff_body = Detection(3, 0, 0.92, (300, 80, 380, 420))
    upright_head = Detection(4, 0, 0.80, (310, 90, 370, 170))
    lying_head = Detection(2, 0, 0.75, (500, 260, 560, 320))
    bed = Detection(1, 59, 0.85, (450, 220, 720, 380))

    standing = f.collect_standing_boxes([staff_body, upright_head, lying_head, bed], 960, 540)
    assert len(standing) == 1
    assert f.head_belongs_to_upright(upright_head.xyxy, standing)
    assert not f.head_belongs_to_upright(lying_head.xyxy, standing)
    assert f.classify_role(0, upright_head.xyxy, 960, 540, standing_boxes=standing) == "person"
    assert f.classify_role(0, lying_head.xyxy, 960, 540, standing_boxes=standing) == "patient_head"

    pairs = f.associate([staff_body, upright_head, lying_head, bed], 960, 540)
    assert len(pairs) == 1
    assert pairs[0].patient_track_id == 2
    assert pairs[0].bed_track_id == 1


def test_tall_face_box_rejected_as_lying_head():
    f = TargetFilter(mode="bed_patient", allow_pseudo_bed=False, lying_head_min_aspect_wh=0.65)
    tall_face = (200, 100, 240, 190)  # aspect ~0.44
    assert not f.is_patient_head_candidate(0, tall_face, 960, 540, standing_boxes=[])


def test_equipment_cart_smaller_than_bed():
    """器械推车整体更小，不能当病床推车配对。"""
    f = TargetFilter(
        mode="bed_patient",
        min_hits=1,
        allow_merged_detection=False,
        allow_pseudo_bed=False,
        reject_small_bed_as_equipment=True,
        demote_bed_without_patient=True,
    )
    f.bind_class_ids(bed_class_ids=(59,), person_class_ids=(0,), equipment_class_ids=(20,))
    big_bed = Detection(1, 59, 0.9, (80, 200, 620, 400))  # large stretcher
    small_cart = Detection(8, 59, 0.8, (700, 300, 820, 390))  # small cart, same bed class
    square_cart = Detection(7, 59, 0.82, (400, 240, 560, 390))  # mid square cart w/o patient
    equip_cls = Detection(9, 20, 0.85, (100, 300, 220, 400))  # explicit equipment class
    head = Detection(2, 0, 0.7, (140, 240, 210, 300))

    roles = f.classify_roles([big_bed, small_cart, square_cart, equip_cls, head], 960, 540)
    assert roles[1] == "bed"
    assert roles[8] == "equipment_cart"
    assert roles[7] == "equipment_cart"
    assert roles[9] == "equipment_cart"
    assert roles[2] == "patient_head"

    pairs = f.associate([big_bed, small_cart, square_cart, equip_cls, head], 960, 540)
    assert len(pairs) == 1
    assert pairs[0].bed_track_id == 1
    assert pairs[0].patient_track_id == 2
    assert not f.should_draw_detection(59, small_cart.xyxy, 960, 540, hide_equipment_carts=True)


def test_lonely_equipment_cart_not_bed():
    """画面里只有器械车时，也不能标成病床。"""
    f = TargetFilter(mode="bed_patient", allow_pseudo_bed=False, demote_bed_without_patient=True)
    f.bind_class_ids(bed_class_ids=(59,), person_class_ids=(0,), equipment_class_ids=())
    cart = Detection(3, 59, 0.88, (420, 250, 600, 400))  # ~aspect 1.2, mid size, no patient
    roles = f.classify_roles([cart], 960, 540)
    assert roles[3] == "equipment_cart"
    assert f.associate([cart], 960, 540) == []


def test_cart_pusher_not_lying_patient():
    """推着器械车的医生：框变宽也不能标成躺着的患者。"""
    f = TargetFilter(
        mode="bed_patient",
        allow_pseudo_bed=False,
        reject_lying_near_equipment=True,
        demote_bed_without_patient=True,
        min_aspect_wh=1.45,
        lying_max_height_ratio=0.28,
        staff_push_max_aspect_wh=1.35,
    )
    f.bind_class_ids(bed_class_ids=(59,), person_class_ids=(0,), equipment_class_ids=(20,))
    # 推车医护：够高但偏宽（旧逻辑易判成 lying）
    wide_pusher = Detection(4, 0, 0.91, (300, 120, 480, 400))  # aspect~1.05, h_ratio~0.52
    cart = Detection(5, 20, 0.86, (470, 280, 620, 420))
    roles = f.classify_roles([wide_pusher, cart], 960, 540)
    assert roles[5] == "equipment_cart"
    assert roles[4] == "person"
    assert f.is_standing_staff(wide_pusher.xyxy, 960, 540)
    assert not f.is_lying_full_body(0, wide_pusher.xyxy, 960, 540, equipment_boxes=[cart.xyxy])
    assert f.associate([wide_pusher, cart], 960, 540) == []


def test_short_wide_box_near_cart_not_lying():
    """靠近器械车的偏矮宽框（推车半身）也不算躺姿患者。"""
    f = TargetFilter(
        mode="bed_patient",
        allow_pseudo_bed=False,
        reject_lying_near_equipment=True,
        min_aspect_wh=1.45,
        lying_max_height_ratio=0.28,
    )
    f.bind_class_ids(bed_class_ids=(59,), person_class_ids=(0,), equipment_class_ids=(20,))
    # 不够高进 standing，但够宽；靠近车 → 仍是医护
    half_body = Detection(6, 0, 0.88, (350, 300, 560, 400))  # aspect~2.1, h_ratio~0.185
    cart = Detection(7, 20, 0.84, (540, 310, 700, 420))
    roles = f.classify_roles([half_body, cart], 960, 540)
    assert roles[6] == "person"
    assert roles[7] == "equipment_cart"


def test_hide_upright_person_from_draw():
    f = TargetFilter(mode="bed_patient", allow_pseudo_bed=False)
    staff = (40, 80, 110, 400)  # tall upright
    head = (140, 230, 200, 290)
    assert f.is_upright_person(0, staff, 960, 540)
    assert not f.is_upright_person(0, head, 960, 540)
    assert not f.should_draw_detection(0, staff, 960, 540, hide_standing_staff=True)
    assert f.should_draw_detection(0, head, 960, 540, hide_standing_staff=True)
    assert f.should_draw_detection(59, (100, 200, 600, 360), 960, 540, hide_standing_staff=True)


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

    bed_ids, person_ids, equip_ids, names = resolve_prompt_class_ids(
        ["person", "human head", "hospital bed", "stretcher", "instrument cart"],
        ["hospital bed", "stretcher"],
        ["person", "human head"],
        ["instrument cart"],
    )
    assert bed_ids == (2, 3)
    assert person_ids == (0, 1)
    assert equip_ids == (4,)
    assert names[2] == "hospital bed"


def test_default_prompt_sets():
    from or_io.model_loader import (
        DEFAULT_BED_PROMPTS,
        DEFAULT_EQUIPMENT_PROMPTS,
        DEFAULT_PERSON_PROMPTS,
        DEFAULT_WORLD_PROMPTS,
        resolve_prompt_class_ids,
    )

    bed_ids, person_ids, equip_ids, _ = resolve_prompt_class_ids(
        DEFAULT_WORLD_PROMPTS,
        DEFAULT_BED_PROMPTS,
        DEFAULT_PERSON_PROMPTS,
        DEFAULT_EQUIPMENT_PROMPTS,
    )
    assert person_ids == (0, 1, 2, 3)
    assert 4 in bed_ids  # hospital bed
    assert len(bed_ids) >= 4
    assert len(equip_ids) >= 1


if __name__ == "__main__":
    test_bed_with_head_only()
    test_empty_bed_not_paired()
    test_staff_beside_bed_not_patient()
    test_merged_box()
    test_roles()
    test_upright_head_not_patient_head()
    test_tall_face_box_rejected_as_lying_head()
    test_equipment_cart_smaller_than_bed()
    test_lonely_equipment_cart_not_bed()
    test_cart_pusher_not_lying_patient()
    test_short_wide_box_near_cart_not_lying()
    test_hide_upright_person_from_draw()
    test_pseudo_bed_from_head_when_stretcher_missing()
    test_open_vocab_class_binding()
    test_prompt_id_resolve()
    test_default_prompt_sets()
    print("stretcher tests passed")
