"""病床 + 盖被患者（常只露头）联合判定（免训练）。

真实入室场景：
- 带栏杆的病床 / 推床推进门口
- 患者平躺、身体被覆盖，通常只露出头部
- 直立医护站在床旁推床，不应单独触发入出室

v1 策略：
1. 病床主体优先来自开放词汇 / COCO 家具弱类
2. 患者证据优先 = 「中心落在床内的躺着的头」（排除直立人的头/脸）
3. 用 患者框面积/床面积 上限排除站在床边的高大医护
4. 推床类仍漏检时：用盖被头（非直立人头）扩成伪床框
5. 仍保留横向全身 / 合并框回退，兼容旧合成验证视频
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

TargetMode = Literal["bed_patient", "stretcher", "person"]
PersonSubMode = Literal["all_persons", "horizontal", "large"]
PatientAppearance = Literal["covered_head", "lying_full", "any"]

# 当检测器检不出推床时，用头部/医护几何关系合成伪床框
PSEUDO_BED_CLASS_ID = 9900
PSEUDO_BED_TRACK_BASE = 900_000


def _aspect_wh(xyxy: Sequence[float]) -> float:
    w = max(float(xyxy[2]) - float(xyxy[0]), 1.0)
    h = max(float(xyxy[3]) - float(xyxy[1]), 1.0)
    return w / h


def _wh(xyxy: Sequence[float]) -> Tuple[float, float]:
    return (
        max(float(xyxy[2]) - float(xyxy[0]), 1.0),
        max(float(xyxy[3]) - float(xyxy[1]), 1.0),
    )


def _area(xyxy: Sequence[float]) -> float:
    w, h = _wh(xyxy)
    return w * h


def _area_ratio(xyxy: Sequence[float], frame_w: int, frame_h: int) -> float:
    return _area(xyxy) / max(frame_w * frame_h, 1)


def _center(xyxy: Sequence[float]) -> Tuple[float, float]:
    return (float(xyxy[0] + xyxy[2]) * 0.5, float(xyxy[1] + xyxy[3]) * 0.5)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _point_in_box(x: float, y: float, box: Sequence[float], margin: float = 0.12) -> bool:
    x1, y1, x2, y2 = map(float, box)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    return x1 <= x <= x2 and y1 <= y <= y2


def _patient_center_in_bed(patient: Sequence[float], bed: Sequence[float], margin: float = 0.12) -> bool:
    px, py = _center(patient)
    return _point_in_box(px, py, bed, margin=margin)


@dataclass
class Detection:
    track_id: int
    class_id: int
    conf: float
    xyxy: Tuple[float, float, float, float]


@dataclass
class BedPatientPair:
    bed_track_id: int
    patient_track_id: int
    bed_xyxy: Tuple[float, float, float, float]
    patient_xyxy: Tuple[float, float, float, float]
    bed_conf: float
    patient_conf: float
    score: float
    evidence: str = "head_on_bed"  # head_on_bed | lying_full | merged

    @property
    def event_track_id(self) -> int:
        return self.bed_track_id

    @property
    def centroid(self) -> Tuple[float, float]:
        # 入出室以病床中心为准更稳（盖被患者头位置会抖）
        return _center(self.bed_xyxy)

    @property
    def conf(self) -> float:
        return float(min(self.bed_conf, self.patient_conf))

    @property
    def union_xyxy(self) -> Tuple[float, float, float, float]:
        return (
            min(self.bed_xyxy[0], self.patient_xyxy[0]),
            min(self.bed_xyxy[1], self.patient_xyxy[1]),
            max(self.bed_xyxy[2], self.patient_xyxy[2]),
            max(self.bed_xyxy[3], self.patient_xyxy[3]),
        )


@dataclass
class TargetFilter:
    mode: TargetMode = "bed_patient"
    person_mode: PersonSubMode = "all_persons"
    patient_appearance: PatientAppearance = "covered_head"
    min_aspect_wh: float = 1.20
    min_area_ratio: float = 0.02
    min_hits: int = 2

    bed_class_ids: Tuple[int, ...] = (59,)
    extra_bed_like_ids: Tuple[int, ...] = (56, 57, 60)  # chair/couch/dining table 弱兜底
    accept_bed_class: bool = True
    person_class_id: int = 0  # 兼容旧单 id
    person_class_ids: Tuple[int, ...] = (0,)
    # 透视下推床可能接近正方形，不宜过严；但须大于器械推车
    bed_min_aspect_wh: float = 0.85
    bed_min_area_ratio: float = 0.028
    bed_min_width_ratio: float = 0.14
    bed_min_height_ratio: float = 0.055
    # 器械推车：整体更小；床类命中但尺寸不够 → equipment_cart
    equipment_class_ids: Tuple[int, ...] = ()
    equipment_max_area_ratio: float = 0.024
    equipment_relative_area_max: float = 0.60  # 相对同框最大床状目标
    reject_small_bed_as_equipment: bool = True

    min_pair_iou: float = 0.01
    max_center_dist_ratio: float = 0.28
    allow_center_in_bed: bool = True
    pair_grace_frames: int = 18
    # 盖被只露头：患者框应明显小于病床
    max_patient_to_bed_area: float = 0.70
    # 过高过大的竖直 person 视为床旁医护
    staff_max_aspect_wh: float = 0.90
    staff_min_height_ratio: float = 0.22
    # 头部候选：相对画面面积上下限
    head_max_area_ratio: float = 0.22
    head_min_area_ratio: float = 0.0006
    head_in_bed_margin: float = 0.22
    # 只认「躺着的头」：排除直立人体上半身的头/脸
    reject_upright_heads: bool = True
    standing_head_upper_ratio: float = 0.55
    lying_head_min_aspect_wh: float = 0.65

    allow_merged_detection: bool = True
    merged_min_aspect_wh: float = 1.25
    merged_min_area_ratio: float = 0.03

    # 推床类经常漏检：用盖被头 + 附近医护推断伪床框
    allow_pseudo_bed: bool = True
    pseudo_bed_require_staff: bool = False  # False=有头即可扩床；True=需附近站立医护
    pseudo_bed_staff_dist_ratio: float = 0.30
    pseudo_bed_width_scale: float = 4.8
    pseudo_bed_height_scale: float = 2.6
    pseudo_bed_min_width_ratio: float = 0.14
    pseudo_bed_max_width_ratio: float = 0.48
    pseudo_bed_min_height_ratio: float = 0.07
    pseudo_bed_max_height_ratio: float = 0.24

    fallback_horizontal_person: bool = True
    require_nearby_person: bool = False
    nearby_person_dist_ratio: float = 0.22

    _hits: Dict[int, int] = field(default_factory=dict, repr=False)
    _pair_grace: Dict[int, Tuple[int, BedPatientPair]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, cfg: dict) -> "TargetFilter":
        t = cfg.get("target") or {}
        legacy = cfg.get("patient_filter") or {}
        mode = str(t.get("mode") or "bed_patient")
        if mode == "stretcher" and t.get("require_bed_and_lying_patient", True):
            mode = "bed_patient"
        if mode not in ("bed_patient", "stretcher", "person"):
            mode = "bed_patient"
        st = t.get("stretcher") or t.get("bed_patient") or {}
        appearance = str(st.get("patient_appearance") or "covered_head")
        if appearance not in ("covered_head", "lying_full", "any"):
            appearance = "covered_head"
        person_ids = st.get("person_class_ids")
        if person_ids is None:
            person_ids = [int(st.get("person_class_id", 0))]
        bed_ids = tuple(st.get("bed_class_ids") or [59])
        extra_ids = tuple(st.get("extra_bed_like_ids") or [56, 57, 60])
        return cls(
            mode=mode,  # type: ignore[arg-type]
            person_mode=str(legacy.get("mode") or t.get("person_mode") or "all_persons"),  # type: ignore[arg-type]
            patient_appearance=appearance,  # type: ignore[arg-type]
            min_aspect_wh=float(st.get("lying_aspect_wh") or st.get("min_aspect_wh") or 1.10),
            min_area_ratio=float(st.get("lying_area_ratio") or st.get("min_area_ratio") or 0.015),
            min_hits=int(st.get("min_hits") or legacy.get("min_hits") or 2),
            bed_class_ids=bed_ids,
            extra_bed_like_ids=extra_ids,
            accept_bed_class=bool(st.get("accept_bed_class", True)),
            person_class_id=int(person_ids[0]) if person_ids else 0,
            person_class_ids=tuple(int(x) for x in person_ids),
            bed_min_aspect_wh=float(st.get("bed_min_aspect_wh", 0.85)),
            bed_min_area_ratio=float(st.get("bed_min_area_ratio", 0.028)),
            bed_min_width_ratio=float(st.get("bed_min_width_ratio", 0.14)),
            bed_min_height_ratio=float(st.get("bed_min_height_ratio", 0.055)),
            equipment_class_ids=tuple(int(x) for x in (st.get("equipment_class_ids") or [])),
            equipment_max_area_ratio=float(st.get("equipment_max_area_ratio", 0.024)),
            equipment_relative_area_max=float(st.get("equipment_relative_area_max", 0.60)),
            reject_small_bed_as_equipment=bool(st.get("reject_small_bed_as_equipment", True)),
            min_pair_iou=float(st.get("min_pair_iou", 0.01)),
            max_center_dist_ratio=float(st.get("max_center_dist_ratio", 0.28)),
            allow_center_in_bed=bool(st.get("allow_center_in_bed", True)),
            pair_grace_frames=int(st.get("pair_grace_frames", 18)),
            max_patient_to_bed_area=float(st.get("max_patient_to_bed_area", 0.70)),
            staff_max_aspect_wh=float(st.get("staff_max_aspect_wh", 0.90)),
            staff_min_height_ratio=float(st.get("staff_min_height_ratio", 0.22)),
            head_max_area_ratio=float(st.get("head_max_area_ratio", 0.22)),
            head_min_area_ratio=float(st.get("head_min_area_ratio", 0.0006)),
            head_in_bed_margin=float(st.get("head_in_bed_margin", 0.22)),
            reject_upright_heads=bool(st.get("reject_upright_heads", True)),
            standing_head_upper_ratio=float(st.get("standing_head_upper_ratio", 0.55)),
            lying_head_min_aspect_wh=float(st.get("lying_head_min_aspect_wh", 0.65)),
            allow_merged_detection=bool(st.get("allow_merged_detection", True)),
            merged_min_aspect_wh=float(st.get("merged_min_aspect_wh", 1.25)),
            merged_min_area_ratio=float(st.get("merged_min_area_ratio", 0.03)),
            allow_pseudo_bed=bool(st.get("allow_pseudo_bed", True)),
            pseudo_bed_require_staff=bool(st.get("pseudo_bed_require_staff", False)),
            pseudo_bed_staff_dist_ratio=float(st.get("pseudo_bed_staff_dist_ratio", 0.30)),
            pseudo_bed_width_scale=float(st.get("pseudo_bed_width_scale", 4.8)),
            pseudo_bed_height_scale=float(st.get("pseudo_bed_height_scale", 2.6)),
            pseudo_bed_min_width_ratio=float(st.get("pseudo_bed_min_width_ratio", 0.14)),
            pseudo_bed_max_width_ratio=float(st.get("pseudo_bed_max_width_ratio", 0.48)),
            pseudo_bed_min_height_ratio=float(st.get("pseudo_bed_min_height_ratio", 0.07)),
            pseudo_bed_max_height_ratio=float(st.get("pseudo_bed_max_height_ratio", 0.24)),
            fallback_horizontal_person=bool(st.get("fallback_horizontal_person", True)),
            require_nearby_person=bool(st.get("require_nearby_person", False)),
            nearby_person_dist_ratio=float(st.get("nearby_person_dist_ratio", 0.22)),
        )

    def bind_class_ids(
        self,
        bed_class_ids: Sequence[int],
        person_class_ids: Sequence[int],
        extra_bed_like_ids: Sequence[int] | None = None,
        equipment_class_ids: Sequence[int] | None = None,
    ) -> None:
        """运行时绑定开放词汇 / COCO 类别 id（pipeline 加载模型后调用）。"""
        beds = [int(x) for x in bed_class_ids]
        if self.allow_pseudo_bed and PSEUDO_BED_CLASS_ID not in beds:
            beds.append(PSEUDO_BED_CLASS_ID)
        self.bed_class_ids = tuple(beds)
        self.person_class_ids = tuple(int(x) for x in person_class_ids) or (0,)
        self.person_class_id = self.person_class_ids[0]
        if extra_bed_like_ids is not None:
            self.extra_bed_like_ids = tuple(int(x) for x in extra_bed_like_ids)
        if equipment_class_ids is not None:
            self.equipment_class_ids = tuple(int(x) for x in equipment_class_ids)

    def is_person_class(self, class_id: int) -> bool:
        return int(class_id) in self.person_class_ids

    def is_equipment_class(self, class_id: int) -> bool:
        return int(class_id) in self.equipment_class_ids

    def is_bed_like_class(self, class_id: int) -> bool:
        cid = int(class_id)
        if cid == PSEUDO_BED_CLASS_ID:
            return True
        return cid in self.bed_class_ids or cid in self.extra_bed_like_ids

    def reset(self) -> None:
        self._hits.clear()
        self._pair_grace.clear()

    def _bed_size_ok(self, xyxy: Sequence[float], frame_w: int, frame_h: int) -> bool:
        """病床推车尺寸门槛（器械推车通常达不到）。"""
        w, h = _wh(xyxy)
        if _area_ratio(xyxy, frame_w, frame_h) < self.bed_min_area_ratio:
            return False
        if w / max(frame_w, 1) < self.bed_min_width_ratio:
            return False
        if h / max(frame_h, 1) < self.bed_min_height_ratio:
            return False
        if _aspect_wh(xyxy) < self.bed_min_aspect_wh:
            return False
        return True

    def is_bed(self, class_id: int, xyxy: Sequence[float] | None = None, frame_w: int = 0, frame_h: int = 0) -> bool:
        if not self.accept_bed_class:
            return False
        cid = int(class_id)
        if cid == PSEUDO_BED_CLASS_ID:
            return True
        # 明确的器械推车类 → 不是病床
        if self.is_equipment_class(cid):
            return False
        if not self.is_bed_like_class(cid):
            return False
        if xyxy is None or frame_w <= 0 or frame_h <= 0:
            return True
        return self._bed_size_ok(xyxy, frame_w, frame_h)

    def is_equipment_cart(
        self,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
        *,
        largest_bed_area: float | None = None,
    ) -> bool:
        """器械推车：专用类，或床状目标但整体明显更小。"""
        cid = int(class_id)
        if cid == PSEUDO_BED_CLASS_ID:
            return False
        area = _area_ratio(xyxy, frame_w, frame_h)
        if self.is_equipment_class(cid):
            # 专用器械车提示：默认都算器械车；若异常巨大则仍可当床（少见）
            return area <= max(self.equipment_max_area_ratio * 2.5, self.bed_min_area_ratio)
        if not self.reject_small_bed_as_equipment:
            return False
        if not self.is_bed_like_class(cid):
            return False
        # 床类命中但尺寸不够 → 器械推车/小车
        if not self._bed_size_ok(xyxy, frame_w, frame_h):
            return True
        # 同框相对更小：明显小于最大病床状目标
        if largest_bed_area is not None and largest_bed_area > 0:
            if area < largest_bed_area * self.equipment_relative_area_max and area <= self.bed_min_area_ratio * 1.35:
                return True
        return False

    def is_standing_staff(self, xyxy: Sequence[float], frame_w: int, frame_h: int) -> bool:
        """高大竖直框 ≈ 床旁行走/推床医护。"""
        w, h = _wh(xyxy)
        aspect = w / h
        height_ratio = h / max(frame_h, 1)
        return aspect <= self.staff_max_aspect_wh and height_ratio >= self.staff_min_height_ratio

    def collect_standing_boxes(
        self,
        detections: Sequence[Detection],
        frame_w: int,
        frame_h: int,
    ) -> List[Tuple[float, float, float, float]]:
        """收集直立全身框，用于排除其头部。"""
        boxes: List[Tuple[float, float, float, float]] = []
        for d in detections:
            if not self.is_person_class(d.class_id):
                continue
            # 先按几何判断直立全身，避免依赖 role（role 可能把头误标成 patient_head）
            if self.is_standing_staff(d.xyxy, frame_w, frame_h):
                boxes.append(d.xyxy)
        return boxes

    def head_belongs_to_upright(
        self,
        head_xyxy: Sequence[float],
        standing_boxes: Sequence[Sequence[float]],
    ) -> bool:
        """头/脸中心落在直立人体上半身，或与上半身明显重叠 → 直立人的头。"""
        if not standing_boxes:
            return False
        hx, hy = _center(head_xyxy)
        for box in standing_boxes:
            x1, y1, x2, y2 = map(float, box)
            bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
            # 略放大直立框，避免头框刚好贴边漏判
            mx, my = bw * 0.08, bh * 0.05
            upper_y2 = y1 + bh * self.standing_head_upper_ratio
            in_upper = (x1 - mx) <= hx <= (x2 + mx) and (y1 - my) <= hy <= (upper_y2 + my)
            if in_upper:
                return True
            upper = (x1, y1, x2, upper_y2)
            if _iou(head_xyxy, upper) >= 0.12:
                return True
        return False

    def is_upright_person(
        self,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
        standing_boxes: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        """直立行人/医护（需屏蔽标注与事件主体）。"""
        if not self.is_person_class(class_id):
            return False
        if self.is_standing_staff(xyxy, frame_w, frame_h):
            return True
        if standing_boxes and self.head_belongs_to_upright(xyxy, standing_boxes):
            return True
        role = self.classify_role(class_id, xyxy, frame_w, frame_h, standing_boxes=standing_boxes)
        return role == "person"

    def should_draw_detection(
        self,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
        *,
        hide_standing_staff: bool = True,
        hide_equipment_carts: bool = True,
        draw_roles: Sequence[str] | None = None,
        standing_boxes: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        """是否在画面上绘制该检测框。默认屏蔽直立的人与器械推车。"""
        role = self.classify_role(class_id, xyxy, frame_w, frame_h, standing_boxes=standing_boxes)
        if hide_equipment_carts and role == "equipment_cart":
            return False
        if hide_standing_staff and (
            role == "person"
            or self.is_upright_person(class_id, xyxy, frame_w, frame_h, standing_boxes=standing_boxes)
        ):
            return False
        if draw_roles is not None:
            return role in set(draw_roles)
        return True

    def is_lying_full_body(self, class_id: int, xyxy: Sequence[float], frame_w: int, frame_h: int) -> bool:
        if not self.is_person_class(class_id):
            return False
        return (
            _aspect_wh(xyxy) >= self.min_aspect_wh
            and _area_ratio(xyxy, frame_w, frame_h) >= self.min_area_ratio
        )

    def is_patient_head_candidate(
        self,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
        standing_boxes: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        """盖被躺着只露头：小框 + 非直立人体的头。"""
        if not self.is_person_class(class_id):
            return False
        # 全身直立框本身绝不是躺着的头
        if self.is_standing_staff(xyxy, frame_w, frame_h):
            return False
        # 落在直立医护上半身的头/脸 → 不算患者头
        if self.reject_upright_heads and standing_boxes is not None:
            if self.head_belongs_to_upright(xyxy, standing_boxes):
                return False
        # 过瘦高的框更像直立人脸，不像枕上侧躺/仰躺露头
        if _aspect_wh(xyxy) < self.lying_head_min_aspect_wh:
            return False
        area = _area_ratio(xyxy, frame_w, frame_h)
        if area > self.head_max_area_ratio or area < self.head_min_area_ratio:
            return False
        return True

    def classify_role(
        self,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
        standing_boxes: Sequence[Sequence[float]] | None = None,
        largest_bed_area: float | None = None,
    ) -> Literal["bed", "equipment_cart", "lying_patient", "patient_head", "person", "other"]:
        if self.is_equipment_cart(
            class_id, xyxy, frame_w, frame_h, largest_bed_area=largest_bed_area
        ):
            return "equipment_cart"
        if self.is_bed(class_id, xyxy, frame_w, frame_h):
            return "bed"
        if not self.is_person_class(class_id):
            return "other"
        if self.is_lying_full_body(class_id, xyxy, frame_w, frame_h):
            return "lying_patient"
        if self.is_patient_head_candidate(class_id, xyxy, frame_w, frame_h, standing_boxes=standing_boxes):
            return "patient_head"
        return "person"

    def classify_roles(
        self,
        detections: Sequence[Detection],
        frame_w: int,
        frame_h: int,
    ) -> Dict[int, str]:
        """分类：直立全身 → 躺头；再按尺寸区分病床推车 vs 器械推车。"""
        standing = self.collect_standing_boxes(detections, frame_w, frame_h)
        # 先估同框最大「床状」面积，用于相对尺寸区分器械车
        bed_like_areas = [
            _area_ratio(d.xyxy, frame_w, frame_h)
            for d in detections
            if self.is_bed_like_class(d.class_id) or self.is_equipment_class(d.class_id)
        ]
        largest = max(bed_like_areas) if bed_like_areas else None
        roles = {
            d.track_id: self.classify_role(
                d.class_id,
                d.xyxy,
                frame_w,
                frame_h,
                standing_boxes=standing,
                largest_bed_area=largest,
            )
            for d in detections
        }
        # 第二遍：若已有明确大病床，把明显更小的床状目标降为器械车
        bed_areas = [
            _area_ratio(d.xyxy, frame_w, frame_h)
            for d in detections
            if roles.get(d.track_id) == "bed" and int(d.class_id) != PSEUDO_BED_CLASS_ID
        ]
        if bed_areas:
            max_bed = max(bed_areas)
            for d in detections:
                if roles.get(d.track_id) != "bed":
                    continue
                if int(d.class_id) == PSEUDO_BED_CLASS_ID:
                    continue
                ar = _area_ratio(d.xyxy, frame_w, frame_h)
                if ar < max_bed * self.equipment_relative_area_max and ar < self.bed_min_area_ratio * 1.5:
                    roles[d.track_id] = "equipment_cart"
        return roles

    def _nearby_staff(
        self,
        head: Detection,
        staff: Sequence[Detection],
        frame_w: int,
        frame_h: int,
    ) -> List[Detection]:
        hx, hy = _center(head.xyxy)
        thr = self.pseudo_bed_staff_dist_ratio * float(np.hypot(frame_w, frame_h))
        near = []
        for s in staff:
            sx, sy = _center(s.xyxy)
            if ((sx - hx) ** 2 + (sy - hy) ** 2) ** 0.5 <= thr:
                near.append(s)
        return near

    def _expand_head_to_bed(
        self,
        head: Detection,
        staff_near: Sequence[Detection],
        frame_w: int,
        frame_h: int,
    ) -> Tuple[float, float, float, float]:
        """把盖被头框扩成推床尺度伪框；若有医护则沿医护方向拉长。"""
        x1, y1, x2, y2 = map(float, head.xyxy)
        hw, hh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = _center(head.xyxy)

        bw = hw * self.pseudo_bed_width_scale
        bh = hh * self.pseudo_bed_height_scale
        bw = float(np.clip(bw, self.pseudo_bed_min_width_ratio * frame_w, self.pseudo_bed_max_width_ratio * frame_w))
        bh = float(np.clip(bh, self.pseudo_bed_min_height_ratio * frame_h, self.pseudo_bed_max_height_ratio * frame_h))

        extend_right = True
        if staff_near:
            sx = float(np.mean([_center(s.xyxy)[0] for s in staff_near]))
            extend_right = sx < cx

        if extend_right:
            bx1 = cx - bw * 0.28
            bx2 = cx + bw * 0.72
        else:
            bx1 = cx - bw * 0.72
            bx2 = cx + bw * 0.28
        by1 = cy - bh * 0.45
        by2 = cy + bh * 0.55

        bx1 = float(np.clip(bx1, 0, frame_w - 1))
        bx2 = float(np.clip(bx2, bx1 + 1, frame_w))
        by1 = float(np.clip(by1, 0, frame_h - 1))
        by2 = float(np.clip(by2, by1 + 1, frame_h))
        return (bx1, by1, bx2, by2)

    def synthesize_pseudo_beds(
        self,
        detections: Sequence[Detection],
        frame_w: int,
        frame_h: int,
        existing_beds: Sequence[Detection] | None = None,
    ) -> List[Detection]:
        """检测器漏检推床时：仅用「躺着的头」合成伪床框。"""
        if not self.allow_pseudo_bed:
            return []
        existing_beds = list(existing_beds or [])
        roles = self.classify_roles(detections, frame_w, frame_h)
        standing = self.collect_standing_boxes(detections, frame_w, frame_h)
        heads: List[Detection] = []
        staff: List[Detection] = []
        for d in detections:
            role = roles.get(d.track_id, "other")
            if role == "patient_head":
                heads.append(d)
            elif self.is_standing_staff(d.xyxy, frame_w, frame_h):
                staff.append(d)

        out: List[Detection] = []
        for head in heads:
            # 双重保险：直立人的头不扩床
            if self.head_belongs_to_upright(head.xyxy, standing):
                continue
            near = self._nearby_staff(head, staff, frame_w, frame_h)
            if self.pseudo_bed_require_staff and not near:
                continue
            bed_box = self._expand_head_to_bed(head, near, frame_w, frame_h)
            if any(_iou(bed_box, b.xyxy) >= 0.25 for b in existing_beds):
                continue
            if any(_iou(bed_box, b.xyxy) >= 0.35 for b in out):
                continue
            out.append(
                Detection(
                    track_id=PSEUDO_BED_TRACK_BASE + int(head.track_id),
                    class_id=PSEUDO_BED_CLASS_ID,
                    conf=float(min(0.55, max(0.2, head.conf * 0.85))),
                    xyxy=bed_box,
                )
            )
        return out

    def _pair_score(
        self,
        bed: Detection,
        patient: Detection,
        frame_w: int,
        frame_h: int,
        evidence: str,
    ) -> Optional[float]:
        frame_diag = float(np.hypot(frame_w, frame_h))
        iou = _iou(bed.xyxy, patient.xyxy)
        bx, by = _center(bed.xyxy)
        px, py = _center(patient.xyxy)
        dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        in_bed = self.allow_center_in_bed and _patient_center_in_bed(
            patient.xyxy, bed.xyxy, margin=self.head_in_bed_margin
        )
        area_ratio = _area(patient.xyxy) / max(_area(bed.xyxy), 1.0)

        if evidence == "head_on_bed":
            # 盖被只露头：必须中心在床内，且框明显小于床（排除床旁医护）
            if not in_bed:
                return None
            if area_ratio > self.max_patient_to_bed_area:
                return None
            if self.is_standing_staff(patient.xyxy, frame_w, frame_h):
                return None
            return float(0.7 + iou + max(0.0, 0.25 - dist / max(frame_diag, 1.0)))

        # lying_full：横向全身，可放宽
        dist_ok = dist <= self.max_center_dist_ratio * max(frame_diag, 1.0)
        if iou >= self.min_pair_iou or in_bed or dist_ok:
            return float(iou + (0.5 if in_bed else 0.0) + max(0.0, 0.3 - dist / max(frame_diag, 1.0)))
        return None

    def associate(
        self,
        detections: Sequence[Detection],
        frame_w: int,
        frame_h: int,
        frame_idx: int = 0,
    ) -> List[BedPatientPair]:
        roles = self.classify_roles(detections, frame_w, frame_h)
        standing = self.collect_standing_boxes(detections, frame_w, frame_h)
        beds: List[Detection] = []
        heads: List[Detection] = []
        lying: List[Detection] = []
        for d in detections:
            role = roles.get(d.track_id, "other")
            if role == "bed":
                beds.append(d)
            elif role == "patient_head":
                heads.append(d)
            elif role == "lying_patient":
                lying.append(d)

        # 推床类漏检时补伪床（仅躺着的头）
        if self.allow_pseudo_bed:
            pseudo = self.synthesize_pseudo_beds(detections, frame_w, frame_h, existing_beds=beds)
            beds.extend(pseudo)

        pairs: List[BedPatientPair] = []
        used_patients: set[int] = set()
        used_beds: set[int] = set()
        candidates: List[Tuple[float, Detection, Detection, str]] = []

        # 1) 优先：床 + 床上躺着的头
        if self.patient_appearance in ("covered_head", "any"):
            for bed in beds:
                for pat in heads:
                    if self.head_belongs_to_upright(pat.xyxy, standing):
                        continue
                    score = self._pair_score(bed, pat, frame_w, frame_h, "head_on_bed")
                    if score is not None:
                        candidates.append((score, bed, pat, "head_on_bed"))
                # 边缘：小框 person 且中心在床内，但仍排除直立人的头
                for d in detections:
                    if not self.is_person_class(d.class_id):
                        continue
                    if d.track_id in {h.track_id for h in heads} or d.track_id in {p.track_id for p in lying}:
                        continue
                    if self.is_standing_staff(d.xyxy, frame_w, frame_h):
                        continue
                    if self.head_belongs_to_upright(d.xyxy, standing):
                        continue
                    if _aspect_wh(d.xyxy) < self.lying_head_min_aspect_wh:
                        continue
                    score = self._pair_score(bed, d, frame_w, frame_h, "head_on_bed")
                    if score is not None:
                        candidates.append((score, bed, d, "head_on_bed"))

        # 2) 兼容：床 + 横向全身
        if self.patient_appearance in ("lying_full", "any", "covered_head"):
            for bed in beds:
                for pat in lying:
                    score = self._pair_score(bed, pat, frame_w, frame_h, "lying_full")
                    if score is not None:
                        candidates.append((score, bed, pat, "lying_full"))

        candidates.sort(key=lambda x: -x[0])
        for score, bed, pat, evidence in candidates:
            if bed.track_id in used_beds or pat.track_id in used_patients:
                continue
            pair = BedPatientPair(
                bed_track_id=bed.track_id,
                patient_track_id=pat.track_id,
                bed_xyxy=bed.xyxy,
                patient_xyxy=pat.xyxy,
                bed_conf=bed.conf,
                patient_conf=pat.conf,
                score=score,
                evidence=evidence,
            )
            pairs.append(pair)
            used_beds.add(bed.track_id)
            used_patients.add(pat.track_id)
            self._pair_grace[bed.track_id] = (frame_idx, pair)

        # 3) 合并框回退：YOLO 把床+患者合成一个横向大框
        if self.allow_merged_detection:
            for pat in lying:
                if pat.track_id in used_patients:
                    continue
                if (
                    _aspect_wh(pat.xyxy) >= self.merged_min_aspect_wh
                    and _area_ratio(pat.xyxy, frame_w, frame_h) >= self.merged_min_area_ratio
                ):
                    pair = BedPatientPair(
                        bed_track_id=pat.track_id,
                        patient_track_id=pat.track_id,
                        bed_xyxy=pat.xyxy,
                        patient_xyxy=pat.xyxy,
                        bed_conf=pat.conf,
                        patient_conf=pat.conf,
                        score=0.4,
                        evidence="merged",
                    )
                    pairs.append(pair)
                    used_patients.add(pat.track_id)
                    self._pair_grace[pat.track_id] = (frame_idx, pair)

        # 短暂丢失头部时，沿用最近成功配对（盖被/遮挡常见）
        active_beds = {p.bed_track_id for p in pairs}
        person_by_id = {d.track_id: d for d in detections if self.is_person_class(d.class_id)}
        for bed_id, (last_f, old_pair) in list(self._pair_grace.items()):
            if bed_id in active_beds:
                continue
            if frame_idx - last_f <= self.pair_grace_frames:
                bed_now = next((b for b in beds if b.track_id == bed_id), None)
                pat_now = person_by_id.get(old_pair.patient_track_id)
                if bed_now is None and old_pair.bed_track_id == old_pair.patient_track_id:
                    bed_now = person_by_id.get(bed_id)
                if bed_now is not None or pat_now is not None:
                    pairs.append(
                        BedPatientPair(
                            bed_track_id=old_pair.bed_track_id,
                            patient_track_id=old_pair.patient_track_id,
                            bed_xyxy=bed_now.xyxy if bed_now else old_pair.bed_xyxy,
                            patient_xyxy=pat_now.xyxy if pat_now else old_pair.patient_xyxy,
                            bed_conf=bed_now.conf if bed_now else old_pair.bed_conf,
                            patient_conf=pat_now.conf if pat_now else old_pair.patient_conf,
                            score=old_pair.score * 0.5,
                            evidence=old_pair.evidence,
                        )
                    )
            elif frame_idx - last_f > self.pair_grace_frames:
                self._pair_grace.pop(bed_id, None)

        return pairs

    def confirm_pair(self, pair: BedPatientPair) -> bool:
        tid = pair.event_track_id
        self._hits[tid] = self._hits.get(tid, 0) + 1
        return self._hits[tid] >= self.min_hits

    def mark_unseen_beds(self, seen_bed_ids: Sequence[int]) -> None:
        seen = set(seen_bed_ids)
        for tid in list(self._hits.keys()):
            if tid not in seen:
                self._hits[tid] = 0

    def accept_track(
        self,
        track_id: int,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
        person_boxes: Sequence[Sequence[float]] | None = None,
    ) -> bool:
        if self.mode == "bed_patient":
            return False
        person_boxes = person_boxes or []
        frame_diag = float(np.hypot(frame_w, frame_h))
        ok = False
        if self.mode == "stretcher":
            role = self.classify_role(class_id, xyxy, frame_w, frame_h)
            if role == "bed" or (self.fallback_horizontal_person and role == "lying_patient"):
                if self.require_nearby_person:
                    bx, by = _center(xyxy)
                    thr = self.nearby_person_dist_ratio * frame_diag
                    ok = any(
                        ((_center(p)[0] - bx) ** 2 + (_center(p)[1] - by) ** 2) ** 0.5 <= thr
                        for p in person_boxes
                    )
                else:
                    ok = True
        else:
            if not self.is_person_class(class_id):
                ok = False
            elif self.person_mode == "horizontal":
                ok = _aspect_wh(xyxy) >= self.min_aspect_wh
            elif self.person_mode == "large":
                ok = _area_ratio(xyxy, frame_w, frame_h) >= self.min_area_ratio
            else:
                ok = True
        if not ok:
            self._hits[track_id] = 0
            return False
        self._hits[track_id] = self._hits.get(track_id, 0) + 1
        return self._hits[track_id] >= self.min_hits


PatientFilter = TargetFilter
