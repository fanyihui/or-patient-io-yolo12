"""病床 + 盖被患者（常只露头）联合判定（免训练）。

真实入室场景：
- 带栏杆的病床 / 推床推进门口
- 患者平躺、身体被覆盖，通常只露出头部
- 直立医护站在床旁推床，不应单独触发入出室

v1 策略：
1. 以 COCO bed（及弱兜底 couch）作为病床主体
2. 患者证据优先 = 「中心落在床内的 person」—— 覆盖只露头的情况
   （不再强制全身横向大框）
3. 用 患者框面积/床面积 上限排除站在床边的高大医护
4. 仍保留横向全身 / 合并框回退，兼容旧合成验证视频
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

TargetMode = Literal["bed_patient", "stretcher", "person"]
PersonSubMode = Literal["all_persons", "horizontal", "large"]
PatientAppearance = Literal["covered_head", "lying_full", "any"]


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
    extra_bed_like_ids: Tuple[int, ...] = (57,)
    accept_bed_class: bool = True
    person_class_id: int = 0
    # 带栏杆推床通常偏横向
    bed_min_aspect_wh: float = 1.05
    bed_min_area_ratio: float = 0.03

    min_pair_iou: float = 0.01
    max_center_dist_ratio: float = 0.20
    allow_center_in_bed: bool = True
    pair_grace_frames: int = 12
    # 盖被只露头：患者框应明显小于病床
    max_patient_to_bed_area: float = 0.55
    # 过高过大的竖直 person 视为床旁医护
    staff_max_aspect_wh: float = 0.85
    staff_min_height_ratio: float = 0.28

    allow_merged_detection: bool = True
    merged_min_aspect_wh: float = 1.5
    merged_min_area_ratio: float = 0.05

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
        return cls(
            mode=mode,  # type: ignore[arg-type]
            person_mode=str(legacy.get("mode") or t.get("person_mode") or "all_persons"),  # type: ignore[arg-type]
            patient_appearance=appearance,  # type: ignore[arg-type]
            min_aspect_wh=float(st.get("lying_aspect_wh") or st.get("min_aspect_wh") or 1.20),
            min_area_ratio=float(st.get("lying_area_ratio") or st.get("min_area_ratio") or 0.02),
            min_hits=int(st.get("min_hits") or legacy.get("min_hits") or 2),
            bed_class_ids=tuple(st.get("bed_class_ids") or [59]),
            extra_bed_like_ids=tuple(st.get("extra_bed_like_ids") or [57]),
            accept_bed_class=bool(st.get("accept_bed_class", True)),
            bed_min_aspect_wh=float(st.get("bed_min_aspect_wh", 1.05)),
            bed_min_area_ratio=float(st.get("bed_min_area_ratio", 0.03)),
            min_pair_iou=float(st.get("min_pair_iou", 0.01)),
            max_center_dist_ratio=float(st.get("max_center_dist_ratio", 0.20)),
            allow_center_in_bed=bool(st.get("allow_center_in_bed", True)),
            pair_grace_frames=int(st.get("pair_grace_frames", 12)),
            max_patient_to_bed_area=float(st.get("max_patient_to_bed_area", 0.55)),
            staff_max_aspect_wh=float(st.get("staff_max_aspect_wh", 0.85)),
            staff_min_height_ratio=float(st.get("staff_min_height_ratio", 0.28)),
            allow_merged_detection=bool(st.get("allow_merged_detection", True)),
            merged_min_aspect_wh=float(st.get("merged_min_aspect_wh", 1.5)),
            merged_min_area_ratio=float(st.get("merged_min_area_ratio", 0.05)),
            fallback_horizontal_person=bool(st.get("fallback_horizontal_person", True)),
            require_nearby_person=bool(st.get("require_nearby_person", False)),
            nearby_person_dist_ratio=float(st.get("nearby_person_dist_ratio", 0.22)),
        )

    def reset(self) -> None:
        self._hits.clear()
        self._pair_grace.clear()

    def is_bed(self, class_id: int, xyxy: Sequence[float] | None = None, frame_w: int = 0, frame_h: int = 0) -> bool:
        if not self.accept_bed_class:
            return False
        cid = int(class_id)
        if cid not in self.bed_class_ids and cid not in self.extra_bed_like_ids:
            return False
        if xyxy is None or frame_w <= 0 or frame_h <= 0:
            return True
        # 过小的床框噪声、明显竖立物体弱过滤
        if _area_ratio(xyxy, frame_w, frame_h) < self.bed_min_area_ratio:
            return False
        if _aspect_wh(xyxy) < self.bed_min_aspect_wh:
            return False
        return True

    def is_standing_staff(self, xyxy: Sequence[float], frame_w: int, frame_h: int) -> bool:
        """高大竖直框 ≈ 床旁行走/推床医护。"""
        w, h = _wh(xyxy)
        aspect = w / h
        height_ratio = h / max(frame_h, 1)
        return aspect <= self.staff_max_aspect_wh and height_ratio >= self.staff_min_height_ratio

    def is_lying_full_body(self, class_id: int, xyxy: Sequence[float], frame_w: int, frame_h: int) -> bool:
        if int(class_id) != self.person_class_id:
            return False
        return (
            _aspect_wh(xyxy) >= self.min_aspect_wh
            and _area_ratio(xyxy, frame_w, frame_h) >= self.min_area_ratio
        )

    def is_patient_head_candidate(self, class_id: int, xyxy: Sequence[float], frame_w: int, frame_h: int) -> bool:
        """盖被只露头：小/中等 person 框，排除高大站立者。"""
        if int(class_id) != self.person_class_id:
            return False
        if self.is_standing_staff(xyxy, frame_w, frame_h):
            return False
        # 头/肩区域通常不会占满半个画面
        return _area_ratio(xyxy, frame_w, frame_h) <= 0.18

    def classify_role(
        self,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
    ) -> Literal["bed", "lying_patient", "patient_head", "person", "other"]:
        if self.is_bed(class_id, xyxy, frame_w, frame_h):
            return "bed"
        if int(class_id) != self.person_class_id:
            return "other"
        if self.is_lying_full_body(class_id, xyxy, frame_w, frame_h):
            return "lying_patient"
        if self.is_patient_head_candidate(class_id, xyxy, frame_w, frame_h):
            return "patient_head"
        return "person"

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
        in_bed = self.allow_center_in_bed and _patient_center_in_bed(patient.xyxy, bed.xyxy)
        area_ratio = _area(patient.xyxy) / max(_area(bed.xyxy), 1.0)

        if evidence == "head_on_bed":
            # 盖被只露头：必须中心在床内，且框明显小于床（排除床旁医护）
            if not in_bed:
                return None
            if area_ratio > self.max_patient_to_bed_area:
                return None
            if self.is_standing_staff(patient.xyxy, frame_w, frame_h) and area_ratio > 0.25:
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
        beds: List[Detection] = []
        heads: List[Detection] = []
        lying: List[Detection] = []
        for d in detections:
            role = self.classify_role(d.class_id, d.xyxy, frame_w, frame_h)
            if role == "bed":
                beds.append(d)
            elif role == "patient_head":
                heads.append(d)
            elif role == "lying_patient":
                lying.append(d)

        pairs: List[BedPatientPair] = []
        used_patients: set[int] = set()
        used_beds: set[int] = set()
        candidates: List[Tuple[float, Detection, Detection, str]] = []

        # 1) 优先：床 + 床上头部/小框（盖被场景）
        if self.patient_appearance in ("covered_head", "any"):
            for bed in beds:
                for pat in heads:
                    score = self._pair_score(bed, pat, frame_w, frame_h, "head_on_bed")
                    if score is not None:
                        candidates.append((score, bed, pat, "head_on_bed"))
                # 也允许普通 person 若中心在床内且足够小（分类成 person 的边缘情况）
                for d in detections:
                    if int(d.class_id) != self.person_class_id:
                        continue
                    if d.track_id in {h.track_id for h in heads} or d.track_id in {p.track_id for p in lying}:
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
        person_by_id = {d.track_id: d for d in detections if int(d.class_id) == self.person_class_id}
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
            if int(class_id) != self.person_class_id:
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
