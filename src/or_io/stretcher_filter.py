"""病床 + 平躺患者 联合判定（免训练）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

TargetMode = Literal["bed_patient", "stretcher", "person"]
PersonSubMode = Literal["all_persons", "horizontal", "large"]


def _aspect_wh(xyxy: Sequence[float]) -> float:
    w = max(float(xyxy[2]) - float(xyxy[0]), 1.0)
    h = max(float(xyxy[3]) - float(xyxy[1]), 1.0)
    return w / h


def _area_ratio(xyxy: Sequence[float], frame_w: int, frame_h: int) -> float:
    w = max(float(xyxy[2]) - float(xyxy[0]), 1.0)
    h = max(float(xyxy[3]) - float(xyxy[1]), 1.0)
    return (w * h) / max(frame_w * frame_h, 1)


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


def _patient_center_in_bed(patient: Sequence[float], bed: Sequence[float], margin: float = 0.15) -> bool:
    px, py = _center(patient)
    x1, y1, x2, y2 = map(float, bed)
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    x1 -= bw * margin
    x2 += bw * margin
    y1 -= bh * margin
    y2 += bh * margin
    return x1 <= px <= x2 and y1 <= py <= y2


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

    @property
    def event_track_id(self) -> int:
        return self.bed_track_id

    @property
    def centroid(self) -> Tuple[float, float]:
        bx, by = _center(self.bed_xyxy)
        px, py = _center(self.patient_xyxy)
        return ((bx + px) * 0.5, (by + py) * 0.5)

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
    min_aspect_wh: float = 1.20
    min_area_ratio: float = 0.02
    min_hits: int = 2

    bed_class_ids: Tuple[int, ...] = (59,)
    extra_bed_like_ids: Tuple[int, ...] = (57,)
    accept_bed_class: bool = True
    person_class_id: int = 0

    min_pair_iou: float = 0.02
    max_center_dist_ratio: float = 0.18
    allow_center_in_bed: bool = True
    pair_grace_frames: int = 8
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
        return cls(
            mode=mode,  # type: ignore[arg-type]
            person_mode=str(legacy.get("mode") or t.get("person_mode") or "all_persons"),  # type: ignore[arg-type]
            min_aspect_wh=float(st.get("lying_aspect_wh") or st.get("min_aspect_wh") or 1.20),
            min_area_ratio=float(st.get("lying_area_ratio") or st.get("min_area_ratio") or 0.02),
            min_hits=int(st.get("min_hits") or legacy.get("min_hits") or 2),
            bed_class_ids=tuple(st.get("bed_class_ids") or [59]),
            extra_bed_like_ids=tuple(st.get("extra_bed_like_ids") or [57]),
            accept_bed_class=bool(st.get("accept_bed_class", True)),
            min_pair_iou=float(st.get("min_pair_iou", 0.02)),
            max_center_dist_ratio=float(st.get("max_center_dist_ratio", 0.18)),
            allow_center_in_bed=bool(st.get("allow_center_in_bed", True)),
            pair_grace_frames=int(st.get("pair_grace_frames", 8)),
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

    def is_bed(self, class_id: int) -> bool:
        if not self.accept_bed_class:
            return False
        cid = int(class_id)
        return cid in self.bed_class_ids or cid in self.extra_bed_like_ids

    def is_lying_patient(
        self, class_id: int, xyxy: Sequence[float], frame_w: int, frame_h: int
    ) -> bool:
        if int(class_id) != self.person_class_id:
            return False
        return (
            _aspect_wh(xyxy) >= self.min_aspect_wh
            and _area_ratio(xyxy, frame_w, frame_h) >= self.min_area_ratio
        )

    def classify_role(
        self,
        class_id: int,
        xyxy: Sequence[float],
        frame_w: int,
        frame_h: int,
    ) -> Literal["bed", "lying_patient", "person", "other"]:
        if self.is_bed(class_id):
            return "bed"
        if self.is_lying_patient(class_id, xyxy, frame_w, frame_h):
            return "lying_patient"
        if int(class_id) == self.person_class_id:
            return "person"
        return "other"

    def _pair_score(
        self,
        bed: Detection,
        patient: Detection,
        frame_diag: float,
    ) -> Optional[float]:
        iou = _iou(bed.xyxy, patient.xyxy)
        bx, by = _center(bed.xyxy)
        px, py = _center(patient.xyxy)
        dist = ((px - bx) ** 2 + (py - by) ** 2) ** 0.5
        dist_ok = dist <= self.max_center_dist_ratio * max(frame_diag, 1.0)
        in_bed = self.allow_center_in_bed and _patient_center_in_bed(patient.xyxy, bed.xyxy)
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
        frame_diag = float(np.hypot(frame_w, frame_h))
        beds: List[Detection] = []
        lying: List[Detection] = []
        for d in detections:
            role = self.classify_role(d.class_id, d.xyxy, frame_w, frame_h)
            if role == "bed":
                beds.append(d)
            elif role == "lying_patient":
                lying.append(d)

        pairs: List[BedPatientPair] = []
        used_patients: set[int] = set()
        used_beds: set[int] = set()

        candidates: List[Tuple[float, Detection, Detection]] = []
        for bed in beds:
            for pat in lying:
                score = self._pair_score(bed, pat, frame_diag)
                if score is not None:
                    candidates.append((score, bed, pat))
        candidates.sort(key=lambda x: -x[0])

        for score, bed, pat in candidates:
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
            )
            pairs.append(pair)
            used_beds.add(bed.track_id)
            used_patients.add(pat.track_id)
            self._pair_grace[bed.track_id] = (frame_idx, pair)

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
                    )
                    pairs.append(pair)
                    used_patients.add(pat.track_id)
                    self._pair_grace[pat.track_id] = (frame_idx, pair)

        active_beds = {p.bed_track_id for p in pairs}
        for bed_id, (last_f, old_pair) in list(self._pair_grace.items()):
            if bed_id in active_beds:
                continue
            if frame_idx - last_f <= self.pair_grace_frames:
                bed_now = next((b for b in beds if b.track_id == bed_id), None)
                pat_now = next((p for p in lying if p.track_id == old_pair.patient_track_id), None)
                if bed_now is None and old_pair.bed_track_id == old_pair.patient_track_id:
                    bed_now = next((p for p in lying if p.track_id == bed_id), None)
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
                        (( _center(p)[0] - bx) ** 2 + (_center(p)[1] - by) ** 2) ** 0.5 <= thr
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
