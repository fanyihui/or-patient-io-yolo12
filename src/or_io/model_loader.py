"""检测模型加载：COCO / YOLO-World / YOLOE / 双模型融合（免训练）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ultralytics import YOLO

# 开放词汇默认提示：病床推车 + 器械推车 + 盖被只露头
DEFAULT_WORLD_PROMPTS: List[str] = [
    "person",
    "human head",
    "face",
    "patient",
    "hospital bed",
    "stretcher",
    "gurney",
    "medical bed",
    "hospital stretcher",
    "wheeled bed",
    "bed with rails",
    "bed",
    "mattress",
    "instrument cart",
    "equipment cart",
    "medical cart",
    "utility cart",
    "mayo stand",
]

# 病床 / 推床（整体更大）
DEFAULT_BED_PROMPTS = (
    "hospital bed",
    "stretcher",
    "gurney",
    "medical bed",
    "hospital stretcher",
    "wheeled bed",
    "bed with rails",
    "bed",
    "mattress",
)
# 器械推车（整体更小，不当作入出室病床）
DEFAULT_EQUIPMENT_PROMPTS = (
    "instrument cart",
    "equipment cart",
    "medical cart",
    "utility cart",
    "mayo stand",
)
DEFAULT_PERSON_PROMPTS = ("person", "human head", "face", "patient")

# COCO 家具弱类别：推床偶尔落在这些类上
COCO_FURNITURE_IDS = (56, 57, 59, 60)  # chair, couch, bed, dining table


@dataclass
class DetectorSpec:
    backend: str  # coco | world | yoloe | fusion
    model: YOLO
    names: Dict[int, str]
    bed_class_ids: Tuple[int, ...]
    person_class_ids: Tuple[int, ...]
    detect_classes: Optional[List[int]]
    equipment_class_ids: Tuple[int, ...] = field(default_factory=tuple)
    secondary_model: Optional[YOLO] = None
    secondary_detect_classes: Optional[List[int]] = None
    secondary_bed_class_ids: Tuple[int, ...] = field(default_factory=tuple)
    secondary_person_class_ids: Tuple[int, ...] = field(default_factory=tuple)
    secondary_names: Dict[int, str] = field(default_factory=dict)
    # fusion 时 secondary class_id 需加偏移再并入主 names
    secondary_id_offset: int = 1000


def _norm_name(s: str) -> str:
    return str(s).strip().lower()


def resolve_prompt_class_ids(
    prompts: Sequence[str],
    bed_prompts: Sequence[str],
    person_prompts: Sequence[str],
    equipment_prompts: Sequence[str] | None = None,
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], Dict[int, str]]:
    names = {i: p for i, p in enumerate(prompts)}
    bed_set = {_norm_name(x) for x in bed_prompts}
    person_set = {_norm_name(x) for x in person_prompts}
    equip_set = {_norm_name(x) for x in (equipment_prompts or ())}
    bed_ids = tuple(i for i, p in enumerate(prompts) if _norm_name(p) in bed_set)
    person_ids = tuple(i for i, p in enumerate(prompts) if _norm_name(p) in person_set)
    equip_ids = tuple(i for i, p in enumerate(prompts) if _norm_name(p) in equip_set)
    if not person_ids:
        person_ids = (0,)
    return bed_ids, person_ids, equip_ids, names


def _load_open_vocab(weights: str, model_cfg: dict, backend_name: str) -> DetectorSpec:
    prompts = list(model_cfg.get("prompts") or DEFAULT_WORLD_PROMPTS)
    bed_prompts = list(model_cfg.get("bed_prompts") or DEFAULT_BED_PROMPTS)
    person_prompts = list(model_cfg.get("person_prompts") or DEFAULT_PERSON_PROMPTS)
    equipment_prompts = list(model_cfg.get("equipment_prompts") or DEFAULT_EQUIPMENT_PROMPTS)

    # YOLOE 优先用 YOLOE 类；失败则 YOLO()
    model = None
    if backend_name == "yoloe":
        try:
            from ultralytics import YOLOE

            model = YOLOE(weights)
        except Exception:  # noqa: BLE001
            model = YOLO(weights)
    else:
        model = YOLO(weights)

    if hasattr(model, "eval"):
        try:
            model.eval()
        except Exception:  # noqa: BLE001
            pass

    if hasattr(model, "set_classes"):
        try:
            # YOLOE 可传 text PE；YOLO-World 仅 prompts
            if backend_name == "yoloe" and hasattr(model, "get_text_pe"):
                model.set_classes(prompts, model.get_text_pe(prompts))
            else:
                model.set_classes(prompts)
        except TypeError:
            model.set_classes(prompts)
    else:
        raise RuntimeError(f"权重 {weights} 不支持 set_classes；请使用 YOLO-World / YOLOE 模型")

    bed_ids, person_ids, equip_ids, names = resolve_prompt_class_ids(
        prompts, bed_prompts, person_prompts, equipment_prompts
    )
    print(f"[model] backend={backend_name} weights={weights}")
    print(f"[model] prompts={prompts}")
    print(f"[model] bed_ids={bed_ids} person_ids={person_ids} equipment_ids={equip_ids}")
    return DetectorSpec(
        backend=backend_name,
        model=model,
        names=names,
        bed_class_ids=bed_ids,
        person_class_ids=person_ids,
        equipment_class_ids=equip_ids,
        detect_classes=None,
    )


def _load_coco(weights: str, model_cfg: dict) -> DetectorSpec:
    model = YOLO(weights)
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    classes = model_cfg.get("classes")
    detect_classes = list(classes) if classes is not None else [0, *COCO_FURNITURE_IDS]
    bed_ids = tuple(model_cfg.get("bed_class_ids") or [59])
    extra = tuple(model_cfg.get("extra_bed_like_ids") or list(COCO_FURNITURE_IDS))
    person_ids = tuple(model_cfg.get("person_class_ids") or [0])
    print(f"[model] backend=coco weights={weights} classes={detect_classes}")
    return DetectorSpec(
        backend="coco",
        model=model,
        names=names,
        bed_class_ids=tuple(sorted(set(bed_ids + extra))),
        person_class_ids=person_ids,
        detect_classes=detect_classes,
    )


def load_detector(model_cfg: dict) -> DetectorSpec:
    """按配置加载检测器，并解析 bed/person 类别 id。"""
    backend = str(model_cfg.get("backend") or "fusion").strip().lower()
    weights = str(model_cfg.get("weights") or "yolov8s-worldv2.pt")

    if backend in ("fusion", "world+coco", "hybrid"):
        # 主：开放词汇；辅：COCO 家具/人，补推床漏检
        primary_backend = str(model_cfg.get("primary_backend") or "world").lower()
        primary_weights = str(model_cfg.get("weights") or "yolov8s-worldv2.pt")
        if primary_backend == "yoloe":
            primary_weights = str(model_cfg.get("weights") or "yoloe-11s-seg.pt")
            primary = _load_open_vocab(primary_weights, model_cfg, "yoloe")
        else:
            primary = _load_open_vocab(primary_weights, model_cfg, "world")

        sec_weights = str(model_cfg.get("secondary_weights") or "yolo12n.pt")
        secondary = _load_coco(sec_weights, model_cfg)
        offset = int(model_cfg.get("secondary_id_offset") or 1000)
        # 合并 names：secondary 用 offset 后的 id
        names = dict(primary.names)
        for cid, nm in secondary.names.items():
            names[offset + int(cid)] = f"coco:{nm}"
        bed_ids = tuple(primary.bed_class_ids) + tuple(offset + i for i in secondary.bed_class_ids)
        person_ids = tuple(primary.person_class_ids) + tuple(offset + i for i in secondary.person_class_ids)
        equip_ids = tuple(primary.equipment_class_ids)
        print(f"[model] backend=fusion primary={primary.backend} secondary=coco({sec_weights})")
        return DetectorSpec(
            backend="fusion",
            model=primary.model,
            names=names,
            bed_class_ids=bed_ids,
            person_class_ids=person_ids,
            equipment_class_ids=equip_ids,
            detect_classes=None,
            secondary_model=secondary.model,
            secondary_detect_classes=secondary.detect_classes,
            secondary_bed_class_ids=secondary.bed_class_ids,
            secondary_person_class_ids=secondary.person_class_ids,
            secondary_names=secondary.names if isinstance(secondary.names, dict) else {},
            secondary_id_offset=offset,
        )

    if backend in ("world", "yolo-world", "open_vocab", "open-vocab"):
        return _load_open_vocab(weights, model_cfg, "world")

    if backend in ("yoloe",):
        return _load_open_vocab(str(model_cfg.get("weights") or "yoloe-11s-seg.pt"), model_cfg, "yoloe")

    return _load_coco(weights, model_cfg)
