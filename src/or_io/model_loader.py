"""检测模型加载：COCO YOLO / YOLO-World 开放词汇（免训练）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ultralytics import YOLO

# 开放词汇默认提示：手术室推床 + 盖被只露头
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
    "bed",
    "mattress",
]

DEFAULT_BED_PROMPTS = (
    "hospital bed",
    "stretcher",
    "gurney",
    "medical bed",
    "hospital stretcher",
    "bed",
    "mattress",
)
DEFAULT_PERSON_PROMPTS = ("person", "human head", "face", "patient")


@dataclass
class DetectorSpec:
    backend: str  # coco | world
    model: YOLO
    names: Dict[int, str]
    bed_class_ids: Tuple[int, ...]
    person_class_ids: Tuple[int, ...]
    detect_classes: Optional[List[int]]


def _norm_name(s: str) -> str:
    return str(s).strip().lower()


def resolve_prompt_class_ids(
    prompts: Sequence[str],
    bed_prompts: Sequence[str],
    person_prompts: Sequence[str],
) -> Tuple[Tuple[int, ...], Tuple[int, ...], Dict[int, str]]:
    names = {i: p for i, p in enumerate(prompts)}
    bed_set = {_norm_name(x) for x in bed_prompts}
    person_set = {_norm_name(x) for x in person_prompts}
    bed_ids = tuple(i for i, p in enumerate(prompts) if _norm_name(p) in bed_set)
    person_ids = tuple(i for i, p in enumerate(prompts) if _norm_name(p) in person_set)
    if not person_ids:
        # 至少保留第一个提示为人
        person_ids = (0,)
    if not bed_ids:
        # 无床提示时退化为空，后续依赖 geometry 兜底
        bed_ids = tuple()
    return bed_ids, person_ids, names


def load_detector(model_cfg: dict) -> DetectorSpec:
    """按配置加载检测器，并解析 bed/person 类别 id。"""
    backend = str(model_cfg.get("backend") or "world").strip().lower()
    weights = str(model_cfg.get("weights") or "yolov8s-worldv2.pt")
    model = YOLO(weights)

    if backend in ("world", "yolo-world", "open_vocab", "open-vocab"):
        prompts = list(model_cfg.get("prompts") or DEFAULT_WORLD_PROMPTS)
        bed_prompts = list(model_cfg.get("bed_prompts") or DEFAULT_BED_PROMPTS)
        person_prompts = list(model_cfg.get("person_prompts") or DEFAULT_PERSON_PROMPTS)
        if hasattr(model, "set_classes"):
            model.set_classes(prompts)
        else:
            raise RuntimeError(
                f"权重 {weights} 不支持 set_classes；请改用 yolov8s-worldv2.pt 等 YOLO-World 模型"
            )
        bed_ids, person_ids, names = resolve_prompt_class_ids(prompts, bed_prompts, person_prompts)
        print(f"[model] backend=world weights={weights}")
        print(f"[model] prompts={prompts}")
        print(f"[model] bed_ids={bed_ids} person_ids={person_ids}")
        return DetectorSpec(
            backend="world",
            model=model,
            names=names,
            bed_class_ids=bed_ids,
            person_class_ids=person_ids,
            detect_classes=None,  # 词汇已由 set_classes 限定
        )

    # COCO 路径（兼容旧配置）
    names = model.names if isinstance(model.names, dict) else {i: n for i, n in enumerate(model.names)}
    classes = model_cfg.get("classes")
    detect_classes = list(classes) if classes is not None else [0, 56, 57, 59, 60]
    bed_ids = tuple(model_cfg.get("bed_class_ids") or [59])
    extra = tuple(model_cfg.get("extra_bed_like_ids") or [56, 57, 60])
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
