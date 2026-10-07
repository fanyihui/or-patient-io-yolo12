"""手术床自动识别单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.or_bed_detect import (
    AutoOrBedConfig,
    AutoOrBedDetector,
    resolve_or_table_class_ids,
)
from or_io.stretcher_filter import Detection


def test_resolve_or_table_ids():
    names = {0: "person", 5: "operating table", 6: "stretcher", 7: "OR table"}
    ids = resolve_or_table_class_ids(names)
    assert 5 in ids and 7 in ids
    assert 6 not in ids


def test_auto_lock_stationary_bed():
    det = AutoOrBedDetector(
        cfg=AutoOrBedConfig(
            enabled=True,
            lock_frames=8,
            max_center_motion_ratio=0.03,
            min_area_ratio=0.03,
            history=20,
        ),
        or_table_class_ids=(10,),
        bed_class_ids={10, 59},
    )
    # 固定手术台类，位置几乎不动
    poly = None
    for f in range(20):
        jitter = (f % 2) * 1.0
        d = Detection(3, 10, 0.9, (400 + jitter, 300, 700 + jitter, 480))
        poly = det.update(frame_idx=f, frame_w=960, frame_h=540, detections=[d], roles={3: "bed"})
    assert poly is not None
    assert det.status == "locked"
    assert det.locked_from == "or_table_class"
    assert det.locked_track_id == 3


def test_moving_stretcher_not_locked_quickly():
    det = AutoOrBedDetector(
        cfg=AutoOrBedConfig(enabled=True, lock_frames=10, max_center_motion_ratio=0.02, min_area_ratio=0.03),
        or_table_class_ids=(),
        bed_class_ids={59},
    )
    poly = None
    for f in range(15):
        # 明显在移动的推床
        x = 100 + f * 25
        d = Detection(8, 59, 0.88, (x, 300, x + 280, 460))
        poly = det.update(frame_idx=f, frame_w=960, frame_h=540, detections=[d], roles={8: "bed"})
    assert poly is None
    assert det.status == "searching"


def test_manual_priority_config_disables_auto_by_default():
    from or_io.or_bed_detect import auto_or_bed_config_from_dict

    cfg = auto_or_bed_config_from_dict({"auto_detect": True, "lock_frames": 12})
    assert cfg.enabled is True
    assert cfg.lock_frames == 12


if __name__ == "__main__":
    test_resolve_or_table_ids()
    test_auto_lock_stationary_bed()
    test_moving_stretcher_not_locked_quickly()
    test_manual_priority_config_disables_auto_by_default()
    print("or_bed auto tests passed")
