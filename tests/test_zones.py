"""门口线 / ROI 穿越单元测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from or_io.events import EventManager
from or_io.zones import DoorLine, DualROIZones, build_zone, side_transition


def test_vertical_line_enter_exit():
    door = DoorLine.from_normalized([0.5, 0.1], [0.5, 0.9], 1000, 600, outside_side="left")
    assert door.classify(200, 300) == "outside"
    assert door.classify(800, 300) == "inside"


def test_dual_roi_classify():
    zone = DualROIZones.from_normalized(
        [[0, 0], [0.45, 0], [0.45, 1], [0, 1]],
        [[0.55, 0], [1, 0], [1, 1], [0.55, 1]],
        1000, 600,
    )
    assert zone.classify(100, 300) == "outside"
    assert zone.classify(900, 300) == "inside"


def test_side_transition_enter():
    evt, stable = side_transition(None, "outside")
    assert stable == "outside"
    evt, stable = side_transition(stable, "inside")
    assert evt == "enter"


if __name__ == "__main__":
    test_vertical_line_enter_exit()
    test_dual_roi_classify()
    test_side_transition_enter()
    print("tests passed")
