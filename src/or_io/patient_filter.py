"""兼容旧导入路径。"""

from .stretcher_filter import TargetFilter as PatientFilter
from .stretcher_filter import TargetFilter

__all__ = ["PatientFilter", "TargetFilter"]
