"""
RKMJ Runtime: Hardware & Model-Scale Adaptive Dynamic Execution Architecture.
"""

from rkmj.runtime.profiler import (
    ExecutionTier,
    SystemHardwareProfiler,
    ModelFootprintEstimator,
    DynamicExecutionRouter,
)
from rkmj.runtime.streamer import (
    MemoryGovernor,
    LayerWiseStreamer,
)

try:
    from rkmj._C import AsyncRingStreamer
except ImportError:
    AsyncRingStreamer = None

__all__ = [
    "ExecutionTier",
    "SystemHardwareProfiler",
    "ModelFootprintEstimator",
    "DynamicExecutionRouter",
    "MemoryGovernor",
    "LayerWiseStreamer",
    "AsyncRingStreamer",
]
