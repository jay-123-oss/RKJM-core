"""
Core memory management, allocator abstractions, and OS supervisor for RKMJ-Core.
"""

from rkmj.core.supervisor import MemorySupervisor, get_supervisor
from rkmj.core.loader import load

def force_heap_trim() -> None:
    """Forces glibc and Python runtime to release unused heap pages back to the OS."""
    import gc
    gc.collect()
    try:
        from rkmj import _C
        if hasattr(_C, "force_heap_trim"):
            _C.force_heap_trim()
            return
    except Exception:
        pass
    import sys
    if sys.platform.startswith("linux"):
        try:
            import ctypes
            libc = ctypes.CDLL("libc.so.6")
            if hasattr(libc, "malloc_trim"):
                libc.malloc_trim(0)
        except Exception:
            pass

__all__ = [
    "MemorySupervisor",
    "get_supervisor",
    "load",
    "force_heap_trim",
]
