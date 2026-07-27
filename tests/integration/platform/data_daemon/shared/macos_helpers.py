import ctypes
import ctypes.util
import sys

_DARWIN = sys.platform == "darwin"
_PERIOD_S = 0.004
_COMPUTATION_S = 0.0025
_CONSTRAINT_S = 0.003

if _DARWIN:
    _lib = ctypes.CDLL(ctypes.util.find_library("c"))

    class _Timebase(ctypes.Structure):
        _fields_ = [("numer", ctypes.c_uint32), ("denom", ctypes.c_uint32)]

    class _TimeConstraint(ctypes.Structure):
        _fields_ = [
            ("period", ctypes.c_uint32),
            ("computation", ctypes.c_uint32),
            ("constraint", ctypes.c_uint32),
            ("preemptible", ctypes.c_int),
        ]

    _lib.mach_timebase_info.argtypes = [ctypes.POINTER(_Timebase)]
    _lib.mach_thread_self.restype = ctypes.c_uint
    _lib.mach_task_self.restype = ctypes.c_uint
    _lib.mach_port_deallocate.argtypes = [ctypes.c_uint, ctypes.c_uint]
    _lib.thread_policy_set.argtypes = [
        ctypes.c_uint,
        ctypes.c_int,
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    _lib.thread_policy_set.restype = ctypes.c_int

    _tb = _Timebase()
    _lib.mach_timebase_info(ctypes.byref(_tb))


def _set_thread_rt(period_s, computation_s, constraint_s, preemptible=True):
    """Apply RT time-constraint policy to the CALLING thread. No-op off macOS.

    Call inside the thread's target function, not at import.
    Returns True if applied, False if not on darwin. Raises on kernel error.
    """
    if not _DARWIN:
        return False

    def to_abs(sec):
        return int(sec * 1e9) * _tb.denom // _tb.numer

    pol = _TimeConstraint(
        period=to_abs(period_s),
        computation=to_abs(computation_s),
        constraint=to_abs(constraint_s),
        preemptible=int(preemptible),
    )
    th = _lib.mach_thread_self()
    try:
        kr = _lib.thread_policy_set(th, 2, ctypes.byref(pol), 4)
        if kr != 0:
            raise OSError(f"thread_policy_set failed: kern_return_t={kr}")
    finally:
        _lib.mach_port_deallocate(_lib.mach_task_self(), th)
    return True


def set_thread_policy_for_macos():
    """MacOs Scheduler doesn't provide the real time performance necessary for the
    preciseness of stochastic tests. Set up a thread policy to enable real time
    performance. Policy has to be set for each new thread.
    """
    _set_thread_rt(_PERIOD_S, _COMPUTATION_S, _CONSTRAINT_S)
