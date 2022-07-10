from typing import Optional

from approx.core.backend._backend import BackendEngine
from approx.core.device._device import DeviceEngine

_APPROX_BACKEND: Optional[BackendEngine] = None
_APPROX_DEVICE: Optional[DeviceEngine] = None


def backend() -> Optional[BackendEngine]:
    global _APPROX_BACKEND
    return _APPROX_BACKEND


def device() -> Optional[DeviceEngine]:
    global _APPROX_DEVICE
    return _APPROX_DEVICE
