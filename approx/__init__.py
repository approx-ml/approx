from approx.core._vars import backend, device
from approx.core.api import auto_quantize, auto_set_backend, auto_set_device

# Please keep this sorted
__all__ = [
    "auto_quantize",
    "auto_set_backend",
    "auto_set_device",
    "backend",
    "device",
]

assert __all__ == sorted(__all__), "Public API methods are not sorted!"
