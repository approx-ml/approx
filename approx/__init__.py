from approx.core._vars import backend, device
from approx.core.api import auto_quantize, auto_set_backend, compare

# Please keep this sorted
__all__ = [
    "auto_quantize",
    "auto_set_backend",
    "backend",
    "compare",
    "device",
]

assert __all__ == sorted(__all__), "Public API methods are not sorted!"
