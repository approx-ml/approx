from approx.core.api import auto_cast_all, auto_set_backend, auto_quantize,\
    auto_set_device

# Please keep this sorted
__all__ = [
    "auto_cast_all",
    "auto_quantize",
    "auto_set_backend",
    "auto_set_device",
]

assert __all__ == sorted(__all__), "Public API methods are not sorted!"
