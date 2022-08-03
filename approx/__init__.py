from approx.core._vars import backend, device
from approx.core.api import (
    auto_quantize,
    auto_set_backend,
    auto_set_device,
    compare,
)
from approx.core.compare import Metric as CompareMetric

# Please keep this sorted
__all__ = [
    "CompareMetric",
    "auto_quantize",
    "auto_set_backend",
    "auto_set_device",
    "backend",
    "compare",
    "device",
]

assert __all__ == sorted(__all__), "Public API methods are not sorted!"
