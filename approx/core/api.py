"""
Contains stuff to be exported in the public API.
"""
from approx.core.backend.common import auto_select_backend
from approx.core.device.common import DeviceEngine, auto_select_device
from approx.core._vars import _APPROX_BACKEND


def auto_quantize(model, pretrained=True):
    """Turns a normal model into a quantized model, using an appropriate backend

    Args:
        model: The model to quantize.
        pretrained: Whether this model is pretrained
    """

    qmodel = _APPROX_BACKEND.auto_quantize(model, pretrained)
    return qmodel


def auto_cast_all(*args) -> None:
    """Automatically casts any tensors for automatic quantization

    Returns:
        None:
    """
    # todo: unimplemented
    pass


def auto_set_backend() -> None:
    """Automatically selects an appropriate backend to utilize.

    Returns:
        None.
    """
    global _APPROX_BACKEND
    _APPROX_BACKEND = auto_select_backend()


def auto_set_device() -> DeviceEngine:
    """Automatically selects an appropriate device to utilize.

    Returns:
        DeviceEngine: An instance of whatever device is most appropriate.
    """
    return auto_select_device()
