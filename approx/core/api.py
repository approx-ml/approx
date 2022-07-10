"""
Contains stuff to be exported in the public API.
"""
from approx.core import _vars
from approx.core.backend.common import auto_select_backend
from approx.core.device.common import DeviceEngine, auto_select_device


def auto_quantize(model, pretrained=True):
    """Turns a normal model into a quantized model, using an appropriate backend

    Args:
        model: The model to quantize.
        pretrained: Whether this model is pretrained
    """

    qmodel = _vars._APPROX_BACKEND.auto_quantize(model, pretrained)
    return qmodel


def auto_cast_all(*args) -> None:
    """Automatically casts any tensors for automatic quantization

    Returns:
        None:
    """
    # todo: unimplemented
    pass


def auto_set_backend() -> None:
    """Automatically sets an appropriate backend for `approx` to use.

    Returns:
        None.
    """

    _vars._APPROX_BACKEND = auto_select_backend()


def auto_set_device() -> None:
    """Automatically sets an appropriate device for `approx` to use.

    Returns:
        None.
    """
    _vars._APPROX_DEVICE = auto_select_device()
