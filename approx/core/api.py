"""
Contains stuff to be exported in the public API.
"""
from approx.core.backend.common import BackendEngine, auto_select_backend
from approx.core.device.common import DeviceEngine, auto_select_device


def auto_cast_all(*args) -> None:
    """Automatically casts any tensors for automatic quantization

    Returns:
        None:
    """
    # todo: unimplemented
    pass


def auto_set_backend() -> BackendEngine:
    """Automatically selects an appropriate backend to utilize.

    Returns:
        BackendEngine: An instance of whatever backend is most appropriate.
    """
    return auto_select_backend()


def auto_set_device() -> DeviceEngine:
    """Automatically selects an appropriate device to utilize.

    Returns:
        DeviceEngine: An instance of whatever device is most appropriate.
    """
    return auto_select_device()
