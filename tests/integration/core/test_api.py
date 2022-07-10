"""
Test Public API
"""


def test_auto_set_backend():
    """Tests that the auto_set_backend function works as expected."""
    from approx.core.api import auto_set_backend
    from approx.core.backend._backend import BackendType

    backend_engine = auto_set_backend()
    assert backend_engine.type == BackendType.NUMPY
    assert str(backend_engine) == "numpy"


def test_set_backend():
    """Tests that the set_backend function works as expected."""
    from approx.core.backend import set_backend
    from approx.core.backend._backend import BackendType

    backend_types = [
        attr
        for attr in dir(BackendType)
        if (
            not callable(getattr(BackendType, attr))
            and not attr.startswith("__")
        )
    ]
    for backend_type in backend_types:
        print(BackendType[backend_type])
        backend_engine = set_backend(BackendType[backend_type])
        assert backend_engine.type == BackendType[backend_type]
        assert str(backend_engine) == backend_type.lower()


def test_set_device():
    """Tests that the set_device function works as expected."""
    from approx.core.device import set_device
    from approx.core.device._device import DeviceType

    device_types = [
        attr
        for attr in dir(DeviceType)
        if (
            not callable(getattr(DeviceType, attr))
            and not attr.startswith("__")
        )
    ]
    for device_type in device_types:
        device_engine = set_device(DeviceType[device_type])
        assert device_engine.type == DeviceType[device_type]
        assert str(device_engine) == device_type.lower()
