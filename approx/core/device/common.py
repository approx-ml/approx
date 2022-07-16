"""
Common backend handling
"""


from approx.core.device._device import (
    DeviceEngine,
    DeviceNotSupported,
    DeviceType,
)


def auto_select_device() -> DeviceEngine:
    """Automatically selects the device necessary

    Raises:
        DeviceNotSupported: Could not find a device.

    Returns:
        DeviceEngine: An instance of whatever device is most appropriate.
    """
    # backend_types = [
    #     attr for attr in dir(BackendType)
    #     if (
    #         not callable(getattr(BackendType, attr))
    #         and not attr.startswith("__")
    #     )
    # ]
    # for backend_type in backend_types:
    #     try:
    #         subprocess.check_output('nvidia-smi')

    raise DeviceNotSupported(
        f"This device is currently not supported. "
        f"Supported devices: {[d.name for d in DeviceType]}"
    )


def set_device(choice: DeviceType) -> DeviceEngine:
    """Sets the device to use.

    Args:
        choice (DeviceType): The device to use. (e.g. DeviceType.CUDA)

    Raises:
        DeviceNotSupported: When the device is not supported.

    Returns:
        DeviceEngine: An instance of whatever device is most appropriate.
    """
    if choice == DeviceType.CUDA:
        from approx.core.device._device import CUDADevice

        return CUDADevice()

    elif choice == DeviceType.XLA:
        from approx.core.device._device import XLADevice

        return XLADevice()

    # if asked for a device that doesn't exist, raise an error
    elif choice not in DeviceType:
        raise DeviceNotSupported(str(choice))

    # if no device was asked for, set device to CPU
    else:
        from approx.core.device._device import CPUDevice

        return CPUDevice()
