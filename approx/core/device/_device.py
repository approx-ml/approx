"""
Device engine definitions
"""
import abc
import enum


class DeviceType(enum.Enum):
    """Device type enum"""

    UNKNOWN = 0
    CPU = 0
    CUDA = 1  # CUDA
    XLA = 2  # XLA / TPU


class DeviceNotSupported(Exception):
    """Device not supported exception"""

    def __init__(self, device: str):
        """
        Args:
            device (str): device name
        """
        super().__init__(f"Device {device} not supported")


class DeviceEngine(abc.ABC):
    """Device engine abstract class"""

    def __init__(self) -> None:
        super().__init__()
        self.type: DeviceType = DeviceType.UNKNOWN

    def __str__(self) -> str:
        return self.type.name.lower()


class CPUDevice(DeviceEngine):
    """CPU device engine"""

    def __init__(self) -> None:
        super().__init__()
        self.type: DeviceType = DeviceType.CPU


class CUDADevice(DeviceEngine):
    """CUDA device engine"""

    def __init__(self) -> None:
        super().__init__()
        self.type: DeviceType = DeviceType.CUDA


class XLADevice(DeviceEngine):
    """XLA device engine"""

    def __init__(self) -> None:
        super().__init__()
        self.type: DeviceType = DeviceType.XLA
