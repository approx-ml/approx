"""
Device engine definitions
"""
import abc
import enum


class DeviceType(enum.Enum):
    """Device type enum"""

    CPU = 0
    CUDA = 1  # CUDA
    XLA = 2  # XLA / TPU


class DeviceNotSupported(Exception):
    """Device not supported exception"""

    def __init__(self, device):
        """
        Args:
            device (str): device name
        """
        super().__init__(f"Device {device} not supported")


class DeviceEngine(abc.ABC):
    """Device engine abstract class"""

    def __init__(self):
        self.type = None

    def __str__(self):
        if self.type:
            return self.type.name.lower()
        return "unknown"


class CPUDevice(DeviceEngine):
    """CPU device engine"""

    def __init__(self):
        self.type = DeviceType.CPU


class CUDADevice(DeviceEngine):
    """CUDA device engine"""

    def __init__(self):
        self.type = DeviceType.CUDA


class XLADevice(DeviceEngine):
    """XLA device engine"""

    def __init__(self):
        self.type = DeviceType.XLA
