"""
Backend engine definitions
"""
import abc
import enum


class BackendType(enum.Enum):
    """Backend type enum"""

    TORCH = 0
    NUMPY = 1


class BackendNotSupported(Exception):
    """Backend not supported exception"""

    def __init__(self, backend):
        """
        Args:
            backend (str): backend name
        """
        super().__init__(f"Backend {backend} not supported")


class BackendEngine(abc.ABC):
    """Backend engine abstract class"""

    def __init__(self):
        self.type = None
        self.module = None

    def __str__(self):
        if self.type:
            return self.type.name.lower()
        return "unknown"


class PyTorchBackend(BackendEngine):
    """PyTorch backend engine"""

    def __init__(self):
        import torch

        self.type = BackendType.TORCH
        self.module = torch


class NumPyBackend(BackendEngine):
    """NumPy backend engine"""

    def __init__(self):
        import numpy

        self.type = BackendType.NUMPY
        self.module = numpy
