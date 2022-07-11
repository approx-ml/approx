"""
Backend engine definitions
"""
import abc
import enum
from typing import Any


class BackendType(enum.Enum):
    """Backend type enum"""

    UNKNOWN = -1
    TORCH = 0
    NUMPY = 1


class BackendNotSupported(Exception):
    """Backend not supported exception"""

    def __init__(self, backend: str) -> None:
        """
        Args:
            backend (str): backend name
        """
        super().__init__(f"Backend {backend} not supported")


class BackendEngine(abc.ABC):
    """Backend engine abstract class"""

    def __init__(self) -> None:
        super().__init__()
        self.type: BackendType = BackendType.UNKNOWN
        self.module: Any = None

    def auto_quantize(self, model: Any, pretrained: bool) -> Any:
        raise NotImplementedError("Need to implement this method")

    def __str__(self) -> str:
        return self.type.name.lower()


class PyTorchBackend(BackendEngine):
    """PyTorch backend engine"""

    def __init__(self) -> None:
        super().__init__()
        import torch

        self.type = BackendType.TORCH
        self.module = torch

    def auto_quantize(self, model: Any, pretrained: bool) -> Any:
        """
        Uses pytorch's dynamic quantization to convert some floating-point
        `nn.Module` to `qint8`
        Args:
            model: The model to quantize.
            pretrained: Whether the model is pretrained.

        Returns:
            The quantized model
        """
        if pretrained:
            qmodel = self.module.quantization.quantize_dynamic(
                model,
                {
                    # todo: this is definitely not enough
                    self.module.nn.Linear,
                    self.module.nn.Conv2d,
                },
                dtype=self.module.qint8,
            )
            return qmodel
        raise NotImplementedError(
            "Currently not implemented for non-pretrained models"
        )


class NumPyBackend(BackendEngine):
    """NumPy backend engine"""

    def __init__(self) -> None:
        super().__init__()
        import numpy

        self.type = BackendType.NUMPY
        self.module = numpy
